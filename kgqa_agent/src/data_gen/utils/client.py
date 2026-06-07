"""Minimal LLM client for plan + top-k filtering (OpenAI-compatible API).

This module is moved from data_gen/llm_filter/client.py to data_gen/utils to
centralize reusable utilities.
"""
from __future__ import annotations

import ast
import json
import os
import re
import time
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()
import requests


def _candidate_key(item: Dict[str, Any]) -> str:
  vid = (item.get("id") or "").strip()
  vname = (item.get("name") or "").strip()
  return vid if vid else vname


@dataclass(frozen=True)
class LLMConfig:
  """Centralized LLM configuration (minimal, readable)."""
  model: str
  base_url: str
  api_key: str
  temperature: float = 0.7
  max_tokens: int = 12288
  use_proxy: bool = True

  @staticmethod
  def from_env() -> "LLMConfig":
    # model = "qwen3-235b-a22b-instruct-2507"
    model = "gemini-2.5-flash"
    base = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    use_proxy = (os.getenv("LLM_USE_PROXY") or "1").lower() not in {"0", "false"}
    return LLMConfig(
      model=model,
      base_url=base.rstrip("/"),
      api_key=api_key,
      use_proxy=use_proxy,
    )


# Single place to resolve defaults from environment
DEFAULT_CONFIG = LLMConfig.from_env()


class LLMClient:
  """Call an OpenAI-compatible Chat Completions API and parse the plan/keep output.

  Configuration: pass an LLMConfig explicitly, or rely on DEFAULT_CONFIG above.
  """

  def __init__(
    self,
    *,
    config: LLMConfig | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    ) -> None:
    # Prefer explicit config; else build from provided args over DEFAULT_CONFIG.
    if config is not None:
      self.config = config
    elif any(v is not None for v in (model, base_url, api_key, temperature, max_tokens)):
      self.config = LLMConfig(
        model=model or DEFAULT_CONFIG.model,
        base_url=(base_url or DEFAULT_CONFIG.base_url).rstrip("/"),
        api_key=api_key or DEFAULT_CONFIG.api_key,
        temperature=DEFAULT_CONFIG.temperature if temperature is None else float(temperature),
        max_tokens=DEFAULT_CONFIG.max_tokens if max_tokens is None else int(max_tokens),
      )
    else:
      self.config = DEFAULT_CONFIG
    # Reuse a single session to reduce TLS handshakes and flakiness
    self._session = requests.Session()
    if not getattr(self.config, "use_proxy", True):
      self._session.trust_env = False

  def _chat(self, messages: List[Dict[str, str]], temperature: float | None = None, max_tokens: int | None = None) -> str:
    """Chat with robust retries for transient HTTP and network errors.

    Retries on: 429, 500, 502, 503, 504 and network errors, with exponential backoff + jitter.
    """
    # Support base_url with or without trailing /v1
    base = self.config.base_url.rstrip("/")
    url = f"{base}/chat/completions" if base.endswith("/v1") else f"{base}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"}
    # Resolve generation params
    temp = self.config.temperature if temperature is None else float(temperature)
    mtoks = self.config.max_tokens if max_tokens is None else int(max_tokens)
    payload = {
      "model": self.config.model,
      "messages": messages,
      "temperature": temp,
      "max_tokens": mtoks,
    }

    def _is_transient(status: int | None) -> bool:
      return status in {429, 500, 502, 503, 504}

    max_attempts = 6
    for attempt in range(1, max_attempts + 1):
      try:
        resp = self._session.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code >= 400:
          # Try to parse body once for helpful context
          body_text = resp.text or ""
          if _is_transient(resp.status_code) and attempt < max_attempts:
            # Exponential backoff with jitter
            backoff = min(2 ** attempt, 30)
            jitter = 0.25 * (attempt)
            time.sleep(backoff + jitter)
            continue
          # Non-transient or exhausted retries
          raise RuntimeError(f"HTTP error from LLM (status={resp.status_code}): {body_text}")

        try:
          data = resp.json()
        except Exception:
          text = resp.text
          # Not JSON — transient? usually not; treat as retryable once if attempts remain
          if attempt < max_attempts:
            backoff = min(2 ** attempt, 30)
            time.sleep(backoff)
            continue
          raise RuntimeError(f"LLM response is not JSON (status={resp.status_code}): {text}")

        # Common OpenAI-like shape: choices -> [ { message: { content: ... } } ]
        if isinstance(data, dict) and "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
          choice = data["choices"][0]
          if isinstance(choice, dict):
            msg = choice.get("message")
            if isinstance(msg, dict) and msg.get("content"):
              return str(msg.get("content")).strip()
            if choice.get("content"):
              return str(choice.get("content")).strip()
            if choice.get("text"):
              return str(choice.get("text")).strip()

        # Provider-specific single-field responses
        if isinstance(data, dict):
          for key in ("output_text", "output", "text", "result"):
            if key in data and isinstance(data[key], str) and data[key].strip():
              return data[key].strip()

        # If we reach here, the response did not contain any expected text fields.
        raise RuntimeError(
          "LLM did not return text in any known field; "
          f"status={resp.status_code}; body={json.dumps(data, ensure_ascii=False)[:4000]}"
        )
      except requests.exceptions.RequestException as e:
        # Network errors (timeouts, DNS, connection) -> retry if attempts remain
        status = getattr(getattr(e, 'response', None), 'status_code', None)
        body = getattr(getattr(e, 'response', None), 'text', None)
        if (status is None or _is_transient(status)) and attempt < max_attempts:
          backoff = min(2 ** attempt, 30)
          jitter = 0.25 * (attempt)
          time.sleep(backoff + jitter)
          continue
        # Exhausted or non-transient: raise with context when available
        if body:
          raise RuntimeError(f"HTTP error from LLM (status={status}): {body}")
        raise

  @staticmethod
  def parse_plan_keep(text: str) -> Tuple[str, List[str]]:
    """Parse LLM output into (plan, keep).

    Supports two formats (tag-first):
    1) Tag format (preferred):
       <plan>...sentence...</plan>\n<keep>["k1", "k2"]</keep>
    2) Legacy two-line format:
       Plan: ...\nKeep: ["k1", "k2"]
    """
    plan = ""
    keep: List[str] = []

    # 1) Tag-based
    plan_m = re.search(r"<\s*plan\s*>(.*?)<\s*/\s*plan\s*>", text, flags=re.I | re.S)
    if plan_m:
      plan = plan_m.group(1).strip()
    keep_m = re.search(r"<\s*keep\s*>(.*?)<\s*/\s*keep\s*>", text, flags=re.I | re.S)
    if keep_m:
      inner = keep_m.group(1).strip()
      list_m = re.search(r"\[.*?\]", inner, flags=re.S)
      if list_m:
        try:
          keep = [str(x) for x in ast.literal_eval(list_m.group(0))]
        except Exception:
          keep = []
    if plan or keep:
      return plan, keep

    # 2) Legacy Plan:/Keep:
    plan_line = re.search(r"(?im)^\s*plan:\s*(.+)$", text)
    keep_line = re.search(r"(?im)^\s*keep:\s*(.+)$", text)
    if plan_line:
      plan = plan_line.group(1).strip()
    if keep_line:
      lm = re.search(r"\[.*?\]", keep_line.group(1))
      if lm:
        try:
          keep = [str(x) for x in ast.literal_eval(lm.group(0))]
        except Exception:
          keep = []
    return plan, keep

  def plan_and_filter(
    self,
    *,
    raw_question: str,
    name_path: str,
    step: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    golden: Dict[str, Any],
    k: int,
    prompt_template: str,
  ) -> Dict[str, Any]:
    # Prepare prompt input block
    user_block = {
      "raw_question": raw_question,
      "name_path": name_path,
      "step": {key: step[key] for key in ("step_index", "query_type", "current", "args") if key in step},
      "candidates": [{"id": c.get("id"), "name": c.get("name")} for c in candidates],
      "golden": {"id": golden.get("id"), "name": golden.get("name")},
      "k": int(k),
    }
    content = f"{prompt_template}\n\nINPUT:\n{json.dumps(user_block, ensure_ascii=False, indent=2)}"

    messages = [
      {"role": "system", "content": "You generate concise plans and top-k keep lists with strict tagged output (<plan>/<keep>)."},
      {"role": "user", "content": content},
    ]
    text = self._chat(messages)
    plan, keep = self.parse_plan_keep(text)

    # Enforce constraints: include golden, at most k, only known keys
    cand_keys = {_candidate_key(c) for c in candidates}
    golden_key = _candidate_key(golden)
    keep = [x for x in keep if x in cand_keys]
    if golden_key and golden_key not in keep and golden_key in cand_keys:
      keep.append(golden_key)
    # Deduplicate (preserve order) and cap to k
    keep_unique = list(dict.fromkeys(keep))[: max(0, int(k))]

    return {"plan": plan, "keep_keys": keep_unique}
