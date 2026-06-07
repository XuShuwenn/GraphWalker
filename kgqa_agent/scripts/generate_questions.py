#!/usr/bin/env python3
from __future__ import annotations
import argparse
import glob
import json
import os
from typing import Any, Dict, Iterable, Set
from kgqa_agent.src.data_gen.generate_questions import LLMClient, LLMConfig, generate_questions, load_prompt_module

def iter_jsonl(pattern: str):
    for p in sorted(glob.glob(pattern)):
        with open(p, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def _parse_jsonl_ids(path: str, id_key: str = "path_id") -> Set[str]:
    seen: Set[str] = set()
    if not os.path.exists(path):
        return seen
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict) and id_key in obj and obj[id_key] is not None:
                    seen.add(str(obj[id_key]))
            except Exception:
                continue
    return seen

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--prompt-module', help='Python module path or file to load PROMPT_SYSTEM and PROMPT_USER')
    args = parser.parse_args()

    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    llm_cfg = LLMConfig(**cfg['llm'])
    client = LLMClient(llm_cfg)
    paths = iter_jsonl(cfg['input_paths'])

    prompt_mod = args.prompt_module or cfg.get('prompt_module')
    if prompt_mod:
        sys_prompt, user_tpl = load_prompt_module(prompt_mod)
    else:
        sys_prompt = cfg['prompt']['system']
        user_tpl = cfg['prompt']['user_template']

    out_file = cfg['output_file']
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    # Resume by path_id: read existing JSONL and skip duplicates
    processed_ids = _parse_jsonl_ids(out_file, id_key='path_id')
    if processed_ids:
        print(f"[RESUME] found {len(processed_ids)} existing records in {out_file}")

    wrote = 0
    since_sync = 0
    with open(out_file, 'a', encoding='utf-8') as f:
        for rec in generate_questions(paths, client, sys_prompt, user_tpl):
            try:
                pid = rec.get('path_id') if isinstance(rec, dict) else None  # type: ignore[attr-defined]
            except Exception:
                pid = None
            if pid is not None and str(pid) in processed_ids:
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            wrote += 1
            since_sync += 1
            if since_sync >= 10:
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
                since_sync = 0
        # final flush
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass

    print(f"wrote {wrote} new QA records -> {out_file}")

if __name__ == '__main__':
    import os
    main()
