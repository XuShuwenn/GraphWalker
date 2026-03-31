#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CWQ predicate extractor (per-file + merged outputs).

What it does:
- For each input JSON (CWQ-style, with a "sparql" field): extract predicates from SPARQL triples
  and emit TWO files per input:
    1) predicate+counts (keys are ns:/rdf: prefixed tokens when possible; otherwise full IRI)
    2) URI+counts (keys are full IRIs)
- After processing all inputs, emit TWO merged files across all inputs:
    - cwq_all_predicates_counts.json
    - cwq_all_uri_counts.json

Notes:
- Supports JSON array and JSONL inputs.
- Deduplication is inherent in the counts maps (one key per predicate/URI).
- Configure input/output directories via CONFIG and run:
    python scripts/extract_cwq_predicates.py
"""

import json
import os
import re
import sys
import glob
import argparse
from typing import Iterable, Dict, Any, Tuple, Set, List, Optional
from tqdm import tqdm


# Remove comments, PREFIX/BASE declarations, and normalize 'a' to rdf:type
PREFIX_DECL_RE = re.compile(r"(?is)\bPREFIX\s+[A-Za-z][\w-]*:\s*<[^>]+>\s*")
BASE_DECL_RE   = re.compile(r"(?is)\bBASE\s*<[^>]+>\s*")
COMMENT_RE     = re.compile(r"#.*?$")
A_PREDICATE_RE = re.compile(r"(?<![\w:])a(?![\w:])")

# Triple pattern: subject SP predicate SP object — capture the predicate (middle position)
TRIPLE_RE = re.compile(
    r"(?is)"                              # case-insensitive, dot matches newline
    r"(?:^|[\{\s.;])"                   # statement or block start
    r"(?:<[^>]+>|\?[A-Za-z_]\w*|[A-Za-z][\w-]*:[A-Za-z0-9_.-]+|:[A-Za-z0-9_.-]+)"  # subject (added empty prefix support)
    r"\s+"
    r"(<[^>]+>|[A-Za-z][\w-]*:[A-Za-z0-9_.-]+|:[A-Za-z0-9_.-]+|a)"                  # predicate (capture, added empty prefix support)
    r"\s+"
    r"(?:<[^>]+>|\?[A-Za-z_]\w*|\".*?\"(?:\^\^<[^>]+>|@[-\w]+)?|[A-Za-z][\w-]*:[A-Za-z0-9_.-]+|:[A-Za-z0-9_.-]+)"  # object (added empty prefix support)
)

# Property path pattern: matches paths like ns:rel1/ns:rel2, ns:rel+, ns:rel*, ns:rel?
# This captures property paths that are not standard triple patterns
PROPERTY_PATH_RE = re.compile(
    r"(?is)"
    r"(?:^|[\{\s.;])"                   # statement or block start
    r"(?:<[^>]+>|\?[A-Za-z_]\w*|[A-Za-z][\w-]*:[A-Za-z0-9_.-]+|:[A-Za-z0-9_.-]+)"  # subject (added empty prefix support)
    r"\s+"
    r"((?:<[^>]+>|[A-Za-z][\w-]*:[A-Za-z0-9_.-]+|:[A-Za-z0-9_.-]+)(?:[/|](?:<[^>]+>|[A-Za-z][\w-]*:[A-Za-z0-9_.-]+|:[A-Za-z0-9_.-]+))*(?:[+*?])?)"  # property path (capture, added empty prefix support)
    r"\s+"
    r"(?:<[^>]+>|\?[A-Za-z_]\w*|\".*?\"(?:\^\^<[^>]+>|@[-\w]+)?|[A-Za-z][\w-]*:[A-Za-z0-9_.-]+|:[A-Za-z0-9_.-]+)"  # object (added empty prefix support)
)

EXCLUDE_PREFIXES = {}  # excluded entity prefixes (e.g. m:, g:)

# Common namespace map (covers Freebase and rdf used in CWQ and GrailQA)
FREEBASE_NS = "http://rdf.freebase.com/ns/"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
DEFAULT_PREFIX_MAP: Dict[str, str] = {
    "ns": FREEBASE_NS,
    "rdf": RDF_NS,
    "": FREEBASE_NS,  # Support empty prefix (used in GrailQA)
}


# ---------------------- CONFIG ----------------------
CONFIG = {
    # Directory that contains input JSON files (can be absolute or relative to repo root)
    "input_dir": "datasets/cwq/cwq_downloads",
    # Glob for selecting input files inside input_dir
    "input_glob": "*.json",
    # Where to write outputs per file and the merged results
    "output_dir": "datasets/cwq/cwq_predicates",
    # Filenames for merged outputs (written under output_dir)
    "merged_predicates": "cwq_all_predicates_counts.json",
    "merged_uris": "cwq_all_uri_counts.json",
    # New: write a merged white list (only URIs, no counts)
    "merged_white_list": "cwq_white_list.json",
    # Extra single-file inputs (auto-added if present). Each item is a path relative to repo root.
    "extra_input_files": [
        "datasets/GrailQA_v1.0/grailqa_v1.0_train.json",
        "datasets/GrailQA_v1.0/grailqa_v1.0_dev.json",
    ],
}


def resolve_path(p: str) -> str:
    """Resolve absolute/relative paths relative to repo root (scripts/..)."""
    if os.path.isabs(p):
        return p
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    return os.path.abspath(os.path.join(repo_root, p))


def clean_sparql(text: str) -> str:
    if not text:
        return ""
    # strip line comments
    text = "\n".join([COMMENT_RE.sub("", line) for line in text.splitlines()])
    # remove PREFIX/BASE declarations
    text = PREFIX_DECL_RE.sub(" ", text)
    text = BASE_DECL_RE.sub(" ", text)
    # normalize 'a' to rdf:type
    text = A_PREDICATE_RE.sub(" rdf:type ", text)
    return text


def normalize_pred_token(token: str) -> str:
    token = token.strip()
    if token == "a":
        return "rdf:type"
    return token


def is_entity_prefixed(token: str) -> bool:
    """Return True if token has an excluded entity prefix like m: or g:."""
    if ":" not in token:
        return False
    prefix, _ = token.split(":", 1)
    return prefix in EXCLUDE_PREFIXES


def extract_predicates_from_path(path_str: str) -> List[str]:
    """Extract individual predicates from a property path like ns:rel1/ns:rel2 or ns:rel+."""
    predicates = []
    # Remove modifiers (+, *, ?) at the end
    path_clean = re.sub(r'[+*?]$', '', path_str.strip())
    # Split by / or | (path separators)
    parts = re.split(r'[/|]', path_clean)
    for part in parts:
        part = part.strip()
        if part:
            pred = normalize_pred_token(part)
            if not is_entity_prefixed(pred):
                predicates.append(pred)
    return predicates


def extract_predicate_tokens(sparql: str) -> List[str]:
    """Extract predicate tokens from a SPARQL string (may include duplicates).
    
    This function extracts predicates from:
    1. Standard triple patterns (subject predicate object)
    2. Property paths (e.g., ns:rel1/ns:rel2, ns:rel+)
    """
    tokens: List[str] = []
    s = clean_sparql(sparql)
    
    # First, extract from standard triple patterns
    for m in TRIPLE_RE.finditer(s):
        p = normalize_pred_token(m.group(1))
        if is_entity_prefixed(p):
            continue
        tokens.append(p)
    
    # Then, extract from property paths (these may overlap with standard patterns, but we want all predicates)
    for m in PROPERTY_PATH_RE.finditer(s):
        path = m.group(1).strip()
        # Extract individual predicates from the path
        path_preds = extract_predicates_from_path(path)
        tokens.extend(path_preds)
    
    return tokens


def to_full_uri(token: str) -> Optional[str]:
    """Convert a token like ns:foo or <uri> into a full URI string without angle brackets.
    Returns None if the token cannot be resolved to a full URI.
    """
    t = token.strip()
    if t.startswith("<") and t.endswith(">"):
        return t[1:-1]
    if ":" in t:
        prefix, local = t.split(":", 1)
        if prefix in EXCLUDE_PREFIXES:
            return None
        base = DEFAULT_PREFIX_MAP.get(prefix)
        if base:
            return base + local
        return None
    return None


def iri_to_ns_prefixed(iri: str) -> str:
    """Compress an IRI to a known prefix (ns: or rdf:) when possible.
    Returns the original IRI if no known prefix matches.
    """
    if iri.startswith(FREEBASE_NS):
        return "ns:" + iri[len(FREEBASE_NS):]
    if iri.startswith(RDF_NS):
        return "rdf:" + iri[len(RDF_NS):]
    return iri


def detect_format(path: str) -> Tuple[str, int]:
    """Return (fmt, total). fmt is 'json' or 'jsonl'.
    For JSONL, total is the number of non-empty lines. For JSON array, total is the number of elements.
    """
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(4096)
        f.seek(0)
        stripped = "".join(ch for ch in head if not ch.isspace())
        if stripped.startswith("["):
            # JSON
            data = json.load(f)
            total = len(data) if isinstance(data, list) else 0
            return "json", total
        else:
            # JSONL: count non-empty lines
            total = sum(1 for line in f if line.strip())
            return "jsonl", total


def iter_records(path: str, fmt: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        if fmt == "json":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Input must be a JSON array or JSON Lines")
            for obj in data:
                yield obj if isinstance(obj, dict) else {}
        else:  # jsonl
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    obj = {}
                yield obj if isinstance(obj, dict) else {}


def iter_sparql_strings(obj: Dict[str, Any]) -> Iterable[str]:
    """Yield all SPARQL strings from a record across supported schemas.

    Supported:
    - CWQ-style: top-level key "sparql".
    - GrailQA: top-level key "sparql_query".
    - WebQSP: either top-level "Sparql" or nested under each item in "Parses".
    """
    if not obj:
        return

    direct_keys = ["sparql", "sparql_query", "Sparql"]
    for k in direct_keys:
        val = obj.get(k)
        if isinstance(val, str) and val.strip():
            yield val

    parses = obj.get("Parses")
    if isinstance(parses, list):
        for p in parses:
            if not isinstance(p, dict):
                continue
            for k in direct_keys:
                val = p.get(k)
                if isinstance(val, str) and val.strip():
                    yield val

def parse_cli_args() -> Optional[Dict[str, Any]]:
    """Parse optional CLI overrides. Returns a dict to update CONFIG, or None if no args."""
    parser = argparse.ArgumentParser(description="Extract predicates from SPARQL queries")
    parser.add_argument("--input-dir", dest="input_dir", help="Input directory for glob matching")
    parser.add_argument("--input-glob", dest="input_glob", help="Glob pattern inside input dir")
    parser.add_argument("--extra-input", dest="extra_input_files", action="append", help="Extra JSON/JSONL file (can repeat)")
    parser.add_argument("--output-dir", dest="output_dir", help="Output directory")
    parser.add_argument("--merged-predicates", dest="merged_predicates", help="Merged predicates filename")
    parser.add_argument("--merged-uris", dest="merged_uris", help="Merged URIs filename")
    parser.add_argument("--merged-white-list", dest="merged_white_list", help="Merged whitelist filename")

    # If no args were given (len(sys.argv)==1), return None to keep defaults
    if len(sys.argv) == 1:
        return None
    args = parser.parse_args()
    cfg: Dict[str, Any] = {}
    for key in [
        "input_dir",
        "input_glob",
        "output_dir",
        "merged_predicates",
        "merged_uris",
        "merged_white_list",
    ]:
        val = getattr(args, key)
        if val:
            cfg[key] = val
    if args.extra_input_files:
        cfg["extra_input_files"] = args.extra_input_files
    return cfg


def main(config_override: Optional[Dict[str, Any]] = None):
    # Merge default CONFIG with overrides
    cfg = dict(CONFIG)
    if config_override:
        cfg.update(config_override)

    def output_group(path: str) -> str:
        """Return logical output group name; GrailQA train/dev collapse to one group."""
        base = os.path.splitext(os.path.basename(path))[0]
        if base.startswith("grailqa_v1.0_"):
            return "grailqa"
        return base

    def add_counts(dst: Dict[str, int], src: Dict[str, int]):
        for k, v in src.items():
            dst[k] = dst.get(k, 0) + v

    # Resolve configured paths
    input_dir = resolve_path(cfg["input_dir"])
    output_dir = resolve_path(cfg["output_dir"])
    pattern = cfg.get("input_glob", "*.json")

    if not os.path.isdir(input_dir):
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)

    inputs = sorted(glob.glob(os.path.join(input_dir, pattern)))

    # Append extra single-file inputs when present
    for extra_rel in cfg.get("extra_input_files", []):
        extra_abs = resolve_path(extra_rel)
        if os.path.isfile(extra_abs):
            inputs.append(extra_abs)
    inputs = sorted(set(inputs))
    if not inputs:
        print(f"No inputs matched: {os.path.join(input_dir, pattern)}", file=sys.stderr)
        sys.exit(1)

    merged_counts_pref: Dict[str, int] = {}
    merged_counts_full: Dict[str, int] = {}
    group_counts_pref: Dict[str, Dict[str, int]] = {}
    group_counts_full: Dict[str, Dict[str, int]] = {}

    for in_path in inputs:
        fname = os.path.basename(in_path)
        base = os.path.splitext(fname)[0]
        print(f"Processing {in_path}...")
        try:
            fmt, total = detect_format(in_path)
        except Exception as e:
            print(f"Failed to detect format for {in_path}: {e}", file=sys.stderr)
            continue

        counts_pref: Dict[str, int] = {}
        counts_full: Dict[str, int] = {}

        with tqdm(total=total or None, desc=f"extracting {fname}", unit="rec") as pbar:
            for obj in iter_records(in_path, fmt):
                sparql_iter = iter_sparql_strings(obj) if isinstance(obj, dict) else []
                for sparql in sparql_iter:
                    tokens = extract_predicate_tokens(sparql)
                    for t in tokens:
                        # predicate key (prefixed when possible; compress <IRI> to ns:/rdf: if possible)
                        if t.startswith("<") and t.endswith(">"):
                            iri = t[1:-1]
                            pref_key = iri_to_ns_prefixed(iri)
                            if pref_key == iri:
                                pref_key = iri  # fallback to full IRI if no known prefix
                            full_key = iri
                        else:
                            pref_key = t
                            full_iri = to_full_uri(t)
                            full_key = full_iri if full_iri else None

                        counts_pref[pref_key] = counts_pref.get(pref_key, 0) + 1
                        if full_key:
                            counts_full[full_key] = counts_full.get(full_key, 0) + 1
                # Even if no SPARQL found, advance progress per record
                pbar.update(1)

        # Update merged totals
        add_counts(merged_counts_pref, counts_pref)
        add_counts(merged_counts_full, counts_full)

        group = output_group(in_path)
        if group not in group_counts_pref:
            group_counts_pref[group] = {}
            group_counts_full[group] = {}
        add_counts(group_counts_pref[group], counts_pref)
        add_counts(group_counts_full[group], counts_full)

    # Grouped per-dataset outputs (e.g., GrailQA train/dev merged)
    for group, g_counts_pref in sorted(group_counts_pref.items()):
        g_counts_full = group_counts_full.get(group, {})
        sorted_pref = sorted(g_counts_pref.items(), key=lambda x: (-x[1], x[0]))
        sorted_full = sorted(g_counts_full.items(), key=lambda x: (-x[1], x[0]))

        per_pred_path = os.path.join(output_dir, f"{group}_predicates_counts.json")
        per_uri_path = os.path.join(output_dir, f"{group}_uri_counts.json")

        with open(per_pred_path, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in sorted_pref}, f, ensure_ascii=False, indent=2)
        with open(per_uri_path, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in sorted_full}, f, ensure_ascii=False, indent=2)

        print(
            f"Wrote grouped outputs for {group}:\n"
            f"  - {per_pred_path} ({len(sorted_pref)} predicates)\n"
            f"  - {per_uri_path} ({len(sorted_full)} URIs)"
        )

    # Merged outputs
    merged_pred_name = cfg.get("merged_predicates", "cwq_all_predicates_counts.json")
    merged_uri_name = cfg.get("merged_uris", "cwq_all_uri_counts.json")
    merged_pred_path = os.path.join(output_dir, merged_pred_name)
    merged_uri_path = os.path.join(output_dir, merged_uri_name)

    sorted_merged_pref = sorted(merged_counts_pref.items(), key=lambda x: (-x[1], x[0]))
    sorted_merged_full = sorted(merged_counts_full.items(), key=lambda x: (-x[1], x[0]))

    with open(merged_pred_path, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in sorted_merged_pref}, f, ensure_ascii=False, indent=2)
    with open(merged_uri_path, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in sorted_merged_full}, f, ensure_ascii=False, indent=2)

    # Additional merged white-list (only URIs, no counts)
    merged_white_name = cfg.get("merged_white_list", "cwq_white_list.json")
    merged_white_path = os.path.join(output_dir, merged_white_name)
    # Extract only the URIs (keys) in sorted order
    white_uris = [k for k, _ in sorted_merged_full]
    with open(merged_white_path, "w", encoding="utf-8") as f:
        json.dump(white_uris, f, ensure_ascii=False, indent=2)

    print(
        "\nMerged outputs written:\n"
        f"  - {merged_pred_path} ({len(sorted_merged_pref)} predicates)\n"
        f"  - {merged_uri_path} ({len(sorted_merged_full)} URIs)\n"
        f"  - {merged_white_path} ({len(white_uris)} URIs)\n"
    )


if __name__ == "__main__":
    cli_cfg = parse_cli_args()
    main(cli_cfg)
