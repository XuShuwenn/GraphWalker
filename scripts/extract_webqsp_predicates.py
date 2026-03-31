import json
import os
import re
from typing import Iterable, List, Set, Union

INPUT_DIR = os.path.join("datasets", "webqsp")
INPUT_FILES = [
    os.path.join(INPUT_DIR, name) for name in ("train.json", "validation.json", "test.json")
]
OUTPUT_FILE = os.path.join(INPUT_DIR, "webqsp_predicates_white_list.json")

# Regex to extract triples from legacy stringified arrays if encountered
TRIPLE_RE = re.compile(r"\['(.*?)',\s*'(.*?)',\s*'(.*?)'\]")


def normalize_predicate(p: str) -> str:
    p = str(p).strip()
    if p.startswith("http://") or p.startswith("https://"):
        return p
    # Map rdf-schema#* to the canonical W3C URI
    if p.startswith("rdf-schema#"):
        return f"http://www.w3.org/2000/01/rdf-schema#{p.split('#', 1)[1]}"
    # Default: treat as Freebase property in ns
    return f"http://rdf.freebase.com/ns/{p}"


def is_meaningful_predicate(uri: str) -> bool:
    """Keep only Freebase namespace predicates; drop schema/meta URIs and odd entries."""
    if not isinstance(uri, str) or not uri:
        return False
    if not uri.startswith("http://rdf.freebase.com/ns/"):
        return False
    # Heuristic: drop if whitespace present (shouldn't be in Freebase property URIs)
    if any(ch.isspace() for ch in uri):
        return False
    return True


def extract_triples_from_graph(graph_value: Union[str, list]) -> Iterable[List[str]]:
    """Yield triples [s, p, o] from a graph field that can be a list or a string."""
    if isinstance(graph_value, list):
        for item in graph_value:
            if (
                isinstance(item, (list, tuple))
                and len(item) == 3
                and all(isinstance(x, str) for x in item)
            ):
                yield [item[0], item[1], item[2]]
    elif isinstance(graph_value, str):
        for s, p, o in TRIPLE_RE.findall(graph_value):
            yield [s, p, o]


def collect_predicates(files: List[str]) -> List[str]:
    preds: Set[str] = set()
    for fp in files:
        if not os.path.exists(fp):
            continue
        with open(fp, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue
        for rec in data:
            graph = rec.get("graph")
            if graph is None:
                continue
            for s, p, o in extract_triples_from_graph(graph):
                uri = normalize_predicate(p)
                if is_meaningful_predicate(uri):
                    preds.add(uri)
    # Return sorted for determinism
    return sorted(preds)


def main():
    predicates = collect_predicates(INPUT_FILES)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(predicates, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(predicates)} predicates to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
