"""Normalize Freebase relation names for SPARQL use.

Provides a small utility that strips common URI prefixes and
normalizes separators so relation strings can be used in queries.
"""
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class RelationNormalizer:
    """Strip prefixes and normalize separators for relation IDs.

    Keeps a simple cache of lowercased inputs.
    """

    def __init__(self):
        self._cache: Dict[str, str] = {}

    def normalize(self, relation: str) -> str:
        """Return a dotted Freebase relation id (no URI prefix).

        Examples:
        - 'http://rdf.freebase.com/ns/people.person.children' -> 'people.person.children'
        - 'ns:people/person.children' -> 'people.person.children'
        """
        if not relation:
            return relation

        key = relation.lower().strip()
        if key in self._cache:
            return self._cache[key]

        cleaned = relation.strip()
        if cleaned.startswith('http://rdf.freebase.com/ns/'):
            cleaned = cleaned[len('http://rdf.freebase.com/ns/'):]
        elif cleaned.startswith('ns:'):
            cleaned = cleaned[len('ns:'):]

        if '/' in cleaned:
            cleaned = cleaned.replace('/', '.')

        self._cache[key] = cleaned
        logger.debug("Normalized relation '%s' -> '%s'", relation, cleaned)
        return cleaned

    def clear_cache(self) -> None:
        """Clear internal cache."""
        self._cache.clear()


# Singleton access
_normalizer_instance: Optional[RelationNormalizer] = None


def get_normalizer() -> RelationNormalizer:
    """Return the global RelationNormalizer instance."""
    global _normalizer_instance
    if _normalizer_instance is None:
        _normalizer_instance = RelationNormalizer()
    return _normalizer_instance


def normalize_relation(relation: str) -> str:
    """Convenience wrapper for `get_normalizer().normalize`."""
    return get_normalizer().normalize(relation)
