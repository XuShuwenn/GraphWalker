"""Unified prompt utilities for KG-Augmented QA
This module consolidates all prompt building helpers into a single place.
It includes:
- Continuation and force-answer prompts
- Search-style prompt template and helpers (single-source prompt used by evaluator)
"""
from __future__ import annotations


CONTINUATION_PROMPT_TEMPLATE = """<information>Here are the query results:
{query_results}
</information>

Review the results. If the answer is found, provide it in <answer> tags. Otherwise, continue reasoning in <think> tags and issue your next <kg-query>.

Reminder: After `get_relations`, you must rank all returned relations and call `get_triples` with the full ranked list. We will execute a query for the top 4.
"""

def build_continuation_prompt(query_results: str) -> str:
    return CONTINUATION_PROMPT_TEMPLATE.format(query_results=query_results)

# Minimal force-answer prompt when max calls reached
FORCE_ANSWER_PROMPT = """You have reached the maximum number of queries. Based on the information gathered, provide your final answer in `<answer>` tags.
The answer(s) must be concise entity names copied exactly from the KG results.
"""


KG_QUERY_SERVER_INSTRUCTION = """
**Available Query Functions:**
- get_relations("entity_name"): Returns all relations (incoming and outgoing) for the entity. Use this to explore the entity's properties.
- get_triples("entity_name", ["relation1", "relation2", ...]): Returns triples for the entity limited to the specified relations.


**KG QUERY SERVER INSTRUCTIONS:**
1. If you encounter a KG-related error, read the error message carefully and correct your query.
2. Use only the two query functions above; do not write natural-language queries inside <kg-query> tags.
3. If Initial Entities are provided, begin your search from them using get_relations("nitial_entity"). If multiple initial entities are given, start with the most specific one. Analyze each systematically.
4. Use the ENTITY NAME EXACTLY as shown in the <information> section. Example: get_relations("Barack Obama").
5. Before calling get_triples(...), you MUST have called get_relations(...) for that entity (or already seen its relations) to ensure the relations exist.
6. When calling get_triples, provide a list of ALL candidate relations returned by the previous step, ranked by relevance to the question. We will automatically use the top 4. Copy each relation EXACTLY as it appears in the <information> list. Example: get_triples("Barack Obama", ["people.person.place_of_birth", "people.person.nationality", ...]).
7. Always use double quotes around all function parameters (entity names and relations).


**ANSWER OUTPUT RULES:**
1. Format: <answer>["Answer1", "Answer2", ...]</answer>. Return a JSON list of strings. The final answer must be a human-readable entity name, not a MID (e.g., m.01234). If the only candidate you have is a MID, do NOT output it as the final answer. Instead, explore the MID's neighbors (use `get_relations`/`get_triples`) to find a named entity to return.
2. No external knowledge: answer ONLY using KG query results you have retrieved. The answer(s) MUST be EXACT and COMPLETE entity names from the retrieved triples. If multiple entities satisfy the question, list them all in the JSON list in <answer> tags.
3. Do NOT include any explanations in the <answer> tags.
4. NEVER include "I don't know" or "Information not available" in <answer> tags. Provide the best possible answer(s) from the entities you have retrieved.
5. CRITICAL: Your final answer must only contain entities that exist in the provided graph triples.
"""


SEARCH_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on knowledge graphs. You can query from knowledge base provided to you to answer the question up to {max_calls} times. 

**GENERAL INSTRUCTIONS:**
1. First, perform careful reasoning inside <think>...</think> tags. In this section, briefly outline your plan, record any working memory or intermediate observations, check for mistakes or ambiguous interpretations.
2. If you need to query knowledge for more information, you can set a query statement between <kg-query>...</kg-query> to query from knowledge graph. The query tool-using rules are provided in "KG Query Server Instruction" part. Each <kg-query> call must occur after <think>...</think>, and you must not issue another <kg-query> until you have received the environment's <information> feedback for the previous query.
3. If you have already found the answer, return it directly in the form <answer>...</answer> and end your search. Your answer must strictly follow the KG Query Server Instruction.

**KG INSTRUCTIONS:**
{kg_instruction}

**Question:** {question}

Begin your search with the following initial entities. Use `get_relations` to explore their relations.

**Initial Entities:** {initial_entities_line}
"""


def _format_initial_entities_line(topic_entities: list | None) -> str:
    if not topic_entities:
        return ""
    names: list[str] = []
    for te in topic_entities:
        if isinstance(te, dict):
            name = te.get("name") or te.get("label")
            if name:
                names.append(name)
        else:
            names.append(str(te))
    if not names:
        return ""
    return ", ".join([f"'{n}'" for n in names])


def build_search_prompt(question: str, max_calls: int = 3, topic_entities: list | None = None) -> str:
    initial_line = _format_initial_entities_line(topic_entities)
    return SEARCH_PROMPT_TEMPLATE.format(
        max_calls=max_calls,
        kg_instruction=KG_QUERY_SERVER_INSTRUCTION,
        question=question,
        initial_entities_line=initial_line,
    )