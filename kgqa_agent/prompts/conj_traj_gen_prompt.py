import os
import random
from pathlib import Path

# Base directory for example files
_EXAMPLE_DIR = Path(__file__).parent / "conj_traj_gen_examples"


def _load_random_example() -> str:
    """Load a random example from available example files in the directory."""
    # Find all example files
    example_files = sorted(_EXAMPLE_DIR.glob("example*.txt"))
    
    # Randomly select one
    example_path = random.choice(example_files)
    
    with open(example_path, "r", encoding="utf-8") as f:
        return f.read().strip()


_PROMPT_TEMPLATE = """=== OVERVIEW ===
You are a helpful assistant that generates Supervised Fine-Tuning samples for a Knowledge Graph question-answering task involving CONJUNCTION reasoning.
Your task is to generate a detailed reasoning thought that explains WHY a specific action was chosen at a given step, based on the historical information available up to that point.

CONJUNCTION reasoning means the question requires exploring from more than one different starting entities (topic entities) that eventually converge to the same answer entity through different paths. The agent is expected to explore all paths to find the common answer.

=== AGENT ROLLOUT RULES ===
These rules describe how the underlying agent behaves. Your thoughts should align with them.

- Tools:
  - get_relations("entity_name"): Returns all relations (both incoming and outgoing) connected to the entity.
  - get_triples("entity_name", ["relation1", "relation2", ...]): Returns triples for the given entity and the specified list of relations.

Entity reference rules (CRITICAL):
1. Always use ENTITY NAME in all function calls. Copy names exactly as shown in the last <information>.
2. If Initial entities are provided, you must start your search from them using get_relations("your_initial_entity").
3. Use double quotes around all function parameters (entity and relations).
4. When calling get_triples, provide a list of relations that are helpful to answer the question, triples related to these relations will be automatically retrieved. Each relation must be copied EXACTLY from the latest <information> list. Example: get_triples("Barack Obama", ["people.person.place_of_birth", "people.person.nationality"]).
5. Only use the two query functions listed above; do not write natural language queries inside <kg-query> tags.
6. Before calling get_triples(...), first call get_relations(...) for your current entity to obtain the predicate list.
7. No outside knowledge: answer ONLY using KG results you have retrieved.

=== WHAT TO WRITE ===
You will be given:
1. The question to answer
2. The topic entities (starting points - may be multiple for conjunction questions)
3. Historical information: all previous steps with their thoughts, actions, and observations
4. The next action that should be taken (this is the "golden action" that leads to the answer)

Your task is to write a concise thought that:
1. Briefly mentions what you learned from the last observation (relations or triples)
2. Directly explains why the given next action is the right choice at this point

For the final step (when the next action is to provide an <answer>):
- Briefly explain how the observations from the previous step provide the answer

Keep your thought concise and direct. Avoid over-summarizing or repeating information. Focus on the essential reasoning that leads to the action, similar to the example style.


=== CRITICAL CONSTRAINTS ===
1. ONLY reference information from previous steps (historical information provided to you). NEVER mention entities, relations, or facts that may appear in future steps.
2. Keep your thought concise and direct - typically 2-5 sentences, similar to the example style. Avoid lengthy summaries or over-explanation.
3. Copy entity/predicate strings EXACTLY as shown in observations.
4. Use natural language only; never include <kg-query> or <answer> tags in your thoughts.
5. Do NOT simply restate the action - explain the REASONING behind choosing it, but keep it brief.

=== INPUT FORMAT ===
{{
  "question": str,
  "topic_entity": ["entity_name1", "entity_name2", ...],
  "history": [
    {{
      "step_index": int,
      "think": "previous reasoning thought",
      "action": "<kg-query>...</kg-query>",
      "observation": ["relation1", "relation2", ...] or ["triple1", "triple2", ...]
    }}, ...
  ],
  "next_action": "<kg-query>...</kg-query>" or "<answer>[\"answer1\", \"answer2\", ...]</answer>"
}}

=== OUTPUT FORMAT ===
<think>
Your concise reasoning thought (typically 2-5 sentences, following the example's brief style)
</think>

=== EXAMPLE ===

{example_content}

=== IMPORTANT NOTE ===
The example above demonstrates the desired concise style. When generating actual reasoning thoughts, you should:
1. Match the example's brevity and directness - avoid lengthy explanations or over-summarizing
2. Use diverse language styles and expressions - avoid simply mimicking the example's exact phrasing
3. Keep it concise: typically 2-5 sentences that directly explain why the action is chosen, without unnecessary elaboration
"""


def CONJUNCTION_TRAJECTORY_GENERATION_PROMPT() -> str:
    example_content = _load_random_example()
    return _PROMPT_TEMPLATE.format(example_content=example_content)
