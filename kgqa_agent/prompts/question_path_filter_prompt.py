PROMPT_TEMPLATE = """You are an expert evaluator for Knowledge Graph Question Answering (KGQA) data quality. Your task is to evaluate question-path pairs based on multiple quality criteria.

## Input Format
You will be provided with two inputs:
1. **name_path**: A reasoning path in the knowledge graph, represented as a sequence of entities and relations connected by "->"
2. **question**: The natural language question that should be answerable by following this path

Input Example:
name_path: "Emap International Limited -> organization.organization.founders -> Richard Winfrey -> people.person.place_of_birth -> Long Sutton, Lincolnshire"
question: "What is the birthplace of a founder of Emap International Limited?"

## Task
For each question-path pair provided, you must:
Evaluate it on the following dimension and assign a score from 0 to 10 based on the scoring guide:

## Evaluation Criteria
You must evaluate each question-path pair based on the following dimension:
{dimension_prompt}

## Important Guidelines
1. **Strict Evaluation**: Be strict but fair.
2. **Output Consistency**: Always output the score in the exact format above. Do not include any text outside the valid structure.
3. Do not include any other text or explanation.

## Output Format
You must output only a single score from 0 to 10:
<score><0-10></score>
"""


REASONING_PATH_QUALITY_PROMPT = """Reasoning Path Quality: "Whether the reasoning path forms a complete, valid, and efficient chain from the starting entity to the answer entity, with all relations semantically correct and entities properly resolved."

## Scoring Guide (0–10)
- **10**: The path is complete, direct, and optimal. All relations are semantically valid and correctly connected. All entities are properly resolved with no ambiguity. The path represents the most efficient route to the answer with no unnecessary steps.
- **8–9**: The path is complete and valid, with only minor inefficiencies or one minor ambiguity. All relations are correct, and the path successfully leads to the answer entity.
- **6–7**: The path is mostly complete and valid, but may have some inefficiencies, minor relation issues, or one ambiguous entity. The path can still lead to the answer, though not optimally.
- **4–5**: The path has noticeable issues: missing intermediate entities, some invalid relations, or multiple ambiguous entities. The path may still reach an answer but with significant problems.
- **1–3**: The path has serious structural problems: circular references, multiple invalid relations, or critical missing entities. The path may not reliably lead to the answer.
- **0**: The path is fundamentally broken: cannot form a valid chain, contains only invalid relations, or fails to connect to any meaningful answer entity.
"""


QUESTION_PATH_RELEVANCE_PROMPT = """Question-Path Relevance: "Whether the question semantically aligns with what the path can answer, and whether the path contains all necessary information to fully and accurately answer the question."

## Scoring Guide (0–10)
- **10**: Perfect alignment. The question directly matches what the path answers. The path contains all necessary information and fully addresses the question. The answer entity at the end of the path is exactly what the question asks for.
- **8–9**: Strong alignment with minor gaps. The path answers the question well, with only very minor information missing or slight semantic nuances not perfectly captured.
- **6–7**: Good alignment but incomplete. The path addresses the core of the question but may miss some aspects or provide partial information. The answer entity is relevant but may not fully satisfy the question.
- **4–5**: Moderate alignment with notable gaps. The path is related to the question but doesn't fully answer it, or answers a different but related aspect. Some necessary information is missing.
- **1–3**: Poor alignment. The path is tangentially related but doesn't answer the question, or answers a different question entirely. Significant information mismatch.
- **0**: No alignment. The path and question are completely unrelated, or the path cannot answer the question at all.
"""


QUESTION_SEMANTIC_COHERENCE_PROMPT = """Question Semantic Coherence: "Whether the question is grammatically correct, semantically clear, naturally phrased, and logically structured like a human would ask it."

## Scoring Guide (0–10)
- **10**: Perfect coherence. The question is grammatically flawless, semantically crystal clear, reads naturally, and has impeccable logical structure. It sounds exactly like a native speaker would ask it.
- **8–9**: Excellent coherence with minor imperfections. The question is grammatically correct, clear, and natural, with only very minor stylistic issues or slight ambiguity.
- **6–7**: Good coherence with some issues. The question is mostly grammatically correct and understandable, but may have minor grammatical errors, slight ambiguity, or somewhat unnatural phrasing.
- **4–5**: Moderate coherence with noticeable problems. The question has grammatical errors, some ambiguity, or unnatural phrasing that affects comprehension, but the core meaning is still discernible.
- **1–3**: Poor coherence. The question has significant grammatical errors, is confusing or ambiguous, or has very unnatural phrasing that makes it difficult to understand.
- **0**: No coherence. The question is grammatically broken, completely ambiguous, or makes no logical sense. It cannot be understood as a valid question.
"""


# Dimension names
REASONING_PATH_QUALITY = "reasoning_path_quality"
QUESTION_PATH_RELEVANCE = "question_path_relevance"
QUESTION_SEMANTIC_COHERENCE = "question_semantic_coherence"

# Mapping from dimension name to dimension prompt
DIMENSION_PROMPTS = {
    REASONING_PATH_QUALITY: REASONING_PATH_QUALITY_PROMPT,
    QUESTION_PATH_RELEVANCE: QUESTION_PATH_RELEVANCE_PROMPT,
    QUESTION_SEMANTIC_COHERENCE: QUESTION_SEMANTIC_COHERENCE_PROMPT,
}


def get_filter_prompt(dimension: str) -> str:
    """Get the complete filter prompt for a specific dimension.
    
    Args:
        dimension: One of "reasoning_path_quality", "question_path_relevance", 
                  or "question_semantic_coherence"
    
    Returns:
        Complete prompt string with the specified dimension prompt inserted
    
    Raises:
        ValueError: If dimension is not recognized
    """
    if dimension not in DIMENSION_PROMPTS:
        raise ValueError(
            f"Unknown dimension: {dimension}. "
            f"Must be one of: {list(DIMENSION_PROMPTS.keys())}"
        )
    
    dimension_prompt = DIMENSION_PROMPTS[dimension]
    return PROMPT_TEMPLATE.format(dimension_prompt=dimension_prompt)


def get_all_dimension_prompts() -> dict:
    """Get all three dimension prompts.
    
    Returns:
        Dictionary mapping dimension names to complete prompt strings
    """
    return {
        dim: get_filter_prompt(dim) 
        for dim in DIMENSION_PROMPTS.keys()
    }