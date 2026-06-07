GENERATE_CONJUNCTION_QUESTION = r"""
You are given:
- Two paths that both lead to the same answer entity:

Your task:
- Compose ONE complete and concise English question whose unique answer is the common entity reached by both paths (the final entity in both paths).
- The question should implicitly reflect that the answer requires exploring from BOTH starting entities (conjunction reasoning), not just one of them.
- Keep it brief and natural (ideally < 120 characters). Avoid enumerations or multiple questions.
- STRICT: The question must be COMPLETE and contain exactly one question mark '?' (only one interrogative). Prefer a single sentence (or two short clauses joined), but only one '?'.
- Do NOT directly mention exact entity names/IDs or raw relation labels. Paraphrase using concrete, meaningful semantics (e.g., country, city, river, award, university, founder, member of, genre, located in, part of, spouse, parent, date, birthplace, capital).
- De-emphasize meta/technical relations (e.g., types, notable_for, generic topic) unless essential. Focus on relations with clear real-world meaning.
- Use only the information implied by the two paths; do not introduce outside facts.
- Try to use diverse styles and sentence structures when generating questions to avoid repetitive patterns and enhance question diversity.
- CRITICAL: Output a COMPLETE question. Do NOT truncate or cut off the question. The question must end with a question mark '?'.
- You MUST output the question in the format <question><your complete question></question>

Example:
Input:
Path 1: Irgun -> organization.organization.founders -> Avraham Tehomi
Path 2: Jewish insurgency in Palestine -> base.culturalevent.event.entity_involved -> Avraham Tehomi
Output:
<question>Which founder of the Irgun was also involved in the Jewish insurgency in Palestine?</question>

Now, please generate the question based on the following paths:
Input:
Path 1: %(path1)s
Path 2: %(path2)s
Output:
<question><your complete question></question>
"""