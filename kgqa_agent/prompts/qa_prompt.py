GENERATE_QUESTION_FREEBASE = r"""
You are given:

- A path represented as an ordered sequence of relations, topic entity and answer entity (e.g.,
  Topic Entity --relationA--> entity1 --> relationB--> entity2 --> ... --> Answer Entity).

You will be provided with the name of the topic entity and the name of the answer entity.
Names of intermediate entities in the path will be replaced by placeholders like entity1, entity2, etc.

**Your task:**
- Compose ONE concise English question whose unique answer is exactly the Answer Entity.
- Implicitly reflect the multi-hop path, but keep it brief and natural. Avoid enumerations or multiple questions.
- STRICT: The question must contain exactly one question mark '?' (only one interrogative). The question must be a single sentence with only one '?'.
- Do NOT directly mention raw relation labels. Paraphrase using concrete, meaningful semantics (e.g., country, city, river, award, university, founder, member of, genre, located in, part of, spouse, parent, date, birthplace, capital).
- Keep the question concise (ideally < 100 characters).
- Use only the information implied by the path; do not introduce outside facts.
- Try to use different styles and sentence structures when generating questions.

**Examples**

Example 1:
Input:
Path: The Twilight Zone -> influence.influence_node.influenced -> entity1 -> music.featured_artist.recordings -> entity2 -> music.recording.releases -> Be Yourself: A Tribute to Graham Nash's Songs for Beginners
Output:
<question>The Twilight Zone influenced an artist whose work is featured on what tribute album to Graham Nash?</question>

Example 2:
Input:
Path: Stainless steel -> law.invention.inventor -> entity1 -> book.author.works_written -> The analysis of steel-works materials
Output:
<question>What is the title of a work written by the inventor of stainless steel?</question>

**Input:**
Path: %(path)s

CRITICAL: Do NOT include any explanations, analysis, or additional text. Output ONLY the one line below, nothing more.
Output format (STRICT — output EXACTLY one line, nothing else):
<question>(one concise question with exactly one '?')</question>

Output:

"""
