# SkillShare Match: Annotation & Evaluation Rubric

This document governs the ground-truth labeling protocol for evaluation queries and helper profiles in **SkillShare Match**. All annotators must follow these guidelines strictly to ensure objective, reproducible relevance evaluation and prevent data leakage or scoring bias.

---

## 1. Ethical & Synthetic Data Disclosure
- All profiles and queries are entirely fictional, synthetic constructs created for academic research prototyping.
- No real individual names, personal contact details, or exact street addresses may be used.
- Ground truth must be established through manual human annotation using this rubric—**never** by querying or extracting predictions from the models being evaluated (TF-IDF, BM25, Sentence-BERT, or Cross-Encoders).

---

## 2. Graded Relevance Scale (0 – 2)

Each `(query, helper_profile)` pair evaluated for semantic skill matching receives a graded relevance score:

| Grade | Label | Operational Definition | Concrete Example |
| :--- | :--- | :--- | :--- |
| **2** | **Relevant (Exact Match)** | The helper explicitly or directly offers the capability needed to satisfy the requester's core goal. Vocabulary may differ (e.g., synonyms or paraphrases), but the functional match is direct. | **Query**: "Need help polishing my CV for software internships."<br>**Helper**: "I review résumés, give placement advice, and conduct mock interviews." *(Grade 2)* |
| **1** | **Partially Relevant (Adjacent / Broad)** | The helper provides an adjacent, foundational, or partial skill that assists the user but does not directly or fully fulfill the specific request. Also applies to multi-skill requests where only one skill is satisfied. | **Query**: "Need someone to teach me React and Redux state management."<br>**Helper**: "Web developer offering introductory HTML, CSS, and basic JavaScript." *(Grade 1)* |
| **0** | **Not Relevant (Mismatch)** | The helper's skills are unrelated, fundamentally unsuitable, or belong to a different functional domain. Shared non-skill words (e.g., "help", "online", "weekend") do not count. | **Query**: "Need someone to improve my resume."<br>**Helper**: "Graphic designer creating posters and brand logos." *(Grade 0)* |

---

## 3. Mandatory Constraints vs. Semantic Relevance

SkillShare Match treats practical constraints strictly according to user intent:

### 3.1 Hard Constraints (`rule: "required"`)
- **Language**: If a language is flagged `required`, a helper **must** list that language in their profile `languages`. If omitted, the helper is **ineligible** for recommendation.
- **Availability**: If days/windows are flagged `required`, the helper's `availability` **must** overlap with the requested window.
- **Session Mode**: If `online` or `in_person` is flagged `required`, the helper **must** support that mode (`both` satisfies either).
- **Active Status**: Inactive profiles (`active: false`) are **never** eligible under any circumstances.

> [!IMPORTANT]
> **Ground Truth Rule for `relevant_helper_ids`:**
> A helper ID appears in a query's `relevant_helper_ids` list **if and only if**:
> 1. The helper has a Skill Relevance Grade of **2** (or Grade **1** if explicitly marked for relaxed retrieval).
> 2. The helper satisfies **100% of all required (hard) constraints**.
> 3. The helper has `active == True`.

### 3.2 Preferred Constraints (`rule: "preferred"`)
- Preferences (e.g., preferred Saturday, preferred Hindi) do **not** disqualify a candidate.
- A helper who matches the skill but lacks a *preferred* constraint remains in `relevant_helper_ids` (and receives a transparent soft adjustment or explicit UI tag during ranking).

---

## 4. Query Categories & Annotation Rules

### 4.1 Paraphrased Queries (`category: "paraphrased"`)
- **Intent**: Uses natural synonyms, descriptive outcomes, or alternative phrasing without using the exact lexical tokens found in profile skill tags.
- **Examples**: "Need CV feedback" vs. "résumé proofreading"; "learn to play six-string acoustic" vs. "guitar lessons".
- **Rule**: Grade 2 if the core competence matches despite zero token overlap.

### 4.2 Vague / Underspecified Queries (`category: "vague"`)
- **Intent**: High-level or outcome-oriented statements where the specific technical skill must be inferred.
- **Examples**: "I want to get better with numbers for my small business" (Accounting, Excel, Bookkeeping).
- **Rule**: Grade 2 for profiles addressing the primary plausible interpretations; Grade 1 for secondary adjacent tools.

### 4.3 Multi-Skill Queries (`category: "multi_skill"`)
- **Intent**: Requesters requiring two distinct capabilities (e.g., "Need Python tutoring and advice on technical interview algorithms").
- **Rule**:
  - Helpers offering **both** skills = Grade 2.
  - Helpers offering **one of two** skills = Grade 1.
  - Helpers offering neither = Grade 0.

### 4.4 Code-Mixed Queries (`category: "code_mixed"`)
- **Intent**: Requesters communicating in mixed Hindi/Punjabi/English using Latin script (Romanized) or native Gurmukhi / Devanagari script.
- **Examples**:
  - Romanized: *"Mujhe interview preparation ke liye CV review karwana hai, weekend pe."*
  - Gurmukhi: *"ਮੈਨੂੰ ਇੰਟਰਵਿਊ ਦੀ ਤਿਆਰੀ ਅਤੇ ਸੀਵੀ ਚੈੱਕ ਕਰਵਾਉਣ ਲਈ ਮਦਦ ਚਾਹੀਦੀ ਹੈ।"*
  - Devanagari: *"मुझे डेटा एनालिसिस और पाइथन सीखने में सहायता चाहिए।"*
- **Rule**: Translated intent determines the match. Language tags in profiles must correspond with requester language requirements if marked required.

### 4.5 No-Match Queries (`category: "no_match"`)
- **Intent**: Queries where no suitable active helper exists in the corpus.
- **Scenarios**:
  1. **Domain Out-of-Corpus**: The skill requested does not exist anywhere in the platform (e.g., "Need helicopter flight lessons", "brain surgery consultation").
  2. **Constraint Over-Constrained**: A helper with the skill exists, but **zero** helpers meet the combination of mandatory constraints (e.g., "Punjabi CV review required on Monday mornings in South Zone" when the only Punjabi CV reviewers are available on Saturdays in North Zone).
- **Ground Truth**: `relevant_helper_ids` **MUST BE EMPTY (`[]`)**.
- **System Goal**: The retrieval system must output an empty list or trigger the "No match found" card, rather than forcing an unrelated recommendation.

---

## 5. Dataset Split & Inter-Annotator Agreement

1. **Validation Split (50%)**: Used exclusively for setting similarity thresholds, tuning soft constraint weights, and error debugging.
2. **Test Split (50%)**: Strictly held out. **Never** inspected during prompt tuning, weight optimization, or algorithm calibration.
3. **Double Annotation & Quality Control**:
   - At least 20% of all queries will be independently annotated by a second reviewer.
   - Cohen's Kappa ($\kappa$) will be calculated and reported.
   - Any score discrepancies ($\Delta \ge 1$) will be resolved through a consensus discussion referencing this rubric.
