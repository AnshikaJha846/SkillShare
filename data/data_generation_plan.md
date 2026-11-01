# SkillShare Match: Data Generation Plan

This document outlines the blueprint for creating the synthetic dataset of **100–200 helper profiles** and **50–60 evaluation queries**. All data is completely synthetic and adheres to the ethical disclosure and annotation rubric in [data/rubric.md](file:///c:/Users/ASUS/SkillShare/data/rubric.md).

---

## 1. Skill Taxonomies (~25 Community Skill Areas)

To reflect a realistic community skill-sharing ecosystem, skills are organized into 6 core domains comprising **26 concrete skill areas**:

| Domain | Skill Tag | Common Synonyms & Vocabulary Mismatches |
| :--- | :--- | :--- |
| **Career & Professional** | `resume_review` | CV, résumé, curriculum vitae, biodata, placement profile, job application |
| | `interview_prep` | mock interview, placement preparation, viva, behavioral interview, case practice |
| | `linkedin_branding` | personal branding, LinkedIn profile optimization, professional networking |
| | `public_speaking` | presentation skills, toastmasters, stage confidence, speech delivery |
| **Tech & Programming** | `python_programming` | Python, scripting, coding, software development, algorithms, backend |
| | `web_development` | frontend, HTML, CSS, JavaScript, React, web design, responsive UI |
| | `data_analysis_excel` | Excel, spreadsheets, Google Sheets, pivot tables, VLOOKUP, data analysis |
| | `sql_databases` | relational databases, queries, MySQL, PostgreSQL, data querying |
| | `machine_learning` | ML, data science, predictive modeling, pandas, scikit-learn |
| **Languages & Communication** | `conversational_english` | spoken English, fluency practice, everyday conversation, pronunciation |
| | `ielts_prep` | IELTS, TOEFL, English test prep, academic writing, band score |
| | `hindi_tutoring` | Hindi speaking, Devanagari reading/writing, conversational Hindi |
| | `punjabi_tutoring` | Punjabi speaking, Gurmukhi script reading/writing, conversational Punjabi |
| **Academic & Sciences** | `high_school_math` | mathematics, algebra, calculus, trigonometry, high school math |
| | `physics_tutoring` | physics, mechanics, optics, thermodynamics, school physics |
| | `chemistry_tutoring` | organic chemistry, inorganic, chemical reactions, chemistry help |
| | `academic_proofreading` | essay editing, thesis proofreading, grammar check, literature review |
| **Creative Arts & Media** | `graphic_design` | poster design, logos, banners, Canva, Photoshop, Illustrator, visual assets |
| | `video_editing` | video production, Premiere Pro, CapCut, reel editing, YouTube editing |
| | `digital_photography` | camera basics, lighting, photo retouching, Lightroom, portrait photography |
| | `creative_writing` | storytelling, fiction, copywriting, content writing, blogging |
| **Music & Lifestyle** | `guitar_lessons` | acoustic guitar, chords, strumming, six-string, tabs, fingerstyle |
| | `piano_keyboard` | keyboard basics, piano chords, synthesizer, scales |
| | `vocal_coaching` | singing lessons, voice modulation, classical singing, pitch training |
| | `home_gardening` | urban gardening, houseplants, composting, balcony plants, organic veggies |
| | `yoga_fitness` | yoga asanas, home workouts, fitness coaching, posture, stretching |

---

## 2. Writing Style Variations (Helpers & Requesters)

Real community posts differ drastically in tone, length, and polish. Bios and queries will be synthesized across 5 distinct stylistic profiles:

1. **Formal / Professional (approx. 25%)**:
   - Polished grammar, structured descriptions, explicit certifications, structured bullet points.
   - *Example*: *"Senior HR professional with 6+ years in corporate tech recruiting. Providing structured CV evaluations, ATS keyword optimization, and mock competency-based interviews."*
2. **Casual / Conversational (approx. 35%)**:
   - First-person, friendly, colloquial idioms, contractions, conversational punctuation.
   - *Example*: *"Hey there! I'm a final-year CS undergrad who loves building web apps and Python bots. Super happy to walk beginners through coding basics or debug assignments."*
3. **Short / Terse (approx. 15%)**:
   - Minimalist, under 15 words, telegraphic phrases, omitted punctuation.
   - *Example*: *"Maths & Physics tutor. Weekends online only. High school level."*
4. **Long / Narrative (approx. 15%)**:
   - Detailed personal stories, motivations, specific toolchains, 60–100 words.
   - *Example*: *"I started playing classical guitar 10 years ago and transitioned to fingerstyle acoustic. When I was learning, having someone break down chord transitions and rhythm made all the difference. In my sessions, we focus on hand posture, finger independence, and your favorite songs."*
5. **Typo / Informal Spelling (approx. 10%)**:
   - Realistic phone-typing errors, missing apostrophes, phonetic spelling.
   - *Example*: *"Offering resumee review and carrer guidance for colledge students. Avaliable sundays."*

---

## 3. Synonym Pairs & Lexical Mismatches

To properly test the research question (*"Does semantic retrieval improve recommendations over lexical baselines?"*), we deliberately engineer query-profile pairs with zero or minimal lexical token overlap:

- `CV` (Query) $\leftrightarrow$ `résumé` / `placement biodata` (Profile)
- `Learn to code` (Query) $\leftrightarrow$ `Python scripting & software development` (Profile)
- `Spreadsheet formulas` (Query) $\leftrightarrow$ `Excel and Google Sheets data modeling` (Profile)
- `Six-string acoustic` (Query) $\leftrightarrow$ `Guitar lessons and chord progression` (Profile)
- `Job interview confidence` (Query) $\leftrightarrow$ `Mock placement viva and behavioral practice` (Profile)
- `Houseplants dying` (Query) $\leftrightarrow$ `Balcony gardening and soil nutrition` (Profile)

---

## 4. Multilingual & Code-Mixed Strategy

The platform specifically evaluates handling of **Hindi**, **Punjabi**, and **English** code-switching:

1. **Native Script Inputs**:
   - Gurmukhi (ਪੰਜਾਬੀ): e.g. *"ਮੈਨੂੰ ਇੰਟਰਵਿਊ ਦੀ ਤਿਆਰੀ ਅਤੇ ਰੈਜ਼ਿਊਮੇ ਚੈੱਕ ਕਰਵਾਉਣ ਲਈ ਮਦਦ ਚਾਹੀਦੀ ਹੈ।"*
   - Devanagari (हिन्दी): e.g. *"मुझे कॉलेज प्लेसमेंट के लिए सीवी बनवाने में मदद चाहिए।"*
2. **Romanized / Transliterated Inputs (Hinglish & Punglish)**:
   - Romanized Hindi: *"Mujhe interview preparation ke liye CV review karwana hai, weekend pe."*
   - Romanized Punjabi: *"Mainu Python sikhna hai shuru to, koi weekend te online sikha sakda hai?"*
3. **Multilingual Helper Profiles**:
   - Profiles listing bilingual/trilingual capabilities (`["English", "Punjabi"]`, `["English", "Hindi"]`, `["Punjabi", "Hindi", "English"]`).
   - A subset of helper bios incorporating natural code-switched phrases to mirror authentic bilingual community settings.

---

## 5. Multi-Skill Helpers & Partial Relevance

To test graded relevance (NDCG@3):
- Helpers with complementary pairs: `[python_programming, data_analysis_excel]`, `[resume_review, interview_prep]`, `[high_school_math, physics_tutoring]`, `[graphic_design, video_editing]`.
- Multi-skill queries test whether models rank full matches (Grade 2) ahead of single-skill matches (Grade 1).

---

## 6. No-Match Archetypes

1. **Out-of-Domain Requests**:
   - Requesting skills completely absent from the platform: e.g. *scuba diving certification*, *helicopter piloting*, *dental surgery consultation*, *stained glass restoration*.
2. **Over-Constrained Practical Requests**:
   - The requested skill exists in the corpus, but the combination of mandatory constraints (e.g. language + day + in-person location) cannot be satisfied by any active helper.
   - Expected output: `relevant_helper_ids = []`.

---

## 7. Sample Batch Specifications (Step 3 Preview)
- **20 Sample Profiles (`H001` – `H020`)**:
  - Covering 12+ skills, varied bio lengths and writing styles (formal, casual, short, typos).
  - 2 inactive profiles (`active = False`) to verify hard filtering.
  - Mix of languages (English, Punjabi, Hindi) and session modes (`online`, `in_person`, `both`).
- **10 Sample Queries (`Q001` – `Q010`)**:
  - 2 Paraphrased (lexical mismatch)
  - 2 Vague / Underspecified
  - 2 Multi-skill
  - 2 Code-mixed (1 Romanized Hinglish/Punglish, 1 Gurmukhi/Devanagari)
  - 2 No-match (1 Out-of-Domain, 1 Over-Constrained)
  - Split evenly: 5 validation, 5 test.
