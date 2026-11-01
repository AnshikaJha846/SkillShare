# SkillShare Match: Explainable Community Skill-Matching Prototype

An applied-NLP capstone prototype that matches community members based on natural language requests and helper profiles, enforces practical constraints as hard filters, and explains recommendations using only profile-grounded facts.

## Research Question
> **Does semantic retrieval improve top-3 recommendations for paraphrased, vague, and code-mixed (Hindi/Punjabi/English) requests compared with lexical baselines, while maintaining constraints?**

---

## Architecture & Pipeline Overview

```
Requester Query (Free-text + Constraints)
                   │
                   ▼
       ┌───────────────────────┐
       │   Hard Constraint     │  ---> Filter out inactive profiles & profiles
       │   Filtering           │       violating mandatory constraints (language,
       └───────────┬───────────┘       availability, session mode)
                   │
                   ▼
       ┌───────────────────────┐
       │   Retrieval & Ranking │  ---> Methods compared:
       │   (Lexical / Semantic)│       1. TF-IDF + Cosine
       └───────────┬───────────┘       2. BM25 (rank_bm25)
                   │                   3. English Sentence-BERT (bi-encoder)
                   ▼                   4. Multilingual Bi-Encoder (MiniLM / E5)
       ┌───────────────────────┐       5. (Optional) Bi-encoder + Cross-encoder
       │ Preference Adjustment │  ---> Transparent score adjustments for preferred
       └───────────┬───────────┘       (soft) constraints
                   │
                   ▼
       ┌───────────────────────┐
       │ Grounded Explanations │  ---> Template-based explanations using strictly
       └───────────────────────┘       profile facts (no hallucinations)
```

---

## Project Structure

```
skillshare-match/
├── data/                  # Synthetic helper profiles, queries, labels, rubric
│   ├── rubric.md          # Annotation rubric (relevance grades & no-match rules)
│   ├── profiles.csv       # Fictional helper profiles
│   ├── queries.json       # Evaluation queries across categories
│   └── labels.json        # Ground-truth relevance annotations
├── src/                   # Core modular logic
│   ├── __init__.py
│   ├── data_loader.py     # Pydantic / Dataclass schemas and validators
│   ├── preprocessing.py   # Information-preserving text normalizer
│   ├── indexing.py        # Vector & inverted index management
│   ├── retrieval.py       # TF-IDF, BM25, and Bi-encoder retrieval models
│   ├── constraints.py     # Hard filters and soft preference scoring
│   ├── explain.py         # Profile-grounded explanation generation
│   ├── metrics.py         # Precision@3, Recall@3, MRR, NDCG@3, constraint sat.
│   └── stats.py           # Bootstrap CIs and paired significance tests
├── app/                   # Interactive Streamlit application
│   ├── streamlit_app.py   # Search interface & demo
│   └── pages/             # Evaluation dashboard & error analysis
├── tests/                 # Automated pytest suite
├── reports/               # Evaluation report and analysis
│   └── evaluation_report.md
├── requirements.txt       # Pinned project dependencies
└── README.md
```

---

## Setup & Local Installation

This project is self-contained, runs offline, and requires no external API keys or LLM services.

```bash
# 1. Create and activate a virtual environment (Python 3.10+)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

# 2. Install pinned dependencies
pip install -r requirements.txt

# 3. Run automated test suite (37 tests across all modules)
pytest -v

# 4. Run the benchmark evaluation suite (generates reports/evaluation_report.md)
python scripts/run_evaluation.py

# 5. Launch the Streamlit interactive app
streamlit run app/streamlit_app.py
```

---

## Benchmark Results (Held-Out Test Set)

*Generated directly by `scripts/run_evaluation.py` on 25 held-out test queries against 120 helper profiles. Numbers report mean and **95% non-parametric bootstrap confidence intervals**.*

| Retrieval Method | Precision@3 | Recall@3 | MRR | NDCG@3 | Constraint Sat. | No-Match Acc. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. TF-IDF + Cosine (Baseline)** | 0.493 [0.360, 0.653] | 0.483 [0.345, 0.645] | 0.613 [0.453, 0.780] | 0.550 [0.405, 0.714] | 100.0% | 100.0% |
| **2. Okapi BM25 (Lexical)** | 0.573 [0.440, 0.707] | 0.511 [0.378, 0.647] | 0.673 [0.540, 0.813] | 0.610 [0.477, 0.753] | 100.0% | 92.0% |
| **3. S-BERT English (`all-MiniLM-L6-v2`)** | **0.693** [0.520, 0.840] | **0.580** [0.438, 0.711] | **0.800** [0.640, 0.960] | **0.778** [0.603, 0.926] | **100.0%** | **92.0%** |
| **4. Multilingual S-BERT (`MiniLM-L12`)** | 0.640 [0.453, 0.813] | 0.494 [0.352, 0.632] | 0.687 [0.506, 0.840] | 0.678 [0.493, 0.840] | **100.0%** | **92.0%** |

### Key Experimental Insights
1. **Semantic superiority on paraphrases & vague queries**: Dense bi-encoders achieve **1.000 NDCG@3** on paraphrased requests (resolving CV vs. résumé token mismatches) and vague requests (matching small business finances to spreadsheet reporting), where TF-IDF achieves only `0.753` and `0.106` respectively.
2. **Hard constraints are essential**: Without hard constraint filters, semantic models recommend language-mismatched or unavailable helpers, causing constraint satisfaction to collapse from **100.0% down to 86.7%** (see [reports/evaluation_report.md](file:///c:/Users/ASUS/SkillShare/reports/evaluation_report.md)).
3. **Sub-second vector retrieval**: Embeddings are precomputed and cached on disk in `.npz` format, achieving < 5ms retrieval response times completely offline.

---

## Hard Rules & Principles

1. **Hard Constraints are Filters**: Required constraints (language, day availability, session mode) are applied strictly as candidate filters, never as soft penalties.
2. **Zero Hallucination Explanations**: Explanations are constructed using deterministic templates grounded strictly in profile fields.
3. **Similarity Scores are Ranking Signals**: Scores are relative ranking indicators, not calibrated probabilities of match success.
4. **Preservative Preprocessing**: Domain-specific terms (CV, résumé, Python, Excel, language names, days) are strictly preserved during tokenization.
5. **No Synthetic Number Claims**: No evaluation numbers are recorded in documentation until generated and validated by the evaluation pipeline on held-out test splits.
6. **Synthetic Data Disclosure**: All helper profiles and queries are entirely fictional and synthetic for research prototyping. No real personal data or exact addresses are used.

