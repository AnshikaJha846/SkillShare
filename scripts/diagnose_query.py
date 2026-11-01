import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_profiles_from_csv
from src.retrieval import get_retriever


profiles = load_profiles_from_csv("data/profiles.csv")
query = "i want review for my project and advice for further improvement"

for method in ["tfidf", "bm25", "sbert_en", "sbert_multi"]:
    retriever = get_retriever(method, profiles)
    results = retriever.search(query, top_k=5, min_threshold=0.0)
    print(f"\n--- Method: {method} ---")
    for r in results:
        print(f"[{r.score:.3f}] {r.helper_id} {r.profile.name} - Skills: {r.profile.skills} - Bio: {r.profile.bio[:70]}...")
