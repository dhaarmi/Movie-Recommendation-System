import random
import numpy as np
from recommender import movies, recommend

# Improved precision@K using stricter relevance rules: a recommended movie counts as relevant
# if it (shares at least 2 genres) OR (shares at least 1 genre AND 1 keyword) with the seed movie.

def is_relevant(seed_genres, seed_keywords, rec_genres, rec_keywords):
    """Determine if a movie is relevant based on stricter rules."""
    shared_genres = seed_genres.intersection(set(rec_genres))
    
    # Relevance rule 1: At least 2 shared genres
    if len(shared_genres) >= 2:
        return True
        
    # Relevance rule 2: At least 1 shared genre AND 1 shared keyword
    shared_keywords = seed_keywords.intersection(set(rec_keywords))
    if len(shared_genres) >= 1 and len(shared_keywords) >= 1:
        return True
        
    return False

def precision_at_k(seed_title, recs, k=6):
    seed_row = movies[movies['title'] == seed_title]
    if seed_row.empty:
        return 0.0
    seed_genres = set(seed_row.iloc[0]['genres'])
    seed_keywords = set(seed_row.iloc[0]['keywords'])
    if not seed_genres:
        return 0.0
    
    relevant = 0
    for r in recs[:k]:
        if is_relevant(seed_genres, seed_keywords, r.get('genres', []), r.get('keywords', [])):
            relevant += 1
    
    return relevant / k if k > 0 else 0.0


def evaluate(sample_size=50, k=6):
    titles = movies['title'].dropna().unique().tolist()
    random.seed(42)
    sample = random.sample(titles, min(sample_size, len(titles)))

    # Define different recommendation approaches to compare
    approaches = [
        {"name": "Original", "normalize": False, "use_metadata": False},
        {"name": "Normalized", "normalize": True, "use_metadata": False},
        {"name": "Normalized+Metadata", "normalize": True, "use_metadata": True},
        {"name": "Overview Only", "normalize": True, "use_metadata": False, 
         "weights": {"genre": 0.0, "overview": 1.0, "keyword": 0.0, "actor": 0.0, "metadata": 0.0}},
        {"name": "Metadata Only", "normalize": True, "use_metadata": True,
         "weights": {"genre": 0.0, "overview": 0.0, "keyword": 0.0, "actor": 0.0, "metadata": 1.0}}
    ]

    # Store scores for each approach
    all_scores = {approach["name"]: [] for approach in approaches}
    
    # Evaluate each approach on each sample movie
    for t in sample:
        print(f"Processing {t}...")
        for approach in approaches:
            weights = approach.get("weights", None)
            recs = recommend(t, top_n=k, 
                            normalize=approach["normalize"],
                            use_metadata=approach["use_metadata"],
                            weights=weights)
            score = precision_at_k(t, recs, k)
            all_scores[approach["name"]].append(score)

    # Print results
    print(f"\nEvaluated {len(sample)} seeds. k={k}")
    print(f"Relevance criteria: >=2 genres OR (>=1 genre AND >=1 keyword)")
    print("-" * 60)
    
    for approach_name, scores in all_scores.items():
        mean_score = np.mean(scores)
        print(f"{approach_name} precision@{k}: {mean_score:.3f}")


if __name__ == '__main__':
    evaluate(sample_size=20, k=6)  # Reduced sample size for faster evaluation
