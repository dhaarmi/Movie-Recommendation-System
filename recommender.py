# recommender.py
# ...existing code...
import os



import os
import pandas as pd
import ast
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Load datasets from SQLite
db_path = os.path.join(os.path.dirname(__file__), "movies.db")
conn = sqlite3.connect(db_path)
movies = pd.read_sql_query("SELECT * FROM movies", conn)
credits = pd.read_sql_query("SELECT * FROM credits", conn)
conn.close()
movies['overview'] = movies['overview'].fillna('')


# Helper function to parse JSON-like columns
def parse_column(column):
    if pd.isna(column):
        return []
    try:
        return [i['name'] for i in ast.literal_eval(column) if i.get('name')]
    except Exception:
        return []

# Helper to parse cast and crew columns (get top 3 actors, top 2 crew by job)
def parse_cast(cast_str):
    if pd.isna(cast_str):
        return []
    try:
        cast = ast.literal_eval(cast_str)
        return [c['name'] for c in cast[:3] if 'name' in c]
    except Exception:
        return []

def parse_crew(crew_str):
    if pd.isna(crew_str):
        return []
    try:
        crew = ast.literal_eval(crew_str)
        # Get director and writer if present
        important = [c['name'] for c in crew if c.get('job') in ['Director', 'Writer'] and 'name' in c]
        return important[:2]
    except Exception:
        return []



# Parse relevant columns
movies['genres'] = movies['genres'].apply(parse_column)
movies['keywords'] = movies['keywords'].apply(parse_column)
movies['production_companies'] = movies['production_companies'].apply(parse_column)

# Default weights (can be overridden at call time)
DEFAULT_WEIGHTS = {
    'genre': 5.0,
    'overview': 3.0,
    'keyword': 2.0,
    'actor': 1.0,
    'metadata': 4.0  # Weight for combined metadata TF-IDF
}

# Merge cast and crew from credits into movies DataFrame
def get_cast_names(movie_id):
    row = credits[credits['movie_id'] == movie_id]
    if row.empty:
        return []
    return parse_cast(row.iloc[0]['cast'])

def get_crew_names(movie_id):
    row = credits[credits['movie_id'] == movie_id]
    if row.empty:
        return []
    return parse_crew(row.iloc[0]['crew'])

movies['cast_names'] = movies['id'].apply(get_cast_names)
movies['crew_names'] = movies['id'].apply(get_crew_names)

# Create metadata text for combined TF-IDF
def create_metadata_text(row):
    """Create a combined metadata string from a movie's features."""
    genres = ' '.join(row['genres']) if isinstance(row['genres'], list) else ''
    keywords = ' '.join(row['keywords']) if isinstance(row['keywords'], list) else ''
    cast = ' '.join(row['cast_names']) if isinstance(row['cast_names'], list) else ''
    companies = ' '.join(row['production_companies']) if isinstance(row['production_companies'], list) else ''
    
    # Give more weight to important features by repeating them
    genres_weighted = ' '.join([genres] * 3)  # Repeat genres 3 times for more weight
    keywords_weighted = ' '.join([keywords] * 2)  # Repeat keywords 2 times
    
    return f"{genres_weighted} {keywords_weighted} {cast} {companies}".strip()

# Add metadata_text column
movies['metadata_text'] = movies.apply(create_metadata_text, axis=1)

# Precompute metadata TF-IDF matrix (will be used in recommend function)
metadata_tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
metadata_matrix = metadata_tfidf.fit_transform(movies['metadata_text'])
metadata_sim = None  # Will be computed on first use

# Recommendation function with basis selection
def recommend(movie_title, top_n=6, normalize=True, weights=None, use_metadata=True):
    """
    Recommend movies similar to `movie_title`.

    Args:
        movie_title (str): title of the movie to base recommendations on
        top_n (int): number of recommendations to return
        normalize (bool): if True, normalize genre/keyword/actor counts to [0..1] before weighting
        weights (dict): optional weights for keys 'genre','overview','keyword','actor','metadata'
        use_metadata (bool): if True, include metadata TF-IDF similarity in scoring

    Returns:
        list of dicts with recommendation metadata
    """
    if movie_title not in movies['title'].values:
        return []

    # Prepare similarity matrix for plot/overview if not already present
    if 'cosine_sim' not in globals():
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(movies['overview'])
        global cosine_sim
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Prepare metadata similarity matrix if not already present and metadata is requested
    global metadata_sim
    if use_metadata and metadata_sim is None:
        metadata_sim = cosine_similarity(metadata_matrix, metadata_matrix)

    if weights is None:
        weights = DEFAULT_WEIGHTS

    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = [x for x in sim_scores if x[0] != idx]

    # Get metadata similarity scores if enabled
    metadata_scores = {}
    if use_metadata and metadata_sim is not None:
        metadata_scores = {i: score for i, score in enumerate(metadata_sim[idx]) if i != idx}

    selected = movies.iloc[idx]
    sel_genres = set(selected['genres'])
    sel_keywords = set(selected['keywords'])
    sel_cast = set(selected['cast_names']) if 'cast_names' in selected else set()

    # Precompute denominators for normalization to avoid divide-by-zero
    max_genres = max(len(sel_genres), 1)
    max_keywords = max(len(sel_keywords), 1)
    max_actors = max(len(sel_cast), 1)

    scored = []
    for i, base_score in sim_scores:
        target = movies.iloc[i]
        shared_genres = list(sel_genres.intersection(target['genres']))
        shared_keywords = list(sel_keywords.intersection(target['keywords']))
        shared_actors = list(sel_cast.intersection(target['cast_names'])) if 'cast_names' in target else []

        # raw counts
        genre_score = len(shared_genres)
        keyword_score = len(shared_keywords)
        actor_score = len(shared_actors)
        
        # Get metadata similarity for this movie (defaults to 0 if not found)
        metadata_score = metadata_scores.get(i, 0.0) if use_metadata else 0.0

        if normalize:
            # Normalize counts into [0..1]
            genre_norm = genre_score / max_genres
            keyword_norm = keyword_score / max_keywords
            actor_norm = actor_score / max_actors
            overview_norm = float(base_score)  # already in [0..1]
            # metadata_score is already a cosine similarity in [0..1]

            total_score = (
                weights.get('genre', DEFAULT_WEIGHTS['genre']) * genre_norm +
                weights.get('overview', DEFAULT_WEIGHTS['overview']) * overview_norm +
                weights.get('keyword', DEFAULT_WEIGHTS['keyword']) * keyword_norm +
                weights.get('actor', DEFAULT_WEIGHTS['actor']) * actor_norm +
                weights.get('metadata', DEFAULT_WEIGHTS['metadata']) * metadata_score if use_metadata else 0
            )
        else:
            # backward-compatible: original integer-count based scoring
            total_score = (
                weights.get('genre', DEFAULT_WEIGHTS['genre']) * genre_score +
                weights.get('overview', DEFAULT_WEIGHTS['overview']) * base_score +
                weights.get('keyword', DEFAULT_WEIGHTS['keyword']) * keyword_score +
                weights.get('actor', DEFAULT_WEIGHTS['actor']) * actor_score +
                weights.get('metadata', DEFAULT_WEIGHTS['metadata']) * metadata_score if use_metadata else 0
            )

        # Add metadata_score to the tuple
        scored.append((i, total_score, base_score, genre_score, keyword_score, actor_score, metadata_scores.get(i, 0.0) if use_metadata else 0.0, shared_genres, shared_keywords, shared_actors))

    # Sort by total_score, then by overview base_score
    scored = sorted(scored, key=lambda x: (x[1], x[2]), reverse=True)[:top_n]

    recommendations = []
    for i, total_score, base_score, genre_score, keyword_score, actor_score, metadata_score, shared_genres, shared_keywords, shared_actors in scored:
        target = movies.iloc[i]
        recommendations.append({
            'title': target['title'],
            'genres': target['genres'],
            'keywords': target['keywords'],
            'production_companies': target['production_companies'],
            'overview': target['overview'],
            'cast': target['cast_names'],
            'genre_score': genre_score,
            'keyword_score': keyword_score,
            'actor_score': actor_score,
            'metadata_score': round(metadata_score, 2),
            'shared_genres': shared_genres,
            'shared_keywords': shared_keywords,
            'shared_actors': shared_actors,
            'similarity': round(base_score, 2),
            'total_score': total_score
        })
    return recommendations
