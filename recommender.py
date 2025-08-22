# recommender.py
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("tmdb_5000_movies.csv")
movies['overview'] = movies['overview'].fillna('')

# Helper function to parse JSON-like columns
def parse_column(column):
    if pd.isna(column):
        return []
    try:
        return [i['name'] for i in ast.literal_eval(column) if i.get('name')]
    except:
        return []

# Parse relevant columns
movies['genres'] = movies['genres'].apply(parse_column)
movies['keywords'] = movies['keywords'].apply(parse_column)
movies['production_companies'] = movies['production_companies'].apply(parse_column)

# Create tags for similarity
def create_tags(row):
    tags = row['overview'] + ' '
    tags += ' '.join(row['genres']*3) + ' '  # emphasize genres
    tags += ' '.join(row['keywords']) + ' '
    tags += ' '.join(row['production_companies'])
    return tags.lower()

movies['tags'] = movies.apply(create_tags, axis=1)

# TF-IDF and cosine similarity
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['tags'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation function
def recommend(movie_title, top_n=5):
    if movie_title not in movies['title'].values:
        return []

    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    recommendations = []
    for i, score in sim_scores:
        target = movies.iloc[i]
        recommendations.append({
            'title': target['title'],
            'genres': target['genres'],
            'keywords': target['keywords'],
            'production_companies': target['production_companies'],
            'overview': target['overview'],
            'similarity': round(score, 2)
        })
    return recommendations
