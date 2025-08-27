import os
import random
import time
from flask import Flask, render_template, request, url_for
from recommender import movies, recommend
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, template_folder="templates")
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-movie-recommender-key')  # fallback for development

# Create a version timestamp for cache busting
version = int(time.time())

# Add cache control for all responses

# Main route for movie recommendations
@app.route('/', methods=['GET', 'POST'])
def index():
    movie_options = sorted(movies['title'].tolist())
    recommendations = []
    selected_movie = None
    selected_basis = 'genre'


    selected_movie_genres = []
    selected_movie_keywords = []
    if request.method == 'POST':
        selected_movie = request.form.get('movie')
        selected_basis = request.form.get('rec_basis', 'genre')
        if selected_movie:

            # Use normalized scoring by default
            recommendations = recommend(selected_movie, top_n=6, normalize=True)
            for rec in recommendations:
                rec['color_h'] = random.randint(200, 260)
                rec['color_s'] = random.randint(70, 90)
                rec['color_l'] = random.randint(40, 60)
            # Get genres and keywords for selected movie for template display
            row = movies[movies['title'] == selected_movie]
            if not row.empty:
                selected_movie_genres = row.iloc[0]['genres']
                selected_movie_keywords = row.iloc[0]['keywords']

    return render_template(
        'index.html',
        movies=movie_options,
        recommendations=recommendations,
        selected_movie=selected_movie,
        selected_basis=selected_basis,
        selected_movie_genres=selected_movie_genres,
        selected_movie_keywords=selected_movie_keywords,
        version=version
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=True)
