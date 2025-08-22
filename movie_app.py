import os
from flask import Flask, render_template, request
from recommender import movies, recommend
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, template_folder="templates")
app.secret_key = os.environ['FLASK_SECRET_KEY']  # no fallback

@app.route('/', methods=['GET', 'POST'])
def home():
    movie_options = sorted(movies['title'].tolist())
    recommendations = []
    selected_movie = None

    if request.method == 'POST':
        selected_movie = request.form.get('movie')
        if selected_movie:
            recommendations = recommend(selected_movie, top_n=5)

    return render_template(
        'index.html',
        movies=movie_options,
        recommendations=recommendations,
        selected_movie=selected_movie
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
