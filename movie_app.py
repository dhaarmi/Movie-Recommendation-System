# movie_app.py
from flask import Flask, render_template, request
from recommender import movies, recommend

# Explicitly specify templates folder
app = Flask(__name__, template_folder="templates")

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

if __name__ == '__main__':
    app.run(debug=True)
