import pandas as pd
import sqlite3

# Load the CSV file
movies = pd.read_csv('tmdb_5000_movies.csv')

# Save to SQLite database
db_path = 'movies.db'
conn = sqlite3.connect(db_path)
movies.to_sql('movies', conn, if_exists='replace', index=False)
conn.close()

print(f"Converted tmdb_5000_movies.csv to {db_path}")
