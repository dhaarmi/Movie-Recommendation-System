import pandas as pd
import sqlite3
import os

# Paths
csv_path = os.path.join(os.path.dirname(__file__), 'tmdb_5000_credits.csv')
db_path = os.path.join(os.path.dirname(__file__), 'movies.db')

def main():
    # Read CSV
    credits = pd.read_csv(csv_path)
    # Connect to SQLite (append to movies.db)
    conn = sqlite3.connect(db_path)
    # Write to new table 'credits'
    credits.to_sql('credits', conn, if_exists='replace', index=False)
    conn.close()
    print('Credits table created in movies.db')

if __name__ == '__main__':
    main()
