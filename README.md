# Movie Recommendation System

Welcome to the Movie Recommendation System!  
This is a Flask-based web application that suggests similar movies based on your selection. The system uses metadata such as genres, keywords, and production companies to find related films and help users discover new titles.

## Features
• Select a movie from a dropdown menu  
• Get top recommended movies based on similarity  
• Each recommendation shows genres, keywords, production companies, and a short overview  
• Simple, responsive, and user-friendly interface  

## How It Works
1. The user selects a movie from the dropdown list.  
2. The system compares it with other movies in the dataset using a similarity algorithm.  
3. Recommendations are displayed as cards with details such as:  
   • Movie title  
   • Similarity score  
   • Genres, keywords, and production companies  
   • Brief description/overview  

## Recommendation Logic
The system calculates similarity between movies using their metadata:  
• Genres  
• Keywords  
• Production companies  

These features are vectorized and compared to suggest the closest matches.  

## Game Rules (Usage Guidelines)
• Only one movie can be selected at a time  
• Recommendations are generated from the dataset (not live data)  
• Results depend on how similar the metadata is  
• Recommendations may vary depending on the dataset used  

## Tech Stack
• Python (Flask)  
• HTML, Bootstrap, CSS  
• Pandas, scikit-learn (for data handling and similarity calculations)  

