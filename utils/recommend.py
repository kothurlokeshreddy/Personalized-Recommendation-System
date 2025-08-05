from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import requests
import streamlit as st
import re

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

def fetch_trending_movies(count=6):
    try:
        api_key = os.getenv("TMDB_API_KEY")
        url = f"https://api.themoviedb.org/3/trending/movie/week?api_key={TMDB_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            trending = data.get("results", [])[:count]
            score = np.random.uniform(low=3.5, high=5.0)
            return [{
                "Title": movie.get("title"),
                "Genres":'',
                "score": score,
                "poster": f"https://image.tmdb.org/t/p/w500{movie.get('poster_path')}" if movie.get("poster_path") else "https://image.tmdb.org/t/p/original/rBxo92GmbsQbinrbJOFnmiKuMXj.jpg"
            } for movie in trending]
        else:
            return []
    except Exception as e:
        print("[ERROR] Could not fetch trending movies:", e)
        return []


def clean_title(title):
    return re.sub(r"\s*\(\d{4}\)$", "", title).strip()

@st.cache_data(show_spinner=False)
def get_poster(title):
    try:
        query_title = clean_title(title)
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={query_title}"
        response = requests.get(url).json()
        if response['results']:
            poster_path = response['results'][0].get('poster_path')
            if poster_path:
                return str(f"https://image.tmdb.org/t/p/w500/{poster_path}").strip()
        return "https://image.tmdb.org/t/p/original/rBxo92GmbsQbinrbJOFnmiKuMXj.jpg"
    except Exception as e:
        print(f"[ERROR] Poster fetch failed for '{title}':", e)
        return "https://image.tmdb.org/t/p/original/rBxo92GmbsQbinrbJOFnmiKuMXj.jpg"

def recommend_collaborative(user_id, model, movies, top_n=10):
    predictions = []
    popular_movies = movies[movies['MovieID'].isin(movies['MovieID'].value_counts().head(50).index)]
    for _, row in popular_movies.iterrows():
        try:
            est = model.predict(user_id, row['MovieID']).est
            predictions.append({
                'Title': row['Title'],
                'score': est,
                'Genres': row['Genres'],
                'poster': str(get_poster(row['Title'])).strip()
            })
        except:
            continue
    recs = pd.DataFrame(predictions).sort_values(by='score', ascending=False).head(top_n)
    return recs, recs[['Title', 'score']]

def recommend_content(fav_movie, cosine_sim, movies, movie_indices, top_n=10):
    idx = movie_indices[fav_movie]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_idxs = [i[0] for i in sim_scores]
    scores = [i[1] * 5 for i in sim_scores]
    recs = movies.iloc[movie_idxs].copy()
    recs['score'] = scores
    recs['poster'] = recs['Title'].apply(lambda x: str(get_poster(x)).strip())
    return recs, recs[['Title', 'score']]

def recommend_hybrid(user_id, fav_movie, model, cosine_sim, movies, movie_indices, top_n=10):
    idx = movie_indices[fav_movie]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+15]
    hybrid = []
    for i, sim in sim_scores:
        row = movies.iloc[i]
        try:
            est = model.predict(user_id, row['MovieID']).est
            final_score = 0.5 * est + 0.5 * sim * 5
            hybrid.append({
                'Title': row['Title'],
                'Genres': row['Genres'],
                'score': final_score,
                'poster': str(get_poster(row['Title'])).strip()
            })
        except:
            continue
    df = pd.DataFrame(hybrid).sort_values(by="score", ascending=False).head(top_n)
    return df, df[['Title', 'score']]