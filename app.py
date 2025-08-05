import streamlit as st
import pandas as pd
import joblib
from utils.recommend import (
    recommend_collaborative, recommend_content, recommend_hybrid, fetch_trending_movies
)
from utils.visuals import (
    display_movie_cards, plot_scores_bar, plot_genre_distribution,
    plot_model_comparison, plot_genre_heatmap
)
import matplotlib as mpl

# 🔧 Dark theme for plots
mpl.rcParams.update({
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white'
})

# 🧠 Cache models and data
@st.cache_resource
def load_models():
    return {
        "svd_model": joblib.load("models/svd_model.pkl"),
        "cosine_sim": joblib.load("models/tfidf_cosine_sim.pkl"),
        "movie_indices": joblib.load("models/movie_indices.pkl")
    }

@st.cache_data
def load_movies():
    return pd.read_csv("data/movies.csv")


for key in ["watchlist", "feedback_log", "recommend_triggered", "active_page", "liked_videos"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key in ["watchlist", "feedback_log", "liked_videos"] else (
            "home" if key == "active_page" else False
        )

# 🧾 Page setup
st.set_page_config(page_title="🎥 Smart Movie Recommender", layout="wide")
st.markdown("""<style>
    .stApp { background-color: #1E1E2F; color: white; }
    .result-box {
        background-color: black;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        margin-top: 20px;
        margin-bottom: 20px;
    }
    h1, h2, h3 { color: #FFA726; }
    .css-18e3th9, .st-c6 { background-color: #2B2B3D !important; }
</style>""", unsafe_allow_html=True)

# 📦 Load model/data
models = load_models()
movies = load_movies()

# 🔧 Sidebar Controls
st.sidebar.header("🔧 Settings")
nav_col1, nav_col2 = st.sidebar.columns(2)
with nav_col1:
    if st.button("🏠 Home"):
        st.session_state.active_page = "home"
    if st.button("❤️ Liked Videos"):
        st.session_state.active_page = "liked_videos"
with nav_col2:
    if st.button("📂 Watchlist"):
        st.session_state.active_page = "watchlist"


user_id = st.sidebar.number_input("👤 Enter User ID", min_value=1, value=10)
model_type = st.sidebar.radio("🤖 Choose Recommendation Type", ["Collaborative", "Content-Based", "Hybrid"])

genres = sorted(list(set('|'.join(movies['Genres'].dropna()).split('|'))))
selected_genre = st.sidebar.selectbox("🎯 Filter by Genre (optional)", ["All"] + genres)

mood_to_genres = {
    "None": [], "Happy 😊": ["Comedy", "Family"], "Sad 😢": ["Drama", "Romance"],
    "Thrilling 😱": ["Thriller", "Action"], "Chill 😌": ["Adventure", "Fantasy"],
    "Romantic ❤️": ["Romance", "Drama"]
}
selected_mood = st.sidebar.selectbox("🎭 Filter by Mood (optional)", list(mood_to_genres.keys()))
title_list = sorted(movies['Title'].dropna().unique().tolist())
fav_movie = st.sidebar.selectbox("🎞️ Search or Select a Movie You Like", title_list)

if st.sidebar.button("🚀 Recommend"):
    st.session_state.recommend_triggered = True

# 💻 Main Area Logic
if st.session_state.active_page == "watchlist":
    st.title("📂 Your Watchlist")
    if st.session_state.watchlist:
        watchlist_df = pd.DataFrame(st.session_state.watchlist)
        display_movie_cards(watchlist_df, key_prefix="watchlist_")
    else:
        st.info("Your watchlist is empty. Add movies to it from recommendations!")

elif st.session_state.active_page == "liked_videos":
    st.title("❤️ Liked Videos")
    if st.session_state.liked_videos:
        liked_videos_df = pd.DataFrame(st.session_state.liked_videos)
        display_movie_cards(liked_videos_df, key_prefix="liked_videos_")
    else:
        st.info("There are no moviees liked. Add movies to it from recommendations!")

elif st.session_state.active_page == "home":
    st.title("🎬 Smart Movie Recommender System")
    st.markdown("Select a model and discover movies you’ll love 🎉")

    with st.expander("🔥 Trending This Week", expanded=True):
        st.markdown("Here are some movies currently trending worldwide:")
        trending = fetch_trending_movies()
        if trending:
            display_movie_cards(pd.DataFrame(trending), key_prefix="trending_")
        else:
            st.warning("Could not fetch trending movies.")

    if st.session_state.recommend_triggered:
        if model_type == "Collaborative":
            recs, scores = recommend_collaborative(user_id, models["svd_model"], movies)
        elif model_type == "Content-Based":
            recs, scores = recommend_content(fav_movie, models["cosine_sim"], movies, models["movie_indices"])
        else:
            recs, scores = recommend_hybrid(
                user_id, fav_movie,
                models["svd_model"], models["cosine_sim"],
                movies, models["movie_indices"]
            )

        # Apply Genre & Mood filters
        if selected_genre != "All":
            recs = recs[recs['Genres'].str.contains(selected_genre, na=False)]

        if selected_mood != "None":
            mood_genres = mood_to_genres[selected_mood]
            recs = recs[recs['Genres'].apply(lambda g: any(mg in g for mg in mood_genres) if isinstance(g, str) else False)]

        display_movie_cards(recs, key_prefix="recs_")
        st.subheader("📊 Recommendation Scores")
        plot_scores_bar(scores)

        st.subheader("🎭 Genre Spread in Recommendations")
        plot_genre_distribution(recs)

        st.subheader("📈 Model Evaluation Snapshot")
        plot_model_comparison()

        st.subheader("🔥 Top Genre Preferences in Recommendations")
        plot_genre_heatmap(recs, movies)

# 📌 Footer
st.markdown("<hr style='margin-top:30px;'>", unsafe_allow_html=True)
st.caption("Built with ❤️ by Lokesh Reddy Kothur")
