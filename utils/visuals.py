import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib as mpl

rocket_cmap = plt.cm.get_cmap('rocket')
dark_palette_for_bars = [rocket_cmap(0.3), rocket_cmap(0.7)]
sns.set_palette("rocket")

feedback_log = []

def display_movie_cards(recs, key_prefix="", title="🎬 Recommended Movies"):
    st.subheader(title)

    for i in range(0, len(recs), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(recs):
                movie = recs.iloc[i + j]
                with cols[j]:
                    st.image(movie['poster'], use_container_width='always')
                    st.caption(f"**{movie['Title']}**")
                    st.write(f"⭐ Score: {movie['score']:.2f}")
                    st.markdown(f"🎭 Genre: `{movie['Genres']}`")

                    print(f"[DEBUG] Rendering movie: {movie['Title']} | Key prefix: {key_prefix}")

                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button(f"👍 Like", key=f"{key_prefix}like_{movie['Title']}"):
                            st.session_state.feedback_log.append((movie['Title'], 'Like'))
                            st.session_state.liked_videos.append(movie.to_dict())
                            st.toast("You liked this movie!", icon="❤️")
                    with col2:
                        if st.button(f"👎 Dislike", key=f"{key_prefix}dislike_{movie['Title']}"):
                            st.session_state.feedback_log.append((movie['Title'], 'DisLike'))
                            st.toast("Noted. We’ll improve your recommendations!", icon="⚠️")

                    if movie['Title'] not in [m['Title'] for m in st.session_state.watchlist]:
                        if st.button("❤️ Add to Watchlist", key=f"{key_prefix}watchlist_{movie['Title']}"):
                            st.session_state.watchlist.append(movie.to_dict())
                            st.toast("Added to your Watchlist!", icon="🎯")
                            st.rerun()
                    else:
                        st.button("✅ Added to Watchlist", key=f"{key_prefix}added_{movie['Title']}", disabled=True)

def plot_scores_bar(score_df):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1E1E2F') 
    ax.set_facecolor('#1E1E2F')
    
    sns.barplot(y='Title', x='score', data=score_df, ax=ax, palette="rocket") 
    
    ax.grid(False)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')

    
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 0.05, 
                p.get_y() + p.get_height() / 2, 
                f'{width:.2f}',
                va='center', 
                color='white', 
                fontsize=10) 

    ax.set_xlim(0, score_df['score'].max() * 1.1) 


    ax.set_title('Recommendation Scores')
    ax.set_xlabel('Score')
    ax.set_ylabel('Movie Title')
    ax.tick_params(colors='white')
    ax.title.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    st.pyplot(fig)

def plot_genre_distribution(recommendations):
    genre_series = recommendations['Genres'].dropna().str.split('|').explode()
    genre_counts = genre_series.value_counts()
    
    colors = sns.color_palette(n_colors=len(genre_counts)) 
    fig, ax = plt.subplots(facecolor='#1E1E2F')
    ax.set_facecolor('#1E1E2F') 

    wedges, texts, autotexts = ax.pie(
        genre_counts,
        labels=genre_counts.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors, 
        wedgeprops=dict(width=1, edgecolor='none')
    )
    
    for text in texts + autotexts:
        text.set_color('white') 

    ax.axis('equal')
    ax.set_title('Genre Distribution') 
    ax.title.set_color('white')
    st.pyplot(fig)

def plot_model_comparison():
    models = ['SVD', 'NMF', 'KNNBasic']
    rmse = [0.87, 0.91, 0.97]
    mae = [0.68, 0.72, 0.77]
    fig, ax = plt.subplots(facecolor='#1E1E2F')
    ax.set_facecolor('#1E1E2F')

    ax.grid(False) 
    ax.tick_params(which='both', axis='both', length=0)
    
    x = range(len(models))

    rmse_color = dark_palette_for_bars[0]
    mae_color = dark_palette_for_bars[1]

    ax.bar(x, rmse, width=0.4, label='RMSE', align='center', color=rmse_color)
    ax.bar([p + 0.4 for p in x], mae, width=0.4, label='MAE', align='center', color=mae_color)
    
    ax.set_xticks([p + 0.2 for p in x])
    ax.set_xticklabels(models)
    ax.set_ylabel('Error Value')
    ax.set_title('Model Evaluation Metrics')
    ax.tick_params(colors='white') 
    ax.legend(facecolor='#1E1E2F', edgecolor='white', labelcolor='white') 
    ax.title.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    st.pyplot(fig)

def plot_genre_heatmap(recommendations, movies):
    
    if 'Genres' not in recommendations.columns:
        movie_data = recommendations.merge(movies[['Title', 'Genres']], on='Title', how='left')
    else:
        movie_data = recommendations.copy()

    movie_data = movie_data.dropna(subset=['Genres'])

    
    genre_dummies = movie_data['Genres'].str.get_dummies(sep='|')
    genre_matrix = pd.concat([movie_data[['Title']], genre_dummies], axis=1).set_index('Title')

    
    top_genres = genre_matrix.sum().sort_values(ascending=False).head(8).index
    genre_matrix_top = genre_matrix[top_genres]

    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1E1E2F')
    ax.set_facecolor('#1E1E2F')

    
    sns.heatmap(
        genre_matrix_top.T,
        cmap='rocket',
        linewidths=0.3,
        linecolor='#333',
        cbar=True,
        ax=ax
    )

    
    ax.set_title("Top Genres in Recommendations", color='white', fontsize=14)
    ax.tick_params(colors='white', labelsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=45, ha='right', color='white')
    plt.yticks(rotation=0, color='white')
    st.pyplot(fig)
