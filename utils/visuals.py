import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib as mpl

rocket_cmap = plt.cm.get_cmap('rocket')
dark_palette_for_bars = [rocket_cmap(0.3), rocket_cmap(0.7)]
sns.set_palette("rocket")

feedback_log = []

def display_movie_cards(recs):
    st.subheader("🎬 Recommended Movies")

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

def plot_scores_bar(score_df):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1E1E2F') # Increased height for better labels
    ax.set_facecolor('#1E1E2F')

    # --- Aesthetic Enhancements ---
    sns.barplot(y='Title', x='score', data=score_df, ax=ax, palette="rocket") # Use "rocket" palette for gradient

    # Remove grid lines
    ax.grid(False)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Ensure left and bottom spines are white
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')

    # Add score labels on the bars
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 0.05, # x position (slightly to the right of the bar)
                p.get_y() + p.get_height() / 2, # y position (center of the bar)
                f'{width:.2f}', # text (the score value)
                va='center', # vertical alignment
                color='white', # text color
                fontsize=10) # font size

    # Set x-axis limits to accommodate labels and provide some padding
    ax.set_xlim(0, score_df['score'].max() * 1.1) # Extend x-axis slightly

    # -----------------------------

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
    # Use the dark_palette for the pie chart colors
    colors = sns.color_palette(n_colors=len(genre_counts)) # Ensure enough colors from the palette
    fig, ax = plt.subplots(facecolor='#1E1E2F')
    ax.set_facecolor('#1E1E2F') # Ensure axes background is also set

    wedges, texts, autotexts = ax.pie(
        genre_counts,
        labels=genre_counts.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors, # Apply the consistent palette
        wedgeprops=dict(width=1, edgecolor='none')
    )
    # Ensure text color for labels and percentages is white
    for text in texts + autotexts:
        text.set_color('white') # Changed from 'transparent' to 'white' based on your request

    ax.axis('equal')
    ax.set_title('Genre Distribution') # Add a title for consistency
    ax.title.set_color('white')
    st.pyplot(fig)

def plot_model_comparison():
    models = ['SVD', 'NMF', 'KNNBasic']
    rmse = [0.87, 0.91, 0.97]
    mae = [0.68, 0.72, 0.77]
    fig, ax = plt.subplots(facecolor='#1E1E2F')
    ax.set_facecolor('#1E1E2F')

    ax.grid(False) # Turn off all grid lines
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
    ax.tick_params(colors='white') # Already set globally, but good to ensure
    ax.legend(facecolor='#1E1E2F', edgecolor='white', labelcolor='white') # Set legend background and text color
    # Set title and label colors
    ax.title.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    st.pyplot(fig)

def plot_genre_heatmap(recommendations, movies):
    # Merge genres if not present
    if 'Genres' not in recommendations.columns:
        movie_data = recommendations.merge(movies[['Title', 'Genres']], on='Title', how='left')
    else:
        movie_data = recommendations.copy()

    movie_data = movie_data.dropna(subset=['Genres'])

    # Multi-hot encode genres
    genre_dummies = movie_data['Genres'].str.get_dummies(sep='|')
    genre_matrix = pd.concat([movie_data[['Title']], genre_dummies], axis=1).set_index('Title')

    # Top 8 genres only
    top_genres = genre_matrix.sum().sort_values(ascending=False).head(8).index
    genre_matrix_top = genre_matrix[top_genres]

    # 📊 Plotting
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1E1E2F')
    ax.set_facecolor('#1E1E2F')

    # Styled heatmap
    sns.heatmap(
        genre_matrix_top.T,
        cmap='rocket',
        linewidths=0.3,
        linecolor='#333',
        cbar=True,
        ax=ax
    )

    # Title and ticks
    ax.set_title("🔥 Top Genres in Recommendations", color='white', fontsize=14)
    ax.tick_params(colors='white', labelsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=45, ha='right', color='white')
    plt.yticks(rotation=0, color='white')
    st.pyplot(fig)
