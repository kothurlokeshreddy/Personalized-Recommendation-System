# 🎬 Smart Movie Recommender System

An interactive AI-powered web application built with **Streamlit** that offers movie recommendations using **Collaborative Filtering**, **Content-Based Filtering**, and **Hybrid Modeling**. The app features a modern UI, interactive visualizations, and user-personalized features like Watchlists and Mood-based filtering.

![App Preview](https://img.shields.io/badge/Streamlit-Movie%20Recommender-%23ff4b4b)

---

## 📌 Table of Contents

- [📌 Table of Contents](#-table-of-contents)
- [🚀 Features](#-features)
- [🛠️ Tech Stack](#️-tech-stack)
- [📂 Project Structure](#-project-structure)
- [📈 Recommendation Models](#-recommendation-models)
- [📊 Visualizations](#-visualizations)
- [🎯 Challenges Faced](#-challenges-faced)
- [🔧 Possible Improvements](#-possible-improvements)
- [▶️ How to Run the App](#️-how-to-run-the-app)

---

## 🚀 Features

- 🔎 **Collaborative Filtering** using SVD (Singular Value Decomposition)
- 🧠 **Content-Based Filtering** using TF-IDF and Cosine Similarity on genres
- 🔄 **Hybrid Model** combining both approaches
- 📂 **Watchlist**: Save your favorite movies
- ❤️ **Like/Dislike Feedback** (Simulated)
- 🔥 **Trending Movies Section** using TMDB API
- 🎭 **Mood-based Recommendations** (Happy, Sad, Thrilling, etc.)
- 📊 **Visual Insights** (bar charts, pie charts, genre heatmaps)
- 🎨 **Dark-themed dashboard** with consistent UI/UX

---

## 🛠️ Tech Stack

- **Frontend & UI**: [Streamlit](https://streamlit.io/)
- **ML Algorithms**: `surprise`, `scikit-learn`
- **Data Handling**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **APIs**: [TMDB API](https://www.themoviedb.org/documentation/api) for posters & trending
- **Model Persistence**: `joblib`
- **Deployment Ready**: Streamlit Web App

---

## 📂 Project Structure

```bash
recommender-app/
│
├── app.py # Main Streamlit App
├── models/
│ ├── svd_model.pkl
│ ├── tfidf_cosine_sim.pkl
│ └── movie_indices.pkl
│
├── data/
│ └── movies.csv # Movie metadata
│
├── utils/
│ ├── recommend.py # Model logic & TMDB integration
│ └── visuals.py # Visualizations & movie card UI
│
├── requirements.txt
└── README.md
```

---


---

## 📈 Recommendation Models

| Model           | RMSE   | MAE   | Precision@5 |
|----------------|--------|-------|-------------|
| SVD            | 0.87   | 0.68  | 0.78        |
| NMF            | 0.91   | 0.72  | 0.75        |
| KNNBasic       | 0.97   | 0.77  | 0.71        |

✅ **SVD** was selected as the final collaborative filtering model for deployment.

---

## 📊 Visualizations

- 📊 **Recommendation Score Bar Plot**
- 🎭 **Pie Chart of Genre Distribution**
- 📈 **Model Evaluation Chart (RMSE vs MAE)**
- 🔥 **Genre Heatmap** across top N recommended movies

---

## 🎯 Challenges Faced

- Handling **duplicate keys** in Streamlit while dynamically generating buttons
- Managing **session state** across pages (Watchlist, Likes)
- Ensuring consistent **dark theme styling** for all components
- TMDB API limitations (missing posters, throttling)
- Performance issues with model prediction/rendering (resolved via caching)

---

## 🔧 Possible Improvements

- 🧾 User login system for persistent watchlists
- 🧠 Train more advanced models like **Autoencoders** or **Neural CF**
- 🗣️ Use **natural language descriptions** or **reviews** for content filtering
- 💾 Save feedback to database and re-train models dynamically
- 🌍 Add **language filtering**, **release year** filters, or **actor-based recommendations**
- 📦 Deploy app using **Streamlit Cloud**, **Render**, or **AWS EC2**

---

## ▶️ How to Run the App

### 1. Clone the repository:
```bash
git clone https://github.com/kothurlokeshreddy/Personalized-Recommendation-System.git
cd Personalized-Recommendation-System
```

### 2. Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Run the app:
```bash
streamlit run app.py
```

##### Ensure that models/ and data/ directories exist and are populated with the necessary .pkl and .csv files.

---
