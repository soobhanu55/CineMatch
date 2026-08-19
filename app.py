import os

import streamlit as st
import pickle
import pandas as pd
import requests

TMDB_API_KEY = os.environ.get("TMDB_API_KEY")

def fetch_poster(movie_id):
    if not TMDB_API_KEY:
        return None
    response = requests.get(
        f"https://api.themoviedb.org/3/movie/{movie_id}",
        params={"api_key": TMDB_API_KEY, "language": "en-US"},
    )
    data = response.json()
    poster_path = data.get("poster_path")
    return f"https://image.tmdb.org/t/p/w500/{poster_path}" if poster_path else None
def recommend(movie):
  movie_index = movies[movies['title']==movie].index[0]
  distances = similarity[movie_index]
  movies_list = sorted(list(enumerate(distances)),reverse = True , key = lambda x:x[1])[1:6]

  recommended_movies = []
  recommended_movie_poster = []
  for i in movies_list:
    movie_id = movies.iloc[i[0]].movie_id
    recommended_movies.append(movies.iloc[i[0]].title)
    recommended_movie_poster.append(fetch_poster(movie_id))
  return recommended_movies,recommended_movie_poster


movies_dict = pickle.load(open('movies_dict.pkl','rb'))
movies = pd.DataFrame(movies_dict)

similarity = pickle.load(open('similarity.pkl','rb'))

st.title('Movie Recommender System')
if not TMDB_API_KEY:
    st.info("Set the TMDB_API_KEY environment variable to show posters (recommendations still work without it).")


selected_movie_name = st.selectbox(
    'Enter the movie name',
    movies['title'].values)

if st.button('Recommend'):
    names,posters = recommend(selected_movie_name)
    col1,col2,col3,col4,col5 = st.columns(5)
    for col, name, poster in zip((col1, col2, col3, col4, col5), names, posters):
        with col:
            st.text(name)
            if poster:
                st.image(poster)
