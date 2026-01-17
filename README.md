# Content-Based Movie Recommendation System

## 📌 Project Overview
CineMatch is a content-based recommendation engine that suggests movies to users based on the similarity of their attributes. By analyzing metadata such as genres, keywords, cast, and crew, the system identifies patterns and recommends films that align with a user's specific interests.

## 🚀 Key Features
* **Metadata Integration**: Merges comprehensive movie details with credits data to create a rich feature set.
* **Text Preprocessing**: Implements a pipeline to clean and transform complex JSON-like columns (genres, keywords, cast) into usable tags.
* **Natural Language Processing**: 
    * Utilizes `CountVectorizer` for text vectorization.
    * Removes stop words to focus on meaningful identifiers.
* **Similarity Engine**: Employs Cosine Similarity to calculate the distance between movie vectors and retrieve the most relevant matches.
* **Deployment Ready**: Includes functionality to export processed data using `pickle` for integration into web applications (e.g., Streamlit).

## 🛠️ Technical Stack
* **Languages**: Python
* **Libraries**: Pandas, NumPy, Scikit-learn (CountVectorizer, Cosine Similarity), NLTK (for stemming).
* **Data Sources**: TMDB 5000 Movies Dataset.

## 📊 How it Works
1. **Data Cleaning**: Irrelevant columns are dropped, and missing values are handled.
2. **Feature Extraction**: Top actors, directors, and genre keywords are extracted and combined into a single "tags" column.
3. **Vectorization**: Each movie is converted into a 5000-dimensional vector.
4. **Recommendation**: When a movie title is entered, the system finds the five movies with the highest cosine similarity scores.

## 📁 Dataset
This project uses the [TMDB 5000 Movies Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata), which includes information on budget, genres, popularity, and credits for nearly 5,000 films.
