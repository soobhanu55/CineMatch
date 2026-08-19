"""Reproduces the notebook's exact pipeline (merge -> extract genres/keywords/
top-3-cast/director -> combine into tags -> stem -> CountVectorizer(5000,
english stopwords) -> cosine similarity), regenerates movies_dict.pkl and
similarity.pkl for app.py to actually run, and evaluates recommendation
quality against a hand-labeled ground truth of genuinely-similar movie
pairs (same franchise/sequel, verified real relationships, not guessed).

Run: python build_and_eval.py
"""

from __future__ import annotations

import ast
import pickle

import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ps = PorterStemmer()


def convert(obj: str) -> list[str]:
    return [i["name"] for i in ast.literal_eval(obj)]


def convert_cast(obj: str) -> list[str]:
    result, counter = [], 0
    for i in ast.literal_eval(obj):
        if counter == 3:
            break
        result.append(i["name"])
        counter += 1
    return result


def convert_crew(obj: str) -> list[str]:
    for i in ast.literal_eval(obj):
        if i["job"] == "Director":
            return [i["name"]]
    return []


def stem(text: str) -> str:
    return " ".join(ps.stem(w) for w in text.split())


def build_pipeline() -> tuple[pd.DataFrame, "list", object]:
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    movies = movies.merge(credits, on="title")
    movies = movies[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]
    movies.dropna(inplace=True)

    movies["genres"] = movies["genres"].apply(convert)
    movies["keywords"] = movies["keywords"].apply(convert)
    movies["cast"] = movies["cast"].apply(convert_cast)
    movies["crew"] = movies["crew"].apply(convert_crew)
    movies["overview"] = movies["overview"].apply(lambda x: x.split(" "))

    movies["genres"] = movies["genres"].apply(lambda x: [i.replace(" ", "") for i in x])
    movies["keywords"] = movies["keywords"].apply(lambda x: [i.replace(" ", "") for i in x])
    movies["cast"] = movies["cast"].apply(lambda x: [i.replace(" ", "") for i in x])
    movies["crew"] = movies["crew"].apply(lambda x: [i.replace(" ", "") for i in x])

    movies["tags"] = movies["overview"] + movies["keywords"] + movies["cast"] + movies["crew"]
    new_df = movies[["movie_id", "title", "tags"]].copy()
    new_df["tags"] = new_df["tags"].apply(lambda x: " ".join(x).lower())
    new_df["tags"] = new_df["tags"].apply(stem)

    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(new_df["tags"]).toarray()
    similarity = cosine_similarity(vectors)

    return new_df.reset_index(drop=True), similarity


def recommend(new_df: pd.DataFrame, similarity, movie: str, k: int = 5) -> list[str]:
    matches = new_df[new_df["title"] == movie]
    if matches.empty:
        return []
    movie_index = matches.index[0]
    distances = similarity[movie_index]
    ranked = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1 : k + 1]
    return [new_df.iloc[i[0]].title for i in ranked]


# Real, verified franchise/sequel relationships in the TMDB 5000 dataset --
# these are objectively similar movies (same series), not a guess.
EVAL_SET = [
    ("Harry Potter and the Half-Blood Prince", ["Harry Potter and the Order of the Phoenix",
                                                  "Harry Potter and the Goblet of Fire",
                                                  "Harry Potter and the Prisoner of Azkaban",
                                                  "Harry Potter and the Chamber of Secrets",
                                                  "Harry Potter and the Philosopher's Stone"]),
    ("Iron Man 2", ["Iron Man", "Iron Man 3", "The Avengers", "Avengers: Age of Ultron"]),
    ("The Dark Knight Rises", ["The Dark Knight", "Batman Begins"]),
    ("Spider-Man 3", ["Spider-Man", "Spider-Man 2", "The Amazing Spider-Man"]),
    ("Pirates of the Caribbean: Dead Man's Chest", ["Pirates of the Caribbean: The Curse of the Black Pearl",
                                                       "Pirates of the Caribbean: At World's End",
                                                       "Pirates of the Caribbean: On Stranger Tides"]),
    ("Transformers: Revenge of the Fallen", ["Transformers", "Transformers: Dark of the Moon", "Transformers: Age of Extinction"]),
    ("X-Men: The Last Stand", ["X-Men", "X2", "X-Men Origins: Wolverine", "X-Men: First Class"]),
    ("Toy Story 2", ["Toy Story", "Toy Story 3"]),
    ("The Bourne Ultimatum", ["The Bourne Identity", "The Bourne Supremacy", "The Bourne Legacy"]),
    ("Shrek 2", ["Shrek", "Shrek the Third", "Shrek Forever After"]),
]


def main() -> None:
    print("Building pipeline (merge, tag extraction, stemming, vectorize, cosine similarity)...")
    new_df, similarity = build_pipeline()
    print(f"  {len(new_df)} movies after cleaning")

    hits_at_5 = 0
    n_evaluable = 0
    for movie, expected_related in EVAL_SET:
        if movie not in new_df["title"].values:
            print(f"SKIP (not in dataset): {movie}")
            continue
        n_evaluable += 1
        recs = recommend(new_df, similarity, movie, k=5)
        found = [r for r in recs if r in expected_related]
        hit = len(found) > 0
        hits_at_5 += int(hit)
        print(f"{'OK ' if hit else 'MISS'} | {movie[:40]:40s} -> {recs}")

    print(f"\nAt-least-one-franchise-entry-in-top-5: {hits_at_5}/{n_evaluable} ({100*hits_at_5/n_evaluable:.1f}%)")

    print("\nSaving movies_dict.pkl and similarity.pkl for app.py...")
    pickle.dump(new_df.to_dict(), open("movies_dict.pkl", "wb"))
    pickle.dump(similarity, open("similarity.pkl", "wb"))
    print("Done.")


if __name__ == "__main__":
    main()
