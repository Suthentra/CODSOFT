import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Load Tamil movies dataset
file_path = 'Tamil_movies_dataset.csv'
tamil_movies = pd.read_csv(file_path)

# Add a unique movieId column
tamil_movies['movieId'] = range(1, len(tamil_movies) + 1)

# Simulate user ratings for Tamil movies
np.random.seed(42)
user_ids = np.random.randint(1, 101, size=500)  # 100 unique users
movie_ids = np.random.choice(tamil_movies['movieId'], size=500, replace=True)
ratings = np.random.uniform(1, 5, size=500).round(1)

# Create a new ratings DataFrame
tamil_ratings = pd.DataFrame({
    'userId': user_ids,
    'movieId': movie_ids,
    'rating': ratings
})

# Function to create the matrix
def create_matrix(df):
    N = len(df['userId'].unique())
    M = len(df['movieId'].unique())
    
    user_mapper = dict(zip(np.unique(df["userId"]), list(range(N))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(M))))
    
    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["userId"])))
    movie_inv_mapper = dict(zip(list(range(M)), np.unique(df["movieId"])))
    
    user_index = [user_mapper[i] for i in df['userId']]
    movie_index = [movie_mapper[i] for i in df['movieId']]

    X = csr_matrix((df["rating"], (movie_index, user_index)), shape=(M, N))
    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

# Create the matrix for Tamil ratings
X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_matrix(tamil_ratings)

# Update movie titles mapping
movie_titles = dict(zip(tamil_movies['movieId'], tamil_movies['MovieName']))

# Function to find similar movies
def find_similar_movies(movie_id, X, k, metric='cosine', show_distance=False):
    neighbour_ids = []
    movie_ind = movie_mapper[movie_id]
    movie_vec = X[movie_ind]
    k += 1
    kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
    kNN.fit(X)
    movie_vec = movie_vec.reshape(1, -1)
    neighbour = kNN.kneighbors(movie_vec, return_distance=show_distance)
    for i in range(0, k):
        n = neighbour.item(i)
        neighbour_ids.append(movie_inv_mapper[n])
    neighbour_ids.pop(0)
    return neighbour_ids

# Function to recommend movies for a user
def recommend_tamil_movies_for_user(user_id, X, user_mapper, movie_mapper, movie_inv_mapper, k=10):
    df1 = tamil_ratings[tamil_ratings['userId'] == user_id]
    
    if df1.empty:
        print(f"User with ID {user_id} does not exist.")
        return

    movie_id = df1[df1['rating'] == max(df1['rating'])]['movieId'].iloc[0]
    similar_ids = find_similar_movies(movie_id, X, k)
    movie_title = movie_titles.get(movie_id, "Movie not found")

    if movie_title == "Movie not found":
        print(f"Movie with ID {movie_id} not found.")
        return

    print(f"Since you watched {movie_title}, you might also like:")
    for i in similar_ids:
        print(movie_titles.get(i, "Movie not found"))

# Example usage
user_id_1 = 52  # Existing user
user_id_2 = 150  # Non-existing user

recommend_tamil_movies_for_user(user_id_1, X, user_mapper, movie_mapper, movie_inv_mapper, k=5)
print("\n")
recommend_tamil_movies_for_user(user_id_2, X, user_mapper, movie_mapper, movie_inv_mapper, k=5)
