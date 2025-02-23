import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Dot

class RecommenderNet(Model):
    def __init__(self, num_users, num_movies, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        # Create embedding layers for users and movies
        self.user_embedding = Embedding(num_users, embedding_size, embeddings_initializer='he_normal', embeddings_regularizer=tf.keras.regularizers.l2(1e-6))
        self.movie_embedding = Embedding(num_movies, embedding_size, embeddings_initializer='he_normal', embeddings_regularizer=tf.keras.regularizers.l2(1e-6))
        # Define a dot product layer to compute the similarity between user and movie embeddings
        self.dot = Dot(axes=1)
    
    def call(self, inputs):
        # Get the embeddings for the users and movies
        user_vector = self.user_embedding(inputs[0])
        movie_vector = self.movie_embedding(inputs[1])
        # Compute the dot product of the user and movie embeddings
        dot_user_movie = self.dot([user_vector, movie_vector])
        return dot_user_movie
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_users': self.num_users,
            'num_movies': self.num_movies,
            'embedding_size': self.embedding_size
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def recommend_top_n(self, user_id, movie_ids, n=20):
        user_array = tf.constant([user_id] * len(movie_ids))
        movie_array = tf.constant(movie_ids)
        predictions = self.call([user_array, movie_array])
        top_n_indices = tf.argsort(predictions, axis=0, direction='DESCENDING')[:n]
        top_n_movie_ids = tf.gather(movie_ids, top_n_indices)
        top_n_ratings = tf.gather(predictions, top_n_indices)
        return top_n_movie_ids.numpy().flatten(), top_n_ratings.numpy().flatten()

# Load the ratings and movies metadata
ratings_df = pd.read_csv('ratings_small.csv')
movies_df = pd.read_csv('movies_metadata.csv')

# Ensure the movie IDs are integers
movies_df['id'] = pd.to_numeric(movies_df['id'], errors='coerce').dropna().astype(int)

# New user added
top_10_movies = movies_df[movies_df['id'].isin(ratings_df['movieId'])].head(10)
new_user_ratings = pd.DataFrame({
    'userId': [672] * 10,
    'movieId': top_10_movies['id'],
    'rating': [5] * 10
})
ratings_df = pd.concat([ratings_df, new_user_ratings], ignore_index=True)

# Select relevant columns from movies_df
movies_df = movies_df[['id', 'original_title', 'release_date']]
movies_df['year'] = pd.to_datetime(movies_df['release_date'], errors='coerce').dt.year
movies_df = movies_df[['id', 'original_title', 'year']]

# Map userId and movieId to continuous range of indices
user_id_mapping = {id: idx for idx, id in enumerate(ratings_df['userId'].unique())}
movie_id_mapping = {id: idx for idx, id in enumerate(ratings_df['movieId'].unique())}

# Apply the mappings to the ratings DataFrame
ratings_df['userId'] = ratings_df['userId'].map(user_id_mapping)
ratings_df['movieId'] = ratings_df['movieId'].map(movie_id_mapping)

# Filter movies_df to include only movies that exist in ratings_df
movies_df = movies_df[movies_df['id'].isin(movie_id_mapping.keys())]

# Add mapped_id to movies_df
movies_df['mapped_id'] = movies_df['id'].map(movie_id_mapping)

# Split the data into training and testing sets
train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

# Initialize and compile the model
num_users = len(ratings_df['userId'].unique())
num_movies = len(ratings_df['movieId'].unique())
embedding_size = 50

model = RecommenderNet(num_users, num_movies, embedding_size)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
train_user_ids = train_df['userId'].values
train_movie_ids = train_df['movieId'].values
train_ratings = train_df['rating'].values

model.fit([train_user_ids, train_movie_ids], train_ratings, batch_size=64, epochs=5, verbose=1)

def recommend_movies(user_id, model, movies_df, movie_id_mapping, num_recommendations=20):
    all_movie_ids = movies_df['id'].map(movie_id_mapping).dropna().astype(int).values
    top_movie_ids, top_ratings = model.recommend_top_n(user_id, all_movie_ids, n=num_recommendations)
    recommended_movies = movies_df[movies_df['id'].map(movie_id_mapping).isin(top_movie_ids)]
    recommended_movies['predicted_rating'] = top_ratings
    return recommended_movies

# Example usage
user_id = 671  # Replace with the actual user ID
recommended_movies = recommend_movies(user_id, model, movies_df, movie_id_mapping)
print("Recommended Movies:")
print(recommended_movies)