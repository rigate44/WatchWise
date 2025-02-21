from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense
from sklearn.model_selection import train_test_split

app = Flask(__name__)
#d

# Load the datasets
movies_df = pd.read_csv('movies_metadata.csv')
ratings_df = pd.read_csv('ratings_small.csv')

# Clean the 'id' column in movies_df to ensure it contains only valid integers
movies_df = movies_df[movies_df['id'].apply(lambda x: str(x).isdigit())]
movies_df['id'] = movies_df['id'].astype(int)

# Filter ratings_df to include only movie IDs that exist in movies_df
valid_movie_ids = movies_df['id'].unique()
ratings_df = ratings_df[ratings_df['movieId'].isin(valid_movie_ids)]

# Filter ratings_df to include only user IDs and movie IDs that exist in movies_df
valid_user_ids = ratings_df['userId'].unique()
ratings_df = ratings_df[ratings_df['userId'].isin(valid_user_ids) & ratings_df['movieId'].isin(valid_movie_ids)]

# Define the number of unique users and movies
num_users = ratings_df['userId'].nunique()
num_movies = ratings_df['movieId'].nunique()

# Add blank rows for new user IDs if the number of unique user IDs is less than 2000
if num_users < 2000:
    new_user_ids = range(num_users, 2000)
    blank_ratings = pd.DataFrame({'userId': new_user_ids, 'movieId': [np.nan] * len(new_user_ids), 'rating': [np.nan] * len(new_user_ids)})
    ratings_df = pd.concat([ratings_df, blank_ratings], ignore_index=True)

# Ensure user IDs and movie IDs are within the range of the embedding layer's indices
ratings_df = ratings_df[ratings_df['userId'] < 2000]
ratings_df = ratings_df[ratings_df['movieId'] < num_movies]

# Define the embedding size
embedding_size = 50

# Define the model
class RecommenderNet(Model):
    def __init__(self, num_users, num_movies, embedding_size):
        super(RecommenderNet, self).__init__()
        self.user_embedding = Embedding(num_users, embedding_size, embeddings_initializer='he_normal', embeddings_regularizer=tf.keras.regularizers.l2(1e-6))
        self.movie_embedding = Embedding(num_movies, embedding_size, embeddings_initializer='he_normal', embeddings_regularizer=tf.keras.regularizers.l2(1e-6))
        self.dot = Dot(axes=1)
    
    def call(self, inputs):
        user_vector = self.user_embedding(inputs[0])
        movie_vector = self.movie_embedding(inputs[1])
        dot_user_movie = self.dot([user_vector, movie_vector])
        return dot_user_movie

# Instantiate the model
model = RecommenderNet(2000, num_movies, embedding_size)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model (this should be done with the full dataset in practice)
train_df, test_df = train_test_split(ratings_df.dropna(), test_size=0.2, random_state=42)
train_user_ids = train_df['userId'].values
train_movie_ids = train_df['movieId'].values
train_ratings = train_df['rating'].values
model.fit([train_user_ids, train_movie_ids], train_ratings, epochs=10, batch_size=64, verbose=1)

@app.route('/')
def index():
    # Convert 'revenue' column to numeric, coercing errors to NaN and then dropping them
    movies_df['revenue'] = pd.to_numeric(movies_df['revenue'], errors='coerce')
    movies_df.dropna(subset=['revenue'], inplace=True)
    
    # Sort movies by revenue and select the top 50
    top_revenue_movies = movies_df.sort_values(by='revenue', ascending=False).head(50)
    
    # Calculate average rating for each movie
    avg_ratings = ratings_df.groupby('movieId')['rating'].mean().reset_index()
    avg_ratings.columns = ['id', 'avg_rating']
    
    # Merge average ratings with movies_df
    movies_with_ratings = pd.merge(movies_df, avg_ratings, on='id')
    
    # Sort movies by average rating and select the top 50
    top_rated_movies = movies_with_ratings.sort_values(by='avg_rating', ascending=False).head(50)
    
    # Combine the two lists and drop duplicates
    combined_movies = pd.concat([top_revenue_movies, top_rated_movies]).drop_duplicates().head(100)
    
    # Add year of release to the title
    combined_movies['title'] = combined_movies.apply(lambda row: f"{row['title']} ({row['release_date'][:4]})" if pd.notnull(row['release_date']) else row['title'], axis=1)

    return render_template('index.html', movies=combined_movies.to_dict(orient='records'))

@app.route('/rate', methods=['POST'])
def rate():
    user_id = int(request.form['user_id'])
    ratings = []
    for movie_id, rating in request.form.items():
        if movie_id != 'user_id' and rating:
            ratings.append((user_id, int(movie_id), float(rating)))
    
    # Debugging: Print the collected ratings
    print("Collected ratings:", ratings)
    
    user_ratings_df = pd.DataFrame(ratings, columns=['userId', 'movieId', 'rating'])
    
    # Filter out invalid user IDs and movie IDs
    user_ratings_df = user_ratings_df[user_ratings_df['userId'] < 2000]
    user_ratings_df = user_ratings_df[user_ratings_df['movieId'] < num_movies]
    
    combined_df = pd.concat([ratings_df, user_ratings_df])
    combined_user_ids = combined_df['userId'].values
    combined_movie_ids = combined_df['movieId'].values
    combined_ratings = combined_df['rating'].values
    
    # Debugging: Print the shapes of the arrays
    print("User IDs shape:", combined_user_ids.shape)
    print("Movie IDs shape:", combined_movie_ids.shape)
    print("Ratings shape:", combined_ratings.shape)
    
    model.fit([combined_user_ids, combined_movie_ids], combined_ratings, epochs=10, batch_size=64, verbose=1)
    
    # Redirect to the recommendations page
    return redirect(url_for('recommend', user_id=user_id))

@app.route('/recommend/<int:user_id>')
def recommend(user_id):
    user_rated_movies = ratings_df[ratings_df['userId'] == user_id]['movieId'].values
    unrated_movies = movies_df[~movies_df['id'].isin(user_rated_movies)]['id'].values
    
    # Ensure movie IDs are within the range of the embedding layer's indices
    unrated_movies = unrated_movies[unrated_movies < num_movies]
    
    predicted_ratings = model.predict([np.array([user_id] * len(unrated_movies)), unrated_movies])
    top_movie_indices = np.argsort(predicted_ratings.flatten())[::-1][:20]
    recommended_movies = movies_df.iloc[top_movie_indices]
    
    # Add year of release and predicted rating to the title
    recommended_movies['title'] = recommended_movies.apply(
        lambda row: f"{row['title']} ({row['release_date'][:4]}) - {predicted_ratings[top_movie_indices][recommended_movies.index.get_loc(row.name)][0]:.2f}" 
        if pd.notnull(row['release_date']) else f"{row['title']} - {predicted_ratings[top_movie_indices][recommended_movies.index.get_loc(row.name)][0]:.2f}", 
        axis=1
    )
    
    return render_template('recommend.html', movies=recommended_movies.to_dict(orient='records'), user_id=user_id)

@app.route('/feedback', methods=['POST'])
def feedback():
    user_id = int(request.form['user_id'])
    ratings = []
    for movie_id, rating in request.form.items():
        if movie_id != 'user_id' and rating:
            ratings.append((user_id, int(movie_id), float(rating)))
    
    # Debugging: Print the collected feedback ratings
    print("Collected feedback ratings:", ratings)
    
    user_ratings_df = pd.DataFrame(ratings, columns=['userId', 'movieId', 'rating'])
    
    # Filter out invalid user IDs and movie IDs
    user_ratings_df = user_ratings_df[user_ratings_df['userId'] < 2000]
    user_ratings_df = user_ratings_df[user_ratings_df['movieId'] < num_movies]
    
    combined_df = pd.concat([ratings_df, user_ratings_df])
    combined_user_ids = combined_df['userId'].values
    combined_movie_ids = combined_df['movieId'].values
    combined_ratings = combined_df['rating'].values
    
    # Debugging: Print the shapes of the arrays
    print("User IDs shape:", combined_user_ids.shape)
    print("Movie IDs shape:", combined_movie_ids.shape)
    print("Ratings shape:", combined_ratings.shape)
    
    model.fit([combined_user_ids, combined_movie_ids], combined_ratings, epochs=10, batch_size=64, verbose=1)
    
    # Return a link to the recommendations page
    return redirect(url_for('recommend', user_id=user_id))

if __name__ == '__main__':
    app.run(debug=True)