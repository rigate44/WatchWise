class RecommenderNet {
    constructor(numUsers, numMovies, embeddingSize) {
        this.numUsers = numUsers;
        this.numMovies = numMovies;
        this.embeddingSize = embeddingSize;
        this.userEmbedding = this.createEmbedding(numUsers, embeddingSize);
        this.movieEmbedding = this.createEmbedding(numMovies, embeddingSize);
    }

    createEmbedding(numEntities, embeddingSize) {
        const embedding = [];
        for (let i = 0; i < numEntities; i++) {
            const entityEmbedding = [];
            for (let j = 0; j < embeddingSize; j++) {
                entityEmbedding.push(Math.random() * 0.01); // Initialize with small random values
            }
            embedding.push(entityEmbedding);
        }
        return embedding;
    }

    dotProduct(vectorA, vectorB) {
        let dotProduct = 0;
        for (let i = 0; i < vectorA.length; i++) {
            dotProduct += vectorA[i] * vectorB[i];
        }
        return dotProduct;
    }

    predict(userId, movieId) {
        const userVector = this.userEmbedding[userId];
        const movieVector = this.movieEmbedding[movieId];
        return this.dotProduct(userVector, movieVector);
    }

    recommendTopN(userId, movieIds, n = 20) {
        const predictions = movieIds.map(movieId => ({
            movieId,
            rating: this.predict(userId, movieId)
        }));
        predictions.sort((a, b) => b.rating - a.rating);
        return predictions.slice(0, n);
    }
}

// Example usage
const numUsers = 1000; // Replace with actual number of users
const numMovies = 100; // Replace with actual number of movies
const embeddingSize = 50;

const recommender = new RecommenderNet(numUsers, numMovies, embeddingSize);

function getRecommendations(userId, userRatings) {
    const movieIds = movies.map(movie => movie.id);
    const recommendations = recommender.recommendTopN(userId, movieIds);
    displayRecommendations(recommendations);
}

function displayRecommendations(recommendations) {
    const recommendationsDiv = document.getElementById('recommendations');
    recommendationsDiv.innerHTML = '<h2>Recommended Movies</h2>';
    recommendations.forEach(rec => {
        const movie = movies.find(movie => movie.id === rec.movieId);
        recommendationsDiv.innerHTML += `<p>${movie.title} (${movie.year}) - Predicted Rating: ${rec.rating.toFixed(2)}</p>`;
    });
}