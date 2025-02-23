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
        console.log("Predicting for userId:", userId, "movieId:", movieId); // Debugging statement
        if (userId >= this.userEmbedding.length || movieId >= this.movieEmbedding.length) {
            console.error("Invalid userId or movieId:", userId, movieId); // Debugging statement
            return 0; // Return a default value or handle the error appropriately
        }
        const userVector = this.userEmbedding[userId];
        const movieVector = this.movieEmbedding[movieId];
        console.log("User Vector:", userVector); // Debugging statement
        console.log("Movie Vector:", movieVector); // Debugging statement
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

const numUsers = 672; // Replace with actual number of users
const numMovies = 607; // Replace with actual number of movies
const embeddingSize = 50;

const recommender = new RecommenderNet(numUsers, numMovies, embeddingSize);

function getRecommendations(userId) {
    const movieIds = movies.map(movie => movie.id);
    console.log("Movie IDs:", movieIds); // Debugging statement
    const recommendations = recommender.recommendTopN(userId, movieIds);
    console.log("Recommendations:", recommendations); // Debugging statement
    displayRecommendations(recommendations);
}

function displayRecommendations(recommendations) {
    const recommendationsDiv = document.getElementById('recommendations');
    // recommendationsDiv.innerHTML = '<h2>Recommended Movies</h2>';
    const maxRating = Math.max(...recommendations.map(rec => rec.rating));
    recommendations.forEach(rec => {
        const movie = movies.find(movie => movie.id === rec.movieId);
        const normalizedRating = (rec.rating / maxRating) * 4.5; // Normalize the rating to a 5-star scale
        recommendationsDiv.innerHTML += `<p>${movie.title} (${movie.year}) - Predicted Rating: ${normalizedRating.toFixed(2)}</p>`;
    });
}

window.onload = function() {
    const userId = localStorage.getItem('userId');
    console.log("User ID from localStorage:", userId); // Debugging statement
    if (userId) {
        getRecommendations(userId);
    }
};