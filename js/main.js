function loadRandomMovies() {
    const shuffledMovies = movies.sort(() => 0.5 - Math.random());
    const selectedMovies = shuffledMovies.slice(0, 10);
    let moviesDiv = document.getElementById('movies');
    moviesDiv.innerHTML = '';
    selectedMovies.forEach(movie => {
        moviesDiv.innerHTML += `
            <label for="movie_${movie.id}">${movie.title} (${movie.year})</label>
            <input type="number" id="movie_${movie.id}" name="movie_${movie.id}" min="0.5" max="5" step="0.5"><br><br>
        `;
    });
}

function saveRatings(event) {
    event.preventDefault();
    const userId = document.getElementById('user_id').value;
    const formData = new FormData(event.target);
    let userRatings = []; // Clear previous ratings
    formData.forEach((value, key) => {
        if (key !== 'user_id') {
            userRatings.push({
                user_id: userId,
                movie_id: key.split('_')[1],
                rating: value
            });
        }
    });
    console.log(userRatings); // For debugging purposes
    alert('Ratings saved successfully!');
    // Send the ratings to the backend and get recommendations
    getRecommendations(userId, userRatings);
}

function getRecommendations(userId, userRatings) {
    // Placeholder for sending data to a web service or API
    // fetch('/recommendations', {
    //     method: 'POST',
    //     headers: {
    //         'Content-Type': 'application/json'
    //     },
    //     body: JSON.stringify({ user_id: userId, ratings: userRatings })
    // })
    // .then(response => response.json())
    // .then(data => {
    //     displayRecommendations(data);
    // });
}

function displayRecommendations(recommendations) {
    let recommendationsDiv = document.getElementById('recommendations');
    recommendationsDiv.innerHTML = '<h2>Recommended Movies</h2>';
    recommendations.forEach(movie => {
        recommendationsDiv.innerHTML += `
            <p>${movie.title} (${movie.year})</p>
        `;
    });
}

window.onload = loadRandomMovies;