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
    console.log("User ID:", userId); // Debugging statement
    console.log("User Ratings:", userRatings); // Debugging statement
    // Save the user ratings to localStorage
    localStorage.setItem('userRatings', JSON.stringify(userRatings));
    localStorage.setItem('userId', userId);
    // Redirect to recommendations.html
    window.location.href = 'recommendations.html';
}

window.onload = loadRandomMovies;