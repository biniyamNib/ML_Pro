document.getElementById('predictionForm').addEventListener('submit', function(event) {
    event.preventDefault();

    const formData = new FormData(event.target);
    const data = {};
    formData.forEach((value, key) => {
        data[key] = value;
    });

    fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    })
    .then(response => response.json())
    .then(result => {
        const resultDiv = document.getElementById('result');
        resultDiv.innerHTML = `
            <p><strong>Prediction:</strong> ${result.prediction}</p>
            <p><strong>Probability:</strong> ${result.probability}</p>
        `;
    })
    .catch(error => {
        console.error('Error:', error);
    });
});