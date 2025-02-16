function toggleTheme() {
    const body = document.body;
    const newTheme = body.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
    body.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
  }
const modeLabel = document.getElementById('mode-label');

// Update prepareDataset function
async function prepareDataset() {
    try {
        const response = await fetch('http://127.0.0.1:5000/api/prepare-dataset', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                binary: !modeToggle.checked 
            })
        });
        const data = await response.json();
        alert(data.message);
    } catch (error) {
        alert('Error preparing dataset: ' + error.message);
    }
}

// Définir le thème sauvegardé
const savedTheme = localStorage.getItem('theme') || 'light';
document.body.setAttribute('data-theme', savedTheme);

// Gestion des fichiers
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('image');
const preview = document.getElementById('preview');
const fileName = document.getElementById('file-name');
const errorMsg = document.getElementById('error-msg');

dropZone.addEventListener('dragover', (e) => {
e.preventDefault();
dropZone.style.borderColor = 'var(--secondary)';
});

dropZone.addEventListener('drop', (e) => {
e.preventDefault();
const file = e.dataTransfer.files[0];
handleFile(file);
});

fileInput.addEventListener('change', (e) => {
const file = e.target.files[0];
handleFile(file);
});

function handleFile(file) {
if (file && file.type.startsWith('image/')) {
    errorMsg.style.display = 'none';
    fileName.textContent = file.name;
    
    const reader = new FileReader();
    reader.onload = (e) => {
    preview.src = e.target.result;
    preview.style.display = 'block';
    };
    reader.readAsDataURL(file);
} else {
    errorMsg.style.display = 'block';
    preview.style.display = 'none';
}
}

// Placeholder functions for buttons
function prepareDataset() {
alert('Préparation du dataset...');
}

function trainModel() {
alert('Entraînement du modèle...');
}

function evaluateModel() {
alert('Évaluation du modèle...');
}

// Update the JavaScript section

async function prepareDataset() {
    try {
        const response = await fetch('http://127.0.0.1:5000/api/prepare-dataset', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ binary: true })
        });
        const text = await response.text();
        console.log('Raw response:', text);
        const data = JSON.parse(text);
        alert(data.message);
    } catch (error) {
        alert('Error preparing dataset: ' + error.message);
    }
}

// Update form submission
document.querySelector('#send').onclick = async (e) => {
    e.preventDefault();
    const loadingOverlay = document.getElementById('loading-overlay');
    const resultSection = document.getElementById('result-section');
    const resultLabel = document.getElementById('result-label');
    const resultConfidence = document.getElementById('result-confidence');
    const confidenceFill = document.getElementById('confidence-fill');
    const heatmapImg = document.getElementById('heatmap');

    loadingOverlay.style.display = 'flex';
    resultSection.style.display = 'none';

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        const response = await fetch('http://127.0.0.1:5000/api/predict', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
    
        loadingOverlay.style.display = 'none';

        if (data.status === 'success') {
            resultLabel.textContent = data.result.label;
            confidence = data.result.confidence;
            if (confidence < 0.5) {
                confidence = 1 - confidence;
            }
            resultConfidence.textContent = `${(confidence * 100).toFixed(1)}%`;
            confidenceFill.style.width = `${confidence * 100}%`;
            heatmapImg.src = data.result.heatmap;
            resultSection.style.display = 'block';
        } else {
            showError(data.message);
        }
    } catch (error) {
        loadingOverlay.style.display = 'none';
        showError('Erreur de connexion au serveur');
    }
};

function showError(message) {
    const errorMsg = document.getElementById('error-msg');
    errorMsg.textContent = message;
    errorMsg.style.display = 'block';
    setTimeout(() => errorMsg.style.display = 'none', 5000);
}