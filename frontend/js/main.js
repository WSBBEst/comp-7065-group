const API_BASE_URL = 'http://localhost:5000';

let uploadedFiles = [];
const MAX_FILES = 5;

function init() {
    setupEventListeners();
    checkBackendHealth();
}

function setupEventListeners() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    
    uploadArea.addEventListener('click', () => fileInput.click());
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        const files = Array.from(e.dataTransfer.files);
        handleFilesSelect(files);
    });
    
    fileInput.addEventListener('change', (e) => {
        const files = Array.from(e.target.files);
        handleFilesSelect(files);
        fileInput.value = '';
    });
    
    document.getElementById('analyzeBtn').addEventListener('click', analyzeOutfits);
    document.getElementById('resetBtn').addEventListener('click', resetToUpload);
}

async function checkBackendHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/health`);
        const data = await response.json();
        console.log('[HEALTH] Backend status:', data.status);
    } catch (error) {
        console.warn('[HEALTH] Backend service may not be running');
    }
}

function handleFilesSelect(files) {
    const validFiles = [];
    
    files.forEach(file => {
        if (!file.type.startsWith('image/')) {
            showError(`Skipped non-image file: ${file.name}`);
            return;
        }

        const maxSize = 16 * 1024 * 1024;
        if (file.size > maxSize) {
            showError(`Image ${file.name} exceeds 16MB limit`);
            return;
        }

        // Check for duplicates
        const isDuplicate = uploadedFiles.some(f => 
            f.name === file.name && f.size === file.size
        );
        if (!isDuplicate) {
            validFiles.push(file);
        }
    });

    if (validFiles.length === 0) return;

    // Limit to maximum 5 files
    const remainingSlots = MAX_FILES - uploadedFiles.length;
    if (remainingSlots <= 0) {
        showError(`Maximum ${MAX_FILES} images allowed`);
        return;
    }

    const filesToAdd = validFiles.slice(0, remainingSlots);
    uploadedFiles = [...uploadedFiles, ...filesToAdd];
    
    updateImagesGrid();
    checkAnalyzeButton();
    removeError();
}

function updateImagesGrid() {
    const grid = document.getElementById('imagesGrid');
    const gridContainer = document.getElementById('uploadedImagesGrid');
    
    if (uploadedFiles.length === 0) {
        gridContainer.style.display = 'none';
        return;
    }
    
    gridContainer.style.display = 'block';
    grid.innerHTML = '';
    
    uploadedFiles.forEach((file, index) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const item = document.createElement('div');
            item.className = 'image-grid-item';
            item.innerHTML = `
                <img src="${e.target.result}" alt="${file.name}">
                <button class="remove-image-btn" onclick="removeImage(${index})" title="Remove">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M6 18L18 6M6 6l12 12"/>
                    </svg>
                </button>
                <div class="image-index">${index + 1}</div>
                ${uploadedFiles.length >= MAX_FILES ? '<div class="max-reached-badge">Limit Reached</div>' : ''}
            `;
            grid.appendChild(item);
        };
        reader.readAsDataURL(file);
    });
}

function removeImage(index) {
    uploadedFiles.splice(index, 1);
    updateImagesGrid();
    checkAnalyzeButton();
}

function checkAnalyzeButton() {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const btnText = analyzeBtn.querySelector('.btn-text');
    
    analyzeBtn.disabled = uploadedFiles.length === 0;
    
    // Update button text to show current count
    if (uploadedFiles.length > 0) {
        btnText.textContent = `Start Smart Analysis (${uploadedFiles.length}/${MAX_FILES})`;
    } else {
        btnText.textContent = 'Start Smart Analysis';
    }
}

async function analyzeOutfits() {
    if (uploadedFiles.length === 0) {
        showError('Please upload at least one full-body photo');
        return;
    }

    setLoading(true);

    const formData = new FormData();
    uploadedFiles.forEach((file, index) => {
        formData.append('images', file, `outfit_${index}_${file.name}`);
    });

    try {
        const response = await fetch(`${API_BASE_URL}/api/analyze-fullbody`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `Server error: ${response.status}`);
        }

        const data = await response.json();

        if (data.success) {
            displayResults(data);
        } else {
            showError(data.error || 'Analysis failed, please try again');
        }
    } catch (error) {
        console.error('[ERROR]', error);
        if (error.message.includes('Failed to fetch')) {
            showError('Network error, please ensure backend service is running (http://localhost:5000)');
        } else {
            showError(error.message || 'Error occurred during analysis');
        }
    } finally {
        setLoading(false);
    }
}

function displayResults(data) {
    const { 
        detected_items, 
        combination, 
        score, 
        evaluation, 
        color_suggestion, 
        style_suggestion, 
        trend_tags, 
        compatibility_score,
        segmentation_info,
        vision_available 
    } = data;

    document.getElementById('resultPlaceholder').style.display = 'none';
    document.getElementById('resultSection').style.display = 'block';


    // Display detected clothing
    const itemsGrid = document.getElementById('itemsGrid');
    itemsGrid.innerHTML = '';
    
    if (detected_items && Object.keys(detected_items).length > 0) {
        let totalItems = 0;
        Object.entries(detected_items).forEach(([category, items]) => {
            items.forEach((item, idx) => {
                totalItems++;
                const itemDiv = document.createElement('div');
                itemDiv.className = 'detected-item';
                itemDiv.innerHTML = `
                    <img src="${item.image}" alt="${category}">
                    <div class="item-info">
                        <div class="item-category">${getCategoryName(category)}</div>
                        <div class="item-name">${item.label || category}</div>
                        ${item.confidence ? `<div class="item-confidence">Confidence: ${(item.confidence * 100).toFixed(0)}%</div>` : ''}
                        ${!item.auto_detected ? '<div class="item-manual">Manually Assigned</div>' : ''}
                        <div class="item-source">From: Photo ${item.source_image}</div>
                    </div>
                `;
                itemsGrid.appendChild(itemDiv);
            });
        });
        
        // Display total count
        const countInfo = document.createElement('div');
        countInfo.className = 'detected-count';
        countInfo.textContent = `Total ${totalItems} items detected`;
        itemsGrid.insertBefore(countInfo, itemsGrid.firstChild);
    } else {
        itemsGrid.innerHTML = '<p class="no-items">No clothing detected. Please try uploading clearer full-body photos</p>';
    }

    // Display recommended outfit
    if (combination) {
        if (combination.top) {
            document.getElementById('comboTop').src = combination.top;
            document.getElementById('comboTopItem').style.display = 'flex';
        }
        if (combination.pants || combination.bottom) {
            document.getElementById('comboPants').src = combination.pants || combination.bottom;
            document.getElementById('comboPantsItem').style.display = 'flex';
        }
        if (combination.shoes) {
            document.getElementById('comboShoes').src = combination.shoes;
            document.getElementById('comboShoesItem').style.display = 'flex';
        }
    }

    // Display score
    const overallScore = Math.round(score * 100) / 10;
    document.getElementById('overallScore').textContent = overallScore.toFixed(1);
    document.getElementById('overallScoreCircle').style.setProperty('--score', score * 100);

    // Score description
    let scoreDesc = '';
    if (score >= 0.85) {
        scoreDesc = 'Excellent match! Colors are harmonious, style is unified';
    } else if (score >= 0.70) {
        scoreDesc = 'Good match, overall coordination is nice';
    } else if (score >= 0.50) {
        scoreDesc = 'Average match, room for improvement';
    } else {
        scoreDesc = 'Poor match, adjustments recommended';
    }
    document.getElementById('scoreDesc').textContent = scoreDesc;

    // Display evaluation
    document.getElementById('evaluationText').textContent = evaluation;
    document.getElementById('colorSuggestion').textContent = color_suggestion;
    document.getElementById('styleSuggestion').textContent = style_suggestion;

    // Display tags
    const tagsContainer = document.getElementById('trendTags');
    tagsContainer.innerHTML = '';
    trend_tags.forEach(tag => {
        const tagElement = document.createElement('span');
        tagElement.className = 'tag';
        tagElement.textContent = tag;
        tagsContainer.appendChild(tagElement);
    });

    console.log('[INFO] Compatibility score:', compatibility_score);
    console.log('[INFO] Vision available:', vision_available);
}

function getCategoryName(category) {
    const names = {
        'top': 'Top',
        'pants': 'Bottom',
        'shoes': 'Shoes',
        'dress': 'Dress',
        'outwear': 'Outerwear',
        'skirt': 'Skirt',
        'bag': 'Bag',
        'accessory': 'Accessory'
    };
    return names[category] || category;
}

function showVisionWarning(message) {
    // Remove existing warning
    const existingWarning = document.querySelector('.vision-warning');
    if (existingWarning) {
        existingWarning.remove();
    }

    const warningDiv = document.createElement('div');
    warningDiv.className = 'vision-warning';
    warningDiv.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width: 16px; height: 16px;">
            <path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
        </svg>
        <span>${message}</span>
    `;
    
    const resultCard = document.querySelector('.result-card');
    if (resultCard) {
        resultCard.insertBefore(warningDiv, resultCard.firstChild);
    }
}

function resetToUpload() {
    uploadedFiles = [];
    document.getElementById('imagesGrid').innerHTML = '';
    document.getElementById('uploadedImagesGrid').style.display = 'none';
    document.getElementById('resultSection').style.display = 'none';
    document.getElementById('resultPlaceholder').style.display = 'flex';
    document.getElementById('analyzeBtn').disabled = true;
    document.querySelector('.btn-text').textContent = 'Start Smart Analysis';
    
    // Clear combination display
    document.getElementById('comboTopItem').style.display = 'none';
    document.getElementById('comboPantsItem').style.display = 'none';
    document.getElementById('comboShoesItem').style.display = 'none';
    
    // Remove Vision warning
    const existingWarning = document.querySelector('.vision-warning');
    if (existingWarning) {
        existingWarning.remove();
    }
    
    removeError();
}

function setLoading(isLoading) {
    const loadingOverlay = document.getElementById('loadingOverlay');
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    if (isLoading) {
        loadingOverlay.style.display = 'flex';
        analyzeBtn.classList.add('loading');
        analyzeBtn.querySelector('.btn-text').style.display = 'none';
        analyzeBtn.querySelector('.btn-loading').style.display = 'flex';
    } else {
        loadingOverlay.style.display = 'none';
        analyzeBtn.classList.remove('loading');
        analyzeBtn.querySelector('.btn-text').style.display = 'block';
        analyzeBtn.querySelector('.btn-loading').style.display = 'none';
    }
}

function showError(message) {
    removeError();
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;
    
    const uploadSection = document.getElementById('uploadSection');
    uploadSection.appendChild(errorDiv);
    
    setTimeout(() => {
        if (errorDiv.parentNode) {
            errorDiv.remove();
        }
    }, 3000);
}

function removeError() {
    const existingErrors = document.querySelectorAll('.error-message');
    existingErrors.forEach(error => error.remove());
}

document.addEventListener('DOMContentLoaded', init);