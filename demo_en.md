# Outfit Match - AI-Powered Outfit Recommendation System

## 📋 Project Overview

Outfit Match is a deep learning-based intelligent outfit recommendation system that automatically identifies clothing items from user-uploaded images, analyzes outfit compatibility, and provides professional fashion advice.

### Core Features

- **Full-Body Image Analysis**: Upload 1-5 full-body outfit photos for AI-powered item recognition
- **Automatic Detection & Segmentation**: Mask R-CNN model for detection and background removal
- **Smart Outfit Recommendation**: Compatibility scoring using trained prediction model
- **Professional Fashion Advice**: Color coordination, style suggestions, and trend tags

---

## 🏗️ System Architecture

### Technology Stack

**Frontend**
- HTML5 + CSS3 (Dark theme, responsive design)
- Vanilla JavaScript (No framework dependencies)
- Drag-and-drop upload with real-time preview

**Backend**
- Python 3.x + Flask
- PyTorch (Deep Learning Framework)
- Dual-model collaboration:
  - **Vision Model**: Mask R-CNN (Clothing detection and segmentation)
  - **Fashion Model**: ResNet + Mean/LSTM (Compatibility prediction)

**Data Storage**
- File system storage (Categorized: top/pants/shoes, etc.)
- Automatic temporary file cleanup

---

## 🔄 Business Flow

### Main Flow: Full-Body Image Analysis (Recommended)

```
User Workflow:
1. Visit http://localhost:5000
2. Drag & drop or click to upload 1-5 full-body outfit photos
3. Click "Start Smart Analysis"
4. Wait for AI processing (10-30 seconds)
5. View analysis results and outfit recommendations

System Processing Flow:
┌─────────────────────────────────────────────┐
│ 1. Frontend receives user-uploaded images    │
│    - Validate file format (JPG/PNG/WEBP)    │
│    - Validate file size (<16MB)             │
│    - Limit quantity (1-5 images)            │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ 2. Send to Backend API                       │
│    POST /api/analyze-fullbody                │
│    Content-Type: multipart/form-data         │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ 3. Vision Model Processing (Mask R-CNN)     │
│    - Load vision/last_ckpt.pth              │
│    - Detect items in each image              │
│    - Identify clothing categories            │
│    - Generate segmentation masks             │
│    - Remove background                       │
│    - Crop with white background              │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ 4. Categorized Saving                        │
│    - Tops → uploads/top/                    │
│    - Pants → uploads/pants/                 │
│    - Shoes → uploads/shoes/                 │
│    - Dresses → uploads/dress/               │
│    - Naming: {category}_{idx}_{timestamp}.png │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ 5. Fashion Model Scoring (CP Model)         │
│    - Load models/cp_best_seed43.pth         │
│    - Select one item from each category     │
│    - Build outfit combination                │
│    - Call score_outfit_paths() for scoring  │
│    - Output compatibility score (0-1)       │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ 6. Generate Evaluation & Suggestions         │
│    - Overall evaluation based on score      │
│    - Color coordination advice              │
│    - Style matching suggestions             │
│    - Trend tags (3-6 tags)                  │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ 7. Return JSON Response                      │
│    {                                         │
│      "success": true,                        │
│      "detected_items": {...},                │
│      "combination": {...},                   │
│      "compatibility_score": 0.85,            │
│      "evaluation": "...",                    │
│      "color_suggestion": "...",              │
│      "style_suggestion": "...",              │
│      "trend_tags": ["...", "..."]            │
│    }                                         │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ 8. Frontend Displays Results                 │
│    - Show all detected items                 │
│    - Display recommended combination        │
│    - Circular progress bar (0-10 scale)     │
│    - Text evaluation and suggestions        │
│    - Style tag cloud                        │
└─────────────────────────────────────────────┘
```

### Alternative Flow 1: Categorized Upload

```
User Action:
1. Upload top, pants, and shoes separately
2. System auto-segments and saves
3. Direct compatibility scoring

API Endpoint: POST /api/recommend
Use Case: When you already have clear item images
```

### Alternative Flow 2: Direct Multi-Image Analysis

```
User Action:
1. Upload 2+ pre-segmented item images
2. Skip segmentation process
3. Direct scoring

API Endpoint: POST /api/upload
Use Case: Quick testing without segmentation
```

---

## 📁 Directory Structure

```
comp-7065-group/
├── frontend/                    # Frontend files
│   ├── index.html              # Main page
│   ├── css/
│   │   └── style.css           # Stylesheet (dark theme)
│   └── js/
│       └── main.js             # Frontend logic
│
├── backend/                     # Backend files
│   ├── app.py                  # Flask application entry
│   ├── config.py               # Configuration
│   ├── model_inference.py      # Core model inference
│   ├── requirements.txt        # Python dependencies
│   ├── vision/                 # Vision model
│   │   ├── last_ckpt.pth      # Mask R-CNN weights (~XXX MB)
│   │   └── load model and weight.py
│   ├── models/                 # Fashion model
│   │   └── cp_best_seed43.pth # CP model weights (96 MB)
│   └── uploads/                # Uploaded files storage
│       ├── top/               # Top images
│       ├── pants/             # Pants images
│       ├── shoes/             # Shoes images
│       ├── dress/             # Dresses
│       └── outwear/           # Outerwear
│
├── polyvore_route_a.py         # Core model implementation
├── closet_recommend.ipynb      # Closet recommendation Notebook
└── vision_preprocess_to_closet.ipynb  # Vision preprocessing Notebook
```

---

## 🚀 Quick Start

### Requirements

- Python 3.8+
- PyTorch 2.1.1+
- Flask 3.0.0+
- At least 4GB RAM (CPU mode)

### Installation Steps

```bash
# 1. Navigate to backend directory
cd backend

# 2. Install dependencies
pip install flask==3.0.0 flask-cors==4.0.0 pillow==10.1.0 \
    torch==2.1.1 torchvision==0.16.1 numpy==1.26.2

# 3. Verify model files exist
# - backend/vision/last_ckpt.pth
# - backend/models/cp_best_seed43.pth

# 4. Start the service
python app.py

# 5. Access the application
# Open browser: http://localhost:5000
```

### Verify Installation

```bash
# Check backend health
curl http://localhost:5000/api/health

# Expected response:
# {"status": "ok", "message": "Outfit Match API is running"}
```

---

## 🎯 Usage Examples

### Example 1: Single Full-Body Image Analysis

**Input:**
- 1 full-body outfit photo (containing top + pants + shoes)

**Processing:**
- Vision model detects 3 items
- Auto-categorize and save
- CP model score: 0.82

**Output:**
```json
{
  "success": true,
  "detected_items": {
    "top": [{"label": "short_sleeve_top", "confidence": 0.92}],
    "pants": [{"label": "trousers", "confidence": 0.88}],
    "shoes": [{"label": "sneakers", "confidence": 0.85}]
  },
  "combination": {
    "top": "/uploads/top/top_0_20260421_230000.png",
    "pants": "/uploads/pants/pants_0_20260421_230001.png",
    "shoes": "/uploads/shoes/shoes_0_20260421_230002.png"
  },
  "compatibility_score": 0.82,
  "evaluation": "This is a nice outfit! The combination shows good coordination...",
  "color_suggestion": "Color coordination is harmonious. Try using similar tones...",
  "style_suggestion": "Overall style direction is correct. Pay attention to fit...",
  "trend_tags": ["Casual Comfort", "Light Business", "Accessory Highlight"]
}
```

### Example 2: Multiple Full-Body Images Combination

**Input:**
- 3 different outfit photos

**Processing:**
- Total 8 items detected (3 tops + 3 pants + 2 shoes)
- System selects best combination
- Final score: 0.91

**Advantages:**
- More options to find the best match
- Mix and match items from different photos

---

## 🔧 API Documentation

### 1. Health Check

```http
GET /api/health
```

**Response:**
```json
{
  "status": "ok",
  "message": "Outfit Match API is running"
}
```

---

### 2. Full-Body Image Analysis (Main Endpoint)

```http
POST /api/analyze-fullbody
Content-Type: multipart/form-data
```

**Request Parameters:**
- `images`: File array (1-5 image files)

**Response:**
```json
{
  "success": true,
  "detected_items": {
    "top": [
      {
        "image": "/uploads/top/top_0_xxx.png",
        "label": "short_sleeve_top",
        "confidence": 0.92,
        "source_image": 1,
        "auto_detected": true
      }
    ],
    "pants": [...],
    "shoes": [...]
  },
  "combination": {
    "top": "/uploads/top/top_0_xxx.png",
    "pants": "/uploads/pants/pants_1_xxx.png",
    "shoes": "/uploads/shoes/shoes_0_xxx.png"
  },
  "compatibility_score": 0.85,
  "overall_evaluation": "Excellent outfit combination!...",
  "color_suggestion": "Color coordination is perfect...",
  "style_suggestion": "Style matching is well-balanced...",
  "trend_tags": ["Premium Style", "Minimalism", "Monochrome"],
  "mode": "real",
  "vision_available": true
}
```

**Error Response:**
```json
{
  "success": false,
  "error": "Maximum 5 images allowed"
}
```

---

### 3. Categorized Upload Recommendation

```http
POST /api/recommend
Content-Type: multipart/form-data
```

**Request Parameters:**
- `top`: Top image
- `pants`: Pants image
- `shoes`: Shoes image

**Response:** Same as full-body analysis

---

### 4. Direct Multi-Image Analysis

```http
POST /api/upload
Content-Type: multipart/form-data
```

**Request Parameters:**
- `images`: File array (minimum 2)

**Note:** No segmentation, direct scoring

---

### 5. Static Resources

```http
GET /uploads/{category}/{filename}
```

**Categories:** top, pants, shoes, dress, outwear, skirt

---

## 📊 Scoring System

### Compatibility Score

- **Range**: 0.0 - 1.0
- **Meaning**: Coordination level of outfit combination
- **Calculation**: Predicted by trained CP model

### Overall Score

- **Range**: 0.0 - 10.0
- **Conversion**: `compatibility_score × 10`
- **Grade Levels**:
  - 8.5 - 10.0: Excellent match
  - 7.0 - 8.4: Good match
  - 5.0 - 6.9: Average match
  - 0.0 - 4.9: Needs improvement

### Dimension Scores

- **Color Match**: Color harmony level
- **Style Match**: Style consistency
- **Trend Score**: Fashion trend alignment

---

## 🎨 Frontend Interface

### Main Components

1. **Upload Area**
   - Drag-and-drop support
   - Real-time preview grid
   - Remove buttons
   - Quantity indicator (x/5)

2. **Result Display**
   - Detected items grid
   - Recommended combination (horizontal layout)
   - Circular score progress bar
   - Detailed evaluation text
   - Suggestion cards (color + style)
   - Tag cloud

3. **Status Indicators**
   - Loading animation
   - Vision model status warning
   - Error messages (auto-dismiss after 3s)

### Responsive Design

- **Desktop**: Side-by-side layout
- **Tablet**: Stacked vertically
- **Mobile**: Single column, touch-optimized

---

## ⚙️ Configuration

### backend/config.py

```python
class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    
    MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), 'models')
    DEFAULT_MODEL = 'cp_best_seed43.pth'
```

### Modify Upload Limit

```python
# In app.py
if len(files) > 5:  # Change to desired number
    return jsonify({'success': False, 'error': 'Maximum X images allowed'}), 400
```

### Modify Frontend Limit

```javascript
// In main.js
const MAX_FILES = 5;  // Change to desired number
```

---

## 🐛 Troubleshooting

### Q1: Vision Model Not Loaded

**Symptom:** Warning "Auto-detection feature disabled"

**Cause:** `backend/vision/last_ckpt.pth` file missing

**Solution:**
1. Obtain pre-trained Mask R-CNN weights
2. Place at `backend/vision/last_ckpt.pth`
3. Restart backend service

---

### Q2: Slow Model Loading

**Cause:** Slower inference in CPU mode

**Solution:**
- First load takes 5-10 seconds (normal)
- Subsequent requests reuse model instance (fast)
- If GPU available, modify `self.device = torch.device("cuda")`

---

### Q3: Upload Failure

**Checklist:**
- [ ] Correct file format (JPG/PNG/WEBP)
- [ ] File size < 16MB
- [ ] Backend service running
- [ ] CORS configured properly (for cross-origin)

---

### Q4: Low Scores

**Possible Causes:**
- Poor image quality, inaccurate detection
- Large style differences between items
- Poor color coordination

**Suggestions:**
- Upload clear full-body photos
- Ensure good lighting
- Try uploading multiple photos for more options

---

