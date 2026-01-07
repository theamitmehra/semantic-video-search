# Semantic Video Search 🎬🔍

**Find exact timestamps in videos using natural language queries.** Type "a person wearing a red hat" and get the precise frame where it happens—powered by OpenAI's CLIP and optimized with vectorized NumPy/PyTorch operations.

## ✨ Features

- **Semantic Search**: Find video moments using natural language descriptions
- **Fast Vectorized Operations**: Leverages NumPy and PyTorch for sub-second similarity computation
- **Exact Timestamps**: Returns precise frame locations with similarity scores
- **Production Ready**: Error handling, progress bars, and formatted output

## 🎯 Use Cases

- **Content Moderation**: Find specific scenes in user-generated content
- **Video Analytics**: Search security footage for specific events
- **Content Creation**: Locate scenes for editing and compilation
- **Compliance**: Find regulatory violations in recorded meetings
- **Media Management**: Intelligent video cataloging without metadata

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/theamitmehra/semantic-video-search.git
cd semantic-video-search

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Basic Usage

```python
from semantic_video_search import SemanticVideoSearch

# Initialize search engine (GPU-enabled)
search_engine = SemanticVideoSearch(device="cuda")

# Extract frames from video
frames, timestamps = search_engine.extract_frames(
    "path/to/video.mp4", 
    fps_sample=2  # Sample every 2 frames
)
search_engine.timestamps = timestamps

# Compute embeddings (vectorized batch processing)
search_engine.compute_frame_embeddings(frames, batch_size=32)

# Search for moments
results = search_engine.search(
    "person wearing a red hat",
    top_k=5,
    threshold=0.25
)

# Display results
search_engine.display_results(results, "person wearing a red hat")
```

### Expected Output

```
======================================================================
Query: 'person wearing a red hat'
Found 5 results

1. [████████████████████████████████████████] 0.942 @ 00:00:12.50
2. [██████████████████████████░░░░░░░░░░░░░░] 0.873 @ 00:00:28.30
3. [██████████████████░░░░░░░░░░░░░░░░░░░░░░] 0.792 @ 00:00:45.80
4. [███████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 0.721 @ 00:01:02.10
5. [█████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 0.681 @ 00:01:11.40
======================================================================
```

## 📊 How It Works

### 1. **Frame Extraction** 🎥
Samples frames from video at configurable rate (e.g., every 2 frames for 30fps video = 0.5s intervals)

### 2. **CLIP Embeddings** 🧠
Converts frames and text into 512-dimensional vectors using OpenAI's CLIP model:
- Image encoder: Vision Transformer (ViT-B/32)
- Text encoder: Transformer-based language model
- Both mapped to shared embedding space

### 3. **Vectorized Similarity** ⚡
Computes cosine similarity using optimized matrix operations:
```
similarities = frame_embeddings @ query_embedding.T
# Shape: (num_frames,) computed in O(n) with GPU acceleration
```

### 4. **Ranking & Filtering** 🏆
Returns top-k results with similarity scores above threshold

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│        Video Input (MP4/MOV/WebM)       │
└──────────────┬──────────────────────────┘
               │
        ┌──────▼────────┐
        │ Frame Extract │  (OpenCV, configurable FPS)
        └──────┬────────┘
               │
        ┌──────▼─────────────┐
        │  CLIP Image Encoder │  (ViT-B/32, GPU batch)
        └──────┬─────────────┘
               │
        ┌──────▼────────────────────┐
        │ Normalized Embeddings     │  (512-dim vectors)
        │ (tensor shape: Nx512)     │
        └──────┬────────────────────┘
               │
        ┌──────▼────────────┐
        │ Text Query        │
        │ CLIP Encoder      │
        └──────┬────────────┘
               │
        ┌──────▼──────────────────────┐
        │ Vectorized @ Operation      │  (PyTorch matmul)
        │ Cosine Similarity Scores    │
        └──────┬──────────────────────┘
               │
        ┌──────▼──────────────┐
        │ Top-K Filtering &   │
        │ Threshold Filtering │
        └──────┬──────────────┘
               │
        ┌──────▼──────────────────┐
        │ Results with Timestamps │
        │ & Similarity Scores     │
        └───────────────────────────┘
```

## 💻 Technology Stack

- **CLIP Model**: `openai/CLIP` (ViT-B/32)
- **Video Processing**: OpenCV (`cv2`)
- **Numerical Computing**: NumPy, PyTorch
