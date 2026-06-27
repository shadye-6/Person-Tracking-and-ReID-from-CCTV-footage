# Person Tracking and Re-Identification from CCTV Footage

## Overview

This project implements a complete person tracking and face re-identification pipeline for CCTV footage. It combines object detection, multi-object tracking, face recognition, and persistent identity management to identify people across video frames and across multiple video sessions.

The system detects people using YOLO, extracts facial embeddings using InsightFace, matches identities using cosine similarity and FAISS, stores facial embeddings for future recognition, and generates appearance logs containing timestamps and durations for every detected individual.

A Streamlit web application is included to simplify video processing, identity management, and result visualization.

---

## Features

- Person detection using YOLO11
- Multi-object person tracking
- Face detection and embedding extraction using InsightFace
- Persistent face database for cross-video recognition
- Fast similarity search using FAISS
- Automatic creation of new identities for unseen individuals
- Appearance logging with timestamps and durations
- Interactive Streamlit interface
- Manual assignment of names to Face IDs
- Downloadable face database, logs, and identity mappings
- Face clustering using DBSCAN for identity analysis

---

## Project Structure

```
Person-Tracking-and-ReID-from-CCTV-footage/
│
├── ID_log/
│   ├── face_db.json          # Persistent face embedding database
│   ├── face_names.json       # FaceID to person name mapping
│   ├── face_log.csv          # Detection logs
│   └── face_clusters.csv     # Clustered identities
│
├── models/
│   ├── yolo11n.pt
│   ├── yolo11s.pt
│   └── yolov8n.pt
│
├── test_videos/
│
├── main.py                   # Standalone processing script
├── main_2.py                 # Streamlit application
├── cluster.py                # Face clustering utility
├── requirements.txt
└── README.md
```

---

# System Pipeline

1. Detect people using YOLO.
2. Track each detected person across frames.
3. Crop the upper body region for face detection.
4. Extract facial embeddings using InsightFace.
5. Normalize embeddings.
6. Search the embedding against the persistent face database.
7. Match existing identities using cosine similarity and FAISS.
8. Assign a new FaceID if no match is found.
9. Store embeddings for future recognition.
10. Generate appearance logs and annotated output video.

---

# Technologies Used

| Component | Library |
|----------|----------|
| Object Detection | YOLO11 (Ultralytics) |
| Tracking | YOLO Track |
| Face Recognition | InsightFace |
| Similarity Search | FAISS |
| Web Interface | Streamlit |
| Computer Vision | OpenCV |
| Numerical Computing | NumPy |
| Data Processing | Pandas |
| Deep Learning | PyTorch |

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Person-Tracking-and-ReID-from-CCTV-footage.git

cd Person-Tracking-and-ReID-from-CCTV-footage
```

---

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Requirements

```
streamlit==1.36.0
ultralytics==8.2.90
insightface==0.7.3
faiss-cpu==1.7.4
numpy==1.26.4
pandas==2.2.2
opencv-python-headless==4.10.0.84
torch==2.2.2
torchvision==0.17.2
onnx==1.16.2
onnxruntime==1.18.0
protobuf==4.25.3
tqdm==4.66.4
```

**Note**

- `faiss-cpu` is used by default.
- `faiss-gpu` can be installed instead if CUDA is available.

---

# Model Files

Place the pretrained YOLO weights inside the `models/` directory.

```
models/
    yolo11n.pt
    yolo11s.pt
    yolov8n.pt
```

The application currently uses:

```
models/yolo11s.pt
```

---

# Running the Project

## Option 1 — Streamlit Application (Recommended)

Launch the web interface:

```bash
streamlit run main_2.py
```

After launching:

1. Open the provided localhost URL in your browser.
2. Upload a CCTV video.
3. Configure processing parameters from the sidebar.
4. Click **Start Processing**.
5. View the annotated output video.
6. Assign names to newly detected Face IDs.
7. Download logs and databases if required.

---

## Option 2 — Standalone Python Script

Edit the input/output paths inside `main.py` if necessary:

```python
INPUT_PATH = "test.mp4"
OUTPUT_PATH = "output.mp4"
```

Run:

```bash
python main.py
```

Outputs generated:

- Annotated video
- `face_db.json`
- `face_log.csv`

---

## Option 3 — Face Clustering

After generating the face database and logs:

```bash
python cluster.py
```

This produces:

```
ID_log/
    face_clusters.csv
```

The clustering groups highly similar identities using DBSCAN and cosine similarity.

---

# Output Files

## face_db.json

Stores normalized facial embeddings for every detected identity.

Example:

```json
{
    "0": [...],
    "1": [...],
    "2": [...]
}
```

---

## face_names.json

Maps Face IDs to user-assigned names.

Example:

```json
{
    "0": "Alice",
    "1": "Bob"
}
```

---

## face_log.csv

Stores appearance information.

| FaceID | Name | StartTime | EndTime | DurationSeconds | RunTimestamp |
|---------|------|-----------|---------|-----------------|--------------|

---

## face_clusters.csv

Groups similar identities.

| ClusterID | FaceIDs | StartFrames | EndFrames | Durations |
|------------|----------|-------------|------------|------------|

---

# Adjustable Parameters

The Streamlit application exposes several configurable parameters:

| Parameter | Description |
|------------|-------------|
| Similarity Threshold | Minimum cosine similarity required for identity matching |
| Embedding Buffer Size | Maximum embeddings retained for each identity |
| Face Detection Interval | Number of frames skipped between face detections |
| Processing FPS | Frame rate used during inference |
| Minimum Track Duration | Minimum tracking duration before assigning an identity |

---

# Identity Assignment

The application supports persistent naming of detected individuals.

Workflow:

1. Detect a new face.
2. Assign a new FaceID.
3. Enter the person's name after processing.
4. Save the mapping.
5. Future detections automatically display the assigned name.

---

# Face Matching Strategy

The recognition pipeline performs:

- Face embedding extraction
- L2 normalization
- FAISS nearest-neighbor search
- Cosine similarity verification
- Brute-force fallback when FAISS is unavailable
- Persistent embedding updates for improved future matching

---

# Future Improvements

Potential extensions include:

- Multi-camera re-identification
- GPU-accelerated FAISS support
- ByteTrack or DeepSORT integration
- Face quality assessment before embedding extraction
- Live RTSP camera support
- Automatic face gallery management
- Export of appearance statistics
- REST API deployment
- Docker support
- PostgreSQL or SQLite backend for identity storage

---

# License

This project is intended for academic and research purposes. Ensure compliance with local privacy regulations and organizational policies when processing CCTV footage containing identifiable individuals.
