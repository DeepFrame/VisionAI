# IMMICH Workflow Analysis

## 📌 Overview
This document details the face recognition pipeline implemented in the Immich photo management system. The pipeline handles face detection, alignment, embedding generation, clustering, and recognition using a series of ONNX-based machine learning models and image processing modules.

---

## 🌟 Features

- ☁️ **Automatic Backup**  
  Automatically backs up your photos and videos.

- 📺 **Google Casting and Screen Mirroring**  
  Cast your media to TVs or smart displays.

- 🧠 **Recognized Face Actions**
  - 📸 Select a feature photo for a person  
  - 📝 Assign a name to a recognized face  
  - 🎂 Enter date of birth  
  - 🔀 Merge people if incorrectly recognized  
  - 🙈 Hide people from the explorer view

- 📤 **Sharing Photos**  
  Easily share your photos or albums with others.

---

## 🔄 Face Recognition Workflow

1. Face Detection  
2. Face Cropping  
3. Face Recognition  
4. Embeddings Indexing  
5. Clustering  

| ![Flowchart of IMMICH Pipeline](immich_face_flow.svg) |
|:--:|
| **Figure 1:** Flowchart of IMMICH Workflow Pipeline. |

---

### 🧍‍♂️ Face Detection

- **Model Used:** ONNX Face Detection Model – RetinaFace (InsightFace model conceptually similar to SSD)
- **Architecture:** Uses a Feature Pyramid Network (FPN) to predict dense facial landmarks.
- **Function:** Detects faces in the input image and returns bounding boxes, confidence scores, and facial landmarks.
- **Post-processing:** Applies Non-Maximum Suppression (NMS) and filters detections using a confidence threshold.

---

### ✂️ Face Cropping

- **Process:**
  - Crop face regions based on bounding boxes.
  - Resize and normalize images to 112×112.
  - Compute and apply affine transformation using facial landmarks.
- **Purpose:** Prepares inputs for the recognition model by standardizing face orientation and size.

---

### 🧠 Face Recognition

- **Model Used:** ArcFace ONNX Model
- **Output:** 512-dimensional embedding vector per face.
- **Steps:**
  - Convert cropped face to blob format `[N, C, H, W]`  
    *(where N = Batch size, C = RGB channels, H = Height, W = Width)*  
  - Run inference using ArcFace.
  - Generate embeddings of 512 dimensions.
  - Apply L2 normalization to the embedding vector.

---

### 📇 Embeddings Indexing

- **Function:** Maintains a searchable index of face embeddings.
- **Usage:** Enables fast matching of new embeddings against stored ones using cosine distance (or cosine similarity).

---

### 🧊 Clustering

- **Algorithm:** DBSCAN (Density-Based Spatial Clustering of Applications with Noise)  
  - `eps = 0.6`  
  - `minPts = 3`
- **Purpose:** Groups similar face embeddings into clusters.
- **Features:**
  - Automatically handles unknown or unclustered faces.
  - Allows manual intervention: merge identities, assign names, or hide individuals.

---
