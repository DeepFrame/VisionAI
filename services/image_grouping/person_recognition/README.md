# Face Recognition and Clustering

## Overview
This module performs **face recognition** by generating embeddings for detected faces and clustering them into distinct persons. It works with faces already detected and stored in a database. The pipeline supports generating embeddings, clustering unlabelled faces, and reclustering all faces when needed.

---

## Features
- Generate embeddings for faces missing them in the database.  
- Cluster unlabelled faces into distinct persons.  
- Recluster all faces from scratch, ignoring previous `PersonId` mappings.  
- Incremental updates: assign new faces to existing clusters using similarity.  
- Detailed logging of embedding generation, clustering, and errors.

---

## Used Models
- **FaceNet** (`keras-facenet`) for face embeddings.  
- **UMAP** for dimensionality reduction of embeddings.  
- **DBSCAN** for clustering faces based on embedding similarity.

---

## Methodology Workflow
1. **Process Missing Embeddings**
   - Fetch faces with missing embeddings from the database.  
   - Crop faces from images using stored bounding boxes.  
   - Generate embeddings using FaceNet.  
   - Update the database with generated embeddings.

2. **Clustering / Reclustering**
   - Fetch unlabelled or reclustered embeddings.  
   - Optionally, reduce dimensionality with UMAP.  
   - Cluster embeddings using DBSCAN.  
   - Assign clusters as `PersonId` in the database.  

3. **Incremental Matching**
   - Compute cosine similarity of new embeddings with labelled faces.  
   - Assign new faces to existing clusters if similarity exceeds threshold.  
   - Create new clusters for unmatched faces.

---

## Code Running

### Install dependencies in Conda
```bash
pip install -r requirements.txt
```

## Run Pipeline Options

All commands are executed via `main.py` CLI.

| Option        | Description                                                   |
|---------------|---------------------------------------------------------------|
| `--recognize` | Generate embeddings and cluster detected faces.              |
| `--recluster` | Rebuild clusters for all faces, ignoring existing PersonId mappings. |
| `--all`       | Run full pipeline (detection + recognition).                |
| `--automate`  | Run full pipeline continuously every 3 minutes.             |

### Example Usage

**Generate embeddings and cluster unlabelled faces**
```bash
python main.py --recognize
```

**Recluster unlabelled faces using labelled faces as reference**
```bash
python main.py --recluster
```

**Run the complete detection and recognition pipeline automatically**
```bash
python main.py --all
python main.py --automate
```
