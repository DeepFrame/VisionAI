# VisionAI - Face Detection and Recognition System

This repository provides a complete pipeline for **face detection, embedding extraction, clustering, and portrait selection** using Python. The system can handle multiple images, remove blurry faces, and organize faces by identity.

---

## Features

- **Face Detection:** Uses [RetinaFace](https://github.com/serengil/retinaface) for high-accuracy detection.
- **Blurry Face Filtering:** Skips faces with low Laplacian variance.
- **Face Cropping:** Maintains aspect ratio, adds padding to square, resizes to 112x112.
- **Embedding Extraction:** Uses [FaceNet](https://github.com/nyoki-mtl/keras-facenet) for 512-dimensional embeddings.
- **Dimensionality Reduction:** Optional UMAP projection for visualization.
- **Clustering:** DBSCAN clustering with reclustering for unassigned faces.
- **Portrait Selection:** Medoid and sharpness-based selection of representative faces.
- **Organized Output:** Saves cropped faces, original images by cluster, and individual portraits.

---

## Installation

### Install Dependencies

```bash
pip install retina-face onnxruntime insightface keras-facenet umap-learn scikit-learn matplotlib opencv-python tensorflow numpy
````

---

## Usage

### 1. Face Detection

```python
from retinaface import RetinaFace
import cv2
from google.colab.patches import cv2_imshow

# Detect, filter, and crop faces from images
process_image_bbox(image_path='/content/images/sample.jpg', output_dir='/content/detected_faces')
```

* **Options:**

  * `min_score`: Minimum confidence threshold (default 0.99)
  * `blur_threshold`: Laplacian variance threshold for blurry faces (default 100)
  * `margin_ratio`: Margin around detected face (default 0.2)
  * `target_size`: Output face size (default (112, 112))

### 2. Full Pipeline

```python
labels, face_files = process_pipeline(images_dir='/content/images', faces_dir='/content/detected_faces')
```

* Processes new images
* Performs embedding extraction
* Clusters faces with DBSCAN
* Optionally reclusters unassigned faces

### 3. Visualize Clusters

```python
face_imgs = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in face_files]
show_cluster_faces(face_imgs, labels)
```

### 4. Save Clustered Faces

```python
save_faces_by_cluster(face_files, labels, faces_dir='/content/individuals_recognized')
save_originals_by_cluster(face_files, labels, original_dir='/content/images', output_dir='/content/individuals_originals')
```

### 5. Portrait Selection (Medoid + Sharpness)

```python
# Compute medoid for each cluster
for cluster_label in set(labels):
    medoid_face = compute_cluster_medoid(face_files, embeddings, cluster_label, labels)
    sharpness = sharpness_score(medoid_face)
```

* Medoid face = most central face in cluster
* Sharpness = variance of Laplacian (higher = sharper)

---

## Output Structure

```
/detected_faces/
    face1.jpg
    face2.jpg
/individuals_recognized/
    person_0/
    person_1/
/individuals_originals/
    person_0/
    person_1/
/individuals_portraits/
    person_0_PORTRAIT.jpg
    person_1_PORTRAIT.jpg
```

---

## Visualization

* **UMAP Projection:** 2D visualization of embeddings colored by cluster.
* **Cluster Display:** Shows images in each cluster, including noise/outliers.

---

## Notes

* Works best on images where faces are frontal and well-lit.
* Blurry or low-confidence faces are skipped.
* You can adjust DBSCAN parameters (`eps` and `min_samples`) to control clustering sensitivity.
* Pipeline supports incremental processing of new images.

---

## References

* [RetinaFace](https://github.com/serengil/retinaface)
* [FaceNet Keras Implementation](https://github.com/nyoki-mtl/keras-facenet)
* [UMAP for Dimensionality Reduction](https://umap-learn.readthedocs.io/)
* [DBSCAN Clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)

---

```
