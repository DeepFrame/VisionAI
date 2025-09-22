![Thumbnail Comparison](Thumbnails_Comparison.png)

| Feature / Component               | Immich                                                                                          | VisionAI                                                                                      |
|----------------------------------|------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| **Model for Detection**           | `InsightFace` | `RetinaFace` |
| **Cropping**  | Tight Crop. | Margin based croping. |
| **Model for Embedding Generation**| **ArcFaceONNX** | **FaceNet** (via `keras_facenet`)  |
| **Model for Clustering**          | Usually DBSCAN on embeddings | **DBSCAN** clustering and reclustering with cosine similarity and DBSCAN, optionally with UMAP dimensionality reduction.   |
| **Handling Outliers**             | Faces below `min_score` or embeddings not assigned remain unclustered until further processing. | Faces below similarity threshold remain unassigned until DBSCAN clustering.                   |
| **Acceleration / Hardware Support**| ONNX Runtime or InsightFace session (CUDA, OpenVINO, CPU).     | TensorFlow GPU/CPU; memory growth enabled for GPU.                                           |
| **Embedding Storage / DB**        | Embeddings serialized (`serialize_np_array`) and stored externally (e.g., DB or vector store). | Embeddings stored in database (varbinary or JSON). Used for clustering and recognition.      |
| **Additional Features**           | ONNX session optimization | Portrait generation (medoid + sharpness) |
