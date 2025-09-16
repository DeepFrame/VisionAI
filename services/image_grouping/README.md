# VisionAI

This repository provides a **face detection and recognition pipeline** that integrates detection, embedding generation, and clustering of faces. It supports single-image testing, batch processing from a database, continuous monitoring, and automated pipelines for recurring processing.

![Face Recognition System](FaceRecognitionSystem.png)

## Features

* **Face Detection:** Detect faces in images or media items stored in a database.
* **Face Recognition:** Generate embeddings for detected faces and cluster them into distinct persons.
* **Automated Pipelines:** Run full detection and recognition periodically.
* **Reclustering:** Recompute clusters for unassigned faces by comparing them against each other or matching them with previously assigned persons.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/DeepFrame/deepframe-backend.git
cd deepframe-backend/services/image_grouping
```

2. Install dependencies (recommended to use a virtual environment with Python v3.8.20):

```bash
pip install -r requirements.txt
```

3. Configure your database connection in `.env`:

```python
SQL_CONNECTION_STRING = "DRIVER={SQL Server};SERVER=your_server;DATABASE=your_db;UID=user;PWD=password"
```

4. Ensure your database contains the required tables with required fields:

* `MediaFile`: Id, FilePath, FileName, Extensions, CreatedAt, ModifiedAt
* `MediaItems`: Id, MediaFileId, Name, IsFacesExtracted, FacesExtractedOn
* `Faces`: Id, MediaItemId, PersonId, BoundingBox, Embedding, FrameNumber, Name, CreatedAt, ModifiedAt
* `Persons`: Id, PortraitMediaFileId, Name, Rank, Appointment, CreatedAt, ModifiedAt

## Usage

The main CLI is `main.py`, supporting multiple modes:

```bash
python main.py [options]
```

### Options

| Option                | Description                                                          |
| --------------------- | -------------------------------------------------------------------- |
| `--test <image_path>` | Run face detection on a single image.                                |
| `--db`                | Run face detection on all database media items once.                 |
| `--watch`             | Continuously monitor and process new media items.                    |
| `--recognize`         | Generate embeddings and cluster detected faces.                      |
| `--all`               | Run full pipeline: detection → recognition once.                     |
| `--automate`          | Run full pipeline every 3 minutes.                                   |
| `--recluster`         | Rebuild clusters for unmatched faces by comparing with existing persons and with each other. |

### Examples

1. **Test a single image:**

```bash
python main.py --test path/to/image.jpg
```

2. **Process all database media items:**

```bash
python main.py --db
```

3. **Run full pipeline once:**

```bash
python main.py --all
```

4. **Run automated pipeline every 3 minutes:**

```bash
python main.py --automate
```

5. **Recluster all faces:**

```bash
python main.py --recluster
```

## Project Structure

```
image_grouping/                                    # Face recognition pipeline service
├── .env                                           # Environment variables (DB, configs)
├── FaceRecognitionSystem.png                      # System architecture diagram
├── Notebooks/                                     # Jupyter notebooks & docs
│   ├── Face_Recognition_System_(FRS).ipynb        # End-to-end recognition pipeline
│   ├── README.md                                  # Notes for notebooks
│   └── face-recognition-system-frs.pdf            # Notebook exported as PDF
├── README.md                                      # Service overview
├── config.py                                      # App & DB configuration
├── docker/                                        # Docker deployment setup
│   ├── .env.example                               # Example environment file
│   ├── Dockerfile                                 # Container build instructions
│   ├── README.md                                  # Docker usage notes
│   ├── docker-entrypoint.sh                       # Startup script
│   ├── healthcheck.py                             # Service health check
│   ├── requirements.txt                           # Python dependencies
│   └── requirements_locked.txt                    # Frozen dependency versions
├── docker-compose.yml                             # Multi-service orchestration
├── image_face_detection/                          # Face detection module
│   ├── Images/                                    # Raw input images
│       ├── GroupM.jpg, … wedding.jpg              # (multiple test samples)
│       └── README.md                              # Notes about test images
│   ├── Thumbnails/                                # Cropped detected faces
│       ├── conference_TN1.jpg … news_TN.jpg       # (face thumbnails)
│   ├── Thumbnails.jpeg                            # Combined thumbnail preview
│   ├── detect_faces.py                            # Face detection + cropping logic
│   ├── logger_config.py                           # Logging setup
│   ├── logs/                                      # Detection logs
│       └── face_detection.log
│   └── README.md                                  # Detection module docs
├── main.py                                        # CLI entry point
├── person_recognition/                            # Person recognition module
│   ├── recognize_persons.py                       # Face embedding + clustering
│   ├── logger_config.py                           # Logging setup
│   ├── logs/                                      # Recognition logs
│       └── embeddings_clustering.log
│   ├── README.md                                  # Recognition module docs
│   └── sql/                                       # SQL queries for recognition
│       ├── link_tables.sql                        # Link face/person tables - SQL Procedure
│       ├── upsert_person.sql                      # Insert/update person records - SQL Procedure
│       └── README.md
├── requirements.txt                               # Service dependencies
├── sample_images/                                 # Example input images
│   ├── conference.jpg, interview.jpg, news.jpg    # (sample images)
│   └── README.md
└── sql/                                           # Core database schema & queries
    ├── Portrait.sql                               # Portrait handling script
    ├── aditionals.sql                             # Extra SQL utilities
    ├── clusters_display.sql                       # View clusters
    ├── create_database&tables.sql                 # DB schema creation
    ├── delete.sql                                 # Delete data queries
    ├── sample_insert_script.sql                   # Insert sample records
    ├── show_tables.sql                            # Show database tables
    └── README.md
```

## Logging

* Logs are stored in `logs/embeddings_clustering.log`.
   - Provides detailed info about embeddings generation and clustering.
* Logs are stored in `logs/face_detection.log`.
   - Provides detailed info details of the face-detection pipeline, including processed images, detected faces, database updates, and any errors or warnings.

## Notes

* **FaceNet** is used for embedding generation.
* **DBSCAN + UMAP** is used for clustering embeddings.
* **Cosine Similarity** assigns unlabelled faces to known persons if similarity ≥ 0.8.
* Ensure images are accessible and bounding boxes are valid for proper embedding generation.
* Reclustering is optional but recommended when significant new faces are added.

---
