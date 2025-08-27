# Face Recognition System

The **Face Recognition System** is designed to provide an end-to-end solution for managing and identifying faces across large collections of images and media files. It combines modern deep learning models with robust database integration, offering both accuracy and scalability. 

At its core, the system leverages **RetinaFace** for face detection and **FaceNet** for generating unique face embeddings. These embeddings are then clustered using **UMAP** for dimensionality reduction and **DBSCAN** for unsupervised grouping, making it possible to automatically discover and organize individuals in media datasets. For incremental recognition, the system applies **cosine similarity** to match new faces against already known persons.

![Face Recognition System](services/image_grouping/FaceRecognitionSystem.png)

### Key Highlights
- Seamless integration with SQL Server for storing faces, persons, and metadata.  
- Automated and manual execution modes, suitable for both small-scale testing and continuous monitoring.  
- A reclustering mechanism that improves accuracy by refining person assignments as more data becomes available.  
- A clear logging and auditing trail to support monitoring, debugging, and traceability.  

With its modular pipeline and configurable architecture, the Face Recognition System can be adapted for diverse use cases including digital asset management, security monitoring, media analysis, and research.
## Project Structure
```bash
├── deepframe-backend/                             # Backend services
│ ├── services/image_grouping/                     # Face recognition pipeline service
│ │ └──  .env                                      # Environment variables (DB credentials, configs)
│ │ ├── config.py                                  # Database connection
│ │ ├── main.py                                    # Entry point for CLI
│ │ ├── requirements.txt                           # Install required dependencies
│ │ ├── image_face_detection/                      # Face detection module
│ │ │   ├── detect_faces.py                        # Face detection, cropping & thumbnails
│ │ │   └── logger_config.py                       # Logs configuration
│ │ │   └── logs/                                  # Detection logs
│ │ ├── person_recognition/                        # Face recognition module
│ │ │   ├── recognize_persons.py                   # Embedding + clustering logic
│ │ │   └── logger_config.py                       # Logs configuration
│ │ │   └── logs/                                  # Recognition logs
│ │ │   └── sql/
│ │ │   │   ├── upsert_person.sql
│ │ │   │   └── link_tables.sql
│ │ ├── Notebooks/                 
│ │ │ └── Face_Recognition_System_(FRS).ipynb      # Face detection and recognition jupyter notebook
│ │ └── README.md
├── docs/
│ ├── Face_Recognition_System_Manual.pdf
│ └── immich_face_recognition.md
├── backend/                                       # Backend files
│ ├── ....
```

## Setup and Installation
### Installation

1. Clone the repository:

```bash
git clone https://github.com/DeepFrame/deepframe-backend.git
cd deepframe-backend/services/image_grouping
```

2. Install dependencies (recommended to use a virtual environment with Python v3.8.20):

```bash
pip install -r requirements.txt
```

3. Configure your database connection in `config.py`:

```python
SQL_CONNECTION_STRING = "DRIVER={SQL Server};SERVER=your_server;DATABASE=your_db;UID=user;PWD=password"
```

4. Ensure your database contains the required tables with required fields:

* `MediaFile`
* `MediaItems`
* `Faces`
* `Persons`

### Usage

The main CLI is `main.py`, supporting multiple modes:

```bash
python main.py [options]
```

#### Options

| Option                | Description                                                          |
| --------------------- | -------------------------------------------------------------------- |
| `--test <image_path>` | Run face detection on a single image.                                |
| `--db`                | Run face detection on all database media items once.                 |
| `--watch`             | Continuously monitor and process new media items.                    |
| `--recognize`         | Generate embeddings and cluster detected faces.                      |
| `--all`               | Run full pipeline: detection → recognition once.                     |
| `--automate`          | Run full pipeline every 3 minutes.                                   |
| `--recluster`         | Rebuild clusters for unmatched faces by comparing with existing persons and with each other. |

