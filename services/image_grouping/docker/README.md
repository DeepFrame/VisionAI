# Image Grouping Service

This service is part of the Deepframe backend. It provides image recognition and detection functionality and is fully dockerized.

## Prerequisites

- Docker
- Docker Compose
- Python 3.8 (for local testing, optional)

## Setup

1. **Clone the repository**

```bash
git clone <repository-url>
cd deepframe-backend/services/image_grouping
````

2. **Configure environment variables**

Create a `.env` file in the service directory. You can start from `.env.example`:

```bash
cp docker/.env.example .env
```

Edit the `.env` file to provide the correct paths and database credentials:

```env
# Database connection string
SQL_CONNECTION_STRING="Driver={ODBC Driver 17 for SQL Server};Server=<YOUR_SERVER>;Database=<YOUR_DB>;Encrypt=no;TrustServerCertificate=no;"

# Paths to save images and thumbnails
THUMBNAIL_SAVE_PATH="<LOCAL_PATH_TO_THUMBNAILS>"
IMAGE_SAVE_PATH="<LOCAL_PATH_TO_IMAGES>"

# Service settings
SERVICE=recognition
MODE=oneshot
INTERVAL_SECONDS=900
RECOGNIZER_ARGS=--recluster
```

**Make sure to replace:**

* `<YOUR_SERVER>` with your SQL Server hostname or IP.
* `<YOUR_DB>` with your database name.
* `<LOCAL_PATH_TO_THUMBNAILS>` with a local path where thumbnails should be saved.
* `<LOCAL_PATH_TO_IMAGES>` with a local path where images should be saved.

3. **Build and start the container**

```bash
docker compose build
docker compose up
```

This will:

* Build the Docker image.
* Start the container with mounted volumes for logs and images.
* Run the service according to your `.env` configuration.

4. **Healthcheck**

The container includes a healthcheck to ensure the service is ready. You can monitor it via:

```bash
docker ps
docker inspect <container_name> --format='{{json .State.Health}}'
```

5. **Logs**

Logs are saved to host-mounted directories:

* `person_recognition/logs/`
* `image_face_detection/logs/`

## Notes

* Ensure the paths you provide exist and have proper permissions.
* Use secure credentials for your database connection.
* Adjust `SERVICE`, `MODE`, and `INTERVAL_SECONDS` in `.env` as needed.
