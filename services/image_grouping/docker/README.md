# Image Grouping Service

This service is part of the Deepframe backend. It provides image recognition and detection functionality and is fully dockerized.

# Image Grouping Service Dockerization

This repository contains the **Image Grouping Service**, which performs **face detection and person recognition** on images stored in a database. This document explains how to build, configure, and run the service using **Docker**.

---

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Environment Variables](#environment-variables)
- [Dockerfile](#dockerfile)
- [docker-compose](#docker-compose)
- [Docker Setup](#docker-setup)
- [Running the Service](#running-the-service)
  - [One-shot mode](#one-shot-mode)
  - [Loop mode](#loop-mode)
- [Healthcheck](#healthcheck)
- [Logging](#logging)

---

## Features

- Detects faces in images stored in a database.
- Generates thumbnails for detected faces.
- Recognizes persons using embeddings and clustering.
- Supports continuous or automated processing.
- Fully dockerized for easy deployment.

---

## Prerequisites

- Docker
- Docker Compose
- Python 3.8 (for local testing, optional)

---

## Environment Variables

The service uses two types of environment variable files:

### 1. `.env` – Static Configuration
Used for database connection, file paths, and system storage:

```env
# Database connection
SQL_CONNECTION_STRING="Driver={ODBC Driver 17 for SQL Server};Server={(localdb)\MSSQLLocalDB};Database=FaceRecognitionSystem;Encrypt=no;TrustServerCertificate=no;"

# File paths
THUMBNAIL_SAVE_PATH="/app/services/image_grouping/image_face_detection/Thumbnails"
IMAGE_SAVE_PATH="/app/services/image_grouping/image_face_detection/Images"
DB_PREFIX="//server_ip/files/location"
CONTAINER_IMAGES_ROOT="/app/services/image_grouping/image_face_detection/Images"
SYSTEM_STORAGE="services/image_grouping/Thumbnails"
````

> **Note:** This file is specific to your server/DB setup and should not change often.

### 2. `.env.example` – Runtime / Service Settings

Used for controlling service behavior, mode, and intervals:

```env
SERVICE=recognition
MODE=oneshot
INTERVAL_SECONDS=900
RECOGNIZER_ARGS=--recluster
```

---
## Dockerfile

The Dockerfile sets up a Python 3.8 environment, installs **ODBC drivers**, **OpenCV dependencies**, and required Python packages.

Key steps:

1. Install ODBC drivers for SQL Server.
2. Install system dependencies for OpenCV.
3. Install Python dependencies from `requirements.txt`.
4. Copy service code into `/app/services/image_grouping`.
5. Setup entrypoint and healthcheck.

---

## docker-compose.yml

```yaml
services:
  image_grouping:
    build:
      context: .
      dockerfile: docker/Dockerfile
    container_name: image-grouping-service
    volumes:
      - ./person_recognition/logs:/app/services/image_grouping/person_recognition/logs
      - ./image_face_detection/logs:/app/services/image_grouping/image_face_detection/logs
      - ./Thumbnails:/app/services/image_grouping/Thumbnails
    env_file:
      - docker/.env.example
      - ./.env
    healthcheck:
      test: ["CMD-SHELL", "python ./docker/healthcheck.py"]
      interval: 10s
      timeout: 5s
      retries: 3
    restart: unless-stopped
    entrypoint: ["docker-entrypoint.sh"]
```

---

## Docker Setup

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

4. **Logs**

Logs are saved to host-mounted directories:

* `person_recognition/logs/`
* `image_face_detection/logs/`

## Notes

* Ensure the paths you provide exist and have proper permissions.
* Use secure credentials for your database connection.
* Adjust `SERVICE`, `MODE`, and `INTERVAL_SECONDS` in `.env` as needed.
* Ensure `SQL_CONNECTION_STRING` points to a valid SQL Server instance accessible from the container.
* `THUMBNAIL_SAVE_PATH` and `IMAGE_SAVE_PATH` are container paths; host volumes are mounted for persistence.
* By default, the service disables CUDA (`os.environ["CUDA_VISIBLE_DEVICES"]=""`) to avoid GPU errors inside the container.
* Use `--dry-run` to test pipeline execution without database modifications.
