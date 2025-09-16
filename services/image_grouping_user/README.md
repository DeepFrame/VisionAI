# VisionAI - Docker Version

## Overview

This project provides an **image face detection and grouping system** that:

* Scans through images.
* Detects human faces.
* Groups similar faces together.
* Stores results (thumbnails, image paths, and face data) in a SQL Server database.

It ties together:

* A **Python-based detection script** (face recognition logic).
* A **SQL Server database** (to save metadata).
* **Docker** (to package and run everything consistently).

---

## Key Files

* **`.env`** → Holds configuration values (database connection, storage paths, container paths).
* **`commands.txt`** → Commands to build and run the Docker container.
* **`face_recognition.py` (or similar)** → Main Python script for:

  * Loading images.
  * Running face detection.
  * Saving cropped face thumbnails.
  * Grouping faces by similarity.
  * Writing results into the database.

---

## How It Works

1. **Load images** from the configured path.
2. **Detect faces** in each image using a face recognition library.
3. **Generate thumbnails** for detected faces.
4. **Group similar faces** (so one person’s images are clustered).
5. **Save data**:

   * Face image paths.
   * Thumbnail paths.
   * Database records.

---

## Running the System

1. Set up environment values in `.env`.
2. Load the SQL Server Docker image (already exported as a .tar file):

   ```bash
   docker load -i image_grouping_sqlserver.tar
   ```
3. Build and start the Docker container:

   ```bash
   docker compose build --no-cache
   docker compose up
   ```
4. The container runs the Python face recognition script automatically.

---
