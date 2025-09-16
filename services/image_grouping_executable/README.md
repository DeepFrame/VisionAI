# Automated Face Recognition System - Executable Version

This folder contains the **executable-ready version** of the Automated Face Recognition System. It is designed to detect, recognize, and cluster faces automatically from media files and update the database accordingly.

---

## 1. Python File Functionalities

- **`main.py`** and supporting scripts include:
  - Detection and cropping of faces from unprocessed media.
  - Thumbnail generation and saving in `Thumbnails/`.
  - Face embedding generation using **FaceNet**.
  - Clustering of unlabelled faces using **DBSCAN** and optional reclustering.
  - Automatic updating of database tables:
    - `dbo.MediaItems`
    - `dbo.Faces`
    - `dbo.Persons`
  - Generation of portraits for each person based on medoid and sharpest face.
  - Logging in `logs/detection_recognition.log`.

---

## 2. Reclustering

- The system **reclusters faces every time** based on existing entries in the `Persons` table.
- Ensures that newly added faces are assigned to the correct clusters automatically.

---

## 3. `.env` File

- Contains SQL Server connection string and credentials:
  ```env
  SQL_CONNECTION_STRING="Driver={ODBC Driver 17 for SQL Server};Server=(localdb)\MSSQLLocalDB;Database=Face_Recognition_System;Encrypt=no;TrustServerCertificate=no;"
  ````

* **Must be updated according to your system** before running the pipeline.

---

## 4. Database Requirement

* A **database must exist** with all required tables:

  * `MediaItems`
  * `MediaFile`
  * `Faces`
  * `Persons`
* Tables should follow the schema used by the Python scripts for proper operation.

---

## 5. Required Stored Procedures

* The system relies on the following **stored procedures** in the database:

  1. `upsert_person` — For inserting or updating person assignments.
  2. `link_tables` — For inserting or updating faces assignments.

---
