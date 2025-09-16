## Video Demonstrations

This repository contains video demonstrations of **VisionAI**. Each video showcases specific parts of the system, from database setup to face detection and recognition pipelines.

---

## 1. `Automated_Face_Recognition_System_with_Re_Clustering.mp4`

Demonstrates the full automated face recognition workflow with re-clustering.

**Key Steps Covered:**
- Creation of database tables and insertion of records.
- Resetting media items before processing:
  ```sql
  SET IsFacesExtracted = FALSE;
  SET FacesExtractedOn = NULL;
    ````

* Activating the conda environment.
* Running the main script with automation:

  ```bash
  python main.py --automate
  ```

  * Automatically fetches unprocessed images.
  * Performs face detection and recognition.
* Updating the database after processing:

  ```sql
  SET IsFacesExtracted = TRUE;
  SET FacesExtractedOn = CURRENT_TIMESTAMP;
  ```
* Logs generated during processing.
* Thumbnails created and saved in the designated folder.

---

## 2. `Database&Tables_Creation.mp4`

Demonstrates database setup and initial data insertion.

**Key Steps Covered:**

* Creating the database and required tables.
* Inserting sample records for testing and demonstration.

---

## 3. `Detection and Recognition Pipeline.mp4`

Shows the face detection and recognition pipeline in action.

**Key Steps Covered:**

* Setting up the database schema.
* Inserting media records for processing.
* Running the detection and recognition pipeline.
* Updating the database with results, including generated thumbnails.

---

## 4. `Face Detection.mp4`

Demonstrates real-time face detection and processing.

**Key Steps Covered:**

* Running the main script with continuous monitoring:

  ```bash
  python main.py --watch
  ```
* Detecting unprocessed media files automatically.
* Performing face detection and recognition.
* Creating and storing thumbnails for processed media.

---



