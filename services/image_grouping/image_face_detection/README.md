# Face Detection Pipeline & Database Update

## 🎯 Objective
Build a Python script that:
1. Reads unprocessed media entries from SQL Server database
2. Performs face detection using RetinaFace model
3. Saves thumbnails of detected faces
4. Updates the database with processing results

## 🔄 Workflow Summary
1. Connect to database and query media items where `IsFacesExtracted = FALSE`
2. For each media file:
   - Detect faces using RetinaFace
   - Align, crop, and resize faces to 112x112 thumbnails
   - Save thumbnails to configured `thumbnails/` directory
   - Update database marking the media item as processed and save thumbnail info
3. Repeat periodically for continuous processing
4. Update database:
   - Set `IsFacesExtracted = TRUE`
   - Set `FacesExtractedOn = CURRENT_TIMESTAMP`
   - Store thumbnail metadata in `ThumbnailStorage` table

---

### 👁️‍🗨️ Cropped Faces Thumbnails

![Thumbnails](Thumbnails.jpeg)

*Cropped Faces Thumbnails that are saved to the `Thumbnails/` directory.*
---
## 📂 Project Structure
```python
services/image_grouping/image_face_detection/
├── config.py # Configuration settings
├── detect_faces.py # Main processing logic
├── main.py # CLI entry point
├── .env # Environment variables
├── Thumbnails/ # Output directory for face crops
├── requirements.txt # Python dependencies
└── README.md # This documentation
```

## 🚀 Setup Instructions

### 🛠️ Prerequisite Tools & Libraries
- **Database**: SQL Server 2019+, SSMS (v21)
- **Python**: 3.8.20
- **Key Libraries**:
  - `retina-face` for face detection
  - `pyodbc` for database connectivity
  - `opencv-python` for image processing
  - `python-dotenv` for configuration
  - `tabulate` for console output formatting

### Installation
1. Create and activate conda environment:
   ```bash
   conda create -n face_detection python=3.10
   conda activate face_detection
   ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Configuration
    - Create .env file with database connection:

        ```ini
        SQL_CONNECTION_STRING="Driver={Driver_Name};Server={Server_name};Database=MetaData;Encrypt=no;TrustServerCertificate=no;"
        THUMBNAIL_SAVE_PATH="Actual Path to store thumbnails"
        ```

4. Ensure database tables exist (see Database Schema section)

## 🚀 Running the Script

### Test Single Image
```bash
python main.py --test <image_path>
```

### Process Database Once
```bash
python main.py --db
```

### Continuous Monitoring
```bash
python main.py --watch
```

(Checks DB every few seconds, pauses if no new data after multiple attempts.)


## 🗄️ How Data is Uploaded to the Database

This pipeline does **not** store the actual image binary in the database — it stores **metadata** about detected faces and the location of their thumbnails on disk.

---

### 1️⃣ Updating `dbo.MediaItems`

When a face is detected, the script marks the media item as processed:

```python
update_query = """
UPDATE dbo.MediaItems 
SET IsFacesExtracted = 1,
    FacesExtractedOn = ?
WHERE Id = ?
"""
cursor.execute(update_query, datetime.now(), media_item_id)
```

**What happens:**

- `IsFacesExtracted` is set to 1 (TRUE) → processed.
- `FacesExtractedOn` is set to the current timestamp.
- Prevents the same image from being processed again.

### 2️⃣ Inserting into `dbo.Faces`

For each cropped face, the script inserts a metadata record:

```python
cursor.execute("""
               INSERT INTO dbo.Faces (MediaItemId, BoundingBox, Name, CreatedAt)
               VALUES (?, ?, ?, ?)
            """, media_item_id, bbox_str, filename, datetime.now())
```

**Stored fields:**

- **Id** → auto generated ID (`TS###`)
- **MediaItemId** → reference to the original media item file
- **Name** → thumbnail file name
- **BoundingBox** → Bounding Box coordinated of face detected
- **CreatedAt** → timestamp

---

### 3️⃣ Database Interaction Flow

**Connect to SQL Server using:**
```python
conn = pyodbc.connect(SQL_CONNECTION_STRING)
cursor = conn.cursor()
```

**Execute `UPDATE` and `INSERT` queries with:**
```python
cursor.execute(query, params...)
```

**Commit changes:**

```python
conn.commit()
```

**Close the connection:**
```python
conn.close()
```
