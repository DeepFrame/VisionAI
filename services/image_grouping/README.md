# 🧠 Database Preparation & Sample Media Setup

## 📌 Objective

Set up the SQL Server database with initial media file entries to be used later for face detection and grouping. Data was inserted manually using SQL Server Management Studio (SSMS) 21.

---

## 🗄️ Database: `MetaData`

This task uses a SQL Server database named `MetaData`.

---

## 📁 Tables

### 1. `dbo.MediaFile`

Stores metadata about media files (images/videos).

| Column       | Type           | Description                                      |
|--------------|----------------|--------------------------------------------------|
| Id           | INT (PK)       | Unique ID for each media file                   |
| FilePath     | NVARCHAR(260)  | Full file path to the image/video               |
| FileName     | NVARCHAR(100)  | Name of the media file (e.g., `news.jpg`)       |
| MediaType    | NVARCHAR(10)   | Type of media — typically `'image'` or `'video'`|
| ThumbnailId  | INT (nullable) | Placeholder for a generated thumbnail reference |

---

### 2. `dbo.MediaItems`

Tracks the processing (face extraction) status of each media file.

| Column            | Type           | Description                                                                 |
|-------------------|----------------|-----------------------------------------------------------------------------|
| Id                | INT (PK)       | Unique ID for the media item record                                        |
| MediaFileId       | INT (FK)       | Foreign key reference to `MediaFile(Id)`                                   |
| FileName          | NVARCHAR(100)  | Redundant name for lookup ease                                             |
| IsFacesExtracted  | BIT            | `0` = not processed, `1` = processed                                        |
| FacesExtractedOn  | DATETIME       | Date/time when face extraction was completed (NULL if not yet processed)   |

---

## 🔗 Relationship

- `MediaItems.MediaFileId` → `MediaFile.Id`
- A foreign key constraint ensures referential integrity.

---

## 📥 Sample Data Inserted

Three unprocessed image entries were manually inserted via SSMS:

| File Name         | Path                                                                 |
|-------------------|----------------------------------------------------------------------|
| `news.jpg`        | `C:/Users/ADMIN/Downloads/FRS_ml/sample_images/news.jpg`            |
| `conference.jpg`  | `C:/Users/ADMIN/Downloads/FRS_ml/sample_images/conference.jpg`      |
| `interview.jpg`   | `C:/Users/ADMIN/Downloads/FRS_ml/sample_images/interview.jpg`       |

These are referenced in both `MediaFile` and `MediaItems` with `IsFacesExtracted = 0`.

---

## 🧾 Manual Steps for Inserting Data Using SSMS

Follow these steps to manually create the database, tables, and insert sample data using **SQL Server Management Studio (SSMS 2022)**.

---

### 🔧 Step 1: Create the Database

```sql
CREATE DATABASE MetaData;
```

---

### 🧱 Step 2: Create the Tables

```sql
USE MetaData;

-- Create MediaFile table
create table dbo.MediaFile (
    Id INT PRIMARY KEY,
    FilePath NVARCHAR(260) NOT NULL,
    FileName NVARCHAR(100) NOT NULL,
    MediaType NVARCHAR(10) NOT NULL,
    ThumbnailId INT NULL 
);

-- Create MediaItems table
create table dbo.MediaItems (
    Id INT PRIMARY KEY,
    MediaFileId INT NOT NULL,
    FileName NVARCHAR(100) NOT NULL,
    IsFacesExtracted BIT NOT NULL DEFAULT 0,
    FacesExtractedOn DATETIME NULL,
    FOREIGN KEY (MediaFileId) REFERENCES dbo.MediaFile(Id)
);
```

---

### 📥 Step 3: Insert Sample Data

```sql
use MetaData;

INSERT INTO dbo.MediaFile (Id, FilePath, FileName, MediaType, ThumbnailId)
VALUES 
(1, 'C:/Users/ADMIN/Downloads/FRS_ml/sample_images/news.jpg', 'news.jpg', 'image', NULL),
(2, 'C:/Users/ADMIN/Downloads/FRS_ml/sample_images/conference.jpg', 'conference.jpg', 'image', NULL),
(3, 'C:/Users/ADMIN/Downloads/FRS_ml/sample_images/interview.jpg', 'interview.jpg', 'image', NULL);

INSERT INTO dbo.MediaItems (Id, MediaFileId, FileName, IsFacesExtracted, FacesExtractedOn)
VALUES 
(1, 1, 'news.jpg', 0, NULL),
(2, 2, 'conference.jpg', 0, NULL),
(3, 3, 'interview.jpg', 0, NULL);
```

---

### 🔎 Step 4: Verify Data
```sql
SELECT * FROM dbo.MediaFile;
```

```sql
SELECT * FROM dbo.MediaItems;
```
---
## 🛠️ Prerequisites

Before running the script, ensure you have:

| Component           | Required Version |
|---------------------|------------------|
| SQL Server          | 2022    |
| SSMS (Management Studio) | (v21)       |
| Image Folder        | Local folder containing test images (see below) |

Example path used for this setup:
`C:/Users/ADMIN/Downloads/FRS_ml/sample_images/`
