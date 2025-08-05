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

## 🛠️ Prerequisites

Before running the script, ensure you have:

| Component           | Required Version |
|---------------------|------------------|
| SQL Server          | 2022    |
| SSMS (Management Studio) | (v21)       |
| Image Folder        | Local folder containing test images (see below) |

Example path used for this setup:
`C:/Users/ADMIN/Downloads/FRS_ml/sample_images/`
