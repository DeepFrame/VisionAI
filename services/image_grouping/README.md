# 🧠 Database Preparation & Sample Media Setup

## 📌 Objective

Set up the SQL Server database with initial media file entries to be used later for face detection and grouping. Data was inserted manually using SQL Server Management Studio (SSMS) 21.

---

## 🗄️ Database: `MetaData`

This task uses a SQL Server database named `MetaData`.

---

## 📁 Tables

### 1. `dbo.MediaFile`

Stores metadata about media files.

| Column      | Type           | Description                                      |
|-------------|----------------|--------------------------------------------------|
| Id          | NVARCHAR(10)   | Unique ID for each media file (e.g., `'MF001'`) |
| FilePath    | NVARCHAR(255)  | Full file path to the image/video               |
| FileName    | NVARCHAR(100)  | Name of the media file                          |
| MediaType   | VARCHAR(10)    | Either `'image'` or `'video'`                   |

---

### 2. `dbo.ThumbnailStorage`

Stores thumbnails generated for media files.

| Column        | Type           | Description                                      |
|---------------|----------------|--------------------------------------------------|
| Id            | NVARCHAR(10)   | Unique ID for each thumbnail (e.g., `'TS001'`)   |
| MediaFileId   | NVARCHAR(10)   | FK to `MediaFile(Id)`                            |
| FileName      | NVARCHAR(100)  | Thumbnail filename                               |
| ThumbnailPath | NVARCHAR(255)  | Full path to the thumbnail file                  |
| CreatedOn     | DATETIME       | Timestamp when the thumbnail was created         |

---

### 3. `dbo.MediaItems`

Tracks face processing status of each media file.

| Column            | Type           | Description                                                               |
|-------------------|----------------|---------------------------------------------------------------------------|
| Id                | NVARCHAR(10)   | Unique ID for the media item (e.g., `'MI001'`)                            |
| MediaFileId       | NVARCHAR(10)   | FK to `MediaFile(Id)`                                                     |
| FileName          | NVARCHAR(100)  | Redundant name for ease of access                                         |
| IsFacesExtracted  | BIT            | `0` = not processed, `1` = processed                                      |
| FacesExtractedOn  | DATETIME       | Timestamp when face extraction was completed (NULL if not processed)      |

---

## 🔗 Relationship

- `MediaItems.MediaFileId` → `MediaFile.Id`
- `ThumbnailStorage.MediaFileId` → `MediaFile.Id`
- A foreign key constraint ensures referential integrity.

---

## 📥 Sample Data Inserted

Three image entries were manually inserted via SSMS:

| File Name         | Path                                                                 |
|-------------------|----------------------------------------------------------------------|
| `news.jpg`        | `C:/Users/ADMIN/Downloads/FRS_ml/sample_images/news.jpg`            |
| `conference.jpg`  | `C:/Users/ADMIN/Downloads/FRS_ml/sample_images/conference.jpg`      |
| `interview.jpg`   | `C:/Users/ADMIN/Downloads/FRS_ml/sample_images/interview.jpg`       |

These are referenced in both `MediaFile` and `MediaItems` with `IsFacesExtracted = 0/1`.

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

-- 1. MediaFile Table
CREATE TABLE dbo.MediaFile (
    Id NVARCHAR(10) PRIMARY KEY, -- e.g., 'MF001'
    FilePath NVARCHAR(255) NOT NULL,
    FileName NVARCHAR(100) NOT NULL,
    MediaType VARCHAR(10) CHECK (MediaType IN ('image', 'video')) NOT NULL
);

-- 2. ThumbnailStorage Table
CREATE TABLE dbo.ThumbnailStorage (
    Id NVARCHAR(10) PRIMARY KEY, -- e.g., 'TS001'
    MediaFileId NVARCHAR(10) FOREIGN KEY REFERENCES dbo.MediaFile(Id),
    FileName NVARCHAR(100) NOT NULL,
    ThumbnailPath NVARCHAR(255) NOT NULL,
    CreatedOn DATETIME NOT NULL DEFAULT GETDATE()
);

-- 3. MediaItems Table
CREATE TABLE dbo.MediaItems (
    Id NVARCHAR(10) PRIMARY KEY, -- e.g., 'MI001'
    MediaFileId NVARCHAR(10) FOREIGN KEY REFERENCES dbo.MediaFile(Id),
    FileName NVARCHAR(100) NOT NULL,
    IsFacesExtracted BIT NOT NULL DEFAULT 0,
    FacesExtractedOn DATETIME NULL
);
```

---

### 📥 Step 3: Insert Sample Data

```sql
use MetaData;

INSERT INTO dbo.MediaFile (Id, FilePath, FileName, MediaType)
VALUES 
('MF001', 'C:/Users/ADMIN/Downloads/FRS_ml/sample_images/conference.jpg', 'conference.jpg', 'image'),
('MF002', 'C:/Users/ADMIN/Downloads/FRS_ml/sample_images/interview.jpg', 'interview.jpg', 'image'),
('MF003', 'C:/Users/ADMIN/Downloads/FRS_ml/sample_images/news.jpg', 'news.jpg', 'image');

-- Insert media items 
INSERT INTO dbo.MediaItems (Id, MediaFileId, FileName, IsFacesExtracted, FacesExtractedOn)
VALUES 
('MI001', 'MF001', 'conference.jpg', 1, GETDATE()),
('MI002', 'MF002', 'interview.jpg', 0, NULL),
('MI003', 'MF003', 'news.jpg', 1, GETDATE());

-- Thumbnails for conference.jpg
INSERT INTO dbo.ThumbnailStorage (Id, MediaFileId, FileName, ThumbnailPath)
VALUES
('TS001', 'MF001', 'conference_TN1.jpg', 'C:/Users/ADMIN/Downloads/FRS_ml/Thumbnails/conference_TN1.jpg'),
('TS002', 'MF001', 'conference_TN2.jpg', 'C:/Users/ADMIN/Downloads/FRS_ml/Thumbnails/conference_TN2.jpg'),
('TS003', 'MF001', 'conference_TN3.jpg', 'C:/Users/ADMIN/Downloads/FRS_ml/Thumbnails/conference_TN3.jpg'),
('TS004', 'MF001', 'conference_TN4.jpg', 'C:/Users/ADMIN/Downloads/FRS_ml/Thumbnails/conference_TN4.jpg'),
('TS005', 'MF001', 'conference_TN5.jpg', 'C:/Users/ADMIN/Downloads/FRS_ml/Thumbnails/conference_TN5.jpg'),
('TS006', 'MF001', 'conference_TN6.jpg', 'C:/Users/ADMIN/Downloads/FRS_ml/Thumbnails/conference_TN6.jpg'),
('TS007', 'MF001', 'conference_TN7.jpg', 'C:/Users/ADMIN/Downloads/FRS_ml/Thumbnails/conference_TN7.jpg'),
('TS008', 'MF001', 'conference_TN8.jpg', 'C:/Users/ADMIN/Downloads/FRS_ml/Thumbnails/conference_TN8.jpg');

-- Thumbnail for news.jpg
INSERT INTO dbo.ThumbnailStorage (Id, MediaFileId, FileName, ThumbnailPath)
VALUES
('TS009', 'MF003', 'news_TN.jpg', 'C:/Users/ADMIN/Downloads/FRS_ml/Thumbnails/news_TN.jpg');
```

---

### 🔎 Step 4: Verify Data
```sql
USE MetaData;
SELECT * FROM dbo.MediaFile;
```

```sql
USE MetaData;
SELECT * FROM dbo.MediaItems;
```

```sql
USE MetaData;
SELECT * FROM dbo.ThumbnailStorage;
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
> `C:/Users/ADMIN/Downloads/FRS_ml/sample_images/`
> 
> `C:/Users/ADMIN/Downloads/FRS_ml/Thumbnails/`
