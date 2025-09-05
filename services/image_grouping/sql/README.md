# 🧠 Database Preparation & Sample Media Setup

## 📌 Objective

Set up the SQL Server database with initial media file entries to be used later for face detection, recognition, and grouping. Data was inserted manually using **SQL Server Management Studio (SSMS 2022)**.

---

## 🗄️ Database: `FaceRecognitionSystem`

This task uses a SQL Server database named `FaceRecognitionSystem`.

---

## 📁 Tables

### 1. `dbo.MediaFile`

Stores metadata about media files.

| Column      | Type           | Description                                      |
|-------------|----------------|--------------------------------------------------|
| Id          | INT (PK)       | Unique identifier for each media file            |
| FilePath    | NVARCHAR(500)  | Full file path to the image/video                |
| FileName    | NVARCHAR(255)  | Name of the media file                           |
| Extensions  | NVARCHAR(10)   | File extension (e.g., `.jpg`, `.mp4`)            |
| CreatedAt   | DATETIME2      | Timestamp when record was created                |
| ModifiedAt  | DATETIME2      | Timestamp when record was last modified          |

---

### 2. `dbo.MediaItems`

Tracks processing status of media files.

| Column           | Type           | Description                                                               |
|------------------|----------------|---------------------------------------------------------------------------|
| Id               | INT (PK)       | Unique identifier for the media item                                      |
| MediaFileId      | INT (FK)       | References `MediaFile(Id)`                                                |
| Name             | NVARCHAR(255)  | Media item name                                                           |
| IsFacesExtracted | BIT            | `0` = not processed, `1` = processed                                      |
| FacesExtractedOn | DATETIME2      | Timestamp when face extraction was completed (NULL if not processed)      |

---

### 3. `dbo.Persons`

Stores details of recognized individuals.

| Column             | Type           | Description                                      |
|--------------------|----------------|--------------------------------------------------|
| Id                 | INT (PK)       | Unique identifier for each person                |
| PortraitMediaFileId| INT (FK)       | References `MediaFile(Id)` for portrait image    |
| Name               | NVARCHAR(255)  | Person’s name                                    |
| Rank               | NVARCHAR(50)   | Rank/position                                    |
| Appointment        | NVARCHAR(100)  | Appointment/role                                 |
| CreatedAt          | DATETIME2      | Timestamp when record was created                |
| ModifiedAt         | DATETIME2      | Timestamp when record was last modified          |

---

### 4. `dbo.Faces`

Stores information about detected faces.

| Column       | Type            | Description                                      |
|--------------|-----------------|--------------------------------------------------|
| Id           | INT (PK)        | Unique identifier for each detected face          |
| MediaItemId  | INT (FK)        | References `MediaItems(Id)`                      |
| PersonId     | INT (FK)        | References `Persons(Id)`                         |
| BoundingBox  | NVARCHAR(255)   | Coordinates of detected face bounding box         |
| Embedding    | VARBINARY(MAX)  | Numerical representation (feature vector) of face |
| FrameNumber  | INT             | Frame number (for video files)                   |
| Name         | NVARCHAR(255)   | Optional label/name for detected face             |
| CreatedAt    | DATETIME2       | Timestamp when record was created                 |
| ModifiedAt   | DATETIME2       | Timestamp when record was last modified           |

---

## 🔗 Relationships

- `MediaItems.MediaFileId` → `MediaFile.Id`
- `Persons.PortraitMediaFileId` → `MediaFile.Id`
- `Faces.MediaItemId` → `MediaItems.Id`
- `Faces.PersonId` → `Persons.Id`

A set of foreign key constraints ensures referential integrity.

---

## 📥 Sample Data Inserted

Three sample images were manually inserted via SSMS:

| File Name         | Path                                                                 |
|-------------------|----------------------------------------------------------------------|
| `news.jpg`        | `C:/Users/ADMIN/Downloads/deepframe-backend/services/image_grouping/sample_images/news.jpg`            |
| `conference.jpg`  | `C:/Users/ADMIN/Downloads/deepframe-backend/services/image_grouping/sample_images/conference.jpg`      |
| `interview.jpg`   | `C:/Users/ADMIN/Downloads/deepframe-backend/services/image_grouping/sample_images/interview.jpg`       |

These are referenced in both `MediaFile` and `MediaItems` with `IsFacesExtracted = 0/1`.

---

## 🧾 Manual Steps for Inserting Data Using SSMS

Follow these steps to manually create the database, tables, and insert sample data.

---

### 🔧 Step 1: Create the Database

```sql
CREATE DATABASE FaceRecognitionSystem;
```

---

### 🧱 Step 2: Create the Tables

```sql
USE FaceRecognitionSystem;

-- MediaFile
CREATE TABLE dbo.MediaFile (
    Id INT IDENTITY(1,1) PRIMARY KEY,
    FilePath NVARCHAR(500) NOT NULL,
    FileName NVARCHAR(255) NOT NULL,
    Extensions NVARCHAR(10) NULL,
    CreatedAt DATETIME2 NOT NULL DEFAULT SYSDATETIME(),
    ModifiedAt DATETIME2 NULL
);

-- MediaItems
CREATE TABLE dbo.MediaItems (
    Id INT IDENTITY(1,1) PRIMARY KEY,
    MediaFileId INT NOT NULL,
    Name NVARCHAR(255) NOT NULL,
    IsFacesExtracted BIT NOT NULL DEFAULT 0,
    FacesExtractedOn DATETIME2 NULL,
    CONSTRAINT FK_MediaItems_MediaFile FOREIGN KEY (MediaFileId) REFERENCES dbo.MediaFile(Id)
);

-- Persons
CREATE TABLE dbo.Persons (
    Id INT IDENTITY(1,1) PRIMARY KEY,
    PortraitMediaFileId INT NULL,
    Name NVARCHAR(255),
    Rank NVARCHAR(50) NULL,
    Appointment NVARCHAR(100) NULL,
    CreatedAt DATETIME2 NOT NULL DEFAULT SYSDATETIME(),
    ModifiedAt DATETIME2 NULL,
    CONSTRAINT FK_Persons_Media FOREIGN KEY (PortraitMediaFileId) REFERENCES dbo.MediaFile(Id)
);

-- Faces
CREATE TABLE dbo.Faces (
    Id INT IDENTITY(1,1) PRIMARY KEY, 
    MediaItemId INT NOT NULL,
    PersonId INT NULL,
    BoundingBox NVARCHAR(255) NULL,
    Embedding VARBINARY(MAX) NULL, 
    FrameNumber INT NULL,
    Name NVARCHAR(255),
    CreatedAt DATETIME2 NOT NULL DEFAULT SYSDATETIME(),
    ModifiedAt DATETIME2 NULL,
    CONSTRAINT FK_Faces_MediaItem FOREIGN KEY (MediaItemId) REFERENCES dbo.MediaItems(Id),
    CONSTRAINT FK_Faces_Person FOREIGN KEY (PersonId) REFERENCES dbo.Persons(Id)
);
```

---

### 📥 Step 3: Insert Sample Data

```sql
USE FaceRecognitionSystem;

-- Insert media files
INSERT INTO dbo.MediaFile (FilePath, FileName, Extensions, CreatedAt, ModifiedAt)
VALUES 
('C:/Users/ADMIN/Downloads/deepframe-backend/services/image_grouping/sample_images/conference.jpg', 'conference.jpg', '.jpg', GETDATE(), GETDATE()),
('C:/Users/ADMIN/Downloads/deepframe-backend/services/image_grouping/sample_images/interview.jpg', 'interview.jpg', '.jpg', GETDATE(), GETDATE()),
('C:/Users/ADMIN/Downloads/deepframe-backend/services/image_grouping/sample_images/news.jpg', 'news.jpg', '.jpg', GETDATE(), GETDATE());

-- Insert media items
INSERT INTO dbo.MediaItems (MediaFileId, Name, IsFacesExtracted, FacesExtractedOn)
SELECT Id, FileName, 0, NULL
FROM dbo.MediaFile
WHERE FileName IN ('conference.jpg','interview.jpg','news.jpg');
```

---

### 🔎 Step 4: Verify Data

```sql
SELECT * FROM dbo.MediaFile;
```

```sql
SELECT * FROM dbo.MediaItems;
```

```sql
SELECT * FROM dbo.Persons;
```

```sql
SELECT * FROM dbo.Faces;
```

---

## 🛠️ Prerequisites

Before running the script, ensure you have:

| Component                | Required Version                    |
| ------------------------ | ----------------------------------- |
| SQL Server               | 2022                                |
| SSMS (Management Studio) | 21                                  |
| Image Folder             | Local folder containing test images |

Example path used for this setup:

> `C:/Users/ADMIN/Downloads/deepframe-backend/services/image_grouping/sample_images/`

```

---
