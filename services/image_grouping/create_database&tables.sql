CREATE DATABASE MetaData;

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
