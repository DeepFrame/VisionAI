create database MetaData;

USE MetaData;

create table dbo.MediaFile (
    Id INT PRIMARY KEY,
    FilePath NVARCHAR(260) NOT NULL,
    FileName NVARCHAR(100) NOT NULL,
    MediaType NVARCHAR(10) NOT NULL,
    ThumbnailId INT NULL 
);

create table dbo.MediaItems (
    Id INT PRIMARY KEY,
    MediaFileId INT NOT NULL,
    FileName NVARCHAR(100) NOT NULL,
    IsFacesExtracted BIT NOT NULL DEFAULT 0,
    FacesExtractedOn DATETIME NULL,
    FOREIGN KEY (MediaFileId) REFERENCES dbo.MediaFile(Id)
);
