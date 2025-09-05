--CREATE DATABASE FaceRecognitionSystem_FRS;
--GO

USE FaceRecognitionSystem_FRS;
GO

-- dbo.MediaFile
CREATE TABLE dbo.MediaFile (
    Id INT IDENTITY(1,1) PRIMARY KEY,
    FilePath NVARCHAR(500) NOT NULL,
    FileName NVARCHAR(255) NOT NULL,
    Extensions NVARCHAR(10) NULL,
    CreatedAt DATETIME2 NOT NULL DEFAULT SYSDATETIME(),
    ModifiedAt DATETIME2 NULL
);
GO

-- dbo.MediaItems
CREATE TABLE dbo.MediaItems (
    Id INT IDENTITY(1,1) PRIMARY KEY,
    MediaFileId INT NOT NULL,
    Name NVARCHAR(255) NOT NULL,
    IsFacesExtracted BIT NOT NULL DEFAULT 0,
    FacesExtractedOn DATETIME2 NULL,
    CONSTRAINT FK_MediaItems_MediaFile FOREIGN KEY (MediaFileId) REFERENCES dbo.MediaFile(Id)
);
GO

-- dbo.Persons
CREATE TABLE dbo.Persons (
    Id INT IDENTITY(1,1) PRIMARY KEY,
    PortraitMediaFileId INT NULL,
    Name NVARCHAR(255),
    Rank NVARCHAR(50) NULL,
    Appointment NVARCHAR(100) NULL,
    CreatedAt DATETIME2 NOT NULL DEFAULT SYSDATETIME(),
    ModifiedAt DATETIME2 NULL
);
GO

-- dbo.Faces
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
GO

ALTER TABLE dbo.Persons
ADD CONSTRAINT FK_Persons_Media FOREIGN KEY (PortraitMediaFileId) REFERENCES dbo.MediaFile(Id);
GO
