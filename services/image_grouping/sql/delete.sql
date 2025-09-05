USE FaceRecognitionSystem_FRS;
GO

-- Drop foreign key constraints
ALTER TABLE dbo.Faces DROP CONSTRAINT FK_Faces_MediaItem;
ALTER TABLE dbo.Faces DROP CONSTRAINT FK_Faces_Person;
ALTER TABLE dbo.Persons DROP CONSTRAINT FK_Persons_Media;
ALTER TABLE dbo.MediaItems DROP CONSTRAINT FK_MediaItems_MediaFile;
GO

-- Drop tables in order
DROP TABLE IF EXISTS dbo.Faces;
DROP TABLE IF EXISTS dbo.Persons;
DROP TABLE IF EXISTS dbo.MediaItems;
DROP TABLE IF EXISTS dbo.MediaFile;
GO
