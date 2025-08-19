USE MetaData;

-- Insert media files
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
