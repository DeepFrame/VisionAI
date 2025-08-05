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
