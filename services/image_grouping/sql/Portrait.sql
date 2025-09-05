USE FaceRecognitionSystem_FRS;

SELECT f.Id AS FaceId,
       mi.MediaFileId,
       f.PersonId,
       mf.FilePath,
       mf.FileName
FROM dbo.Faces f
JOIN dbo.MediaItems mi ON f.MediaItemId = mi.Id
JOIN dbo.MediaFile mf ON mi.MediaFileId = mf.Id
WHERE mi.MediaFileId=1;
