USE FaceRecognitionSystem_FRS;

SELECT 
    F.Id AS FaceId,
    F.PersonId,
    MF.FilePath,
    MF.FileName,
    F.BoundingBox
FROM dbo.Faces F
JOIN dbo.MediaItems MI ON F.MediaItemId = MI.Id
JOIN dbo.MediaFile MF ON MI.MediaFileId = MF.Id
LEFT JOIN dbo.Persons P ON F.PersonId = P.Id
WHERE F.PersonId IS NOT NULL
ORDER BY F.PersonId, F.Id;