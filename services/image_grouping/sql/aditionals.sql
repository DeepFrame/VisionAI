USE FaceRecognitionSystem_FRS;
GO

-- 1. Insert into dbo.MediaFile
INSERT INTO dbo.MediaFile (FilePath, FileName, Extensions, CreatedAt, ModifiedAt)
VALUES
('C:/Users/ADMIN/Downloads/DeepFrame-UP/services/image_grouping/image_face_detection/Images/hamidMir.jpg', 'hamidMir.jpg', '.jpg', GETDATE(), GETDATE()),
('C:/Users/ADMIN/Downloads/DeepFrame-UP/services/image_grouping/image_face_detection/Images/imran_khan_img.jpg', 'imran_khan_img.jpg', '.jpg', GETDATE(), GETDATE()),
('C:/Users/ADMIN/Downloads/DeepFrame-UP/services/image_grouping/image_face_detection/Images/Shaista-Lodhi.jpg', 'Shaista-Lodhi.jpg', '.jpg', GETDATE(), GETDATE()),
('C:/Users/ADMIN/Downloads/DeepFrame-UP/services/image_grouping/image_face_detection/Images/shaista.jpg', 'shaista.jpg', '.jpg', GETDATE(), GETDATE()),
('C:/Users/ADMIN/Downloads/DeepFrame-UP/services/image_grouping/image_face_detection/Images/IK.png', 'IK.png', '.png', GETDATE(), GETDATE()),
('C:/Users/ADMIN/Downloads/DeepFrame-UP/services/image_grouping/image_face_detection/Images/lodhi.jpg', 'lodhi.jpg', '.jpg', GETDATE(), GETDATE()),
('C:/Users/ADMIN/Downloads/DeepFrame-UP/services/image_grouping/image_face_detection/Images/SLK.jpg', 'SLK.jpg', '.jpg', GETDATE(), GETDATE()),
('C:/Users/ADMIN/Downloads/DeepFrame-UP/services/image_grouping/image_face_detection/Images/IK_image.png', 'IK_image.png', '.png', GETDATE(), GETDATE()),
('C:/Users/ADMIN/Downloads/DeepFrame-UP/services/image_grouping/image_face_detection/Images/Imran_Khan.jpg', 'Imran_Khan.jpg', '.jpg', GETDATE(), GETDATE()),
('C:/Users/ADMIN/Downloads/DeepFrame-UP/services/image_grouping/image_face_detection/Images/Imran-Khan.jpg', 'Imran-Khan.jpg', '.jpg', GETDATE(), GETDATE()),
('C:/Users/ADMIN/Downloads/DeepFrame-UP/services/image_grouping/image_face_detection/Images/Hamid_MIR.jpg', 'Hamid_MIR.jpg', '.jpg', GETDATE(), GETDATE());

-- 2. Insert into dbo.MediaItems for each new MediaFile row
INSERT INTO dbo.MediaItems (MediaFileId, Name, IsFacesExtracted, FacesExtractedOn)
SELECT Id, FileName, 0, NULL
FROM dbo.MediaFile
WHERE FileName IN ('hamidMir.jpg','imran_khan_img.jpg','Shaista-Lodhi.jpg','shaista.jpg','IK.png','lodhi.jpg','SLK.jpg','IK_image.png','Imran_Khan.jpg','Imran-Khan.jpg','Hamid_MIR.jpg');
