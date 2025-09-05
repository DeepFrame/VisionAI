USE FaceRecognitionSystem_FRS;
GO

-- Insert into dbo.MediaFile with FULL file paths
INSERT INTO dbo.MediaFile (FilePath, FileName, Extensions, CreatedAt, ModifiedAt)
VALUES
('C:/Users/ADMIN/Downloads/DeepFrame-UP/services/image_grouping/image_face_detection/Images/conference.jpg', 'conference.jpg', '.jpg', '2025-08-06 11:59:00', '2025-08-06 11:59:00'),
('C:/Users/ADMIN/Downloads/DeepFrame-UP/services/image_grouping/image_face_detection/Images/conference_hamid.png', 'conference_hamid.png', '.png', '2025-08-15 09:53:00', '2025-08-15 09:53:00'),
('C:/Users/ADMIN/Downloads/DeepFrame-UP/services/image_grouping/image_face_detection/Images/family.jpg', 'family.jpg', '.jpg', '2025-08-11 10:12:00', '2025-08-11 10:12:00'),
('C:/Users/ADMIN/Downloads/DeepFrame-UP/services/image_grouping/image_face_detection/Images/female_anchor.jpg', 'female_anchor.jpg', '.jpg', '2025-08-05 13:15:00', '2025-08-05 13:15:00'),
('C:/Users/ADMIN/Downloads/DeepFrame-UP/services/image_grouping/image_face_detection/Images/GroupM.jpg', 'GroupM.jpg', '.jpg', '2025-08-08 08:42:00', '2025-08-08 08:42:00'),
('C:/Users/ADMIN/Downloads/DeepFrame-UP/services/image_grouping/image_face_detection/Images/Hamid_Mir_at_Studio.png', 'Hamid_Mir_at_Studio.png', '.png', '2025-08-15 09:55:00', '2025-08-15 09:55:00'),
('C:/Users/ADMIN/Downloads/DeepFrame-UP/services/image_grouping/image_face_detection/Images/image-talk-show.jpg', 'image-talk-show.jpg', '.jpg', '2025-08-15 09:54:00', '2025-08-15 09:54:00'),
('C:/Users/ADMIN/Downloads/DeepFrame-UP/services/image_grouping/image_face_detection/Images/image.jpg', 'image.jpg', '.jpg', '2025-08-15 09:53:00', '2025-08-15 09:53:00'),
('C:/Users/ADMIN/Downloads/DeepFrame-UP/services/image_grouping/image_face_detection/Images/image_1.jpg', 'image_1.jpg', '.jpg', '2025-08-13 09:21:00', '2025-08-13 09:21:00'),
('C:/Users/ADMIN/Downloads/DeepFrame-UP/services/image_grouping/image_face_detection/Images/interview.jpg', 'interview.jpg', '.jpg', '2025-08-06 11:59:00', '2025-08-06 11:59:00'),
('C:/Users/ADMIN/Downloads/DeepFrame-UP/services/image_grouping/image_face_detection/Images/interview_anchor.jpg', 'interview_anchor.jpg', '.jpg', '2025-08-15 09:54:00', '2025-08-15 09:54:00'),
('C:/Users/ADMIN/Downloads/DeepFrame-UP/services/image_grouping/image_face_detection/Images/Jenny.jpg', 'Jenny.jpg', '.jpg', '2025-08-11 09:33:00', '2025-08-11 09:33:00'),
('C:/Users/ADMIN/Downloads/DeepFrame-UP/services/image_grouping/image_face_detection/Images/kid.jpg', 'kid.jpg', '.jpg', '2025-08-12 12:00:00', '2025-08-12 12:00:00'),
('C:/Users/ADMIN/Downloads/DeepFrame-UP/services/image_grouping/image_face_detection/Images/lunch.jpg', 'lunch.jpg', '.jpg', '2025-08-12 11:55:00', '2025-08-12 11:55:00'),
('C:/Users/ADMIN/Downloads/DeepFrame-UP/services/image_grouping/image_face_detection/Images/marriage.jpg', 'marriage.jpg', '.jpg', '2025-08-11 09:12:00', '2025-08-11 09:12:00'),
('C:/Users/ADMIN/Downloads/DeepFrame-UP/services/image_grouping/image_face_detection/Images/memory.jpg', 'memory.jpg', '.jpg', '2025-08-12 12:00:00', '2025-08-12 12:00:00'),
('C:/Users/ADMIN/Downloads/DeepFrame-UP/services/image_grouping/image_face_detection/Images/news.jpg', 'news.jpg', '.jpg', '2025-08-06 11:59:00', '2025-08-06 11:59:00'),
('C:/Users/ADMIN/Downloads/DeepFrame-UP/services/image_grouping/image_face_detection/Images/political.jpg', 'political.jpg', '.jpg', '2025-08-07 14:30:00', '2025-08-07 14:30:00'),
('C:/Users/ADMIN/Downloads/DeepFrame-UP/services/image_grouping/image_face_detection/Images/political_leader.png', 'political_leader.png', '.png', '2025-08-15 11:32:00', '2025-08-15 11:32:00'),
('C:/Users/ADMIN/Downloads/DeepFrame-UP/services/image_grouping/image_face_detection/Images/protest.jpg', 'protest.jpg', '.jpg', '2025-08-13 09:20:00', '2025-08-13 09:20:00'),
('C:/Users/ADMIN/Downloads/DeepFrame-UP/services/image_grouping/image_face_detection/Images/radio_pakisatn.jpg', 'radio_pakisatn.jpg', '.jpg', '2025-08-07 15:08:00', '2025-08-07 15:08:00'),
('C:/Users/ADMIN/Downloads/DeepFrame-UP/services/image_grouping/image_face_detection/Images/random.jpg', 'random.jpg', '.jpg', '2025-08-15 09:54:00', '2025-08-15 09:54:00'),
('C:/Users/ADMIN/Downloads/DeepFrame-UP/services/image_grouping/image_face_detection/Images/smiles.jpg', 'smiles.jpg', '.jpg', '2025-08-11 09:55:00', '2025-08-11 09:55:00'),
('C:/Users/ADMIN/Downloads/DeepFrame-UP/services/image_grouping/image_face_detection/Images/strike.jpg', 'strike.jpg', '.jpg', '2025-08-13 09:20:00', '2025-08-13 09:20:00'),
('C:/Users/ADMIN/Downloads/DeepFrame-UP/services/image_grouping/image_face_detection/Images/talk.jpg', 'talk.jpg', '.jpg', '2025-08-13 09:20:00', '2025-08-13 09:20:00'),
('C:/Users/ADMIN/Downloads/DeepFrame-UP/services/image_grouping/image_face_detection/Images/trip.jpg', 'trip.jpg', '.jpg', '2025-08-12 10:33:00', '2025-08-12 10:33:00'),
('C:/Users/ADMIN/Downloads/DeepFrame-UP/services/image_grouping/image_face_detection/Images/wedding.jpg', 'wedding.jpg', '.jpg', '2025-08-12 10:25:00', '2025-08-12 10:25:00')
;

-- Now insert into MediaItems only for these new rows
INSERT INTO dbo.MediaItems (MediaFileId, Name, IsFacesExtracted, FacesExtractedOn)
SELECT Id, FileName, 0, NULL
FROM dbo.MediaFile
WHERE FileName IN (
    'conference.jpg','conference_hamid.png','family.jpg','female_anchor.jpg','GroupM.jpg',
    'Hamid_Mir_at_Studio.png','image-talk-show.jpg','image.jpg','image_1.jpg','interview.jpg',
    'interview_anchor.jpg','Jenny.jpg','kid.jpg','lunch.jpg','marriage.jpg','memory.jpg',
    'news.jpg','political.jpg','political_leader.png','protest.jpg','radio_pakisatn.jpg',
    'random.jpg','smiles.jpg','strike.jpg','talk.jpg','trip.jpg','wedding.jpg'
);
