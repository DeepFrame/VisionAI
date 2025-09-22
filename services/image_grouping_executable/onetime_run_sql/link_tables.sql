USE FaceRecognitionSystem;
GO

CREATE OR ALTER PROCEDURE dbo.UpsertFace
    @FaceId INT = NULL,         
    @MediaItemId INT = NULL,
    @PersonId INT = NULL,   
    @BoundingBox NVARCHAR(255) = NULL,
    @Name NVARCHAR(255) = NULL,
    @Embedding VARBINARY(MAX) = NULL
AS
BEGIN
    SET NOCOUNT ON;

    DECLARE @Now DATETIME2 = SYSDATETIME();

    IF @FaceId IS NULL
    BEGIN
        INSERT INTO dbo.Faces (MediaItemId, PersonId, BoundingBox, Name, Embedding, CreatedAt, ModifiedAt)
        VALUES (@MediaItemId, @PersonId, @BoundingBox, @Name, @Embedding, @Now, @Now);

        SELECT SCOPE_IDENTITY() AS FaceId;
    END
    ELSE
    BEGIN
        UPDATE dbo.Faces
        SET 
            PersonId = ISNULL(@PersonId, PersonId), 
            BoundingBox = ISNULL(@BoundingBox, BoundingBox),
            Name = ISNULL(@Name, Name),
            Embedding = ISNULL(@Embedding, Embedding),
            ModifiedAt = @Now
        WHERE Id = @FaceId;

        SELECT @FaceId AS FaceId;
    END
END;
GO
