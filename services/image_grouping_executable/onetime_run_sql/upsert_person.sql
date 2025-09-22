USE FaceRecognitionSystem;
GO

CREATE OR ALTER PROCEDURE dbo.UpsertPerson
    @PersonId INT = NULL,   -- If NULL, insert new
    @Name NVARCHAR(255),
    @Rank NVARCHAR(50) = NULL,
    @Appointment NVARCHAR(100) = NULL
AS
BEGIN
    SET NOCOUNT ON;

    IF @PersonId IS NULL
    BEGIN
        -- Insert new person
        INSERT INTO dbo.Persons (Name, Rank, Appointment, CreatedAt)
        VALUES (@Name, @Rank, @Appointment, SYSDATETIME());

        SET @PersonId = SCOPE_IDENTITY();
    END
    ELSE
    BEGIN
        -- Update existing, but only overwrite if NOT NULL
        UPDATE dbo.Persons
        SET Name        = COALESCE(@Name, Name),
            Rank        = COALESCE(@Rank, Rank),
            Appointment = COALESCE(@Appointment, Appointment),
            ModifiedAt  = SYSDATETIME()
        WHERE Id = @PersonId;
    END

    SELECT @PersonId AS PersonId;
END
GO
