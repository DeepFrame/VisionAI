# SQL Procedures

This document describes the stored procedures used in the **`FaceRecognitionSystem`** database for managing persons and faces in the **image grouping / recognition service**.

---

## 1. `UpsertFace` Procedure

**Location:** `person_recognition/sql/link_tables.sql`

**Purpose:**  
Handles both insertion and update of a record in the `Faces` table. It ensures that a face record is either created if it doesn't exist or updated if it already exists.

**Definition:**

```sql
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
````

**Parameters:**

| Parameter      | Type           | Description                                           |
| -------------- | -------------- | ----------------------------------------------------- |
| `@FaceId`      | INT            | ID of the face to update. If NULL, inserts new.       |
| `@MediaItemId` | INT            | Reference to the media item associated with the face. |
| `@PersonId`    | INT            | Reference to the person associated with the face.     |
| `@BoundingBox` | NVARCHAR(255)  | Face bounding box coordinates.                        |
| `@Name`        | NVARCHAR(255)  | Name of the person (optional).                        |
| `@Embedding`   | VARBINARY(MAX) | Facial embedding vector for recognition.              |

**Behavior:**

* If `@FaceId` is `NULL` → Inserts a new record into `Faces`.
* If `@FaceId` is provided → Updates the existing record, only replacing fields that are not NULL.
* Returns the `FaceId` of the inserted or updated face.

---

## 2. `UpsertPerson` Procedure

**Location:** `person_recognition/sql/upsert_person.sql`

**Purpose:**
Handles insertion and update of a person in the `Persons` table.

**Definition:**

```sql
CREATE OR ALTER PROCEDURE dbo.UpsertPerson
    @PersonId INT = NULL,
    @Name NVARCHAR(255),
    @Rank NVARCHAR(50) = NULL,
    @Appointment NVARCHAR(100) = NULL
AS
BEGIN
    SET NOCOUNT ON;

    IF @PersonId IS NULL
    BEGIN
        INSERT INTO dbo.Persons (Name, Rank, Appointment, CreatedAt)
        VALUES (@Name, @Rank, @Appointment, SYSDATETIME());
        SET @PersonId = SCOPE_IDENTITY();
    END
    ELSE
    BEGIN
        UPDATE dbo.Persons
        SET Name        = COALESCE(@Name, Name),
            Rank        = COALESCE(@Rank, Rank),
            Appointment = COALESCE(@Appointment, Appointment),
            ModifiedAt  = SYSDATETIME()
        WHERE Id = @PersonId;
    END

    SELECT @PersonId AS PersonId;
END;
```

**Parameters:**

| Parameter      | Type          | Description                                       |
| -------------- | ------------- | ------------------------------------------------- |
| `@PersonId`    | INT           | ID of the person to update. If NULL, inserts new. |
| `@Name`        | NVARCHAR(255) | Name of the person.                               |
| `@Rank`        | NVARCHAR(50)  | Rank or designation of the person (optional).     |
| `@Appointment` | NVARCHAR(100) | Appointment or position (optional).               |

**Behavior:**

* If `@PersonId` is `NULL` → Inserts a new person record.
* If `@PersonId` is provided → Updates the existing record, only overwriting non-NULL values.
* Returns the `PersonId` of the inserted or updated person.

---

## Notes

1. Both procedures implement **UPSERT behavior** (Insert if missing, Update if exists).
2. `SYSDATETIME()` is used to track `CreatedAt` and `ModifiedAt`.
3. `COALESCE` or `ISNULL` functions ensure that NULL values do not overwrite existing data.


Do you want me to add that?
```
