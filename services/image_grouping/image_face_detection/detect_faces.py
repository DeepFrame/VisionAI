import os
import time
import pyodbc
import cv2
import numpy as np
from datetime import datetime
from retinaface import RetinaFace
from tabulate import tabulate
from config import SQL_CONNECTION_STRING, THUMBNAIL_SAVE_PATH

import sys
import time

conn_str = SQL_CONNECTION_STRING
thumbnail_base_path = THUMBNAIL_SAVE_PATH

# Ensure the thumbnail directory exists
os.makedirs(thumbnail_base_path, exist_ok=True)

# DATABASE Unprocessed files and Thumbnails Query Processing
def get_unprocessed_files():
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        query = """
        SELECT MI.Id, MF.FilePath, MI.FileName
        FROM dbo.MediaItems MI
        JOIN dbo.MediaFile MF ON MI.MediaFileId = MF.Id
        WHERE MI.IsFacesExtracted = 0
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        return rows
    
    except Exception as e:
        print(f"[ERROR] Database fetch failed: {e}")
        return []

def get_next_thumbnail_id(cursor):
    cursor.execute("""
        SELECT TOP 1 Id 
        FROM dbo.ThumbnailStorage 
        WHERE Id LIKE 'TS%' 
        ORDER BY Id DESC
    """)
    row = cursor.fetchone()
    if row:
        last_id = row[0]  
        last_num = int(last_id[2:]) 
        next_num = last_num + 1
    else:
        next_num = 1

    return f"TS{next_num:03d}" 

# DATABASE Update Query Processing
def update_database(media_item_id, thumbnail_filename, thumbnail_path):
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        update_query = """
        UPDATE dbo.MediaItems 
        SET IsFacesExtracted = 1,
            FacesExtractedOn = ?
        WHERE Id = ?
        """
        cursor.execute(update_query, datetime.now(), media_item_id)

        cursor.execute("SELECT MediaFileId FROM dbo.MediaItems WHERE Id = ?", media_item_id)
        result = cursor.fetchone()
        if not result:
            raise Exception("MediaFileId not found")

        media_file_id = result[0]
        #thumbnail_id = f"TS{int(time.time()) % 100000}"
        thumbnail_id = get_next_thumbnail_id(cursor)

        insert_query = """
        INSERT INTO dbo.ThumbnailStorage 
        (Id, MediaFileId, FileName, ThumbnailPath, CreatedOn)
        VALUES (?, ?, ?, ?, ?)
        """
        cursor.execute(insert_query,
                       thumbnail_id,
                       media_file_id,
                       thumbnail_filename,
                       thumbnail_path,
                       datetime.now())

        conn.commit()
        cursor.close()
        conn.close()

        print(f"[INFO] DB updated for MediaItemId {media_item_id}")
    except Exception as e:
        print(f"[ERROR] Database update failed: {e}")

# Countdown Timer for Continuous Processing
def countdown_timer(seconds, message="Waiting"):
    for remaining in range(seconds, 0, -1):
        mins, secs = divmod(remaining, 60)
        sys.stdout.write(f"\r{message}: {mins:02d}:{secs:02d} remaining ")
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write("\r" + " " * 50 + "\r")

# CROP & ALIGN with Surroundings
def align_face(image, landmarks, margin=0.1):
    eye1 = landmarks['right_eye']
    eye2 = landmarks['left_eye']

    if eye1[0] < eye2[0]:
        left_eye = eye1
        right_eye = eye2
    else:
        left_eye = eye2
        right_eye = eye1

    eyes_center = ((left_eye[0] + right_eye[0]) / 2,
                   (left_eye[1] + right_eye[1]) / 2)

    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    desired_width = 256
    desired_height = 256

    margin_factor = 1 + margin
    desired_width_margin = int(desired_width * margin_factor)
    desired_height_margin = int(desired_height * margin_factor)

    desired_left_eye = (0.35, 0.4)

    dist = np.sqrt(dx ** 2 + dy ** 2)
    desired_dist = (1.0 - 2 * desired_left_eye[0]) * desired_width
    scale = desired_dist / dist

    scale /= margin_factor

    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

    tX = desired_width_margin * 0.5
    tY = desired_height_margin * desired_left_eye[1]
    M[0, 2] += (tX - eyes_center[0])
    M[1, 2] += (tY - eyes_center[1])

    aligned_face = cv2.warpAffine(image, M, (desired_width_margin, desired_height_margin),
                                  flags=cv2.INTER_CUBIC)

    return aligned_face

# FACE DETECTION (RetinaFace)
def detect_and_crop_faces(image_path, media_item_id=None):
    try:
        faces = RetinaFace.detect_faces(image_path)
        img = cv2.imread(image_path)

        if not isinstance(faces, dict) or len(faces) == 0:
            print(f"[INFO] No faces detected in {image_path}")
            return None

        base_name = os.path.basename(image_path)
        name_without_ext, ext = os.path.splitext(base_name)
        ext = ext.lower() if ext else ".jpg"

        for idx, (key, face_data) in enumerate(faces.items()):
            landmarks = face_data["landmarks"]
            
            aligned = align_face(img, landmarks, margin=0.1)

            aligned_resized = cv2.resize(aligned, (112, 112), interpolation=cv2.INTER_AREA)

            if aligned_resized.shape[0] < 112 or aligned_resized.shape[1] < 112:
                print(f"[WARN] Face {idx+1} in {image_path} is too small after alignment. Skipping.")
                continue

            filename = f"{name_without_ext}_TN{idx + 1}{ext}"
            save_path = os.path.join(thumbnail_base_path, filename)
            cv2.imwrite(save_path, aligned_resized)
            print(f"[INFO] Saved aligned face {idx+1} to {save_path}")

            if media_item_id is not None:
                update_database(media_item_id, filename, save_path)

        return True

    except Exception as e:
        print(f"[ERROR] Face processing failed for {image_path}: {e}")
        return False

# MAIN BATCH PROCESSOR
def batch_process_from_db():
    print("[INFO] Starting batch face detection from DB...")

    rows = get_unprocessed_files()
    if not rows:
        print("[INFO] No unprocessed files found.")
        return

    print(tabulate(rows, headers=["MediaItemId", "FilePath", "FileName"], tablefmt="grid"))

    for row in rows:
        media_item_id, file_path, file_name = row
        full_path = file_path

        print(f"\n[INFO] Processing MediaItemId {media_item_id}: {full_path}")
        detect_and_crop_faces(full_path, media_item_id=media_item_id)

    print("[INFO] Batch processing complete.")

# TEST SINGLE IMAGE
def test_single_image(image_path):
    print(f"[INFO] Testing face detection on single image: {image_path}")
    test_detect_and_crop_faces(image_path)

# CONTINUOUS BATCH PROCESSING
def continuous_batch_process():
    print("[INFO] Starting continuous face detection monitoring...")
    no_data_attempts = 0

    while True:
        rows = get_unprocessed_files()

        if not rows:
            no_data_attempts += 1
            print(f"[INFO] No unprocessed files found. Attempt {no_data_attempts}/5")
            if no_data_attempts >= 5:
                print("[INFO] No data after 5 attempts. Pausing for 4 minutes before next check...")
                #time.sleep(240)
                countdown_timer(4 * 60, "Next check in")
                no_data_attempts = 0
            else:
                #time.sleep(10) 
                countdown_timer(10, "Next check in")
            continue
        else:
            no_data_attempts = 0 

        print(tabulate(rows, headers=["MediaItemId", "FilePath", "FileName"], tablefmt="grid"))

        for row in rows:
            media_item_id, file_path, file_name = row
            full_path = file_path

            print(f"\n[INFO] Processing MediaItemId {media_item_id}: {full_path}")
            success = detect_and_crop_faces(full_path, media_item_id=media_item_id)
            if not success:
                print(f"[WARN] Skipped MediaItemId {media_item_id} due to processing error.")

        print("[INFO] Completed current batch. Checking again in 10 seconds...\n")
        time.sleep(10)

# Test Detection and Cropping (Single file)
def test_detect_and_crop_faces(image_path, media_item_id=None):
    try:
        save_path = ''.join([thumbnail_base_path, os.path.basename(image_path)])
        faces = RetinaFace.detect_faces(image_path)
        img = cv2.imread(image_path)

        if not isinstance(faces, dict) or len(faces) == 0:
            print(f"[INFO] No faces detected in {image_path}")
            return False

        for idx, (key, face_data) in enumerate(faces.items()):
            landmarks = face_data["landmarks"]
            
            aligned = align_face(img, landmarks, margin=0.1)

            aligned_resized = cv2.resize(aligned, (112, 112), interpolation=cv2.INTER_AREA)

            if aligned_resized.shape[0] < 112 or aligned_resized.shape[1] < 112:
                print(f"[WARN] Face {idx+1} in {image_path} is too small after cropping. Skipping.")
                continue

            filename = f"thumb_{media_item_id or 'manual'}_{idx+1}_{int(time.time())}.jpg"
            cv2.imwrite(os.path.join(thumbnail_base_path, filename), aligned_resized)

            print(f"[INFO] Saved face {idx+1} to {filename}")

            if media_item_id is not None:
                update_database(media_item_id, filename, save_path)
        
        print(f"[INFO] Successfully processed {len(faces)} faces in {image_path}")

        return True

    except Exception as e:
        print(f"[ERROR] Face processing failed for {image_path}: {e}")
        return False
