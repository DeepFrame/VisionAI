import os
import time
import pyodbc
import cv2
import numpy as np
from datetime import datetime
from retinaface import RetinaFace
from tabulate import tabulate
#from .config import SQL_CONNECTION_STRING, THUMBNAIL_SAVE_PATH
from config import SQL_CONNECTION_STRING, THUMBNAIL_SAVE_PATH

import sys
import time

import json

import logging

# logger setup
from .logger_config import get_logger
logger = get_logger()

# Load environment variables
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
        logger.error(f"Database update failed: {e}")
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

def get_unassigned_faces():
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT Id, Embedding
            FROM dbo.Faces
            WHERE PersonId IS NULL
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"Could not load unassigned faces: {e}")
        #print(f"[ERROR] Could not load unassigned faces: {e}")
        return []
    
def parse_embedding(raw_value):
    try:
        if isinstance(raw_value, str):
            return np.array(json.loads(raw_value), dtype=np.float32)
        # If stored as varbinary
        elif isinstance(raw_value, bytes):
            return np.frombuffer(raw_value, dtype=np.float32)
        else:
            raise ValueError("Unknown embedding format")
    except Exception as e:
        logger.error(f"Failed to parse embedding: {e}")
        return None

# Filter Non-Blurry Images
def is_blurry(image, threshold=100.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold, variance

# DATABASE Update Query Processing
def update_database(media_item_id, face_bboxes, filename=None):
    """
    Inserts detected faces into dbo.Faces (BoundingBox only, no embedding yet).
    Avoids duplicates for same MediaItemId + BoundingBox.
    Updates ModifiedAt if duplicate exists.
    """
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        # Update MediaItems status
        cursor.execute("""
            UPDATE dbo.MediaItems
            SET IsFacesExtracted = 1,
                FacesExtractedOn = ?
            WHERE Id = ?
        """, datetime.now(), media_item_id)

        inserted_count = 0
        updated_count = 0

        for bbox in face_bboxes:
            # Ensure bbox is in the correct format [x1, y1, x2, y2]
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                logger.warning(f"Invalid bounding box format: {bbox}")
                continue

            try:
                bbox_norm = [round(float(c), 3) for c in bbox] 
                bbox_str = json.dumps(bbox_norm) 

                cursor.execute("""
                    SELECT Id FROM dbo.Faces
                    WHERE MediaItemId = ? AND BoundingBox = ?
                """, media_item_id, bbox_str)
                row = cursor.fetchone()

                if row:
                    cursor.execute("""
                        UPDATE dbo.Faces
                        SET BoundingBox = ?, Name = ?, ModifiedAt = ?
                        WHERE Id = ?
                    """, bbox_str, filename, datetime.now(), row[0])
                    updated_count += 1
                else:
                    cursor.execute("""
                        INSERT INTO dbo.Faces (MediaItemId, BoundingBox, Name, CreatedAt)
                        VALUES (?, ?, ?, ?)
                    """, media_item_id, bbox_str, filename, datetime.now())
                    inserted_count += 1
            except Exception as e:
                logger.warning(f"Failed to process bounding box {bbox}: {e}")
                continue

        conn.commit()
        cursor.close()
        conn.close()

        logger.info(f"Inserted {inserted_count} new face(s), updated {updated_count} existing face(s) for MediaItemId {media_item_id}")

    except Exception as e:
        logger.error(f"Database update failed: {e}")
        if 'conn' in locals():
            conn.rollback()

# Countdown Timer for Continuous Processing
def countdown_timer(seconds, message="Waiting"):
    for remaining in range(seconds, 0, -1):
        mins, secs = divmod(remaining, 60)
        sys.stdout.write(f"\r{message}: {mins:02d}:{secs:02d} remaining ")
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write("\r" + " " * 50 + "\r")

# SQUARE CROP with margin & expansion/shrink
def process_face_square(img, face, margin_ratio=0.2, target_size=(112, 112)):
    #score = face.get("score", 0.0)
    
    h, w = img.shape[:2]
    x1, y1, x2, y2 = face["facial_area"]

    bw = x2 - x1
    bh = y2 - y1
    margin_x = int(bw * margin_ratio)
    margin_y = int(bh * margin_ratio)

    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(w, x2 + margin_x)
    y2 = min(h, y2 + margin_y)

    crop_w = x2 - x1
    crop_h = y2 - y1
    if crop_w > crop_h:
        diff = crop_w - crop_h
        expand_top = diff // 2
        expand_bottom = diff - expand_top
        if y1 - expand_top >= 0 and y2 + expand_bottom <= h:
            y1 -= expand_top
            y2 += expand_bottom
        else:
            x1 += diff // 2
            x2 -= (diff - diff // 2)
    elif crop_h > crop_w:
        diff = crop_h - crop_w
        expand_left = diff // 2
        expand_right = diff - expand_left
        if x1 - expand_left >= 0 and x2 + expand_right <= w:
            x1 -= expand_left
            x2 += expand_right
        else:
            y1 += diff // 2
            y2 -= (diff - diff // 2)

    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)

    cropped_face = img[y1:y2, x1:x2]

    if cropped_face.size == 0:
        return None, None

    resized_face = cv2.resize(cropped_face, target_size, interpolation=cv2.INTER_AREA)
    updated_bbox = [int(x1), int(y1), int(x2), int(y2)]  
    return resized_face, updated_bbox

# FACE DETECTION (RetinaFace)
def detect_and_crop_faces(image_path, media_item_id=None):
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not read image: {image_path}")
            return False

        faces = RetinaFace.detect_faces(image_path)
        if not isinstance(faces, dict) or len(faces) == 0:
            logger.info(f"No faces detected in {image_path}")
            return False

        base_name = os.path.basename(image_path)
        name_without_ext, ext = os.path.splitext(base_name)
        ext = ext.lower() if ext else ".jpg"

        bounding_boxes = []
        
        for idx, (key, face_data) in enumerate(faces.items()):
            processed_face, bbox = process_face_square(img, face_data, margin_ratio=0.2, target_size=(112, 112))
            if processed_face is None or bbox is None:
                logger.warning(f"Face not found.")
                continue

            filename = f"{name_without_ext}_TN{idx + 1}{ext}"
            save_path = os.path.join(thumbnail_base_path, filename)
            
            try:
                cv2.imwrite(save_path, processed_face)
                logger.info(f"Saved square (112x112) face {idx+1} to {save_path}")
                # Immediately update DB per face
                update_database(media_item_id, [bbox], filename)
            except Exception as e:
                logger.error(f"Failed to save face {idx+1}: {e}")

        if media_item_id is not None and bounding_boxes:
            update_database(media_item_id, bounding_boxes, filename)

        return True

    except Exception as e:
        logger.error(f"Face processing failed for {image_path}: {e}")
        return False
    
# MAIN BATCH PROCESSOR
def batch_process_from_db():
    logger.info("Starting batch face detection from DB...")

    rows = get_unprocessed_files()
    if not rows:
        logger.info("No unprocessed files found.")
        return

    print(tabulate(rows, headers=["MediaItemId", "FilePath", "FileName"], tablefmt="grid"))
    logger.info(tabulate(rows, headers=["MediaItemId", "FilePath", "FileName"], tablefmt="grid"))
    logger.info(f"Found {len(rows)} unprocessed files.")

    for row in rows:
        media_item_id, file_path, file_name = row
        full_path = file_path

        logger.info(f"\nProcessing MediaItemId {media_item_id}: {full_path}")
        detect_and_crop_faces(full_path, media_item_id=media_item_id)

    print("[INFO] Batch processing complete.")
    logger.info("Batch processing complete.")

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

        #print(tabulate(rows, headers=["MediaItemId", "FilePath", "FileName"], tablefmt="grid"))

        logger.info(tabulate(rows, headers=["MediaItemId", "FilePath", "FileName"], tablefmt="grid"))
        logger.info(f"Found {len(rows)} unprocessed files.")

        for row in rows:
            media_item_id, file_path, file_name = row
            full_path = file_path

            logger.info(f"\nProcessing MediaItemId {media_item_id}: {full_path}")
            success = detect_and_crop_faces(full_path, media_item_id=media_item_id)
            if not success:
                logger.warning(f"Skipped MediaItemId {media_item_id} due to processing error.")

        print("[INFO] Completed current batch. Checking again in 10 seconds...\n")
        time.sleep(10)

# Test Detection and Cropping (Single file)
def test_detect_and_crop_faces(image_path, media_item_id=None):
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"[ERROR] Could not read image: {image_path}")
            return False

        faces = RetinaFace.detect_faces(image_path)
        if not isinstance(faces, dict) or len(faces) == 0:
            print(f"[INFO] No faces detected in {image_path}")
            return False

        for idx, (key, face_data) in enumerate(faces.items()):
            processed_face, boundings = process_face_square(img, face_data, margin_ratio=0.2, target_size=(112, 112))
            if processed_face is None:
                print(f"[WARN] Face not found.")
                continue

            filename = f"thumb_{media_item_id or 'manual'}_{idx+1}_{int(time.time())}.jpg"
            save_path = os.path.join(thumbnail_base_path, filename)
            cv2.imwrite(save_path, processed_face)
            logger.info(f"[INFO] Saved face {idx+1} to {filename}")

            if media_item_id is not None and boundings is not None:
                update_database(media_item_id, boundings, filename)

        print(f"[INFO] Successfully processed {len(faces)} faces in {image_path}")
        return True

    except Exception as e:
        print(f"[ERROR] Face processing failed for {image_path}: {e}")
        return False
