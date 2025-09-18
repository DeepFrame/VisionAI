# ***************************** IMPORTS *****************************
import argparse
import time
import pyodbc

import os
from dotenv import load_dotenv

import cv2
import json
import logging
import numpy as np
from datetime import datetime

from retinaface import RetinaFace

import sys
import time

from keras_facenet import FaceNet
from sklearn.cluster import DBSCAN
import umap
from tabulate import tabulate
import logging

from collections import defaultdict

from sklearn.metrics.pairwise import cosine_similarity
import shutil

import re, posixpath
import tensorflow as tf

print("""
 __   __  ___   _______  ___   _______  __    _  _______  ___  
|  | |  ||   | |       ||   | |       ||  |  | ||   _   ||   | 
|  |_|  ||   | |  _____||   | |   _   ||   |_| ||  |_|  ||   | 
|       ||   | | |_____ |   | |  | |  ||       ||       ||   | 
|       ||   | |_____  ||   | |  |_|  ||  _    ||       ||   | 
 |     | |   |  _____| ||   | |       || | |   ||   _   ||   | 
  |___|  |___| |_______||___| |_______||_|  |__||__| |__||___| 

""")

# ***************************** GPU SETUP *****************************
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[INFO] {len(gpus)} GPU(s) detected: {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(f"[ERROR] GPU setup failed: {e}")
else:
    print("[INFO] No GPU detected, running on CPU")

# ***************************** CONFIG.py *****************************
#if getattr(sys, 'frozen', False):  
#    exe_dir = os.path.dirname(sys.executable)
#    dotenv_path = os.path.join(exe_dir, ".env")
#else:
#    dotenv_path = os.path.join(os.getcwd(), ".env")

load_dotenv()

SQL_CONNECTION_STRING = os.getenv("SQL_CONNECTION_STRING")

CURRENT_DIR = os.getcwd()
THUMBNAIL_SAVE_PATH = os.path.join(CURRENT_DIR, "Thumbnails")

SYSTEM_THUMBNAILS_PATH = THUMBNAIL_SAVE_PATH  

os.makedirs(THUMBNAIL_SAVE_PATH, exist_ok=True)

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load environment variables
conn_str = SQL_CONNECTION_STRING
thumbnail_base_path = THUMBNAIL_SAVE_PATH
system_thumb_path = SYSTEM_THUMBNAILS_PATH

# Ensure the thumbnail directory exists
os.makedirs(thumbnail_base_path, exist_ok=True)
os.makedirs(system_thumb_path, exist_ok=True)

# ***************************** LOGGER_CONFIG.py *****************************
LOG_DIR = os.path.join(CURRENT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_PATH = os.path.join(LOG_DIR, "detection_recognition.log")

def get_logger(name="detection_recognition"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.FileHandler(LOG_PATH)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

# ***************************** DETECT_FACES.py *****************************
# logger setup
logger = get_logger()

def reprocess_media_missing_faces(dry_run: bool = False):
    """
    Find MediaItems marked as extracted (IsFacesExtracted=1 and FacesExtractedOn IS NOT NULL)
    but with no corresponding rows in dbo.Faces. Re-run detection and write into dbo.Faces.
    """
    try:
        with pyodbc.connect(SQL_CONNECTION_STRING) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT mi.Id AS MediaItemId, mf.FilePath, mf.FileName
                    FROM dbo.MediaItems mi
                    INNER JOIN dbo.MediaFile mf ON mi.MediaFileId = mf.Id
                    WHERE mi.IsFacesExtracted = 1
                      AND mi.FacesExtractedOn IS NOT NULL
                      AND NOT EXISTS (
                          SELECT 1
                          FROM dbo.Faces f
                          WHERE f.MediaItemId = mi.Id
                      )
                """)
                rows = cur.fetchall()

        if not rows:
            logger.info("All processed MediaItems have Faces records. Nothing to reprocess.")
            return

        logger.info(f"Reprocessing {len(rows)} MediaItems with missing Faces rows...")
        for media_item_id, file_path, file_name in rows:
            try:
                full_path = file_path
            except Exception as e:
                logger.error(f"Path mapping failed for MediaItemId {media_item_id}: {e}")
                continue

            ok = detect_and_crop_faces(full_path, media_item_id=media_item_id, dry_run=dry_run)
            if not ok:
                logger.warning(f"Re-detect skipped/failed for MediaItemId {media_item_id}")

    except Exception as e:
        logger.error(f"reprocess_media_missing_faces failed: {e}")

def check_thumbnails(dry_run=False):
    """
    Check if thumbnails exist for each entry in dbo.Faces.
    If missing, recreate them using the stored bounding box.
    """

    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        query = """
        SELECT f.Id, f.BoundingBox, f.Name, mf.FilePath, mf.FileName
        FROM dbo.Faces f
        INNER JOIN dbo.MediaItems mi ON f.MediaItemId = mi.Id
        INNER JOIN dbo.MediaFile mf ON mi.MediaFileId = mf.Id
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        if not rows:
            logger.info("No faces found in DB.")
            return

        for face_id, bbox_str, face_name, file_path, file_name in rows:
            try:
                full_path = file_path

                base_name, _ = os.path.splitext(file_name)
                thumb_name = face_name if face_name else f"{base_name}_TN{face_id}.jpg"
                thumb_path = os.path.join(thumbnail_base_path, thumb_name)

                if os.path.exists(thumb_path):
                    logger.debug(f"Thumbnail exists: {thumb_name}")
                    continue  # skip

                logger.info(f"[Missing] Recreating thumbnail: {thumb_name}")

                img = cv2.imread(full_path)
                if img is None:
                    logger.error(f"Could not read original image: {full_path}")
                    continue

                if not bbox_str:
                    logger.warning(f"No bounding box for FaceId {face_id}")
                    continue

                bbox = json.loads(bbox_str) if isinstance(bbox_str, str) else None
                if not bbox or len(bbox) != 4:
                    logger.warning(f"Invalid bbox for FaceId {face_id}: {bbox_str}")
                    continue

                x1, y1, x2, y2 = map(int, bbox)
                cropped = img[y1:y2, x1:x2]

                if cropped.size == 0:
                    logger.warning(f"Empty crop for FaceId {face_id}")
                    continue

                if not dry_run:
                    cv2.imwrite(thumb_path, cropped)
                    logger.info(f"[OK] Created {thumb_path}")
                else:
                    logger.info(f"[DryRun] Would create {thumb_path}")

            except Exception as e:
                logger.error(f"Error processing FaceId {face_id}: {e}")

    except Exception as e:
        logger.error(f"check_thumbnails() failed: {e}")

# DATABASE Unprocessed files and Thumbnails Query Processing
def get_unprocessed_files():
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        query = """
        SELECT MI.Id, MF.FilePath, MI.Name
        FROM dbo.MediaItems MI
        JOIN dbo.MediaFile MF ON MI.MediaFileId = MF.Id
        WHERE MI.IsFacesExtracted = 0 AND LOWER(MF.Extension) IN ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        return rows
    
    except Exception as e:
        logger.error(f"Database update failed: {e}")
        return []

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
                        INSERT INTO dbo.Faces (MediaItemId, BoundingBox, Name, CreatedAt, IsUserVerified)
                        VALUES (?, ?, ?, ?, ?)
                    """, media_item_id, bbox_str, filename, datetime.now(), 0)  # 0 = false

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
def detect_and_crop_faces(image_path, media_item_id=None, dry_run=False):
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
            #sys_save_path = os.path.join(system_thumb_path, filename)
            
            try:
                cv2.imwrite(save_path, processed_face)
                #cv2.imwrite(sys_save_path, processed_face)

                logger.info(f"Saved in system, the face {idx+1} to \n{save_path}")

                if not dry_run:
                    logger.info(f"Saved square (112x112) face {idx+1} to {save_path}")

                    update_database(media_item_id, [bbox], filename)
                else:
                    logger.info(f"[Dry Run] Would save face {idx+1} to {save_path}")
            except Exception as e:
                logger.error(f"Failed to save face {idx+1}: {e}")

        if media_item_id is not None and bounding_boxes and not dry_run:
            update_database(media_item_id, bounding_boxes, filename)

        return True

    except Exception as e:
        logger.error(f"Face processing failed for {image_path}: {e}")
        return False
    
# MAIN BATCH PROCESSOR
def batch_process_from_db(dry_run=False):
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
        detect_and_crop_faces(full_path, media_item_id=media_item_id, dry_run=dry_run)

    print("[INFO] Batch processing complete.")
    logger.info("Batch processing complete.")

# TEST SINGLE IMAGE
def test_single_image(image_path, dry_run=False):
    print(f"[INFO] Testing face detection on single image: {image_path}")
    test_detect_and_crop_faces(image_path, dry_run=dry_run)

# CONTINUOUS BATCH PROCESSING
def continuous_batch_process(dry_run=False):
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
            success = detect_and_crop_faces(full_path, media_item_id=media_item_id, dry_run=dry_run)
            if not success:
                logger.warning(f"Skipped MediaItemId {media_item_id} due to processing error.")

        print("[INFO] Completed current batch. Checking again in 10 seconds...\n")
        time.sleep(10)

# Test Detection and Cropping (Single file)
def test_detect_and_crop_faces(image_path, media_item_id=None, dry_run=False):
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

            #sys_save_path = os.path.join(system_thumb_path, filename)
            #cv2.imwrite(sys_save_path, processed_face)
            #logger.info(f"Saved in system, the face {idx+1} to \n{sys_save_path} and \n{save_path}")
            
            logger.info(f"[INFO] Saved face {idx+1} to {filename}")

            if media_item_id is not None and boundings is not None and not dry_run:
                update_database(media_item_id, boundings, filename)

        print(f"[INFO] Successfully processed {len(faces)} faces in {image_path}")
        return True

    except Exception as e:
        print(f"[ERROR] Face processing failed for {image_path}: {e}")
        return False

# ***************************** RECOGNIZE_PERSONS.py *****************************    
# FaceNet Embedder
embedder = FaceNet()

# DB Utilities
def get_faces_with_bboxes():
    """Fetch faces missing embeddings (NULL Embedding)."""
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            F.Id, MF.FilePath, MF.FileName, F.BoundingBox
            FROM dbo.Faces F
            JOIN dbo.MediaItems MI ON F.MediaItemId = MI.Id
            JOIN dbo.MediaFile MF ON MI.MediaFileId = MF.Id
            WHERE F.Embedding IS NULL
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    results = []
    for row in rows:
        face_id, file_path, file_name, bbox_json = row
        try:
            bbox = json.loads(bbox_json) if bbox_json else None
        except:
            bbox = None
        full_path = file_path
        logger.info(f"[DB] FaceId={face_id}, File={full_path}, BBox={bbox}")
        results.append({
            "FaceId": face_id,
            "FullPath": full_path,
            "BoundingBox": bbox
        })
    return results

def update_face_embedding(face_id, embedding):
    """Update DB with generated embedding."""
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        emb_bytes = embedding.astype(np.float32).tobytes()
        cursor.execute("""
            UPDATE dbo.Faces
            SET Embedding = ?, ModifiedAt = ?
            WHERE Id = ?
        """, emb_bytes, datetime.now(), face_id)
        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"Updated embedding for FaceId={face_id}")
    except Exception as e:
        logger.error(f"Failed to update embedding for FaceId {face_id}: {e}")

# Compute medoid of a cluster - PORTRAIT GENERATION
def compute_cluster_medoid(face_files, embeddings, cluster_label, labels):
    """
    Return the path of the medoid face for a given cluster.
    Medoid = face whose embedding is closest on average to all others in the cluster.
    """
    cluster_indices = [i for i, lbl in enumerate(labels) if lbl == cluster_label]
    cluster_embeddings = embeddings[cluster_indices]
    cluster_faces = [face_files[i] for i in cluster_indices]

    sim_matrix = cosine_similarity(cluster_embeddings)
    dist_matrix = 1 - sim_matrix 
    total_distances = dist_matrix.sum(axis=1)
    medoid_idx = np.argmin(total_distances)
    
    return cluster_faces[medoid_idx]

def calculate_sharpness(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0 
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return laplacian_var

def generate_portraits(rows, portrait_dir, dry_run=False):
    """
    Generate portraits for each person using existing MediaFiles:
    - Select medoid + sharpest face
    - Update Persons.PortraitMediaFileId and PortraitFaceId
    - Update MediaFile.ModifiedAt
    - No new file creation or MediaFile insertion
    """

    if not os.path.exists(portrait_dir) and not dry_run: 
        os.makedirs(portrait_dir)

    faces_by_person = defaultdict(list)
    for row in rows:
        person_id = row["PersonId"]
        faces_by_person[person_id].append({
            "FaceId": row["FaceId"],
            "FaceImagePath": row["FaceImagePath"],
            "Embedding": parse_embedding(row["Embedding"]),
        })

    logger.info(f"Total persons to process: {len(faces_by_person)}")

    for person_id, face_entries in faces_by_person.items():
        cluster_embeddings = np.array([f["Embedding"] for f in face_entries])
        cluster_faces = [f["FaceImagePath"] for f in face_entries]

        sim_matrix = cosine_similarity(cluster_embeddings)
        dist_matrix = 1 - sim_matrix
        total_distances = dist_matrix.sum(axis=1)
        medoid_idx = np.argmin(total_distances)

        medoid_path = cluster_faces[medoid_idx]

        medoid_sharpness = calculate_sharpness(medoid_path)
        for i, path in enumerate(cluster_faces):
            candidate_sharpness = calculate_sharpness(path)
            if candidate_sharpness > medoid_sharpness:
                medoid_path = path
                medoid_sharpness = candidate_sharpness

        medoid_face_id = None
        for f in face_entries:
            if f["FaceImagePath"] == medoid_path:
                medoid_face_id = f["FaceId"]
                break

        if medoid_face_id is None:
            logger.error(f"Could not find FaceId for medoid image {medoid_path} of Person {person_id}")
            continue

        logger.info(f"Selected portrait for Person {person_id}: {medoid_path} (Sharpness={medoid_sharpness}, FaceId={medoid_face_id})")

        if not dry_run and medoid_path: 
            output_path = os.path.join(portrait_dir, f"Person_{person_id}_portrait.jpg") 
            shutil.copy(medoid_path, output_path) 
            logger.info(f"Saved portrait for Person {person_id}: {output_path}\n\n")
            update_portrait_with_existing_mediafile(person_id, medoid_face_id)

def update_portrait_with_existing_mediafile(person_id, face_id):
    """
    Fetch the MediaFileId for a given FaceId and update:
    - Persons.PortraitMediaFileId
    - Persons.PortraitFaceId
    - MediaFile.ModifiedAt
    """
    try:
        with pyodbc.connect(conn_str) as conn:
            with conn.cursor() as cursor:
                # Get MediaFileId from the face
                cursor.execute("""
                    SELECT mf.Id AS MediaFileId
                    FROM dbo.Faces f
                    JOIN dbo.MediaItems mi ON f.MediaItemId = mi.Id
                    JOIN dbo.MediaFile mf ON mi.MediaFileId = mf.Id
                    WHERE f.Id = ?
                """, face_id)
                row = cursor.fetchone()
                if not row:
                    logger.error(f"No MediaFile found for FaceId={face_id}")
                    return
                mediafile_id = row[0]

                # Update Persons with MediaFileId + FaceId
                cursor.execute("""
                    UPDATE dbo.Persons
                    SET PortraitMediaFileId = ?, 
                        ModifiedAt = ?
                    WHERE Id = ?
                """, mediafile_id, datetime.now(), person_id)

                # Update MediaFile.ModifiedAt
                cursor.execute("""
                    UPDATE dbo.MediaFile
                    SET ModifiedAt = ?
                    WHERE Id = ?
                """, datetime.now(), mediafile_id)

                conn.commit()
                logger.info(f"[DB] Linked PersonId={person_id} to FaceId={face_id}, MediaFileId={mediafile_id}")
    except Exception as e:
        logger.error(f"[ERROR] Failed updating Portrait for PersonId={person_id}, FaceId={face_id}: {e}")

def update_portrait_face_id(person_id, face_id):
    """Update Persons.PortraitFaceId with the chosen FaceId"""
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE dbo.Persons
            SET PortraitFaceId = ?
            WHERE Id = ?;
        """, (face_id, person_id))
        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"[DB] Updated PersonId={person_id} with PortraitFaceId={face_id}")
    except Exception as e:
        logger.info(f"[ERROR] Failed to update PortraitFaceId for PersonId={person_id}: {e}")

# Fetch faces to cluster
def get_unassigned_faces(recluster=False, labelled=False):
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()

    if recluster:
        cursor.execute("""
            SELECT Id, Embedding
            FROM dbo.Faces
            WHERE Embedding IS NOT NULL
        """)
    elif labelled:
        cursor.execute("""
            SELECT Id, Embedding, PersonId
            FROM dbo.Faces
            WHERE PersonId IS NOT NULL AND Embedding IS NOT NULL
        """)
    else:
        cursor.execute("""
            SELECT Id, Embedding
            FROM dbo.Faces
            WHERE PersonId IS NULL AND Embedding IS NOT NULL
        """)

    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

def get_faces_with_paths(conn_str=conn_str):
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT 
            p.Id AS PersonId,
            f.Id AS FaceId,
            f.Name AS FaceName,
            f.Name AS FaceFileName,
            f.Embedding
        FROM dbo.Faces f
        INNER JOIN dbo.Persons p 
            ON f.PersonId = p.Id
        ORDER BY p.Id, f.Id;
    """)

    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    results = []
    for row in rows:
        person_id, face_id, face_name, file_name, embedding = row
        db_thumb_path = os.path.join(THUMBNAIL_SAVE_PATH, file_name)

        logger.info(f"Thumbnail Path -> {db_thumb_path}")
        results.append({
            "PersonId": person_id,
            "FaceId": face_id,
            "FaceName": face_name,
            "FaceImagePath": db_thumb_path,
            "Embedding": embedding,
        })
    return results, ["PersonId", "FaceId", "FaceName", "FaceImagePath", "Embedding"]


def parse_embedding(raw_value):
    try:
        if isinstance(raw_value, bytes):
            return np.frombuffer(raw_value, dtype=np.float32)
        elif isinstance(raw_value, str):
            return np.array(json.loads(raw_value), dtype=np.float32)
        return None
    except Exception as e:
        logger.error(f"Embedding parse error: {e}")
        return None

# Update Database with clusters
def assign_clusters(labels, face_ids, recluster=False, existing_person_id=None, dry_run=False):
    """
    Assign clusters to faces.
    - If existing_person_id is provided, assign faces directly to that person.
    - Otherwise, create new person entries per cluster.
    """
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        # === Direct assignment to existing person ===
        if existing_person_id is not None:
            if dry_run:
                logger.info(f"Dry run: Assigning {len(face_ids)} faces to PersonId={existing_person_id}")
            else:
                logger.info(f"Assigning {len(face_ids)} faces to existing PersonId={existing_person_id}")
                for face_id in face_ids:
                    cursor.execute("EXEC dbo.UpsertFace @FaceId=?, @PersonId=?", (face_id, existing_person_id))
                    cursor.execute("EXEC dbo.UpsertPerson ?, ?, ?, ?", (int(existing_person_id), None, None, None))

            conn.commit()
            cursor.close()
            conn.close()
            return

        # === Normal clustering mode ===
        logger.info(f"Assigning {len(set(labels))} clusters to {len(face_ids)} faces...")
        for cluster_id in set(labels):
            if cluster_id == -1: 
                continue
            if not dry_run:
                cursor.execute("""
                    INSERT INTO dbo.Persons (Name, Rank, Appointment, CreatedAt, Type)
                    OUTPUT INSERTED.Id
                    VALUES (?, ?, ?, ?, ?)
                """, f"Unknown-{int(cluster_id)}", None, None, datetime.now(), 0)
            person_id = int(cursor.fetchone()[0])

            logger.info(f"Created new PersonId={person_id} for cluster {cluster_id}")

            for face_id, label in zip(face_ids, labels):
                if label == cluster_id and not dry_run:
                    cursor.execute("""
                        UPDATE dbo.Faces
                        SET PersonId = ?, ModifiedAt = ?
                        WHERE Id = ?
                    """, person_id, datetime.now(), int(face_id))

            if not dry_run:
                cursor.execute("""
                    UPDATE dbo.Persons
                    SET ModifiedAt = ?
                    WHERE Id = ?
                """, datetime.now(), person_id)

        conn.commit()
        cursor.close()
        conn.close()

    except Exception as e:
        logger.error(f"DB update failed (assign clusters): {e}")

# Processing Pipeline
def process_missing_embeddings(dry_run=False):
    """Find faces with NULL embedding, generate, and update."""
    logger.info("Starting embedding process...")

    rows = get_faces_with_bboxes()
    if not rows:
        print("[INFO] No missing embeddings found.")
        return

    for row in rows:
        face_id, full_path, bbox = row["FaceId"], row["FullPath"], row["BoundingBox"]

        if not os.path.exists(full_path):
            logger.warning(f"File not found: {full_path}")
            continue

        img = cv2.imread(full_path)
        if img is None or bbox is None:
            logger.warning(f"Invalid image/bbox for FaceId={face_id}")
            continue

        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            face_crop = img[y1:y2, x1:x2]
            if face_crop.size == 0:
                logger.warning(f"Empty crop for FaceId={face_id}")
                continue

            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_crop = cv2.resize(face_crop, (160, 160))

            emb = embedder.embeddings([face_crop])[0]
            if dry_run:
                logger.info(f"[Dry Run] Generated embedding for FaceId={face_id} (not updating DB)")
            else:
                update_face_embedding(face_id, emb)

        except Exception as e:
            logger.error(f"Embedding failed for FaceId={face_id}: {e}")

def recluster_unlabelled_faces(eps=0.35, min_samples=3, similarity_threshold=0.75, dry_run=False):
    labelled_rows = get_unassigned_faces(labelled=True) 
    labelled_ids, labelled_embeddings, labelled_persons = [], [], []
    for row in labelled_rows:
        face_id, raw_emb, person_id = row 
        emb = parse_embedding(raw_emb)
        if emb is not None:
            labelled_ids.append(face_id)
            labelled_embeddings.append(emb)
            labelled_persons.append(person_id)

    labelled_embeddings = np.array(labelled_embeddings)

    logger.info(f"Found {len(labelled_ids)} labelled faces for reclustering.")
    if labelled_embeddings.size == 0:
        print("[INFO] No labelled embeddings found for reclustering.")
        #return
    
    unlabelled_rows = get_unassigned_faces(recluster=False)
    unlabelled_ids, unlabelled_embeddings = [], []
    for row in unlabelled_rows:
        face_id, raw_emb = row
        emb = parse_embedding(raw_emb)
        if emb is not None:
            unlabelled_ids.append(face_id)
            unlabelled_embeddings.append(emb)

    logger.info(f"Found {len(unlabelled_ids)} unlabelled faces for reclustering.")
    if not unlabelled_embeddings:
        print("[INFO] No unlabelled embeddings to recluster.")
        return

    unlabelled_embeddings = np.array(unlabelled_embeddings)
    logger.info(f"Unlabelled embeddings shape: {unlabelled_embeddings.shape}")
    logger.info(f"Unlabelled embeddings ids: " + ", ".join(map(str, unlabelled_ids)))
    for idx, emb in enumerate(unlabelled_embeddings):
        if labelled_embeddings.size == 0:
            break
        sims = cosine_similarity([emb], labelled_embeddings)[0]
        
        max_idx = np.argmax(sims)
        
        logger.info(f"FaceId={unlabelled_ids[idx]} similarity with PersonId={labelled_persons[max_idx]}: {sims[max_idx]:.4f}")
        if sims[max_idx] >= similarity_threshold:
            logger.info(f"Assigning FaceId={unlabelled_ids[idx]} to PersonId={labelled_persons[max_idx]}")
            assign_clusters(labels=None, face_ids=[unlabelled_ids[idx]], existing_person_id=labelled_persons[max_idx], dry_run=dry_run)

            labelled_ids.append(unlabelled_ids[idx])
        else:
            logger.info(f"FaceId={unlabelled_ids[idx]} does not meet similarity threshold, skipping.")
            continue

    remaining_ids = [fid for fid in unlabelled_ids if fid not in labelled_ids]
    remaining_embeddings = np.array([emb for i, emb in enumerate(unlabelled_embeddings) if unlabelled_ids[i] in remaining_ids])
    
    logger.info(f"Remaining unlabelled faces to cluster: {len(remaining_ids)}")
    
    if remaining_embeddings.size > 0:
        logger.info("Running DBSCAN on remaining unlabelled faces...")
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
        labels = clustering.fit_predict(remaining_embeddings)
        logger.info(f'Labels found: {set(labels)}')
        if len(set(labels)) <= 1:
            logger.info("No significant clusters found in remaining unlabelled faces.")
            return
        assign_clusters(labels, remaining_ids, recluster=False, dry_run=dry_run)
        logger.info(f"Reclustering complete. Assigned {len(set(labels))} clusters to remaining unlabelled faces.")

# Recognition pipeline: embeddings + clustering.
def recognize_persons_main(recluster=False, dry_run=False):
    print("[Step 1:] Generating embeddings for missing faces...")
    process_missing_embeddings(dry_run=dry_run)

    if recluster:
        print("[Step 2:] Reclustering unlabelled faces...")
        recluster_unlabelled_faces(dry_run=dry_run)
    else:
        print("[Step 2:] Clustering unlabelled faces...")
        rows = get_unassigned_faces(recluster=False)
        if not rows:
            print("[INFO] No faces available for clustering.")
            return

        face_ids, embeddings = [], []
        for row in rows:
            face_id, raw_emb = row
            emb = parse_embedding(raw_emb)
            if emb is not None:
                face_ids.append(face_id)
                embeddings.append(emb)

        embeddings = np.array(embeddings)
        if embeddings.size == 0:
            print("[WARN] No valid embeddings found.")
            return

        # Dimensionality reduction
        reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, metric="cosine")
        embeddings_2d = reducer.fit_transform(embeddings)

        clustering = DBSCAN(eps=0.35, min_samples=3, metric="cosine")
        labels = clustering.fit_predict(embeddings)

        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        logger.info(f"Found {num_clusters} clusters")

        assign_clusters(labels, face_ids, recluster=False, dry_run=dry_run)

        print(f"[INFO] Finished. Found {num_clusters} clusters.")
    
    portrait_dir = "individuals_portraits"

    if not os.path.exists(portrait_dir):
        os.makedirs(portrait_dir)

    rows, columns = get_faces_with_paths()
    logger.info(tabulate([list(r.values()) for r in rows], headers=columns, tablefmt="psql"))


    print("[Step 3:] Generating portraits for each person...")
    generate_portraits(rows, portrait_dir, dry_run=dry_run)
    print("[INFO] Portraits saved to 'individuals_portraits/' directory.")
    logger.info("Portrait generation complete.")

# MAIN.py
def should_recluster():
    """Check if there are entries in the Persons table."""
    try:
        conn = pyodbc.connect(SQL_CONNECTION_STRING)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM dbo.Persons")
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return count > 0 
    except Exception as e:
        print(f"[ERROR] Failed to check Persons table: {e}")
        return False

def full_pipeline_once(recluster=False, dry_run=False):
    """Run detection + recognition once."""

    print("[00] Reprocessing missing faces and thumbnails...")
    reprocess_media_missing_faces()
    check_thumbnails()

    print("[01] Detecting faces...")
    batch_process_from_db(dry_run=dry_run)
    
    print("[02] Running recognition pipeline...")
    recognize_persons_main(recluster=recluster, dry_run=dry_run)
    
    print("[INFO] Full pipeline completed.")

def automated_pipeline(interval_minutes=3, recluster=False, dry_run=False):
    """Run full pipeline every N minutes."""
    
    print(f"[AUTO] Pipeline execution started...")

    while True:
        start_time = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{start_time}] Running full pipeline...")

        full_pipeline_once(recluster=recluster, dry_run=dry_run)

        wait_seconds = interval_minutes * 60
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Sleeping for {interval_minutes} minutes...")

        countdown_timer(wait_seconds, message="Next run starts in")

def main():  
    print("Recluster flag set. This will affect the next recognition run.")
    recluster_needed = should_recluster()

    print(f"Persons table entries found: {recluster_needed}. Setting recluster={recluster_needed}")
    automated_pipeline(interval_minutes=3, recluster=recluster_needed)

if __name__ == "__main__":
    main()
