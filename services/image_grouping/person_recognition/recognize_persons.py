import os
import cv2
import json
import logging
import pyodbc
import numpy as np
from datetime import datetime
from keras_facenet import FaceNet
from sklearn.cluster import DBSCAN
import umap
from tabulate import tabulate

from collections import defaultdict

from sklearn.metrics.pairwise import cosine_similarity
import shutil

import re, posixpath

# logger setup
from .logger_config import get_logger
logger = get_logger()
    
# DB connection
from config import SQL_CONNECTION_STRING, OLD_PREFIX, NEW_PREFIX

conn_str = SQL_CONNECTION_STRING

# FaceNet Embedder
embedder = FaceNet()

# DB Utilities
def to_container_path(db_path: str) -> str:
    # normalize slashes and trim spaces
    p = db_path.strip().replace("\\", "/")
    old = OLD_PREFIX.rstrip("/")
    new = NEW_PREFIX.rstrip("/")

    if p.lower().startswith(old.lower()):
        suffix = p[len(old):].lstrip("/")
        logger.info(f"[DOCKER] CONTAINER Path -> {posixpath.join(new, suffix)}")
        return posixpath.join(new, suffix)

    m = re.search(r"/(\d{1,})/([^/]+)$", p)
    if m:
        return posixpath.join(new, m.group(1), m.group(2))

    raise ValueError(f"Cannot map path: {db_path}")

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
        full_path = to_container_path(file_path)
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
        print(f"[DB] Updated PersonId={person_id} with PortraitFaceId={face_id}")
    except Exception as e:
        print(f"[ERROR] Failed to update PortraitFaceId for PersonId={person_id}: {e}")

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
        db_thumb_path = os.path.join(NEW_PREFIX, "Thumbnails", file_name)
        face_image_path = to_container_path(db_thumb_path)
        results.append({
            "PersonId": person_id,
            "FaceId": face_id,
            "FaceName": face_name,
            "FaceImagePath": face_image_path,
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
                    INSERT INTO dbo.Persons (Name, Rank, Appointment, CreatedAt)
                    OUTPUT INSERTED.Id
                    VALUES (?, ?, ?, ?)
                """, f"Unknown-{int(cluster_id)}", None, None, datetime.now())
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
        return
    
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
def main(recluster=False, dry_run=False):
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
