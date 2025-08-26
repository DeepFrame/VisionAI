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

# logger setup
from .logger_config import get_logger
logger = get_logger()
    
# DB connection
from config import SQL_CONNECTION_STRING

conn_str = SQL_CONNECTION_STRING

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
        
        cursor.execute("EXEC dbo.UpsertFace @FaceId=?, @Embedding=?", (face_id, emb_bytes)) 
        
        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"Updated embedding for FaceId={face_id}")
    except Exception as e:
        logger.error(f"Failed to update embedding for FaceId {face_id}: {e}")

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
def assign_clusters(labels, face_ids, recluster=False, existing_person_id=None):
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

            cursor.execute("EXEC dbo.UpsertPerson ?, ?, ?, ?", 
                        (None, f"Person-{int(cluster_id)}", f"Rank-{int(cluster_id)}", f"Appointment-{int(cluster_id)}"))

            person_id = int(cursor.fetchone()[0])

            logger.info(f"Created new PersonId={person_id} for cluster {cluster_id}")

            for face_id, label in zip(face_ids, labels):
                if label == cluster_id:
                    cursor.execute("EXEC dbo.UpsertFace @FaceId=?, @PersonId=?", (face_id, existing_person_id))

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
def process_missing_embeddings():
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
            update_face_embedding(face_id, emb)

        except Exception as e:
            logger.error(f"Embedding failed for FaceId={face_id}: {e}")

from sklearn.metrics.pairwise import cosine_similarity

def recluster_unlabelled_faces(eps=0.35, min_samples=3, similarity_threshold=0.8):
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
    for idx, emb in enumerate(unlabelled_embeddings):
        if labelled_embeddings.size == 0:
            break
        sims = cosine_similarity([emb], labelled_embeddings)[0]
        
        max_idx = np.argmax(sims)
        
        logger.info(f"FaceId={unlabelled_ids[idx]} similarity with PersonId={labelled_persons[max_idx]}: {sims[max_idx]:.4f}")
        if sims[max_idx] >= similarity_threshold:
            logger.info(f"Assigning FaceId={unlabelled_ids[idx]} to PersonId={labelled_persons[max_idx]}")
            assign_clusters(labels=None, face_ids=[unlabelled_ids[idx]], existing_person_id=labelled_persons[max_idx])

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
        assign_clusters(labels, remaining_ids, recluster=False)
        logger.info(f"Reclustering complete. Assigned {len(set(labels))} clusters to remaining unlabelled faces.")

# Recognition pipeline: embeddings + clustering.
def main(recluster=False):
    print("[Step 1:] Generating embeddings for missing faces...")
    process_missing_embeddings()

    if recluster:
        print("[Step 2:] Reclustering unlabelled faces...")
        recluster_unlabelled_faces()
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

        assign_clusters(labels, face_ids, recluster=False)

        print(f"[INFO] Finished. Found {num_clusters} clusters.")
