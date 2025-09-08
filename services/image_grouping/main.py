import argparse
import time
import pyodbc

from image_face_detection.detect_faces import (
    test_single_image,
    batch_process_from_db,
    continuous_batch_process,
    countdown_timer,
    check_thumbnails,
    reprocess_media_missing_faces
)
from person_recognition.recognize_persons import main as recognize_persons_main

from config import SQL_CONNECTION_STRING

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
    parser = argparse.ArgumentParser(description="Face Recognition System CLI")
    parser.add_argument("--test", type=str, help="Run face detection on single image")
    parser.add_argument("--db", action="store_true", help="Run face detection on DB media items once")
    parser.add_argument("--watch", action="store_true", help="Run continuous monitoring and processing (detection)")
    parser.add_argument("--recognize", action="store_true", help="Run embeddings + clustering pipeline")
    parser.add_argument("--all", action="store_true", help="Run full pipeline: detection -> recognition once")
    parser.add_argument("--automate", action="store_true", help="Run full pipeline every 3 minutes: detection + recognition")
    parser.add_argument("--recluster", action="store_true", help="Rebuild clusters for all faces (ignore old PersonId mappings)")
    parser.add_argument("--dry-run", action="store_true", help="Produces logs without DB changes")

    args = parser.parse_args()
    
    if args.test:
        test_single_image(args.test, dry_run=args.dry_run)
    elif args.db:
        batch_process_from_db()
    elif args.watch:
        continuous_batch_process(dry_run=args.dry_run)
    elif args.recognize:
        recognize_persons_main(recluster=args.recluster)
    elif args.all:
        full_pipeline_once(recluster=args.recluster)
    elif args.automate:
        recluster_needed = should_recluster()
        print(f"Persons table entries found: {recluster_needed}. Setting recluster={recluster_needed}")
        automated_pipeline(interval_minutes=3, recluster=recluster_needed, dry_run=args.dry_run)
    elif args.dry_run:
        print("Dry run mode activated. No DB changes will be made.")
        recluster_needed = should_recluster()
        full_pipeline_once(recluster=recluster_needed, dry_run=True)
    elif args.recluster:
        print("Recluster flag set. This will affect the next recognition run.")
        recluster_needed = should_recluster()
        print(f"Persons table entries found: {recluster_needed}. Setting recluster={recluster_needed}")
        automated_pipeline(interval_minutes=3, recluster=recluster_needed, dry_run=args.dry_run)
    else:
        print("Use --test <image_path>, --db, --watch, --recognize, --all, --automate, --dry-run, or --recluster")

if __name__ == "__main__":
    main()

