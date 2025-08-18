import argparse
import time
from image_face_detection.detect_faces import (
    test_single_image,
    batch_process_from_db,
    continuous_batch_process,
)
from person_recognition.recognize_persons import main as recognize_persons_main


def full_pipeline_once(recluster=False):
    """Run detection + recognition once."""
    print("[01] Detecting faces...")
    batch_process_from_db()
    print("[02] Running recognition pipeline...")
    recognize_persons_main(recluster=recluster)
    print("[INFO] Full pipeline completed.")


def automated_pipeline(interval_minutes=3, recluster=False):
    """Run full pipeline every N minutes."""
    print(f"[AUTO] Pipeline execution started (interval={interval_minutes}m).")
    while True:
        start_time = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{start_time}] Running full pipeline...")
        full_pipeline_once(recluster=recluster)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [Info] Sleeping {interval_minutes} minutes...\n")
        time.sleep(interval_minutes * 60)


def main():
    parser = argparse.ArgumentParser(description="Face Recognition System CLI")
    parser.add_argument("--test", type=str, help="Run face detection on single image")
    parser.add_argument("--db", action="store_true", help="Run face detection on DB media items once")
    parser.add_argument("--watch", action="store_true", help="Run continuous monitoring and processing (detection)")
    parser.add_argument("--recognize", action="store_true", help="Run embeddings + clustering pipeline")
    parser.add_argument("--all", action="store_true", help="Run full pipeline: detection -> recognition once")
    parser.add_argument("--automate", action="store_true", help="Run full pipeline every 3 minutes: detection + recognition")
    parser.add_argument("--recluster", action="store_true", help="Rebuild clusters for all faces (ignore old PersonId mappings)")

    args = parser.parse_args()

    if args.test:
        test_single_image(args.test)
    elif args.db:
        batch_process_from_db()
    elif args.watch:
        continuous_batch_process()
    elif args.recognize:
        recognize_persons_main(recluster=args.recluster)
    elif args.all:
        full_pipeline_once(recluster=args.recluster)
    elif args.automate or args.recluster:
        automated_pipeline(interval_minutes=3, recluster=args.recluster)
    else:
        print("Use --test <image_path>, --db, --watch, --recognize, --all, --automate, or --recluster")


if __name__ == "__main__":
    main()
