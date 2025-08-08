import argparse
from detect_faces import test_single_image, batch_process_from_db, continuous_batch_process

def main():
    parser = argparse.ArgumentParser(description="RetinaFace Batch Processor")
    parser.add_argument("--test", type=str, help="Run face detection on single image")
    parser.add_argument("--db", action="store_true", help="Run face detection on DB media items once")
    parser.add_argument("--watch", action="store_true", help="Run continuous monitoring and processing")

    args = parser.parse_args()

    if args.test:
        test_single_image(args.test)
    elif args.db:
        batch_process_from_db()
    elif args.watch:
        continuous_batch_process()
    else:
        print("Use --test <image_path>, --db or --watch")

if __name__ == "__main__":
    main()
