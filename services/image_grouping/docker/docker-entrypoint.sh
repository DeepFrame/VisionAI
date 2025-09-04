#!/bin/bash
set -e
set -x

# Load environment variables (DB credentials, etc.)
SERVICE=${SERVICE:-recognition}
MODE=${MODE:-loop}
INTERVAL_SECONDS=${INTERVAL_SECONDS:-900}
RECOGNIZER_ARGS=${RECOGNIZER_ARGS:--dry-run}

mkdir -p person_recognition/logs image_face_detection/logs

run_service() {
    case $SERVICE in
        recognition)
            LOGFILE=person_recognition/logs/face_detection.log
            echo "[$(date)] Running $SERVICE with args: $RECOGNIZER_ARGS" >> "$LOGFILE"
            python main.py $RECOGNIZER_ARGS
            ;;
        detection)
            LOGFILE=image_face_detection/logs/embeddings_clustering.log
            echo "[$(date)] Running $SERVICE with args: $RECOGNIZER_ARGS" >> "$LOGFILE"
            python main.py $RECOGNIZER_ARGS
            ;;
        *)
            echo "Unknown SERVICE: $SERVICE" >&2
            exit 1
            ;;
    esac
}

if [ "$MODE" = "loop" ]; then
    while true; do
        run_service
        echo "[$(date)] Sleeping for $INTERVAL_SECONDS seconds..."
        sleep $INTERVAL_SECONDS
    done
else
    run_service
fi
