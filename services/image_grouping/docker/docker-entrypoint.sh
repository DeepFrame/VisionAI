#!/bin/bash
set -x

# Set default environment variables if not provided
SERVICE=${SERVICE:-recognition}
MODE=${MODE:-loop}
INTERVAL_SECONDS=${INTERVAL_SECONDS:-900}
RECOGNIZER_ARGS=${RECOGNIZER_ARGS:--recluster}

run_service() {
    case $SERVICE in
        recognition)
            LOGFILE=person_recognition/logs/run.log
            mkdir -p "$(dirname "$LOGFILE")"
            echo "[$(date)] Running $SERVICE" >> "$LOGFILE"
            python main.py $RECOGNIZER_ARGS
            ;;
        detection)
            LOGFILE=image_face_detection/logs/run.log
            mkdir -p "$(dirname "$LOGFILE")"
            echo "[$(date)] Running $SERVICE" >> "$LOGFILE"
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
        sleep $INTERVAL_SECONDS
    done
else
    run_service
fi
