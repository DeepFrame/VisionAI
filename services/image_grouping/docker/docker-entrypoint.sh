#!/bin/bash
set -e
set -x

# Load environment variables (DB credentials, etc.)
export DB_HOST=${DB_HOST:-sqlserver}
export DB_PORT=${DB_PORT:-1433}
export DB_USER=${DB_USER:-sa}
export DB_PASSWORD=${DB_PASSWORD:-YourStrong@Passw0rd}
export DB_NAME=${DB_NAME:-MediaDB}

SERVICE=${SERVICE:-detection}
MODE=${MODE:-loop}
INTERVAL_SECONDS=${INTERVAL_SECONDS:-900}
RECOGNIZER_ARGS=${RECOGNIZER_ARGS:--recluster}

mkdir -p person_recognition/logs image_face_detection/logs

run_service() {
    case $SERVICE in
        recognition)
            LOGFILE=person_recognition/logs/run.log
            echo "[$(date)] Running $SERVICE with args: $RECOGNIZER_ARGS" >> "$LOGFILE"
            python main.py $RECOGNIZER_ARGS
            ;;
        detection)
            LOGFILE=image_face_detection/logs/run.log
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
