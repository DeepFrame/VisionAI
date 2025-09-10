#!/bin/bash
set -Eeuo pipefail
set -x

SERVICE=${SERVICE:-recognition}
MODE=${MODE:-loop}
INTERVAL_SECONDS=${INTERVAL_SECONDS:-900}
RECOGNIZER_ARGS=${RECOGNIZER_ARGS:---recluster}

mkdir -p /logs

run_service() {
  case "$SERVICE" in
    recognition)
      python -m FaceRecognitionSystem $RECOGNIZER_ARGS >> /logs/detection_recognition.log 2>&1
      ;;
    *)
      echo "Unknown SERVICE: $SERVICE" >&2; return 64 ;;
  esac
}

if [ "${MODE:-loop}" = "loop" ]; then
  while true; do run_service; echo "[$(date)] Sleeping $INTERVAL_SECONDS…"; sleep "$INTERVAL_SECONDS"; done
else
  case "$SERVICE" in
    recognition) exec python -m FaceRecognitionSystem $RECOGNIZER_ARGS >> /logs/detection_recognition.log 2>&1 ;;
  esac
fi
