#!/bin/bash

set -e

MODEL_DIR="/app/YPACm/weights/pose_estimation"
CONVERTED_MODEL="${MODEL_DIR}/yolov8l-pose.engine"

if [ ! -f "${CONVERTED_MODEL}" ]; then
    echo "Converting model to TensorRT format..."
    cd "./YPACm/utils/"
    python3 "export_yolo.py"
    cd "/app"
fi

echo "Starting FastAPI application..."
exec fastapi run main.py --port 9000
