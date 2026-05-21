#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <path-to-tflite-model>"
  exit 1
fi

MODEL_PATH="$1"
if [ ! -f "$MODEL_PATH" ]; then
  echo "Model file not found: $MODEL_PATH"
  exit 1
fi

ASSET_DIR="android-app/app/src/main/assets"
mkdir -p "$ASSET_DIR"
cp "$MODEL_PATH" "$ASSET_DIR/model_int8.tflite"

echo "Copied $MODEL_PATH -> $ASSET_DIR/model_int8.tflite"
