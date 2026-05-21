# KWS Week1-4 Implementation

This repository implements the Week1-Week4 scope for:

- Speech Commands data preparation (`8 commands + unknown + silence`)
- Feature extraction (`log-mel` / `MFCC`)
- Keras model training (`small_cnn`, optional `ds_cnn`)
- TFLite export (`float32`, `int8` quantized)
- Quantization benchmark and comparison report
- Android Compose realtime inference app (`AudioRecord` + rolling buffer + live UI)

## 1) Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 2) Main CLI Interfaces

```bash
python -m kws.data.prepare --config configs/default_config.json --download
python -m kws.train --config configs/default_config.json --model small_cnn
python -m kws.eval --config configs/default_config.json --model small_cnn
python -m kws.export.tflite --config configs/default_config.json --model small_cnn
python -m kws.benchmark --config configs/default_config.json --model small_cnn
```

Optional DS-CNN run:

```bash
python -m kws.train --config configs/default_config.json --model ds_cnn
```

## 3) Expected Artifacts

- `artifacts/data/manifest.csv`
- `artifacts/data/label_map.json`
- `artifacts/models/<model>/best_model.keras`
- `artifacts/models/<model>/model_float32.tflite`
- `artifacts/models/<model>/model_int8.tflite`
- `artifacts/reports/<model>/eval_report.json`
- `artifacts/reports/<model>/benchmark_report.json`
- `artifacts/reports/<model>/quantization_comparison.md`

## 4) Android Realtime App (Week4)

Project path: `android-app/`

What it does:

- Captures microphone PCM via `AudioRecord`
- Maintains rolling 1-second buffer and runs inference every ~320ms
- Displays realtime waveform, top label, confidence bars, timestamp, and latency
- Includes a simple voice-command panel (`up/down/left/right/go/stop/yes/no`)

Before running Android app, copy a model into assets:

```bash
bash scripts/copy_model_to_android_assets.sh artifacts/models/small_cnn_qat/model_int8_qat.tflite
```

Then open `android-app` in Android Studio and run the app.
Allow microphone permission when prompted and press `Start Listening`.

## 5) Configuration Schema (Key Fields)

- `sample_rate`, `clip_duration_ms`, `num_classes`
- `frame_length`, `frame_step`, `n_mfcc` / `num_mel_bins`
- `augment.*`, `train.*`, `quantization.*`
- `data.use_feature_cache`, `data.feature_cache_dir`, `data.cache_train_augmented_features`

See `configs/default_config.json` for the full schema.

## 6) Training Speed Notes

- Default config uses feature caching (`data.use_feature_cache=true`).
- First run builds cache files under `artifacts/data/feature_cache/` and can be slow.
- Later runs reuse cached features and are much faster on CPU.
- Default keeps training augmentation fresh each epoch (`cache_train_augmented_features=false`) to protect accuracy.
- If you need maximum speed and can tolerate less augmentation diversity, set:

```json
"data": {
  "use_feature_cache": true,
  "cache_train_augmented_features": true
}
```

## 7) Tests

Python tests:

```bash
pytest
```

Current test coverage includes:

- fixed class mapping order validation
- manifest reproducibility with fixed seed
- float32/int8 TFLite loadability smoke check

Android smoke tests are included under:

- `android-app/app/src/androidTest/...`
- `android-app/app/src/test/...`

## 8) Notes

- The dataset command supports `--download` via `kagglehub` or direct local `--dataset-root`.
- If DS-CNN is unstable on your setup, `small_cnn` is the default deployable model.
