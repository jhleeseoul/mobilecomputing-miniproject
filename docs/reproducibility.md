# Reproducibility Runbook (Week1-Week4)

## A. Prepare Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## B. Prepare Data Manifest

```bash
python -m kws.data.prepare --config configs/default_config.json --download
```

Or with local dataset:

```bash
python -m kws.data.prepare --config configs/default_config.json --dataset-root /path/to/speech_commands
```

## C. Train Model

```bash
python -m kws.train --config configs/default_config.json --model small_cnn
```

Note:
- With default config, feature cache files are created on first run at `artifacts/data/feature_cache/`.
- Validation/test cache is reused on later runs for speed.
- Training keeps stochastic augmentation by default (accuracy-friendly), so it remains slower than val/test.

Optional:

```bash
python -m kws.train --config configs/default_config.json --model ds_cnn
```

## D. Evaluate

```bash
python -m kws.eval --config configs/default_config.json --model small_cnn
```

## E. Export TFLite Models

```bash
python -m kws.export.tflite --config configs/default_config.json --model small_cnn
```

## F. Benchmark and Comparison Table

```bash
python -m kws.benchmark --config configs/default_config.json --model small_cnn
```

Review:

- `artifacts/reports/small_cnn/benchmark_report.json`
- `artifacts/reports/small_cnn/quantization_comparison.md`

## G. Android Realtime Inference

```bash
bash scripts/copy_model_to_android_assets.sh artifacts/models/small_cnn_qat/model_int8_qat.tflite
```

Open `android-app` in Android Studio and run the app on emulator/device.
Grant microphone permission and press `Start Listening` in the app.
