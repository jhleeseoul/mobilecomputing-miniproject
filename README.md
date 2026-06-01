# On-device Real-time Keyword Spotting for Android

This project implements a real-time keyword spotting (KWS) system that runs directly on an Android smartphone. It recognizes short spoken commands from the microphone, converts audio into MFCC features, performs TensorFlow Lite inference on-device, and stabilizes the result for practical real-time interaction.

The final app is designed for low-latency, offline, privacy-preserving command recognition using a compact quantized neural network.

## Project Goals

- Build an end-to-end keyword spotting pipeline from dataset preparation to Android deployment.
- Recognize eight spoken commands: `yes`, `no`, `up`, `down`, `left`, `right`, `stop`, and `go`.
- Include `unknown` and `silence` classes so the model can reject non-command audio.
- Optimize the model for mobile inference with TensorFlow Lite int8 quantization.
- Improve real-time usability with confidence smoothing, trigger thresholds, and command adoption logic.

## Usage Scenario

The target scenario is hands-free or low-touch mobile interaction. A user speaks short commands into the phone, and the Android app immediately displays the recognized command and confidence score. Because inference runs locally, the app does not require network access and does not send microphone audio to a server.

Example use cases:

- Voice command control for simple mobile interfaces.
- Accessibility-oriented interaction where touch input is inconvenient.
- Offline command recognition for embedded or edge AI prototypes.
- Real-time mobile computing demo for audio sensing and model optimization.

## Key Ideas

### MFCC Feature Extraction

Raw waveform audio is not fed directly into the deployed model. Each 1-second audio window is transformed into Mel-frequency cepstral coefficients (MFCCs), which compactly represent speech-related frequency patterns.

- Audio sample rate: `16 kHz`
- Window length: `1 second`
- Default deployed feature shape: `49 x 13 x 1`
- Feature type: MFCC by default, with log-mel support in the Python pipeline

MFCCs reduce input dimensionality while preserving useful speech structure, making them suitable for a small mobile CNN.

### Quantization and QAT

The project compares floating-point and int8 TensorFlow Lite models. Post-training quantization (PTQ) reduces model size but can reduce accuracy. Quantization-aware training (QAT) simulates quantization during training, allowing the model to recover accuracy while keeping the int8 deployment benefits.

The final deployed baseline is `small_cnn_qat`, an int8 QAT model.

### EMA Stabilization

Real-time microphone predictions can fluctuate from frame to frame. The Android app applies exponential moving average (EMA) smoothing to model scores before selecting a command.

- EMA alpha: `0.6`
- Purpose: reduce jitter and make displayed predictions more stable
- Applied inside the real-time inference path before command adoption

### Command Adoption Logic

The app does not accept every top-1 prediction immediately. It uses confidence, margin, command-specific thresholds, cooldown, and rearm logic to avoid noisy repeated triggers.

- Default trigger: top score `>= 0.30`, margin `>= 0.05`
- Sensitive trigger for `down` and `go`: top score `>= 0.24`, margin `>= 0.03`
- `unknown` and `silence` are excluded from command adoption.
- Cooldown prevents rapid duplicate triggers.
- Latch/rearm behavior helps the UI keep a stable accepted command.

## Final Results

The final Android deployment uses `small_cnn_qat` because it provides the best overall balance between accuracy, per-command reliability, size, and latency.

| Model | TFLite Type | Accuracy | `down` Recall | `go` Recall | Size | Avg. Latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `small_cnn_qat` | int8 QAT | `0.9375` | `0.8063` | `0.8446` | `63.8 KB` | `0.1967 ms` |
| `ds_cnn_qat` | int8 QAT | `0.9232` | `0.8498` | `0.8008` | `54.5 KB` | `0.1037 ms` |

Although `ds_cnn_qat` is smaller and faster, it was not selected as the final deployment model because its overall accuracy and `go` recall were lower, and the `down` recall improvement was not large enough to justify switching the baseline.

Earlier quantization experiments also showed why QAT was necessary:

| Model Variant | Accuracy | Size |
| --- | ---: | ---: |
| Float32 TFLite | `0.9088` | `214.3 KB` |
| PTQ int8 | `0.8313` | smaller int8 model |
| Final QAT int8 (`small_cnn_qat`) | `0.9375` | `63.8 KB` |

## App Functionality

The Android app is implemented with Kotlin and Jetpack Compose. It captures microphone audio continuously and performs real-time keyword spotting on-device.

Main app features:

- Microphone capture with `AudioRecord`
- 16 kHz mono PCM audio stream
- Rolling 1-second inference buffer
- Inference every `200 ms`
- Lightweight automatic gain control for quiet speech
- MFCC extraction on Android to match the training pipeline
- TFLite int8 model inference
- EMA-smoothed confidence scores
- Command adoption logic with thresholds and cooldown
- UI for waveform, top prediction, confidence bars, latency, and command status

The UI exposes the eight target commands:

`up`, `down`, `left`, `right`, `go`, `stop`, `yes`, `no`

## Repository Structure

```text
.
|-- configs/
|   `-- default_config.json
|-- kws/
|   |-- data/
|   |-- models/
|   |-- export/
|   |-- audio.py
|   |-- features.py
|   |-- train.py
|   |-- eval.py
|   |-- benchmark.py
|   `-- qat.py
|-- android-app/
|   `-- app/src/main/java/com/example/kwsapp/
|       |-- MainActivity.kt
|       `-- inference/
|           |-- KeywordSpotter.kt
|           |-- RealtimeKeywordSpotter.kt
|           |-- MfccExtractor.kt
|           |-- PredictionResult.kt
|           `-- WavReader.kt
|-- scripts/
|-- tests/
`-- docs/
```

## End-to-end Pipeline

The Python pipeline covers data preparation, model training, evaluation, TensorFlow Lite export, benchmarking, and QAT.

```text
Google Speech Commands
        |
        v
Manifest and label map generation
        |
        v
Audio loading and augmentation
        |
        v
MFCC / log-mel feature extraction
        |
        v
CNN training
        |
        v
Evaluation and reports
        |
        v
TFLite export and int8 quantization
        |
        v
Benchmarking
        |
        v
Android asset deployment
        |
        v
Real-time microphone inference
```

## Setup

Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Core Python dependencies include:

- TensorFlow `>=2.21,<2.22`
- TensorFlow Model Optimization Toolkit `>=0.8,<0.9`
- NumPy, SciPy, scikit-learn, matplotlib
- KaggleHub and PyYAML
- pytest

## Reproducing the Model

Prepare the dataset:

```bash
python -m kws.data.prepare --config configs/default_config.json --download
```

Train the default CNN:

```bash
python -m kws.train --config configs/default_config.json --model small_cnn
```

Evaluate the model:

```bash
python -m kws.eval --config configs/default_config.json --model small_cnn
```

Export TensorFlow Lite models:

```bash
python -m kws.export.tflite --config configs/default_config.json --model small_cnn
```

Benchmark the exported models:

```bash
python -m kws.benchmark --config configs/default_config.json --model small_cnn
```

Run quantization-aware training:

```bash
python -m kws.qat \
  --config configs/default_config.json \
  --base-model small_cnn \
  --qat-model-name small_cnn_qat \
  --qat-epochs 20 \
  --qat-learning-rate 1e-5 \
  --fit-verbose 2
```

Expected generated artifacts:

```text
artifacts/data/manifest.csv
artifacts/data/label_map.json
artifacts/models/<model>/best_model.keras
artifacts/models/<model>/model_float32.tflite
artifacts/models/<model>/model_int8.tflite
artifacts/models/small_cnn_qat/model_int8_qat.tflite
artifacts/reports/<model>/eval_report.json
artifacts/reports/<model>/benchmark_report.json
artifacts/reports/<model>/quantization_comparison.md
```

## Android Deployment

Copy the final QAT model into the Android app assets:

```bash
bash scripts/copy_model_to_android_assets.sh artifacts/models/small_cnn_qat/model_int8_qat.tflite
```

The deployed model should be available at:

```text
android-app/app/src/main/assets/model_int8.tflite
```

Open `android-app/` in Android Studio, connect an Android device or emulator with microphone support, and run the app.

Android stack:

- Kotlin
- Jetpack Compose
- Android Gradle Plugin `8.5.2`
- Kotlin plugin `1.9.24`
- `minSdk 26`
- `compileSdk 34`
- TensorFlow Lite Java `2.17.0`

## Configuration

Most experiment settings are controlled by `configs/default_config.json`.

Important fields:

- `sample_rate`, `clip_duration_ms`
- `frame_length`, `frame_step`, `fft_length`
- `num_mel_bins`, `n_mfcc`, `feature_type`
- `commands`
- `augment.*`
- `train.*`
- `quantization.*`
- `data.use_feature_cache`
- `paths.*`

The Android real-time inference settings are implemented in `MainActivity.kt` and the `inference/` package.

## Tests

Run the Python test suite:

```bash
pytest -q
```

The tests cover:

- deterministic label mapping
- manifest generation with fixed seeds
- class-weight computation
- TensorFlow Lite model loading smoke checks
- audio and feature-processing behavior

Android unit and instrumentation tests are located under the Android app test directories.

## Documentation

Additional project notes are available in `docs/`.

- `docs/reproducibility.md`: reproducibility checklist and experiment notes
- `docs/development_handover.md`: development summary and handover details
- `docs/week4_kws_accuracy_sensitivity_patch.md`: final sensitivity and model-selection patch notes
- `docs/experiment_template.md`: experiment logging template
- `docs/presentation_report.md`: presentation script and content planning notes
- `docs/presentation/`: final presentation-related materials

## Limitations and Future Work

- Accuracy may vary depending on microphone quality, distance, background noise, and speaker accent.
- The real-time thresholds are tuned for the current model and may need retuning after retraining.
- More robust noise augmentation could improve deployment performance in noisy environments.
- A larger speaker-diverse evaluation set would make the final recall estimates more reliable.
- Future work could add wake-word support, personalization, or command-specific calibration in the app.
