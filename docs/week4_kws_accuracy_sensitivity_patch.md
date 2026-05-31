# Week4 KWS Accuracy/Sensitivity Patch Report

## Scope
- Android app only: realtime decision policy and microphone sensitivity path.
- Python pipeline: robust retraining and QAT refresh for `small_cnn`, plus `ds_cnn` comparison run.

## Applied Patches
1. Android realtime policy
- `RealtimeKeywordSpotter`: score aggregation changed from window average to `EMA(alpha=0.6)`.
- Inference interval changed `320ms -> 200ms`.
- Command acceptance score now uses stabilized EMA scores.

2. Android microphone sensitivity
- Added lightweight AGC in capture loop.
- Parameters:
  - `targetRms=0.06`
  - gain clamp `1.0 ~ 4.0`
  - clipping-aware gain suppression.

3. Android trigger thresholds (class-dependent)
- Default classes:
  - `topScore >= 0.30`
  - `top1-top2 >= 0.05`
- Relaxed classes (`down`, `go`):
  - `topScore >= 0.24`
  - `top1-top2 >= 0.03`
- `unknown/silence` excluded, cooldown `350ms` 유지.

4. Python training/QAT robustness
- Augment update:
  - `volume_min: 0.3`
  - `volume_max: 1.6`
  - `noise_level_max: 0.2`
- Added `train.class_weights` support and applied to both standard training and QAT.
- Added `train.fit_verbose` and CLI `--fit-verbose` for stable long-run execution logs.

5. DS-CNN QAT support
- QAT path extended to support `ds_cnn`.
- Added `DepthwiseConv2D` annotation in QAT.
- Fixed CPU kernel compatibility in DS-CNN stride (`(2,1) -> (2,2)`).

## Key Output Metrics
Dataset/test split: `artifacts/data/manifest.csv` 기준.

### `small_cnn_qat` (current deploy baseline)
- int8 QAT accuracy: `0.9375`
- int8 QAT down recall: `0.8063` (204/253)
- int8 QAT go recall: `0.8446` (212/251)
- int8 size: `63.8 KB`
- int8 avg latency: `0.1967 ms` (local benchmark env)

### `ds_cnn_qat` (comparison candidate)
- int8 QAT accuracy: `0.9232`
- int8 QAT down recall: `0.8498` (215/253)
- int8 QAT go recall: `0.8008` (201/251)
- int8 size: `54.5 KB`
- int8 avg latency: `0.1037 ms` (local benchmark env)

## Adoption Decision (per plan criteria)
Plan criteria:
- `down` recall: `small_cnn_qat` 대비 `+5%p` 이상
- streaming detection average: `+5%p` 이상
- size/latency increase: 20% 이내

Result:
- `down` recall 개선폭: `+4.35%p` (`0.8063 -> 0.8498`)로 기준 미달
- `go` recall은 하락 (`0.8446 -> 0.8008`)
- 정확도도 `small_cnn_qat`가 더 높음 (`0.9375 > 0.9232`)

Decision:
- 배포 기준 모델은 `small_cnn_qat` 유지.
- `ds_cnn_qat`는 후보 실험 결과로 보관.

## Files Changed (this patch line)
- Android:
  - `android-app/app/src/main/java/com/example/kwsapp/inference/RealtimeKeywordSpotter.kt`
  - `android-app/app/src/main/java/com/example/kwsapp/MainActivity.kt`
- Python/config:
  - `kws/train.py`
  - `kws/qat.py`
  - `kws/models/ds_cnn.py`
  - `kws/class_weights.py`
  - `configs/default_config.json`
  - `configs/quant_calib_2000.json`
  - `tests/test_class_weights.py`
  - `tests/conftest.py`

## Current Android Asset
- `android-app/app/src/main/assets/model_int8.tflite`
- Source synced from:
  - `artifacts/models/small_cnn_qat/model_int8_qat.tflite`
