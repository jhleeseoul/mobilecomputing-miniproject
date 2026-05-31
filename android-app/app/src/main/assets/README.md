# Android Assets

Place exported TFLite model here before running inference on device:

- `model_int8.tflite` (preferred, currently `artifacts/models/small_cnn_qat/model_int8_qat.tflite`)
- or `model_float32.tflite` and update `MainActivity` model path accordingly

`sample_audio/` files are kept only for optional offline smoke checks.

Sync command:

```bash
cp artifacts/models/small_cnn_qat/model_int8_qat.tflite android-app/app/src/main/assets/model_int8.tflite
```
