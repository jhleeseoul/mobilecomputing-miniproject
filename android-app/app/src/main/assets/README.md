# Android Assets

Place exported TFLite model here before running inference on device:

- `model_int8.tflite` (preferred, currently QAT int8)
- or `model_float32.tflite` and update `MainActivity` model path accordingly

`sample_audio/` files are kept only for optional offline smoke checks.
