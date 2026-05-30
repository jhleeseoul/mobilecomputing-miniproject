# KWS Android Real-Time App 개발 문서

이 문서는 현재 저장소의 구현 사양과 코드 변경 사항을 단일 문서로 정리한 개발 인수인계 문서다.

## 1. 프로젝트 개요

### 1.1 목적
- 스마트폰 온디바이스 키워드 스포팅(Keyword Spotting) 시스템 구현
- 고정 어휘 10개 클래스 분류
- 실시간 마이크 입력 기반 Android 앱 추론 및 시각화

### 1.2 분류 클래스
- 명령어 8개: `yes`, `no`, `up`, `down`, `left`, `right`, `stop`, `go`
- 보조 클래스 2개: `unknown`, `silence`

### 1.3 핵심 기능
- 데이터 준비: Speech Commands 기반 manifest/split 생성
- 오디오 전처리: 1초 클립 정규화 + MFCC/log-mel 특성 추출
- 학습: `small_cnn`(기본), `ds_cnn`(비교용)
- 모델 변환: float32 / int8(TFLite) 변환
- 모델 최적화: PTQ + QAT 경로 제공
- 비교 리포트: 정확도, confusion matrix, 모델 크기, 지연시간
- Android 실시간 추론: 마이크 수집, rolling buffer, confidence UI, latency 표시

## 2. 기술 스택

### 2.1 Python/ML
- Python `3.10+` (현재 WSL/venv 기반)
- TensorFlow `>=2.21,<2.22`
- Keras (TF 내장)
- TensorFlow Model Optimization Toolkit `>=0.8,<0.9` (QAT)
- NumPy, SciPy, scikit-learn, matplotlib
- KaggleHub (데이터셋 다운로드)
- PyYAML (YAML config 지원)
- pytest (테스트)

### 2.2 Android
- Kotlin + Jetpack Compose
- Android Gradle Plugin `8.5.2`
- Kotlin `1.9.24`
- minSdk `26`, target/compileSdk `34`
- TensorFlow Lite Java `2.17.0`

### 2.3 운영 환경
- VS Code + WSL2 + Python venv
- Android Studio (Gradle wrapper 미포함이므로 IDE 빌드 기준)

## 3. 아키텍처 및 폴더 구조

### 3.1 전체 데이터/모델/앱 흐름
1. `kws.data.prepare`가 dataset root를 스캔해 manifest/split/label_map 생성
2. `kws.train`이 manifest 기반 dataset 구성 후 모델 학습
3. `kws.eval`이 Keras 모델 정확도 + confusion matrix 생성
4. `kws.export.tflite`가 float32/int8 TFLite 생성
5. `kws.benchmark`가 float32/int8 추론 정확도/지연 비교 리포트 생성
6. `kws.qat`가 base model 기반 QAT fine-tuning + int8 QAT 모델 생성
7. Android 앱이 `model_int8.tflite`를 로드해 실시간 추론 수행

### 3.2 폴더 구조
```text
.
├── configs/
│   ├── default_config.json
│   └── quant_calib_2000.json
├── kws/
│   ├── data/                # manifest 생성, dataset 구성, representative set
│   ├── models/              # small_cnn, ds_cnn
│   ├── export/              # tflite 변환
│   ├── audio.py             # wav 로드/리샘플/증강
│   ├── features.py          # MFCC/log-mel 추출
│   ├── train.py             # 학습 엔트리
│   ├── eval.py              # 평가 엔트리
│   ├── benchmark.py         # TFLite 벤치마크 엔트리
│   └── qat.py               # QAT 엔트리
├── android-app/
│   └── app/src/main/java/com/example/kwsapp/
│       ├── MainActivity.kt
│       └── inference/
│           ├── KeywordSpotter.kt
│           ├── RealtimeKeywordSpotter.kt
│           ├── MfccExtractor.kt
│           ├── PredictionResult.kt
│           └── WavReader.kt
├── scripts/
│   └── copy_model_to_android_assets.sh
├── tests/
└── docs/
```

### 3.3 주요 파일 역할
- `kws/data/prepare.py`: known/unknown/silence 샘플링 규칙 적용 manifest 작성
- `kws/data/dataset.py`: tf.data 파이프라인, feature cache, augmentation, split 샘플링
- `kws/features.py`: TF signal 기반 MFCC/log-mel 추출 및 정규화
- `kws/train.py`: 모델 컴파일/학습, early stopping/model checkpoint
- `kws/export/tflite.py`: representative dataset 기반 int8 양자화
- `kws/benchmark.py`: float32/int8 동시 벤치마크 + 마크다운 비교표 작성
- `kws/qat.py`: small_cnn base 모델을 tfmot로 QAT fine-tuning
- `android-app/.../KeywordSpotter.kt`: TFLite I/O quant/dequant + softmax + latency
- `android-app/.../RealtimeKeywordSpotter.kt`: AudioRecord 캡처, rolling buffer, 주기 추론
- `android-app/.../MainActivity.kt`: 권한 요청, 실시간 파형/신뢰도/UI 패널 렌더링

## 4. 핵심 기능 및 구현 세부사항

### 4.1 데이터 준비 및 라벨 정책
- `validation_list.txt`, `testing_list.txt`를 기준으로 split 고정
- 라벨 순서 고정: `commands + ["unknown", "silence"]`
- `unknown_ratio`만큼 split별 unknown 샘플링
- `silence_ratio`만큼 silence 샘플 추가
- `_background_noise_`가 존재하면 noise segment(`start_sample`)로 silence 구성

핵심 로직 (`kws/data/prepare.py`):
```python
unknown_target_count = int(len(known_entries) * unknown_ratio)
silence_count = int(current_count * silence_ratio)
```

### 4.2 데이터셋 생성, 증강, 캐시
- 입력 길이: `sample_rate * clip_duration_ms` (기본 16000 x 1초)
- 증강: time shift, volume scale, noise mix
- split별 seed 오프셋으로 재현성 확보
- `max_items` 제한 시 무작위 샘플링(클래스 바이어스 방지)
- feature cache 사용 시 `.npy` 메모리맵으로 반복 실행 속도 개선
- train split은 기본적으로 `cache_train_augmented_features=false` 설정으로 매 epoch 증강 다양성 유지

핵심 로직 (`kws/data/dataset.py`):
```python
if max_items is not None and max_items < len(filtered):
    rng = np.random.default_rng(_split_seed(seed, split) + 101)
    indices = rng.choice(len(filtered), size=max_items, replace=False)
```

### 4.3 특성 추출
- Python: `tf.signal.stft -> mel filter bank -> log -> mfcc`
- Android: Kotlin 직접 구현(`MfccExtractor.kt`), FFT/DCT 포함
- normalize 옵션 기본 활성화
- Tensor 경로/NumPy 경로 모두 제공 (`extract_features_tensor`, `extract_features_np`)

### 4.4 모델 학습
- 모델 선택: `small_cnn` 또는 `ds_cnn`
- 손실함수: `SparseCategoricalCrossentropy`
- 옵티마이저: Adam
- `steps_per_execution`를 dataset cardinality에 맞게 자동 clamp
- 산출물:
  - `best_model.keras`
  - `last_model.keras`
  - `history.json`
  - `train_summary.json`
  - `model_summary.txt`

### 4.5 평가/변환/벤치마크
- `kws.eval`:
  - test split 예측
  - `eval_report.json`
  - confusion matrix CSV/PNG
- `kws.export.tflite`:
  - `model_float32.tflite`
  - `model_int8.tflite` (representative dataset 기반)
- `kws.benchmark`:
  - TFLite interpreter로 샘플별 지연시간 측정
  - avg/p95 latency, accuracy, size, delta 산출
  - `quantization_comparison.md` 자동 생성

### 4.6 QAT 구현
- 진입점: `python -m kws.qat`
- 대상 모델: 현재 `small_cnn` 전용
- 절차:
  1. base `tf.keras` 모델 로드
  2. 동일 구조를 `tf_keras`로 재구성
  3. Conv2D/Dense에 selective quantize annotation
  4. `quantize_apply` 후 짧은 fine-tuning
  5. int8 QAT TFLite 생성 + 벤치마크

핵심 코드 (`kws/qat.py`):
```python
def annotate(layer):
    if isinstance(layer, (keras.layers.Conv2D, keras.layers.Dense)):
        return tfmot.quantization.keras.quantize_annotate_layer(layer)
    return layer
```

### 4.7 Android 실시간 추론
- `KeywordSpotter.predict(featureOrAudio)` 인터페이스 유지
- `PredictionResult(topLabel, topScore, scores, latencyMs)` 반환
- `RealtimeKeywordSpotter`:
  - `AudioRecord`(16kHz, mono, PCM16) 시작
  - 1초 rolling buffer 유지 (`windowSamples=16000`)
  - `inferenceIntervalMs=320` 주기 추론
  - 최근 `smoothingWindow=3` 평균으로 점수 안정화
  - `StateFlow<StreamingInferenceState>`로 UI 반영
- `MainActivity`:
  - `RECORD_AUDIO` 런타임 권한 요청
  - waveform 캔버스 렌더링
  - class confidence progress bar
  - timestamp/latency 표시
  - `go/stop/up/down/left/right/yes/no` 데모 패널

### 4.8 구현 변경 이력 (커밋 단위)
- `ac3e70c`: Python KWS 파이프라인 전체(prepare/train/eval/export/benchmark/QAT/test)
- `008f3db`: Android 실시간 추론 앱(마이크/롤링 버퍼/UI) + 모델 복사 스크립트
- `9f114be`: 문서/재현 가이드 업데이트

## 5. 실행 및 빌드 방법

### 5.1 Python 환경 변수 및 의존성
- 필수 환경 변수는 없음
- 권장(선택):
```bash
export PYTHONHASHSEED=42
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=-1
```

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 5.2 Python 파이프라인 실행
```bash
# 1) 데이터 준비
python -m kws.data.prepare --config configs/default_config.json --download

# 2) 학습
python -m kws.train --config configs/default_config.json --model small_cnn --epochs 20

# 3) 평가
python -m kws.eval --config configs/default_config.json --model small_cnn

# 4) TFLite 변환
python -m kws.export.tflite --config configs/default_config.json --model small_cnn

# 5) 벤치마크
python -m kws.benchmark --config configs/default_config.json --model small_cnn

# 6) QAT (선택)
python -m kws.qat --config configs/default_config.json --base-model small_cnn --qat-model-name small_cnn_qat --qat-epochs 3 --qat-learning-rate 1e-5
```

### 5.3 테스트 실행
```bash
source .venv/bin/activate
pytest -q
```

### 5.4 Android 빌드/실행
1. 모델 복사:
```bash
bash scripts/copy_model_to_android_assets.sh artifacts/models/small_cnn_qat/model_int8_qat.tflite
```
2. Android Studio에서 `android-app` 열기
3. Gradle sync 후 디바이스/에뮬레이터 실행
4. 앱에서 마이크 권한 허용 후 `Start Listening` 클릭

참고:
- 저장소에 `gradlew` wrapper가 포함되어 있지 않음
- CLI 빌드가 필요하면 로컬 Gradle 설치 또는 Android Studio 실행 환경 사용

### 5.5 주요 산출물 경로
- 데이터: `artifacts/data/`
- 모델: `artifacts/models/small_cnn/`, `artifacts/models/small_cnn_qat/`
- 리포트: `artifacts/reports/small_cnn/`, `artifacts/reports/small_cnn_qat/`
- Android 자산: `android-app/app/src/main/assets/model_int8.tflite`

## 6. 이슈 및 해결 과정 (Troubleshooting)

### 6.1 낮은 정확도 문제
문제:
- `--max-items` 스모크 실행 시 앞부분 데이터만 잘리는 방식으로 샘플 바이어스 발생

해결:
- split 내부 무작위 샘플링으로 변경 (`kws/data/dataset.py`, `kws/benchmark.py`)
- 결과: 클래스 분포 왜곡 감소, 학습/평가 일관성 개선

### 6.2 CPU 학습 속도 저하
문제:
- 매 epoch마다 전체 샘플 특성 재계산으로 병목 발생

해결:
- feature cache(`artifacts/data/feature_cache`) 도입
- validation/test 캐시는 기본 활성화
- train cache는 정확도 보존을 위해 기본 비활성(`cache_train_augmented_features=false`)
- 비캐시 경로도 배치 텐서 기반 특성 추출(`extract_features_tensor`)로 최적화

### 6.3 PTQ int8 정확도 손실
문제:
- PTQ int8 정확도 하락폭이 목표치 대비 큼

해결:
- QAT 파이프라인(`kws/qat.py`) 추가
- 현재 artifact 기준:
  - float32 TFLite accuracy: `0.9088`
  - PTQ int8 accuracy: `0.8313`
  - QAT int8 accuracy: `0.9271`
  - QAT int8 size: `63.8KB` (float32 `214.3KB`)

### 6.4 tfmot/Keras 호환성 문제
문제:
- `tensorflow-model-optimization` 사용 시 `keras`/`tf_keras` 로더 충돌
- BatchNorm 포함 전체 모델 양자화 시 오류 가능

해결:
- base 모델을 `tf.keras`로 먼저 로드한 뒤 tfmot import
- `tf_keras`로 동일 구조 재구성 후 가중치 이식
- Conv2D/Dense만 selective quantization 적용

### 6.5 경고 메시지 해석
다음 경고는 현재 동작에 치명적이지 않음:
- `Could not find cuda drivers` / `GPU will not be used`
- `TF-TRT Warning: Could not find TensorRT`
- `WavFileWarning: Chunk (non-data) not understood, skipping it`
- `tf.lite.Interpreter is deprecated` (향후 LiteRT 마이그레이션 필요)

### 6.6 Android 관련 주의사항
- 마이크 권한 미허용 시 실시간 추론 시작 불가
- 에뮬레이터 마이크 환경에 따라 실시간 데모 품질이 낮을 수 있음
- 실제 디바이스에서 latency 및 명령 반응성 검증 권장

