# 온디바이스 키워드 스포팅(KWS) 프로젝트 발표 보고서

## 목차
1. 프로젝트 목적
2. 문제 정의와 성공 기준
3. 전체 개발 여정 (Week1~Week4)
4. 시스템 구조를 쉽게 이해하기
5. 시행착오와 해결 과정
6. 모델 선택 의사결정 (`small_cnn` vs `ds_cnn`)
7. 최종 결과
8. 한계와 다음 단계
9. 재현 방법 (부록)

---

## 1) 프로젝트 목적

### 1.1 왜 이 프로젝트를 했는가
이 프로젝트의 목표는 스마트폰에서 인터넷 연결 없이도 음성 명령을 빠르게 인식하는 기능을 구현하는 것이다.  
핵심 키워드는 다음 3가지다.

- 지연시간: 서버 왕복 없이 기기 내부에서 즉시 반응
- 프라이버시: 음성이 외부 서버로 전송되지 않음
- 오프라인 동작: 네트워크가 없어도 동작 가능

### 1.2 최종적으로 만들고자 한 것
- 10개 클래스(명령 8개 + `unknown` + `silence`)를 분류하는 모델
- Android 앱에서 실시간 마이크 입력을 받아 명령을 인식하는 데모
- 모델 학습부터 배포(TFLite int8)까지 이어지는 재현 가능한 파이프라인

---

## 2) 문제 정의와 성공 기준

### 2.1 분류 대상
- 명령어: `yes`, `no`, `up`, `down`, `left`, `right`, `stop`, `go`
- 기타 클래스: `unknown`, `silence`

### 2.2 성공 기준
- 모델 관점
  - 테스트 정확도: 약 0.92 이상 유지
  - int8 양자화 후 정확도 저하 최소화
  - 모델 크기/지연시간이 모바일 배포 가능 수준
- 앱 관점
  - 실시간 명령 반응(데모 패널 이동) 가능
  - `unknown/silence` 오동작 억제
  - 장시간 실행 시 끊김/크래시 없이 동작

### 2.3 Week별 목표
- Week1: 데이터 준비/전처리 기반 구축
- Week2: 학습/평가 자동화
- Week3: TFLite 변환/양자화 + Android 오프라인 추론
- Week4: 실시간 마이크 통합 + 반응성/인식률 개선

---

## 3) 전체 개발 여정 (Week1~Week4)

### Week1: 데이터 파이프라인 구축
- Speech Commands 기반 데이터 준비 스크립트 구현
- 고정 라벨 인덱스(10클래스) 체계 확립
- `unknown`, `silence` 샘플링 규칙 정의
- train/val/test split 재현성 보장(seed 고정)

### Week2: 학습/평가 루프 완성
- `small_cnn` 기본 모델 학습 루프 구현
- `ds_cnn` 비교 모델 추가
- confusion matrix, class-wise 지표 생성
- 증강(shift/noise/volume) 포함한 학습 자동화

### Week3: 배포 모델 준비
- SavedModel/TFLite(float32, int8) 변환
- representative dataset 기반 PTQ 구현
- 정확도/모델크기/지연시간 비교 리포트 자동 생성
- Android Compose 스모크 앱에서 고정 샘플 추론 확인

### Week4: 실시간 통합 및 품질 개선
- 실시간 마이크 입력 + rolling buffer + 주기 추론
- 인식률/반응성 이슈 수정:
  - 중복 softmax 제거
  - 트리거 정책 재설계(임계값 + margin)
  - 실시간 score 집계를 avg3 -> EMA로 전환
  - AGC(자동 이득 보정)로 원거리 발화 대응
- `small_cnn` 재학습 + QAT 재생성
- `ds_cnn` 학습/QAT 비교 실험 후 배포 기준 모델 결정

---

## 4) 시스템 구조를 쉽게 이해하기

## 4.1 한 줄 요약
`음성(1초)` -> `특징(MFCC)` -> `TFLite 모델` -> `점수 안정화(EMA)` -> `명령 채택 규칙` -> `앱 동작`

### 4.2 단계별 설명
1. 마이크에서 16kHz 오디오를 계속 받는다.
2. 최근 1초(16000샘플)를 rolling buffer로 유지한다.
3. 오디오를 MFCC 특징으로 바꿔 모델 입력(49x13x1)을 만든다.
4. TFLite int8 모델이 10개 클래스 점수를 출력한다.
5. 점수는 EMA로 안정화한다(갑작스런 튐 감소).
6. 임계값/마진/쿨다운 조건을 만족하면 명령으로 채택한다.
7. 채택된 명령으로 데모 패널 상태를 업데이트한다.

### 4.3 왜 이 구조가 필요한가
- 원시 점수는 프레임마다 흔들릴 수 있어 바로 쓰면 오동작이 많다.
- EMA와 마진 조건을 쓰면 오탐과 미탐 사이를 균형 있게 맞출 수 있다.
- 모바일 환경에서는 속도와 메모리(할당/GC)를 같이 관리해야 한다.

---

## 5) 시행착오와 해결 과정

## 5.1 문제 1: confidence가 비정상적으로 낮음 (0.1~0.2대)
### 원인
- Android 후처리에서 softmax를 이미 확률인 출력에 다시 적용(중복 softmax)

### 해결
- 출력이 확률분포인지 검사 후, 필요할 때만 softmax 적용
- 결과: 실제 confidence 스케일 복원

## 5.2 문제 2: 앱이 거의 반응하지 않음
### 원인
- 트리거 임계값이 실제 출력 분포와 불일치
- 초기 정책이 보수적이라 미반응 빈도 증가

### 해결
- Balanced 정책으로 재설계
- 이후 `down/go` 약세 대응을 위해 클래스별 임계값 완화
  - 기본: `topScore>=0.30`, `top1-top2>=0.05`
  - `down/go`: `topScore>=0.24`, `top1-top2>=0.03`

## 5.3 문제 3: 실시간 경로의 지연 변동과 효율 저하
### 원인
- 반복 할당과 복사(버퍼/배열)로 GC 부담 가능성

### 해결
- 입력/출력 버퍼 재사용
- 불필요 복사 제거
- 인터프리터 종료 경로 정리

## 5.4 문제 4: 마이크에 매우 가까워야만 인식
### 원인
- 실제 입력 음량 분포와 학습 분포 차이
- 원거리 발화 시 신호 에너지가 낮아 점수 상승이 어려움

### 해결
- 캡처 루프에 경량 AGC 추가
  - `targetRms=0.06`, gain clamp `1.0~4.0`
  - clipping 감지 시 gain 감소
- 학습 증강 강화(저음량/고음량/노이즈 범위 확장)

## 5.5 문제 5: WSL 장시간 실행 시 작업 불안정
### 원인
- 학습 진행 로그가 과도하게 길어 세션/터미널 부담

### 해결
- `train.fit_verbose`, `--fit-verbose` 옵션 추가
- 장시간 실행 시 epoch 단위 로그로 안정화

---

## 6) 모델 선택 의사결정 (`small_cnn` vs `ds_cnn`)

### 6.1 배경 가설
일반적으로 DS-CNN은 모바일 친화적 구조(Depthwise Separable Conv)라 경량화에 유리하다.  
따라서 본 프로젝트에서도 DS-CNN이 배포에 더 적합할 가능성을 검증했다.

### 6.2 실험 결과(이번 프로젝트 기준)
| 모델 | int8 정확도 | down recall | go recall | int8 크기 |
|---|---:|---:|---:|---:|
| `small_cnn_qat` | 0.9375 | 0.8063 | 0.8446 | 63.8 KB |
| `ds_cnn_qat` | 0.9232 | 0.8498 | 0.8008 | 54.5 KB |

### 6.3 왜 이번에는 `small_cnn`이 더 좋았는가
1. 입력 해상도 특성  
   입력이 `49x13x1`로 이미 작다. DS-CNN은 초기에 다운샘플링이 빨라 정보 손실 리스크가 커질 수 있다.

2. 표현력 차이  
   현재 구현의 DS-CNN은 파라미터가 더 작아(noisy/unknown 비중이 큰 10클래스 문제에서) 표현력이 부족했을 가능성이 있다.

3. 구현 제약 영향  
   CPU 호환성 문제로 stride를 `(2,1)`에서 `(2,2)`로 바꿨고, 이 변경이 성능에 불리하게 작용했을 수 있다.

4. 튜닝 성숙도 차이  
   실서비스 경로(임계값, 후처리, 자산 교체)는 `small_cnn_qat` 기준으로 먼저 안정화되었다.

### 6.4 최종 선택
- 배포 기준 모델: `small_cnn_qat`
- DS-CNN: 후보 모델로 유지, 추가 튜닝 후 재평가

---

## 7) 최종 결과

### 7.1 파이프라인 결과
- 데이터 준비 -> 학습 -> 평가 -> 변환 -> 벤치마크 -> Android 반영까지 자동화 완성
- float32/int8 모델 산출 및 비교 리포트 생성 완료

### 7.2 앱 결과
- 실시간 마이크 추론 동작
- command panel 데모 동작(명령 채택 시 이동)
- 오동작 억제 규칙과 반응성 균형 정책 적용

### 7.3 핵심 수치 요약
- `small_cnn` 학습 test accuracy: 약 0.93
- `small_cnn_qat` int8 accuracy: 약 0.94
- 모델 크기: int8 기준 수십 KB 수준
- 실시간 정책 개선 후 미반응 문제 상당 부분 해소

### 7.4 산출물
- Python 패키지형 KWS 학습/배포 코드
- Android 실시간 KWS 데모 앱
- 재현 문서 및 비교 리포트

---

## 8) 한계와 다음 단계

### 8.1 현재 한계
- `down/go`는 환경/발화 습관에 따라 변동성이 상대적으로 큼
- 디바이스별 마이크/오디오 HAL 차이로 성능 편차 발생 가능
- Android MFCC와 Python MFCC 정합은 개선했지만 완전 자동 검증 체계는 아직 미완

### 8.2 다음 단계 (실행 우선순위)
1. 스트리밍 오프라인 시뮬레이터 추가
- `silence->keyword->silence` 시나리오로 검출률/오탐률 자동 측정

2. 디바이스별 오디오 입력 A/B 실험
- AUDIO_SOURCE, AGC 파라미터, threshold를 디바이스군별 비교

3. DS-CNN 재튜닝
- stride/채널 수/블록 수 재설계
- `down/go` 중심 recall 개선 조건 만족 시 재승격 검토

4. 정합 검증 자동화
- Python vs Android feature/probability 허용오차 기반 회귀 테스트 추가

---

## 9) 재현 방법 (부록)

### 9.1 Python
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

python -m kws.data.prepare --config configs/default_config.json --download
python -m kws.train --config configs/default_config.json --model small_cnn --epochs 20
python -m kws.eval --config configs/default_config.json --model small_cnn
python -m kws.export.tflite --config configs/default_config.json --model small_cnn
python -m kws.qat --config configs/default_config.json --base-model small_cnn --qat-model-name small_cnn_qat --qat-epochs 20 --qat-learning-rate 1e-5
python -m kws.benchmark --config configs/default_config.json --model small_cnn
```

### 9.2 Android
```bash
cp artifacts/models/small_cnn_qat/model_int8_qat.tflite android-app/app/src/main/assets/model_int8.tflite
```
- Android Studio에서 `android-app` 실행
- 마이크 권한 허용 후 `Start Listening`

---

## 발표용 핵심 메시지 요약
- 이 프로젝트는 "모바일에서 바로 동작하는 음성 명령 인식"을 실제로 끝까지 연결한 사례다.
- 단순히 모델 정확도만 본 것이 아니라, 실시간 제품 동작(반응성/오탐/미탐/지연)을 같이 최적화했다.
- DS-CNN이 이론적으로 경량화에 유리해도, 실제 입력 조건과 구현 제약에서는 `small_cnn_qat`가 더 나은 선택이 될 수 있음을 실험으로 확인했다.
