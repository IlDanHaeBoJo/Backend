# 🎵 wav2vec2-large-xlsr-korean 3개 클래스 음성 감정 분석 파인튜닝

이 프로젝트는 Hugging Face의 `wav2vec2-large-xlsr-korean` 모델을 사용하여 3개 클래스 음성 감정 분류를 위한 파인튜닝을 수행합니다.

## 📋 목차

- [개요](#개요)
- [감정 클래스](#감정-클래스)
- [데이터 구조](#데이터-구조)
- [설치 및 요구사항](#설치-및-요구사항)
- [사용법](#사용법)
- [데이터 증강 기법](#데이터-증강-기법)
- [하이퍼파라미터 튜닝](#하이퍼파라미터-튜닝)
- [결과 해석](#결과-해석)
- [문제 해결](#문제-해결)

## 🎯 개요

본 프로젝트는 음성에서 감정을 분류하는 모델을 구축합니다. 기존 15개 클래스에서 3개 클래스로 간소화하여 더 높은 정확도와 일반화 성능을 목표로 합니다.

### 주요 특징

- ✨ **사전 훈련된 모델**: `kresnik/wav2vec2-large-xlsr-korean` 사용
- 🎯 **3개 클래스 분류**: Anxious, Dry, Kind
- 🔄 **고급 데이터 증강**: SpecAugment, MixUp, CutMix 등
- 📊 **일반화 성능 최적화**: 드롭아웃, 가중치 감쇠, 라벨 스무딩
- 🚀 **간편한 실행**: 원클릭 실행 스크립트 제공

## 🏷️ 감정 클래스

| 파일명 번호 | 감정 라벨 | 설명 |
|-------------|-----------|------|
| 000021-000030 | **Anxious** | 불안, 초조함 |
| 000031-000040 | **Kind** | 친절함, 따뜻함 |
| 000091-000100 | **Dry** | 건조함, 무미건조함 |

## 📁 데이터 구조

```
/data/ghdrnjs/SER/small/
├── F2001/
│   └── wav_48000/
│       ├── F2001_000001.wav
│       ├── F2001_000021.wav  # Anxious
│       ├── F2001_000031.wav  # Kind
│       ├── F2001_000091.wav  # Dry
│       └── ... (총 160개 파일)
├── F2002/
│   └── wav_48000/
│       ├── F2002_000021.wav  # Anxious
│       └── ... (동일 패턴)
├── F2003/
├── M2001/
├── M2002/
└── ... (다양한 person 폴더)
```

## 🛠️ 설치 및 요구사항

### 시스템 요구사항

- Python 3.8+
- CUDA 11.0+ (GPU 사용 시)
- 메모리: 16GB+ RAM 권장
- GPU: 8GB+ VRAM 권장

### 패키지 설치

```bash
# 필수 패키지 설치
pip install -r requirements.txt

# 또는 개별 설치
pip install torch torchaudio transformers librosa scikit-learn pandas numpy
```

### GPU 설정 확인

```bash
python -c "import torch; print(f'CUDA 사용 가능: {torch.cuda.is_available()}')"
```

## 🚀 사용법

### 1. 기본 실행

```bash
# 기본 설정으로 파인튜닝 실행
python run_finetune.py
```

### 2. 데이터 로딩 테스트

```bash
# 실제 훈련 전 데이터 확인
python run_finetune.py --test
```

### 3. 커스텀 설정으로 실행

```bash
# 하이퍼파라미터 조정
python run_finetune.py \
    --epochs 20 \
    --batch_size 8 \
    --lr 1e-5 \
    --weight_decay 0.1
```

### 4. 고급 옵션

```bash
# 데이터 경로 및 분할 비율 조정
python run_finetune.py \
    --data_dir /custom/data/path \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --no_augmentation
```

### 5. GPU 설정

```bash
# 특정 GPU 사용
python run_finetune.py --gpu 0

# CPU만 사용
python run_finetune.py --no_cuda
```

## 🔄 데이터 증강 기법

본 프로젝트는 일반화 성능 향상을 위해 다양한 데이터 증강 기법을 제공합니다:

### 기본 증강 기법

1. **노이즈 추가** (AddNoise)
   - 가우시안 노이즈 추가
   - SNR 기반 노이즈 레벨 조정

2. **시간 스트레칭** (TimeStretch)
   - 0.8~1.2배 속도 변경
   - Phase vocoder 사용

3. **피치 시프트** (PitchShift)
   - ±2 반음 피치 변경
   - librosa 기반 구현

4. **볼륨 변경** (VolumeChange)
   - 0.5~1.5배 볼륨 조정

### 고급 증강 기법

1. **SpecAugment**
   - 주파수/시간 마스킹
   - SOTA 음성 인식 기법

2. **MixUp**
   - 두 오디오 샘플 선형 결합
   - 부드러운 라벨 처리

3. **CutMix**
   - 오디오 일부 교체
   - 지역적 특성 학습

4. **RandomErase**
   - 랜덤 영역 마스킹
   - 과적합 방지

### 증강 적용 확률 조정

```python
# preprocessing.py에서 확률 조정
# 50% 확률로 첫 번째 증강
# 20% 확률로 두 번째 증강 (조합)
```

## ⚙️ 하이퍼파라미터 튜닝

### 기본 설정 (일반화 성능 최적화)

```python
# config.py
learning_rate = 2e-5          # 안정적 학습을 위한 낮은 학습률
num_epochs = 10               # 충분한 학습을 위한 에포크
weight_decay = 0.05           # 과적합 방지
dropout_rate = 0.3            # 드롭아웃 강화
label_smoothing = 0.1         # 라벨 스무딩
```

### 성능 향상을 위한 권장 설정

| 상황 | 학습률 | 에포크 | 배치 크기 | 가중치 감쇠 |
|------|--------|--------|-----------|-------------|
| **데이터 부족** | 1e-5 | 20+ | 2-4 | 0.1 |
| **데이터 충분** | 2e-5 | 10-15 | 4-8 | 0.05 |
| **과적합 발생** | 1e-5 | 조기 종료 | 작게 | 0.1+ |
| **학습 불안정** | 5e-6 | 더 많이 | 작게 | 0.05 |

### 클래스 불균형 처리

```python
# 자동 클래스 가중치 계산
from sklearn.utils.class_weight import compute_class_weight

# balanced 가중치 자동 적용
# 소수 클래스에 높은 가중치 부여
```

## 📊 결과 해석

### 훈련 중 모니터링

```
🎯 최종 결과:
   - 정확도: 0.8500
   - F1 스코어: 0.8320
   - 손실: 0.4231
```

### 성능 지표 해석

- **정확도**: 전체 예측 중 맞춘 비율
- **F1 스코어**: 정밀도와 재현율의 조화평균
- **손실**: 모델의 예측 오차 (낮을수록 좋음)

### 클래스별 성능 확인

```python
# 혼동 행렬로 클래스별 성능 확인
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred, target_names=['Anxious', 'Dry', 'Kind']))
```

## 🔧 문제 해결

### 일반적인 문제들

#### 1. GPU 메모리 부족

```bash
# 배치 크기 줄이기
python run_finetune.py --batch_size 2

# Mixed precision 사용 (자동 적용됨)
# fp16=True가 기본 설정
```

#### 2. 데이터 로딩 실패

```bash
# 데이터 경로 확인
python run_finetune.py --test

# 권한 문제 해결
chmod -R 755 /data/ghdrnjs/SER/small/
```

#### 3. 과적합 현상

```bash
# 정규화 강화
python run_finetune.py \
    --weight_decay 0.1 \
    --epochs 15
```

#### 4. 학습 불안정

```bash
# 학습률 낮추기
python run_finetune.py \
    --lr 1e-5 \
    --batch_size 2
```

### 디버깅 팁

1. **데이터 분포 확인**
   ```bash
   python run_finetune.py --test
   ```

2. **GPU 사용량 모니터링**
   ```bash
   nvidia-smi -l 1
   ```

3. **로그 파일 확인**
   ```bash
   tail -f results_3class/training_log.txt
   ```

## 📈 성능 최적화 가이드

### 1. 데이터 품질 개선

- 노이즈가 적은 고품질 오디오 사용
- 일관된 녹음 환경 유지
- 클래스 균형 맞추기

### 2. 모델 튜닝

- 학습률 스케줄링 활용
- Early stopping으로 과적합 방지
- 앙상블 기법 적용

### 3. 하드웨어 최적화

- 충분한 GPU 메모리 확보
- SSD 사용으로 데이터 로딩 속도 개선
- Multi-GPU 사용 (필요시)

## 📝 라이센스

이 프로젝트는 MIT 라이센스를 따릅니다.

## 🤝 기여하기

버그 리포트나 기능 제안은 Issues를 통해 알려주세요.

## 📞 연락처

문의사항이 있으시면 언제든 연락해주세요.

---

**Happy Fine-tuning! 🎉**