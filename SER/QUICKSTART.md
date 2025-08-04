# 🚀 빠른 시작 가이드

## 💨 1분 만에 시작하기

### 1단계: 데이터 확인
```bash
cd CPX/Backend/SER
python run_finetune.py --test
```

### 2단계: 훈련 시작
```bash
# 방법 1: 간단한 실행
python run_finetune.py

# 방법 2: 메뉴 방식
./start_training.sh
```

## 📂 필요한 데이터 구조

```
/data/ghdrnjs/SER/small/
├── F2001/wav_48000/
│   ├── F2001_000021.wav → Anxious
│   ├── F2001_000031.wav → Kind
│   ├── F2001_000091.wav → Dry
│   └── ... (총 160개 파일)
├── F2002/wav_48000/ (동일 패턴)
├── M2001/wav_48000/ (동일 패턴)
└── ... (person별 폴더)
```

**파일명 번호별 감정:**
- 000021-000030: Anxious
- 000031-000040: Kind  
- 000091-000100: Dry

## ⚡ 빠른 테스트

```bash
# 데이터만 확인
python run_finetune.py --test

# 빠른 훈련 (3 에포크)
python run_finetune.py --epochs 3 --batch_size 2
```

## 🎯 추천 설정

### 일반적인 환경
```bash
python run_finetune.py --epochs 15 --batch_size 4 --lr 2e-5
```

### GPU 메모리 부족 시
```bash
python run_finetune.py --epochs 20 --batch_size 2 --lr 1e-5
```

### 최고 성능 추구
```bash
python run_finetune.py --epochs 25 --batch_size 8 --lr 1e-5 --weight_decay 0.1
```

## 📊 예상 결과

- **정확도**: 80-90%
- **훈련 시간**: 30분-2시간 (데이터 크기에 따라)
- **모델 크기**: ~1.2GB

## 🆘 문제 해결

| 문제 | 해결책 |
|------|--------|
| GPU 메모리 부족 | `--batch_size 2` |
| 데이터 없음 | 경로 확인 `/data/ghdrnjs/SER/small/` |
| 너무 느림 | `--no_augmentation` |
| 정확도 낮음 | `--epochs 30 --weight_decay 0.1` |

## 🎉 성공적인 실행 후

1. `results_3class/final_model/` 폴더에 모델 저장됨
2. 정확도 80% 이상이면 성공!
3. 추론 테스트 결과 확인

---

**질문이 있으시면 README_3class.md를 참고하세요!**