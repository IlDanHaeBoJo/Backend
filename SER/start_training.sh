#!/bin/bash

# 3개 클래스 음성 감정 분석 파인튜닝 실행 스크립트

echo "🎵 wav2vec2-large-xlsr-korean 3개 클래스 감정 분석 파인튜닝"
echo "=================================================================="

# Python 환경 확인
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3가 설치되어 있지 않습니다."
    exit 1
fi

# GPU 확인
echo "🔍 시스템 환경 확인 중..."
python3 -c "
import torch
print(f'🐍 Python: {torch.version.__version__}')
print(f'🔥 PyTorch: {torch.__version__}')
print(f'🎮 CUDA 사용 가능: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'🎮 GPU 개수: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'   GPU {i}: {torch.cuda.get_device_name(i)}')
"

echo ""

# 메뉴 표시
echo "📋 실행 옵션을 선택하세요:"
echo "1. 데이터 로딩 테스트"
echo "2. 기본 설정으로 훈련 시작"
echo "3. 커스텀 설정으로 훈련 시작"
echo "4. 종료"

read -p "선택 (1-4): " choice

case $choice in
    1)
        echo "🧪 데이터 로딩 테스트 실행 중..."
        python3 run_finetune.py --test
        ;;
    2)
        echo "🚀 기본 설정으로 훈련 시작..."
        python3 run_finetune.py
        ;;
    3)
        echo "⚙️ 커스텀 설정 입력:"
        
        read -p "에포크 수 (기본값: 15): " epochs
        epochs=${epochs:-15}
        
        read -p "배치 크기 (기본값: 4): " batch_size
        batch_size=${batch_size:-4}
        
        read -p "학습률 (기본값: 2e-5): " learning_rate
        learning_rate=${learning_rate:-2e-5}
        
        read -p "가중치 감쇠 (기본값: 0.05): " weight_decay
        weight_decay=${weight_decay:-0.05}
        
        read -p "특정 GPU 사용 (예: 0, 비어두면 자동): " gpu
        
        echo ""
        echo "🚀 커스텀 설정으로 훈련 시작..."
        echo "   - 에포크: $epochs"
        echo "   - 배치 크기: $batch_size"
        echo "   - 학습률: $learning_rate"
        echo "   - 가중치 감쇠: $weight_decay"
        
        cmd="python3 run_finetune.py --epochs $epochs --batch_size $batch_size --lr $learning_rate --weight_decay $weight_decay"
        
        if [ ! -z "$gpu" ]; then
            cmd="$cmd --gpu $gpu"
            echo "   - 사용 GPU: $gpu"
        fi
        
        echo ""
        eval $cmd
        ;;
    4)
        echo "👋 종료합니다."
        exit 0
        ;;
    *)
        echo "❌ 잘못된 선택입니다. 1-4 중에서 선택해주세요."
        exit 1
        ;;
esac

echo ""
echo "✨ 실행 완료!"