"""
3개 클래스 음성 감정 분석 파인튜닝 스크립트
- Anxious (21-30 폴더)
- Kind (31-40 폴더) 
- Dry (91-100 폴더)
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # 패키지로 실행될 때
    from .model import create_model
    from .data_loader import load_dataset_from_numbered_folders, create_data_splits, create_dataloaders
    from .trainer import create_trainer
    from .config import model_config, training_config, data_config
    from .preprocessing import preprocessor
except ImportError:
    # 직접 실행될 때
    from model import create_model
    from data_loader import load_dataset_from_numbered_folders, create_data_splits, create_dataloaders
    from trainer import create_trainer
    from config import model_config, training_config, data_config
    from preprocessing import preprocessor

def setup_environment(seed=42):
    """환경 설정"""
    # 랜덤 시드 설정
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # GPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  사용 디바이스: {device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    return device

def print_data_summary(audio_paths, labels):
    """데이터 요약 정보 출력"""
    from collections import Counter
    label_counts = Counter(labels)
    
    print(f"\n{'='*50}")
    print(f"📊 데이터셋 요약")
    print(f"{'='*50}")
    print(f"총 파일 수: {len(audio_paths):,}개")
    print(f"\n클래스별 분포:")
    for label, count in label_counts.items():
        percentage = (count / len(audio_paths)) * 100
        print(f"  {label}: {count:,}개 ({percentage:.1f}%)")
    
    # 클래스 균형 확인
    min_count = min(label_counts.values())
    max_count = max(label_counts.values())
    imbalance_ratio = max_count / min_count
    print(f"\n클래스 불균형 비율: {imbalance_ratio:.2f}")
    if imbalance_ratio > 2.0:
        print("⚠️  클래스 불균형이 있습니다. 가중치 조정을 고려해보세요.")

def create_class_weights(labels):
    """클래스 가중치 계산 (불균형 데이터 대응)"""
    from collections import Counter
    from sklearn.utils.class_weight import compute_class_weight
    
    label_counts = Counter(labels)
    classes = list(label_counts.keys())
    class_weights = compute_class_weight(
        'balanced',
        classes=np.array(classes),
        y=np.array(labels)
    )
    
    weight_dict = {classes[i]: class_weights[i] for i in range(len(classes))}
    print(f"\n🏋️  클래스 가중치:")
    for label, weight in weight_dict.items():
        print(f"  {label}: {weight:.3f}")
    
    return weight_dict

def main():
    """메인 실행 함수"""
    
    print("🎵 wav2vec2-large-xlsr-korean 3개 클래스 감정 분석 파인튜닝")
    print("="*70)
    
    # 1. 환경 설정
    device = setup_environment()
    
    # 2. 데이터 경로 설정
    data_dir = "/data/ghdrnjs/SER/small/"
    output_dir = "./results_3class"
    
    print(f"📁 데이터 디렉토리: {data_dir}")
    print(f"💾 결과 저장 경로: {output_dir}")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. 데이터 로드
    print(f"\n📂 파일명 번호 기반으로 데이터 로드 중...")
    print(f"   - 000021-000030 → Anxious")
    print(f"   - 000031-000040 → Kind")
    print(f"   - 000091-000100 → Dry")
    print(f"   - 데이터 구조: [PERSON]/wav_48000/[PERSON]_[6자리번호].wav")
    
    audio_paths, labels = load_dataset_from_numbered_folders(data_dir)
    
    if len(audio_paths) == 0:
        print("❌ 데이터를 찾을 수 없습니다. 경로를 확인해주세요.")
        return
    
    # 4. 데이터 요약 출력
    print_data_summary(audio_paths, labels)
    
    # 5. 클래스 가중치 계산
    class_weights = create_class_weights(labels)
    
    # 6. 데이터 분할
    print(f"\n🔀 데이터 분할 중...")
    train_data, val_data, test_data = create_data_splits(
        audio_paths, labels,
        train_ratio=0.7,    # 훈련용 70%
        val_ratio=0.15,     # 검증용 15%
        test_ratio=0.15,    # 테스트용 15%
        random_state=42
    )
    
    # 7. 모델 생성
    print(f"\n🤖 모델 생성 중...")
    print(f"   - 모델: {model_config.model_name}")
    print(f"   - 클래스 수: {model_config.num_labels}")
    print(f"   - 감정 라벨: {model_config.emotion_labels}")
    
    model = create_model()
    model.to(device)
    
    # 8. 데이터로더 생성
    print(f"\n🔄 데이터로더 생성 중...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data, model.processor
    )
    
    # 9. 훈련 설정 업데이트
    training_config.output_dir = output_dir
    training_config.num_epochs = 15  # 3개 클래스이므로 충분한 에포크
    
    print(f"\n⚙️  훈련 설정:")
    print(f"   - 에포크: {training_config.num_epochs}")
    print(f"   - 배치 크기: {training_config.batch_size}")
    print(f"   - 학습률: {training_config.learning_rate}")
    print(f"   - 가중치 감쇠: {training_config.weight_decay}")
    print(f"   - 드롭아웃: {training_config.dropout_rate}")
    print(f"   - 라벨 스무딩: {training_config.label_smoothing}")
    
    # 10. 훈련기 생성
    print(f"\n🏋️  훈련기 생성 중...")
    trainer = create_trainer(model, class_weights=class_weights)
    
    # 11. 훈련 시작
    print(f"\n🚀 훈련 시작!")
    print("="*50)
    
    try:
        # 훈련 실행
        trainer.train(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            output_dir=output_dir
        )
        
        print(f"\n✅ 훈련 완료!")
        
        # 12. 최종 평가
        print(f"\n📊 최종 평가 중...")
        eval_results = trainer.evaluate()
        
        if eval_results:
            print(f"\n🎯 최종 결과:")
            print(f"   - 정확도: {eval_results.get('eval_accuracy', 'N/A'):.4f}")
            print(f"   - F1 스코어: {eval_results.get('eval_f1', 'N/A'):.4f}")
            print(f"   - 손실: {eval_results.get('eval_loss', 'N/A'):.4f}")
        
        # 13. 모델 저장
        model_save_path = os.path.join(output_dir, "final_model")
        model.save_model(model_save_path)
        print(f"\n💾 모델 저장 완료: {model_save_path}")
        
        # 14. 간단한 추론 테스트
        print(f"\n🧪 추론 테스트...")
        if len(test_data[0]) > 0:
            test_audio_path = test_data[0][0]
            predicted_emotion = model.predict(test_audio_path)
            actual_emotion = test_data[1][0]
            print(f"   - 테스트 파일: {os.path.basename(test_audio_path)}")
            print(f"   - 예측 감정: {predicted_emotion}")
            print(f"   - 실제 감정: {actual_emotion}")
            print(f"   - 결과: {'✅ 정확' if predicted_emotion == actual_emotion else '❌ 틀림'}")
        
        print(f"\n🎉 모든 과정이 성공적으로 완료되었습니다!")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  훈련이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 훈련 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

def test_data_loading():
    """데이터 로딩 테스트 함수"""
    data_dir = "/data/ghdrnjs/SER/small/"
    print(f"📂 데이터 로딩 테스트: {data_dir}")
    print(f"   파일명 패턴: [PERSON]_[6자리번호].wav")
    print(f"   감정 매핑: 000021-030(Anxious), 000031-040(Kind), 000091-100(Dry)")
    
    audio_paths, labels = load_dataset_from_numbered_folders(data_dir)
    print_data_summary(audio_paths, labels)
    
    if len(audio_paths) > 0:
        print(f"\n📄 첫 5개 파일 예시:")
        for i in range(min(5, len(audio_paths))):
            filename = os.path.basename(audio_paths[i])
            print(f"   {i+1}. {filename} → {labels[i]}")
    else:
        print(f"\n💡 데이터 확인 팁:")
        print(f"   1. 경로가 올바른지 확인: {data_dir}")
        print(f"   2. person 폴더 하위에 wav_48000 폴더가 있는지 확인")
        print(f"   3. 파일명이 [PERSON]_[6자리번호].wav 패턴인지 확인")
        print(f"   4. 번호가 21-30, 31-40, 91-100 범위에 있는지 확인")

if __name__ == "__main__":
    # 명령행 인자로 테스트 모드 실행 가능
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_data_loading()
    else:
        main()