"""
SER 모듈 사용 예시
"""

import os
import sys

# SER 모듈을 사용하기 위해 Backend 디렉토리를 Python path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SER.config import model_config, training_config, data_config
from SER.model import create_model
from SER.trainer import create_trainer
from SER.inference import create_inference_engine
from SER.data_loader import load_dataset_from_directory, create_data_splits
from SER.utils import setup_logging, print_system_info

def example_training():
    """훈련 예시"""
    
    print("=" * 60)
    print("📚 음성 감정 분석 모델 훈련 예시")
    print("=" * 60)
    
    # 로깅 설정
    logger = setup_logging(log_level="INFO")
    
    # 시스템 정보 출력
    print_system_info()
    
    # 설정 확인
    print(f"🎯 사용할 모델: {model_config.model_name}")
    print(f"📊 감정 클래스 수: {model_config.num_labels}")
    print(f"🎵 샘플링 레이트: {model_config.sampling_rate}Hz")
    print(f"⏱️ 최대 길이: {model_config.max_duration}초")
    print(f"💾 모델 크기: 317M 파라미터 (kresnik 한국어 ASR 모델)")
    print(f"🏆 원본 성능: WER 4.74%, CER 1.78% (Zeroth-Korean)")
    print()
    
    # 데이터 경로 설정 (실제 경로로 변경 필요)
    data_directory = "./data/emotions"  # 실제 데이터 디렉토리 경로
    
    if not os.path.exists(data_directory):
        print(f"⚠️ 데이터 디렉토리를 찾을 수 없습니다: {data_directory}")
        print("📝 실제 데이터를 준비한 후 경로를 수정해주세요.")
        print("\n예상 구조:")
        print("data/emotions/")
        print("├── Neutral/")
        print("│   ├── file1.wav")
        print("│   └── file2.wav")
        print("├── Angry/")
        print("│   ├── file3.wav")
        print("│   └── file4.wav")
        print("└── ...")
        return
    
    try:
        # 데이터 로드
        print("📁 데이터 로드 중...")
        audio_paths, labels = load_dataset_from_directory(data_directory)
        
        if len(audio_paths) == 0:
            print("❌ 오디오 파일을 찾을 수 없습니다.")
            return
        
        print(f"✅ {len(audio_paths)}개의 오디오 파일을 찾았습니다.")
        
        # 데이터 분할
        print("🔄 데이터 분할 중...")
        train_data, val_data, test_data = create_data_splits(
            audio_paths, labels,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1
        )
        
        # kresnik 모델에 최적화된 훈련 설정
        training_config.batch_size = 2  # 317M 파라미터 모델용 작은 배치
        training_config.gradient_accumulation_steps = 8  # 효과적 배치 크기 = 16
        training_config.learning_rate = 3e-5  # 큰 모델에 적합한 낮은 학습률
        training_config.num_epochs = 3  # 빠른 테스트를 위해
        training_config.warmup_steps = 500  # 짧은 테스트용 웜업
        training_config.output_dir = "./results"
        
        print(f"⚙️ kresnik 모델 최적화 설정:")
        print(f"  배치 크기: {training_config.batch_size}")
        print(f"  Gradient 누적: {training_config.gradient_accumulation_steps}")
        print(f"  효과적 배치 크기: {training_config.batch_size * training_config.gradient_accumulation_steps}")
        print(f"  학습률: {training_config.learning_rate}")
        print()
        
        # 모델 생성
        print("🤖 모델 생성 중...")
        model = create_model()
        
        # 훈련기 생성
        print("🏋️ 훈련 준비 중...")
        trainer = create_trainer(model)
        
        # 훈련 실행
        print("🚀 훈련 시작!")
        trainer.train(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data
        )
        
        # 평가
        print("📊 모델 평가 중...")
        eval_results = trainer.evaluate()
        
        if eval_results:
            print(f"✅ 최종 정확도: {eval_results.get('eval_accuracy', 'N/A'):.4f}")
            print(f"✅ 최종 F1 스코어: {eval_results.get('eval_f1', 'N/A'):.4f}")
        
        print("🎉 훈련 완료!")
        
    except Exception as e:
        print(f"❌ 훈련 중 오류 발생: {e}")
        raise e

def example_inference():
    """추론 예시"""
    
    print("=" * 60)
    print("🔮 음성 감정 분석 추론 예시")
    print("=" * 60)
    
    model_path = "./results"  # 훈련된 모델 경로
    
    if not os.path.exists(model_path):
        print(f"⚠️ 훈련된 모델을 찾을 수 없습니다: {model_path}")
        print("📝 먼저 모델을 훈련하거나 올바른 경로를 설정해주세요.")
        return
    
    try:
        # 추론 엔진 생성
        print("🧠 추론 엔진 로드 중...")
        inference = create_inference_engine(model_path)
        
        # 모델 정보 출력
        model_info = inference.get_model_info()
        print("📋 모델 정보:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        print()
        
        # 단일 파일 예측 예시
        audio_file = "sample_audio.wav"  # 실제 오디오 파일 경로
        
        if os.path.exists(audio_file):
            print(f"🎵 단일 파일 예측: {audio_file}")
            result = inference.predict_single(audio_file)
            
            if result['status'] == 'success':
                print(f"  예측 감정: {result['predicted_emotion']}")
                print(f"  신뢰도: {result['confidence']:.4f}")
                print("  모든 감정별 확률:")
                for emotion, prob in result['probabilities'].items():
                    print(f"    {emotion}: {prob:.4f}")
            else:
                print(f"  예측 실패: {result['error']}")
        else:
            print(f"⚠️ 테스트 오디오 파일을 찾을 수 없습니다: {audio_file}")
        
        # 디렉토리 배치 예측 예시
        test_directory = "./test_audio"  # 테스트 오디오 디렉토리
        
        if os.path.exists(test_directory):
            print(f"\n📁 디렉토리 배치 예측: {test_directory}")
            results = inference.predict_directory(test_directory)
            
            # 감정 분포 분석
            distribution = inference.analyze_emotions_distribution(results)
            
            if 'error' not in distribution:
                print(f"  총 샘플: {distribution['total_samples']}개")
                print(f"  평균 신뢰도: {distribution['average_confidence']:.4f}")
                print("  감정별 분포:")
                
                for emotion, count in distribution['emotion_counts'].items():
                    percentage = distribution['emotion_percentages'][emotion]
                    print(f"    {emotion}: {count}개 ({percentage:.1f}%)")
            else:
                print(f"  분석 실패: {distribution['error']}")
        else:
            print(f"⚠️ 테스트 디렉토리를 찾을 수 없습니다: {test_directory}")
        
        print("🎉 추론 완료!")
        
    except Exception as e:
        print(f"❌ 추론 중 오류 발생: {e}")
        raise e

def example_configuration():
    """설정 예시"""
    
    print("=" * 60)
    print("⚙️ 설정 커스터마이징 예시")
    print("=" * 60)
    
    # 기본 설정 확인
    print("📋 현재 모델 설정:")
    print(f"  모델명: {model_config.model_name}")
    print(f"  감정 클래스: {model_config.emotion_labels}")
    print(f"  샘플링 레이트: {model_config.sampling_rate}")
    print(f"  최대 길이: {model_config.max_duration}")
    print()
    
    print("📋 현재 훈련 설정:")
    print(f"  배치 크기: {training_config.batch_size}")
    print(f"  학습률: {training_config.learning_rate}")
    print(f"  에포크 수: {training_config.num_epochs}")
    print(f"  출력 디렉토리: {training_config.output_dir}")
    print()
    
    print("📋 현재 데이터 설정:")
    print(f"  정규화: {data_config.normalize_audio}")
    print(f"  노이즈 제거: {data_config.apply_noise_reduction}")
    print(f"  데이터 증강: {data_config.data_augmentation}")
    print()
    
    # 설정 수정 예시
    print("🔧 설정 수정 예시:")
    
    # 모델 설정 수정
    original_duration = model_config.max_duration
    model_config.max_duration = 8.0
    print(f"  최대 오디오 길이: {original_duration} → {model_config.max_duration}")
    
    # 훈련 설정 수정
    original_batch_size = training_config.batch_size
    training_config.batch_size = 16
    print(f"  배치 크기: {original_batch_size} → {training_config.batch_size}")
    
    original_lr = training_config.learning_rate
    training_config.learning_rate = 5e-5
    print(f"  학습률: {original_lr} → {training_config.learning_rate}")
    
    # 데이터 설정 수정
    original_augmentation = data_config.data_augmentation
    data_config.data_augmentation = True
    print(f"  데이터 증강: {original_augmentation} → {data_config.data_augmentation}")
    
    print("\n✅ 설정이 수정되었습니다!")

def example_utils():
    """유틸리티 함수 예시"""
    
    print("=" * 60)
    print("🛠️ 유틸리티 함수 예시")
    print("=" * 60)
    
    from SER.utils import (
        validate_audio_files, 
        create_emotion_distribution_plot,
        save_predictions,
        create_summary_report
    )
    
    # 오디오 파일 검증 예시
    test_files = ["audio1.wav", "audio2.mp3", "nonexistent.wav", "invalid.txt"]
    valid_files, invalid_files = validate_audio_files(test_files)
    
    print("📁 파일 검증 결과:")
    print(f"  유효한 파일: {valid_files}")
    print(f"  무효한 파일: {invalid_files}")
    print()
    
    # 더미 예측 결과 생성
    dummy_predictions = [
        {'predicted_emotion': 'Joy', 'confidence': 0.85, 'status': 'success'},
        {'predicted_emotion': 'Neutral', 'confidence': 0.92, 'status': 'success'},
        {'predicted_emotion': 'Sad', 'confidence': 0.78, 'status': 'success'},
        {'predicted_emotion': 'Angry', 'confidence': 0.89, 'status': 'success'},
    ]
    
    # 예측 결과 저장 예시
    print("💾 예측 결과 저장:")
    save_predictions(dummy_predictions, "example_predictions.json", format='json')
    print("  JSON 형식으로 저장 완료")
    
    # 요약 보고서 생성 예시
    print("\n📊 요약 보고서 생성:")
    model_info = {
        'model_name': model_config.model_name,
        'num_labels': model_config.num_labels,
        'emotion_labels': model_config.emotion_labels
    }
    
    training_results = {
        'final_accuracy': 0.85,
        'final_f1_score': 0.82,
        'training_time': '2h 30m'
    }
    
    report = create_summary_report(model_info, training_results)
    print(report)

def main():
    """메인 함수"""
    
    print("🎤 SER (Speech Emotion Recognition) 모듈 예시")
    print("=" * 60)
    
    while True:
        print("\n선택하세요:")
        print("1. 📚 훈련 예시")
        print("2. 🔮 추론 예시") 
        print("3. ⚙️ 설정 예시")
        print("4. 🛠️ 유틸리티 예시")
        print("0. 종료")
        
        choice = input("\n번호를 입력하세요: ").strip()
        
        if choice == '1':
            example_training()
        elif choice == '2':
            example_inference()
        elif choice == '3':
            example_configuration()
        elif choice == '4':
            example_utils()
        elif choice == '0':
            print("👋 프로그램을 종료합니다.")
            break
        else:
            print("❌ 잘못된 선택입니다. 다시 시도해주세요.")

if __name__ == "__main__":
    main()