"""
음성 감정 분석 모델 훈련 실행 스크립트
"""

import os
import argparse
import torch
from typing import Optional

from .model import create_model
from .trainer import create_trainer
from .data_loader import load_dataset_from_directory, load_dataset_from_csv, create_data_splits
from .config import model_config, training_config, data_config

def parse_arguments():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description="음성 감정 분석 모델 훈련")
    
    # 데이터 관련 인자
    parser.add_argument("--data_dir", type=str, help="데이터 디렉토리 경로")
    parser.add_argument("--csv_path", type=str, help="CSV 파일 경로")
    parser.add_argument("--output_dir", type=str, default="./results", help="결과 저장 디렉토리")
    
    # 모델 관련 인자
    parser.add_argument("--model_name", type=str, default=model_config.model_name, help="사용할 모델 이름")
    parser.add_argument("--max_duration", type=float, default=model_config.max_duration, help="최대 오디오 길이")
    
    # 훈련 관련 인자
    parser.add_argument("--batch_size", type=int, default=training_config.batch_size, help="배치 크기")
    parser.add_argument("--learning_rate", type=float, default=training_config.learning_rate, help="학습률")
    parser.add_argument("--num_epochs", type=int, default=training_config.num_epochs, help="훈련 에포크 수")
    parser.add_argument("--warmup_steps", type=int, default=training_config.warmup_steps, help="웜업 스텝")
    
    # 기타 설정
    parser.add_argument("--no_cuda", action="store_true", help="CUDA 사용 안함")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--resume_from_checkpoint", type=str, help="체크포인트에서 재개")
    
    return parser.parse_args()

def setup_environment(args):
    """환경 설정"""
    
    # 랜덤 시드 설정
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # GPU 설정
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
        print("CPU를 사용합니다.")
    else:
        device = torch.device("cuda")
        print(f"GPU를 사용합니다: {torch.cuda.get_device_name()}")
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    return device

def load_data(args):
    """데이터 로드"""
    
    audio_paths = []
    labels = []
    
    if args.data_dir:
        print(f"디렉토리에서 데이터 로드: {args.data_dir}")
        audio_paths, labels = load_dataset_from_directory(args.data_dir)
        
    elif args.csv_path:
        print(f"CSV에서 데이터 로드: {args.csv_path}")
        audio_paths, labels = load_dataset_from_csv(args.csv_path)
        
    else:
        print("데이터 경로가 지정되지 않았습니다. --data_dir 또는 --csv_path를 사용하세요.")
        return None, None, None
    
    if len(audio_paths) == 0:
        print("로드된 데이터가 없습니다.")
        return None, None, None
    
    # 데이터 분할
    train_data, val_data, test_data = create_data_splits(
        audio_paths, labels,
        train_ratio=data_config.train_ratio,
        val_ratio=data_config.val_ratio,
        test_ratio=data_config.test_ratio
    )
    
    return train_data, val_data, test_data

def update_configs(args):
    """설정 업데이트"""
    
    # 모델 설정 업데이트
    if args.model_name != model_config.model_name:
        model_config.model_name = args.model_name
    
    if args.max_duration != model_config.max_duration:
        model_config.max_duration = args.max_duration
    
    # 훈련 설정 업데이트
    training_config.batch_size = args.batch_size
    training_config.learning_rate = args.learning_rate
    training_config.num_epochs = args.num_epochs
    training_config.warmup_steps = args.warmup_steps
    training_config.output_dir = args.output_dir

def main():
    """메인 함수"""
    
    print("=" * 60)
    print("음성 감정 분석 모델 훈련")
    print("=" * 60)
    
    # 명령행 인자 파싱
    args = parse_arguments()
    
    # 환경 설정
    device = setup_environment(args)
    
    # 설정 업데이트
    update_configs(args)
    
    # 데이터 로드
    train_data, val_data, test_data = load_data(args)
    
    if train_data is None:
        print("데이터 로드에 실패했습니다.")
        return
    
    # 모델 생성
    print(f"모델 생성: {model_config.model_name}")
    model = create_model()
    model.to(device)
    
    # 훈련기 생성
    trainer = create_trainer(model)
    
    # 훈련 실행
    try:
        trainer.train(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            output_dir=args.output_dir
        )
        
        # 평가 실행
        print("\n모델 평가 시작...")
        eval_results = trainer.evaluate()
        
        if eval_results:
            print(f"최종 정확도: {eval_results.get('eval_accuracy', 'N/A'):.4f}")
            print(f"최종 F1 스코어: {eval_results.get('eval_f1', 'N/A'):.4f}")
        
        print("\n훈련이 성공적으로 완료되었습니다!")
        
    except KeyboardInterrupt:
        print("\n훈련이 사용자에 의해 중단되었습니다.")
        
    except Exception as e:
        print(f"\n훈련 중 오류 발생: {e}")
        raise e

if __name__ == "__main__":
    main()