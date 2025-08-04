#!/usr/bin/env python3
"""
3개 클래스 음성 감정 분석 파인튜닝 실행 스크립트
간편한 실행을 위한 래퍼 스크립트
"""

import os
import sys
import argparse
from pathlib import Path

def parse_arguments():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="wav2vec2-large-xlsr-korean 3개 클래스 감정 분석 파인튜닝",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python run_finetune.py                           # 기본 설정으로 실행
  python run_finetune.py --test                    # 데이터 로딩 테스트만 실행
  python run_finetune.py --data_dir /custom/path   # 커스텀 데이터 경로
  python run_finetune.py --epochs 20 --lr 1e-5    # 하이퍼파라미터 조정
  python run_finetune.py --gpu 1                   # 특정 GPU 사용

감정 클래스 매핑:
  21-30 폴더 → Anxious (불안)
  31-40 폴더 → Kind (친절)  
  91-100 폴더 → Dry (건조/무미건조)
        """
    )
    
    # 데이터 관련
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="/data/ghdrnjs/SER/small/",
        help="데이터 디렉토리 경로 (기본값: /data/ghdrnjs/SER/small/)"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./results_3class",
        help="결과 저장 디렉토리 (기본값: ./results_3class)"
    )
    
    # 훈련 설정
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=15,
        help="훈련 에포크 수 (기본값: 15)"
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4,
        help="배치 크기 (기본값: 4)"
    )
    
    parser.add_argument(
        "--lr", "--learning_rate",
        type=float, 
        default=2e-5,
        help="학습률 (기본값: 2e-5)"
    )
    
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.05,
        help="가중치 감쇠 (기본값: 0.05)"
    )
    
    # 데이터 분할 비율
    parser.add_argument(
        "--train_ratio", 
        type=float, 
        default=0.7,
        help="훈련 데이터 비율 (기본값: 0.7)"
    )
    
    parser.add_argument(
        "--val_ratio", 
        type=float, 
        default=0.15,
        help="검증 데이터 비율 (기본값: 0.15)"
    )
    
    parser.add_argument(
        "--test_ratio", 
        type=float, 
        default=0.15,
        help="테스트 데이터 비율 (기본값: 0.15)"
    )
    
    # 기타 설정
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="랜덤 시드 (기본값: 42)"
    )
    
    parser.add_argument(
        "--gpu", 
        type=int, 
        default=None,
        help="사용할 GPU 번호 (기본값: 자동 선택)"
    )
    
    parser.add_argument(
        "--no_cuda", 
        action="store_true",
        help="GPU 사용 안함 (CPU만 사용)"
    )
    
    # 모드 설정
    parser.add_argument(
        "--test", 
        action="store_true",
        help="데이터 로딩 테스트만 실행"
    )
    
    parser.add_argument(
        "--resume_from", 
        type=str,
        help="체크포인트에서 훈련 재개"
    )
    
    # 데이터 증강 설정
    parser.add_argument(
        "--no_augmentation", 
        action="store_true",
        help="데이터 증강 비활성화"
    )
    
    parser.add_argument(
        "--augmentation_prob", 
        type=float, 
        default=0.5,
        help="데이터 증강 적용 확률 (기본값: 0.5)"
    )
    
    return parser.parse_args()

def validate_arguments(args):
    """인자 유효성 검사"""
    errors = []
    
    # 데이터 디렉토리 존재 확인
    if not os.path.exists(args.data_dir):
        errors.append(f"데이터 디렉토리를 찾을 수 없습니다: {args.data_dir}")
    
    # 비율 합계 확인
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        errors.append(f"데이터 분할 비율의 합이 1.0이 아닙니다: {total_ratio:.3f}")
    
    # 하이퍼파라미터 범위 확인
    if args.lr <= 0 or args.lr > 1:
        errors.append(f"학습률이 유효한 범위(0, 1]를 벗어났습니다: {args.lr}")
    
    if args.epochs <= 0:
        errors.append(f"에포크 수는 양수여야 합니다: {args.epochs}")
    
    if args.batch_size <= 0:
        errors.append(f"배치 크기는 양수여야 합니다: {args.batch_size}")
    
    if errors:
        print("❌ 인자 유효성 검사 실패:")
        for error in errors:
            print(f"   - {error}")
        sys.exit(1)

def setup_gpu(args):
    """GPU 설정"""
    import torch
    
    if args.no_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("🖥️  CPU 모드로 실행")
        return
    
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"🎮 GPU {args.gpu} 사용")
    
    if torch.cuda.is_available():
        print(f"🎮 사용 가능한 GPU: {torch.cuda.device_count()}개")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("🖥️  CUDA를 사용할 수 없습니다. CPU 모드로 실행")

def update_configs(args):
    """설정 업데이트"""
    # 현재 디렉토리를 Python 경로에 추가
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        from .config import training_config, data_config
    except ImportError:
        from config import training_config, data_config
    
    # 훈련 설정 업데이트
    training_config.num_epochs = args.epochs
    training_config.batch_size = args.batch_size
    training_config.learning_rate = args.lr
    training_config.weight_decay = args.weight_decay
    training_config.output_dir = args.output_dir
    
    # 데이터 설정 업데이트
    data_config.train_ratio = args.train_ratio
    data_config.val_ratio = args.val_ratio
    data_config.test_ratio = args.test_ratio
    data_config.data_augmentation = not args.no_augmentation
    
    print(f"⚙️  설정 업데이트 완료")
    print(f"   - 에포크: {training_config.num_epochs}")
    print(f"   - 배치 크기: {training_config.batch_size}")
    print(f"   - 학습률: {training_config.learning_rate}")
    print(f"   - 가중치 감쇠: {training_config.weight_decay}")
    print(f"   - 데이터 증강: {'활성화' if data_config.data_augmentation else '비활성화'}")

def run_training(args):
    """훈련 실행"""
    try:
        from .finetune_3class import main as train_main
    except ImportError:
        from finetune_3class import main as train_main
    
    print(f"🚀 훈련 시작!")
    print(f"   - 데이터: {args.data_dir}")
    print(f"   - 출력: {args.output_dir}")
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 훈련 실행
    train_main()

def run_test(args):
    """데이터 로딩 테스트 실행"""
    try:
        from .finetune_3class import test_data_loading
    except ImportError:
        from finetune_3class import test_data_loading
    
    print(f"🧪 데이터 로딩 테스트 실행")
    test_data_loading()

def main():
    """메인 함수"""
    print("🎵 wav2vec2-large-xlsr-korean 3개 클래스 감정 분석 파인튜닝")
    print("="*70)
    
    # 인자 파싱 및 검증
    args = parse_arguments()
    validate_arguments(args)
    
    # GPU 설정
    setup_gpu(args)
    
    # 설정 업데이트
    update_configs(args)
    
    try:
        if args.test:
            # 테스트 모드
            run_test(args)
        else:
            # 훈련 모드
            run_training(args)
            
    except KeyboardInterrupt:
        print(f"\n⏹️  실행이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()