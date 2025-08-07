import os
import re
import json
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import yaml

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """로깅 설정"""
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 로그 레벨 설정
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # 핸들러 설정
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers,
        force=True
    )
    
    return logging.getLogger(__name__)

def save_config_to_file(config_dict: Dict[str, Any], file_path: str):
    """설정을 파일로 저장"""
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if file_path.endswith('.json'):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    elif file_path.endswith('.yaml') or file_path.endswith('.yml'):
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    else:
        raise ValueError("지원되지 않는 파일 형식입니다. .json 또는 .yaml/.yml을 사용하세요.")

def load_config_from_file(file_path: str) -> Dict[str, Any]:
    """파일에서 설정 로드"""
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {file_path}")
    
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif file_path.endswith('.yaml') or file_path.endswith('.yml'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError("지원되지 않는 파일 형식입니다. .json 또는 .yaml/.yml을 사용하세요.")

def create_directory(dir_path: str, exist_ok: bool = True):
    """디렉토리 생성"""
    os.makedirs(dir_path, exist_ok=exist_ok)
    return dir_path

def get_device() -> torch.device:
    """사용 가능한 디바이스 반환"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU 사용: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("CPU 사용")
    
    return device

def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """모델 파라미터 수 계산"""
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }

def format_time(seconds: float) -> str:
    """초를 시:분:초 형식으로 변환"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"

def plot_training_history(history: Dict[str, List[float]], 
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (12, 4)):
    """훈련 히스토리 시각화"""
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Loss 그래프
    if 'train_loss' in history and 'val_loss' in history:
        axes[0].plot(history['train_loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Validation Loss')
        axes[0].set_title('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
    
    # Accuracy/F1 그래프
    if 'val_accuracy' in history:
        axes[1].plot(history['val_accuracy'], label='Accuracy', color='green')
    if 'val_f1' in history:
        axes[1].plot(history['val_f1'], label='F1 Score', color='red')
    
    axes[1].set_title('Metrics')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"훈련 히스토리 그래프가 저장되었습니다: {save_path}")
    
    plt.show()

def calculate_class_weights(labels: List[str], emotion_labels: List[str]) -> torch.Tensor:
    """클래스 불균형을 위한 가중치 계산"""
    
    from collections import Counter
    from sklearn.utils.class_weight import compute_class_weight
    
    # 라벨 카운트
    label_counts = Counter(labels)
    
    # 모든 감정에 대한 카운트 (없는 경우 0)
    class_counts = [label_counts.get(emotion, 0) for emotion in emotion_labels]
    
    # sklearn을 사용한 가중치 계산
    try:
        weights = compute_class_weight(
            'balanced',
            classes=np.arange(len(emotion_labels)),
            y=[emotion_labels.index(label) for label in labels]
        )
        return torch.FloatTensor(weights)
    except:
        # 균등 가중치 반환
        return torch.ones(len(emotion_labels))

def save_predictions(predictions: List[Dict], 
                    save_path: str,
                    format: str = 'json'):
    """예측 결과 저장"""
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if format.lower() == 'json':
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
    
    elif format.lower() == 'csv':
        import pandas as pd
        
        # 예측 결과를 DataFrame으로 변환
        df_data = []
        for pred in predictions:
            if pred.get('status') == 'success':
                row = {
                    'audio_path': pred['audio_path'],
                    'predicted_emotion': pred['predicted_emotion'],
                    'confidence': pred.get('confidence', 0)
                }
                
                # 각 감정별 확률 추가
                if 'probabilities' in pred:
                    for emotion, prob in pred['probabilities'].items():
                        row[f'prob_{emotion}'] = prob
                
                df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv(save_path, index=False, encoding='utf-8')
    
    else:
        raise ValueError("지원되지 않는 형식입니다. 'json' 또는 'csv'를 사용하세요.")
    
    print(f"예측 결과가 저장되었습니다: {save_path}")

def load_predictions(file_path: str) -> List[Dict]:
    """저장된 예측 결과 로드"""
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
    
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    elif file_path.endswith('.csv'):
        import pandas as pd
        df = pd.read_csv(file_path)
        
        predictions = []
        for _, row in df.iterrows():
            pred = {
                'audio_path': row['audio_path'],
                'predicted_emotion': row['predicted_emotion'],
                'confidence': row.get('confidence', 0),
                'status': 'success'
            }
            
            # 감정별 확률 복원
            probabilities = {}
            for col in df.columns:
                if col.startswith('prob_'):
                    emotion = col.replace('prob_', '')
                    probabilities[emotion] = row[col]
            
            if probabilities:
                pred['probabilities'] = probabilities
            
            predictions.append(pred)
        
        return predictions
    
    else:
        raise ValueError("지원되지 않는 파일 형식입니다. .json 또는 .csv를 사용하세요.")

def create_emotion_distribution_plot(predictions: List[Dict], 
                                   save_path: Optional[str] = None,
                                   figsize: Tuple[int, int] = (10, 6)):
    """감정 분포 시각화"""
    
    # 성공적인 예측만 필터링
    successful_preds = [p for p in predictions if p.get('status') == 'success']
    
    if not successful_preds:
        print("시각화할 데이터가 없습니다.")
        return
    
    # 감정별 카운트
    from collections import Counter
    emotion_counts = Counter([p['predicted_emotion'] for p in successful_preds])
    
    # 그래프 생성
    emotions = list(emotion_counts.keys())
    counts = list(emotion_counts.values())
    
    plt.figure(figsize=figsize)
    bars = plt.bar(emotions, counts, alpha=0.8)
    
    # 각 바 위에 카운트 표시
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom')
    
    plt.title('Predicted Emotion Distribution')
    plt.xlabel('Emotions')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"감정 분포 그래프가 저장되었습니다: {save_path}")
    
    plt.show()

def validate_audio_files(file_paths: List[str], 
                        valid_extensions: List[str] = None) -> Tuple[List[str], List[str]]:
    """오디오 파일 경로 검증"""
    
    if valid_extensions is None:
        valid_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
    
    valid_files = []
    invalid_files = []
    
    for file_path in file_paths:
        # 파일 존재 확인
        if not os.path.exists(file_path):
            invalid_files.append(f"{file_path} (파일 없음)")
            continue
        
        # 확장자 확인
        _, ext = os.path.splitext(file_path.lower())
        if ext not in valid_extensions:
            invalid_files.append(f"{file_path} (지원되지 않는 형식)")
            continue
        
        # 파일 크기 확인 (0바이트가 아닌지)
        if os.path.getsize(file_path) == 0:
            invalid_files.append(f"{file_path} (빈 파일)")
            continue
        
        valid_files.append(file_path)
    
    return valid_files, invalid_files

def create_summary_report(model_info: Dict, 
                         training_results: Optional[Dict] = None,
                         evaluation_results: Optional[Dict] = None) -> str:
    """종합 보고서 생성"""
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("음성 감정 분석 모델 종합 보고서")
    report_lines.append("=" * 80)
    report_lines.append(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # 모델 정보
    report_lines.append("📋 모델 정보")
    report_lines.append("-" * 40)
    for key, value in model_info.items():
        if isinstance(value, list):
            report_lines.append(f"  {key}: {', '.join(map(str, value))}")
        else:
            report_lines.append(f"  {key}: {value}")
    report_lines.append("")
    
    # 훈련 결과
    if training_results:
        report_lines.append("🔥 훈련 결과")
        report_lines.append("-" * 40)
        for key, value in training_results.items():
            if isinstance(value, float):
                report_lines.append(f"  {key}: {value:.4f}")
            else:
                report_lines.append(f"  {key}: {value}")
        report_lines.append("")
    
    # 평가 결과
    if evaluation_results:
        report_lines.append("📊 평가 결과")
        report_lines.append("-" * 40)
        
        if 'overall_metrics' in evaluation_results:
            overall = evaluation_results['overall_metrics']
            report_lines.append("  전체 성능:")
            for metric, value in overall.items():
                if isinstance(value, float):
                    report_lines.append(f"    {metric}: {value:.4f}")
                else:
                    report_lines.append(f"    {metric}: {value}")
        
        report_lines.append("")
    
    return "\n".join(report_lines)

class ModelCheckpoint:
    """모델 체크포인트 관리"""
    
    def __init__(self, checkpoint_dir: str, save_best_only: bool = True):
        self.checkpoint_dir = checkpoint_dir
        self.save_best_only = save_best_only
        self.best_score = -float('inf')
        
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save(self, model, optimizer, epoch: int, score: float, 
             is_best: bool = False, extra_info: Optional[Dict] = None):
        """체크포인트 저장"""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'score': score,
            'timestamp': datetime.now().isoformat()
        }
        
        if extra_info:
            checkpoint.update(extra_info)
        
        # 최신 체크포인트 저장
        if not self.save_best_only:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save(checkpoint, checkpoint_path)
        
        # 최고 성능 체크포인트 저장
        if is_best or score > self.best_score:
            self.best_score = score
            best_path = os.path.join(self.checkpoint_dir, 'best_checkpoint.pt')
            torch.save(checkpoint, best_path)
            print(f"최고 성능 체크포인트 저장: {best_path} (score: {score:.4f})")
    
    def load(self, checkpoint_path: str, model, optimizer=None):
        """체크포인트 로드"""
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        score = checkpoint.get('score', 0)
        
        print(f"체크포인트 로드 완료: epoch {epoch}, score {score:.4f}")
        
        return epoch, score, checkpoint

def print_system_info():
    """시스템 정보 출력"""
    
    print("=" * 60)
    print("시스템 정보")
    print("=" * 60)
    
    # Python 정보
    import sys
    print(f"Python 버전: {sys.version}")
    
    # PyTorch 정보
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA 버전: {torch.version.cuda}")
        print(f"GPU 개수: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 메모리 정보
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"시스템 메모리: {memory.total // (1024**3):.1f}GB (사용률: {memory.percent}%)")
    except ImportError:
        pass
    
    print("=" * 60)

def extract_number_from_filename(filename: str) -> Optional[int]:
    """F####_######.wav 형식에서 마지막 한 자리 추출"""
    try:
        # F로 시작하는 모든 파일에서 마지막 숫자의 마지막 한 자리 추출
        match = re.search(r'F\d+_\d*(\d)\.wav', filename)
        if match:
            return int(match.group(1))
        return None
    except (ValueError, AttributeError):
        return None

def get_emotion_from_filename(filename: str) -> Optional[str]:
    """파일명에서 번호를 추출하여 감정 라벨 반환"""
    try:
        match = re.search(r'_(\d{6})\.', filename)
        if not match:
            return None
        
        file_num = int(match.group(1))
        
        if 21 <= file_num <= 30:
            return "Anxious"
        elif 31 <= file_num <= 40:
            return "Kind"
        elif 91 <= file_num <= 100:
            return "Dry"
        else:
            return None
    except (ValueError, AttributeError):
        return None