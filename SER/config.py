"""
음성 감정 분석 모델 설정
"""

import os
from dataclasses import dataclass
from typing import List

@dataclass
class ModelConfig:
    """모델 설정"""
    model_name: str = "kresnik/wav2vec2-large-xlsr-korean"
    num_labels: int = 3  # 3개 클래스로 변경
    sampling_rate: int = 16000
    max_duration: float = 10.0  # 최대 오디오 길이 (초)
    
    # 감정 라벨 정의 (3개 클래스)
    emotion_labels: List[str] = None
    
    def __post_init__(self):
        if self.emotion_labels is None:
            self.emotion_labels = [
                "Anxious", "Dry", "Kind"  # 3개 클래스만 사용
            ]
    
    @property
    def label2id(self):
        return {label: i for i, label in enumerate(self.emotion_labels)}
    
    @property
    def id2label(self):
        return {i: label for i, label in enumerate(self.emotion_labels)}

@dataclass
class TrainingConfig:
    """훈련 설정 (kresnik/wav2vec2-large-xlsr-korean 최적화, 일반화 성능 향상)"""
    output_dir: str = "./results"
    
    # kresnik 모델(317M 파라미터)에 최적화된 학습률 (일반화 개선)
    learning_rate: float = 2e-5  # 더 낮은 학습률로 안정적 학습
    batch_size: int = 4  # GPU 메모리 고려하여 작은 배치 크기
    gradient_accumulation_steps: int = 8  # 효과적인 배치 크기 = 4 * 8 = 32 (더 큰 배치)
    
    num_epochs: int = 10  # 3개 클래스이므로 더 많은 에포크로 안정적 학습
    warmup_steps: int = 500  # 웜업 기간 조정
    weight_decay: float = 0.05  # 정규화 강화로 오버피팅 방지
    
    # 평가 및 저장 설정
    eval_steps: int = 100  # 더 자주 평가하여 오버피팅 방지
    save_steps: int = 200
    logging_steps: int = 25
    save_total_limit: int = 5
    
    # GPU 최적화 설정
    fp16: bool = True  # Mixed precision으로 메모리 절약
    dataloader_num_workers: int = 2  # 메모리 사용량 고려
    
    # Early stopping (일반화 성능 향상)
    early_stopping_patience: int = 8  # 더 긴 patience로 충분한 학습
    early_stopping_threshold: float = 0.001  # 더 엄격한 threshold
    
    # kresnik 모델 특화 설정 (일반화 개선)
    max_grad_norm: float = 0.5  # 더 엄격한 gradient clipping
    adam_epsilon: float = 1e-8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    
    # 일반화 성능 향상을 위한 추가 설정
    dropout_rate: float = 0.3  # 드롭아웃 강화
    layer_drop: float = 0.1  # 레이어 드롭으로 정규화
    label_smoothing: float = 0.1  # 라벨 스무딩으로 일반화 개선
    
    # 학습률 스케줄링
    lr_scheduler_type: str = "cosine"  # 코사인 스케줄러로 안정적 학습
    num_cycles: float = 0.5

@dataclass 
class DataConfig:
    """데이터 설정"""
    train_data_path: str = "./data/train"
    val_data_path: str = "./data/val"
    test_data_path: str = "./data/test"
    
    # 데이터 전처리 설정
    normalize_audio: bool = True
    apply_noise_reduction: bool = False
    data_augmentation: bool = True
    
    # 데이터셋 분할 비율
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

# 전역 설정 인스턴스
model_config = ModelConfig()
training_config = TrainingConfig()
data_config = DataConfig()