"""
Speech Emotion Recognition (SER) Package
음성 감정 인식 패키지
"""

from .config import config, Config, ModelConfig, TrainingConfig, DataConfig, PathConfig

__version__ = "1.0.0"
__all__ = [
    "config",
    "Config", 
    "ModelConfig", 
    "TrainingConfig", 
    "DataConfig", 
    "PathConfig"
]

# 패키지 초기화 시 설정 검증
def _validate_config():
    """설정 유효성 검증"""
    if not config.paths.ser_root.exists():
        raise RuntimeError(f"SER 루트 디렉터리를 찾을 수 없습니다: {config.paths.ser_root}")
    
    print(f"✅ SER 패키지 초기화 완료")
    print(f"   - 루트 경로: {config.paths.ser_root}")
    print(f"   - 모델: {config.model.model_name}")
    print(f"   - 감정 라벨: {config.model.emotion_labels}")

_validate_config()