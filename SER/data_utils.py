import numpy as np

def simple_augmentation(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """간단한 오디오 증강 (NumPy 2.x 호환)"""
    if np.random.random() < 0.3:  # 30% 확률로 노이즈 추가
        noise = np.random.normal(0, 0.005, audio.shape)
        audio = audio + noise
    
    if np.random.random() < 0.3:  # 30% 확률로 볼륨 조정
        volume_factor = np.random.uniform(0.8, 1.2)
        audio = audio * volume_factor
    
    return audio