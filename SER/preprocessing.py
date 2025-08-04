"""
음성 데이터 전처리 모듈
"""

import numpy as np
import librosa
import torch
import torchaudio
import torchaudio.transforms as T
from typing import Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

from .config import model_config, data_config

class AudioPreprocessor:
    """음성 데이터 전처리 클래스"""
    
    def __init__(self, 
                 target_sr: int = 16000,
                 max_duration: float = 10.0,
                 normalize: bool = True):
        self.target_sr = target_sr
        self.max_duration = max_duration
        self.normalize = normalize
        
        # 데이터 증강을 위한 변환들 (다양한 기법 추가)
        self.noise_transform = AddNoise(noise_factor=0.005)
        self.time_stretch = TimeStretch()
        self.pitch_shift = PitchShift()
        self.volume_change = VolumeChange()
        self.speed_change = SpeedChange()
        self.formant_shift = FormantShift()
        self.reverb = SimpleReverb()
        
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """오디오 파일 로드"""
        try:
            # librosa로 로드
            audio, sr = librosa.load(file_path, sr=self.target_sr)
            return audio, sr
        except Exception as e:
            print(f"오디오 로드 실패: {file_path}, 에러: {e}")
            return None, None
    
    def resample_audio(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """오디오 리샘플링"""
        if orig_sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.target_sr)
        return audio
    
    def trim_or_pad_audio(self, audio: np.ndarray) -> np.ndarray:
        """오디오 길이 조정 (자르기 또는 패딩)"""
        target_length = int(self.target_sr * self.max_duration)
        
        if len(audio) > target_length:
            # 길면 자르기 (랜덤 위치에서 시작)
            start_idx = np.random.randint(0, len(audio) - target_length + 1)
            audio = audio[start_idx:start_idx + target_length]
        elif len(audio) < target_length:
            # 짧으면 패딩
            pad_length = target_length - len(audio)
            audio = np.pad(audio, (0, pad_length), mode='constant', constant_values=0)
        
        return audio
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """오디오 정규화"""
        if self.normalize:
            # RMS 정규화
            rms = np.sqrt(np.mean(audio**2))
            if rms > 0:
                audio = audio / (rms * 10)  # 적절한 스케일링
            
            # 클리핑 방지
            audio = np.clip(audio, -1.0, 1.0)
        
        return audio
    
    def remove_silence(self, audio: np.ndarray, 
                      top_db: int = 20) -> np.ndarray:
        """묵음 제거"""
        try:
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
            return audio_trimmed if len(audio_trimmed) > 0 else audio
        except:
            return audio
    
    def apply_noise_reduction(self, audio: np.ndarray) -> np.ndarray:
        """간단한 노이즈 리덕션"""
        # Spectral subtraction 기반 간단한 노이즈 제거
        try:
            # STFT
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # 노이즈 추정 (첫 10프레임의 평균)
            noise_profile = np.mean(magnitude[:, :10], axis=1, keepdims=True)
            
            # Spectral subtraction
            alpha = 2.0  # 오버서브트랙션 계수
            magnitude_clean = magnitude - alpha * noise_profile
            magnitude_clean = np.maximum(magnitude_clean, 0.1 * magnitude)
            
            # 역변환
            stft_clean = magnitude_clean * np.exp(1j * phase)
            audio_clean = librosa.istft(stft_clean)
            
            return audio_clean
        except:
            return audio
    
    def preprocess(self, file_path: str, 
                  apply_augmentation: bool = False) -> Optional[np.ndarray]:
        """전체 전처리 파이프라인"""
        # 오디오 로드
        audio, sr = self.load_audio(file_path)
        if audio is None:
            return None
        
        # 리샘플링
        audio = self.resample_audio(audio, sr)
        
        # 묵음 제거
        audio = self.remove_silence(audio)
        
        # 노이즈 리덕션 (설정에 따라)
        if data_config.apply_noise_reduction:
            audio = self.apply_noise_reduction(audio)
        
        # 길이 조정
        audio = self.trim_or_pad_audio(audio)
        
        # 정규화
        audio = self.normalize_audio(audio)
        
        # 데이터 증강 (훈련 시에만)
        if apply_augmentation and data_config.data_augmentation:
            audio = self.apply_augmentation(audio)
        
        return audio
    
    def apply_augmentation(self, audio: np.ndarray) -> np.ndarray:
        """데이터 증강 적용 (일반화 성능 향상을 위한 다양한 기법)"""
        # 랜덤하게 증강 기법 선택 (확률 기반으로 여러 기법 조합 가능)
        augmentations = [
            lambda x: self.noise_transform(torch.tensor(x)).numpy(),
            lambda x: self.time_stretch(torch.tensor(x)).numpy(),
            lambda x: self.pitch_shift(torch.tensor(x)).numpy(),
            lambda x: self.volume_change(torch.tensor(x)).numpy(),
            lambda x: self.speed_change(torch.tensor(x)).numpy(),
            lambda x: self.formant_shift(torch.tensor(x)).numpy(),
            lambda x: self.reverb(torch.tensor(x)).numpy(),
        ]
        
        # 50% 확률로 하나의 증강 적용
        if np.random.random() < 0.5:
            aug_func = np.random.choice(augmentations)
            try:
                audio = aug_func(audio)
            except Exception as e:
                pass  # 증강 실패 시 원본 반환
        
        # 20% 확률로 두 번째 증강 적용 (조합)
        if np.random.random() < 0.2:
            aug_func = np.random.choice(augmentations)
            try:
                audio = aug_func(audio)
            except Exception as e:
                pass
        
        return audio

class AddNoise(torch.nn.Module):
    """노이즈 추가 변환"""
    
    def __init__(self, noise_factor: float = 0.005):
        super().__init__()
        self.noise_factor = noise_factor
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(audio) * self.noise_factor
        return audio + noise

class TimeStretch(torch.nn.Module):
    """시간 스트레칭 변환"""
    
    def __init__(self, stretch_factors: Tuple[float, float] = (0.8, 1.2)):
        super().__init__()
        self.min_factor, self.max_factor = stretch_factors
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        factor = np.random.uniform(self.min_factor, self.max_factor)
        try:
            # Phase vocoder를 사용한 시간 스트레칭
            if len(audio.shape) == 1:
                audio = audio.unsqueeze(0)
            
            stretched = torchaudio.functional.phase_vocoder(
                audio.unsqueeze(0),
                rate=factor,
                phase_advance=torch.linspace(0, np.pi * audio.size(-1), audio.size(-1) // 2 + 1)
            )
            return stretched.squeeze(0)
        except:
            return audio

class PitchShift(torch.nn.Module):
    """피치 시프트 변환"""
    
    def __init__(self, shift_range: Tuple[float, float] = (-2, 2)):
        super().__init__()
        self.min_shift, self.max_shift = shift_range
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        shift = np.random.uniform(self.min_shift, self.max_shift)
        try:
            # 간단한 피치 시프트 (librosa 사용)
            audio_np = audio.numpy() if isinstance(audio, torch.Tensor) else audio
            shifted = librosa.effects.pitch_shift(audio_np, sr=16000, n_steps=shift)
            return torch.tensor(shifted, dtype=audio.dtype)
        except:
            return audio

class VolumeChange(torch.nn.Module):
    """볼륨 변경 변환"""
    
    def __init__(self, volume_range: Tuple[float, float] = (0.5, 1.5)):
        super().__init__()
        self.min_volume, self.max_volume = volume_range
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        volume_factor = np.random.uniform(self.min_volume, self.max_volume)
        return audio * volume_factor

class SpeedChange(torch.nn.Module):
    """속도 변경 변환 (시간 스케일링)"""
    
    def __init__(self, speed_range: Tuple[float, float] = (0.9, 1.1)):
        super().__init__()
        self.min_speed, self.max_speed = speed_range
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        speed_factor = np.random.uniform(self.min_speed, self.max_speed)
        try:
            audio_np = audio.numpy() if isinstance(audio, torch.Tensor) else audio
            # librosa를 사용한 속도 변경
            changed = librosa.effects.time_stretch(audio_np, rate=speed_factor)
            return torch.tensor(changed, dtype=audio.dtype)
        except:
            return audio

class FormantShift(torch.nn.Module):
    """포먼트 시프트 변환 (음성 특성 변경)"""
    
    def __init__(self, shift_range: Tuple[float, float] = (-0.1, 0.1)):
        super().__init__()
        self.min_shift, self.max_shift = shift_range
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # 간단한 포먼트 시프트 (주파수 도메인에서 변경)
        shift_factor = np.random.uniform(self.min_shift, self.max_shift)
        try:
            audio_np = audio.numpy() if isinstance(audio, torch.Tensor) else audio
            # STFT를 이용한 주파수 변조
            stft = librosa.stft(audio_np)
            # 주파수 축 시프트
            n_freq = stft.shape[0]
            shift_bins = int(n_freq * shift_factor)
            if shift_bins != 0:
                stft_shifted = np.roll(stft, shift_bins, axis=0)
                audio_shifted = librosa.istft(stft_shifted)
                return torch.tensor(audio_shifted, dtype=audio.dtype)
        except:
            pass
        return audio

class SimpleReverb(torch.nn.Module):
    """간단한 리버브 효과"""
    
    def __init__(self, reverb_factor: Tuple[float, float] = (0.1, 0.3)):
        super().__init__()
        self.min_reverb, self.max_reverb = reverb_factor
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        reverb_strength = np.random.uniform(self.min_reverb, self.max_reverb)
        try:
            # 간단한 딜레이 기반 리버브
            delay_samples = int(0.05 * 16000)  # 50ms 딜레이
            reverb_audio = audio.clone()
            
            if len(audio) > delay_samples:
                reverb_audio[delay_samples:] += audio[:-delay_samples] * reverb_strength
            
            return reverb_audio
        except:
            return audio

def extract_features(audio: np.ndarray, sr: int = 16000) -> dict:
    """오디오에서 특성 추출"""
    features = {}
    
    try:
        # MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfcc, axis=1)
        features['mfcc_std'] = np.std(mfcc, axis=1)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # Chroma
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        features['chroma_mean'] = np.mean(chroma, axis=1)
        
        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(rolloff)
        
    except Exception as e:
        print(f"특성 추출 중 오류: {e}")
    
    return features

# 전역 전처리기 인스턴스
preprocessor = AudioPreprocessor(
    target_sr=model_config.sampling_rate,
    max_duration=model_config.max_duration,
    normalize=data_config.normalize_audio
)