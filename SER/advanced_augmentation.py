"""
고급 음성 데이터 증강 기법
SpecAugment, MixUp 등 SOTA 기법들 포함
"""

import torch
import torch.nn as nn
import numpy as np
import librosa
from typing import Tuple, Optional, Union
import random

class SpecAugment(nn.Module):
    """
    SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition
    https://arxiv.org/abs/1904.08779
    """
    
    def __init__(self, 
                 freq_mask_num: int = 2,
                 freq_mask_width: int = 27,
                 time_mask_num: int = 2, 
                 time_mask_width: int = 100):
        super().__init__()
        self.freq_mask_num = freq_mask_num
        self.freq_mask_width = freq_mask_width
        self.time_mask_num = time_mask_num
        self.time_mask_width = time_mask_width
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        SpecAugment 적용
        Args:
            audio: (batch_size, time) 또는 (time,) 형태의 오디오
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # STFT 변환
        stft = torch.stft(
            audio, 
            n_fft=512, 
            hop_length=160, 
            win_length=400,
            return_complex=True
        )
        
        # magnitude spectrogram
        magnitude = torch.abs(stft)  # (batch, freq, time)
        
        # Frequency masking
        for _ in range(self.freq_mask_num):
            freq_mask_start = random.randint(0, max(0, magnitude.size(1) - self.freq_mask_width))
            freq_mask_end = min(freq_mask_start + self.freq_mask_width, magnitude.size(1))
            magnitude[:, freq_mask_start:freq_mask_end, :] = 0
        
        # Time masking
        for _ in range(self.time_mask_num):
            time_mask_start = random.randint(0, max(0, magnitude.size(2) - self.time_mask_width))
            time_mask_end = min(time_mask_start + self.time_mask_width, magnitude.size(2))
            magnitude[:, :, time_mask_start:time_mask_end] = 0
        
        # 원래 phase와 결합하여 복원
        phase = torch.angle(stft)
        stft_masked = magnitude * torch.exp(1j * phase)
        
        # ISTFT로 오디오 복원
        audio_masked = torch.istft(
            stft_masked,
            n_fft=512,
            hop_length=160,
            win_length=400
        )
        
        return audio_masked.squeeze(0) if audio_masked.size(0) == 1 else audio_masked

class MixUp(nn.Module):
    """
    MixUp 데이터 증강
    두 오디오 샘플을 선형 결합하여 새로운 샘플 생성
    """
    
    def __init__(self, alpha: float = 0.2):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, audio1: torch.Tensor, audio2: torch.Tensor, 
                lambda_mix: Optional[float] = None) -> Tuple[torch.Tensor, float]:
        """
        MixUp 적용
        Args:
            audio1, audio2: 믹스할 오디오 텐서
            lambda_mix: 믹싱 비율 (None이면 베타 분포에서 샘플링)
        """
        if lambda_mix is None:
            lambda_mix = np.random.beta(self.alpha, self.alpha)
        
        # 길이 맞추기
        min_len = min(len(audio1), len(audio2))
        audio1 = audio1[:min_len]
        audio2 = audio2[:min_len]
        
        # 선형 결합
        mixed_audio = lambda_mix * audio1 + (1 - lambda_mix) * audio2
        
        return mixed_audio, lambda_mix

class CutMix(nn.Module):
    """
    CutMix 오디오 버전
    한 오디오의 일부를 다른 오디오의 일부로 교체
    """
    
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, audio1: torch.Tensor, audio2: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        CutMix 적용
        """
        # 베타 분포에서 lambda 샘플링
        lambda_cut = np.random.beta(self.alpha, self.alpha)
        
        # 길이 맞추기
        min_len = min(len(audio1), len(audio2))
        audio1 = audio1[:min_len]
        audio2 = audio2[:min_len]
        
        # 잘라낼 영역 크기 결정
        cut_len = int(min_len * lambda_cut)
        
        # 랜덤 시작 위치
        start_pos = random.randint(0, max(0, min_len - cut_len))
        end_pos = start_pos + cut_len
        
        # CutMix 적용
        mixed_audio = audio1.clone()
        mixed_audio[start_pos:end_pos] = audio2[start_pos:end_pos]
        
        return mixed_audio, lambda_cut

class RandomResizedCrop(nn.Module):
    """
    RandomResizedCrop의 오디오 버전
    오디오의 랜덤한 부분을 크롭하고 원래 길이로 리샘플링
    """
    
    def __init__(self, scale: Tuple[float, float] = (0.8, 1.0)):
        super().__init__()
        self.scale = scale
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        RandomResizedCrop 적용
        """
        original_len = len(audio)
        
        # 크롭할 비율 결정
        crop_scale = random.uniform(self.scale[0], self.scale[1])
        crop_len = int(original_len * crop_scale)
        
        # 랜덤 시작 위치
        start_pos = random.randint(0, max(0, original_len - crop_len))
        cropped_audio = audio[start_pos:start_pos + crop_len]
        
        # 원래 길이로 리샘플링 (선형 보간)
        if crop_len != original_len:
            # 간단한 리샘플링 (실제로는 librosa.resample 사용 권장)
            indices = torch.linspace(0, crop_len - 1, original_len)
            resampled_audio = torch.zeros(original_len)
            
            for i, idx in enumerate(indices):
                idx_low = int(idx.floor())
                idx_high = min(idx_low + 1, crop_len - 1)
                weight = idx - idx_low
                
                resampled_audio[i] = (1 - weight) * cropped_audio[idx_low] + weight * cropped_audio[idx_high]
            
            return resampled_audio
        
        return cropped_audio

class RandomErase(nn.Module):
    """
    RandomErase의 오디오 버전
    오디오의 랜덤한 부분을 0으로 마스킹
    """
    
    def __init__(self, 
                 prob: float = 0.5,
                 scale: Tuple[float, float] = (0.02, 0.33),
                 ratio: Tuple[float, float] = (0.3, 3.3)):
        super().__init__()
        self.prob = prob
        self.scale = scale
        self.ratio = ratio
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        RandomErase 적용
        """
        if random.random() > self.prob:
            return audio
        
        audio_len = len(audio)
        
        for _ in range(100):  # 최대 100번 시도
            # 마스킹할 영역 크기 결정
            erase_scale = random.uniform(self.scale[0], self.scale[1])
            erase_len = int(audio_len * erase_scale)
            
            # 시작 위치 결정
            start_pos = random.randint(0, max(0, audio_len - erase_len))
            
            # 마스킹 적용
            audio_erased = audio.clone()
            audio_erased[start_pos:start_pos + erase_len] = 0
            
            return audio_erased
        
        return audio

class AdditiveNoise(nn.Module):
    """
    다양한 종류의 노이즈 추가
    """
    
    def __init__(self, 
                 noise_types: list = ['gaussian', 'uniform', 'pink'],
                 snr_range: Tuple[float, float] = (10, 30)):
        super().__init__()
        self.noise_types = noise_types
        self.snr_range = snr_range
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        노이즈 추가
        """
        noise_type = random.choice(self.noise_types)
        snr_db = random.uniform(self.snr_range[0], self.snr_range[1])
        
        # 신호 전력 계산
        signal_power = torch.mean(audio ** 2)
        
        # SNR에 따른 노이즈 전력 계산
        noise_power = signal_power / (10 ** (snr_db / 10))
        
        # 노이즈 생성
        if noise_type == 'gaussian':
            noise = torch.randn_like(audio) * torch.sqrt(noise_power)
        elif noise_type == 'uniform':
            noise = (torch.rand_like(audio) - 0.5) * 2 * torch.sqrt(3 * noise_power)
        elif noise_type == 'pink':
            # Pink noise (1/f noise) 근사
            noise = self._generate_pink_noise(audio.shape[0]) * torch.sqrt(noise_power)
        else:
            noise = torch.randn_like(audio) * torch.sqrt(noise_power)
        
        return audio + noise
    
    def _generate_pink_noise(self, length: int) -> torch.Tensor:
        """Pink noise 생성"""
        # 간단한 pink noise 근사 (실제로는 더 정교한 알고리즘 필요)
        white_noise = torch.randn(length)
        
        # 간단한 1차 IIR 필터로 근사
        pink_noise = torch.zeros_like(white_noise)
        b = 0.02
        for i in range(1, length):
            pink_noise[i] = b * white_noise[i] + (1 - b) * pink_noise[i-1]
        
        return pink_noise

class AdvancedAudioAugmenter:
    """
    고급 오디오 증강 기법들을 조합한 증강기
    """
    
    def __init__(self, 
                 use_specaugment: bool = True,
                 use_mixup: bool = True,
                 use_cutmix: bool = True,
                 use_random_erase: bool = True,
                 use_additive_noise: bool = True,
                 augment_prob: float = 0.5):
        
        self.augment_prob = augment_prob
        self.augmentations = []
        
        if use_specaugment:
            self.augmentations.append(('specaugment', SpecAugment()))
        if use_mixup:
            self.augmentations.append(('mixup', MixUp()))
        if use_cutmix:
            self.augmentations.append(('cutmix', CutMix()))
        if use_random_erase:
            self.augmentations.append(('random_erase', RandomErase()))
        if use_additive_noise:
            self.augmentations.append(('additive_noise', AdditiveNoise()))
    
    def __call__(self, audio: torch.Tensor, 
                 audio_mix: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        랜덤하게 증강 기법 적용
        """
        if random.random() > self.augment_prob:
            return audio
        
        # 랜덤하게 증강 기법 선택
        aug_name, aug_func = random.choice(self.augmentations)
        
        try:
            if aug_name in ['mixup', 'cutmix'] and audio_mix is not None:
                # 두 오디오가 필요한 증강
                augmented_audio, _ = aug_func(audio, audio_mix)
            else:
                # 단일 오디오 증강
                augmented_audio = aug_func(audio)
            
            return augmented_audio
            
        except Exception as e:
            # 증강 실패 시 원본 반환
            print(f"증강 실패 ({aug_name}): {e}")
            return audio
    
    def get_available_augmentations(self) -> list:
        """사용 가능한 증강 기법 목록 반환"""
        return [name for name, _ in self.augmentations]