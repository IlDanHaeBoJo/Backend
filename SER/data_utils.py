import numpy as np
from typing import Tuple, List, Optional, Literal
import re
import os

EMOTION_LABELS = ["Anxious", "Dry", "Kind"]

def simple_augmentation(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """간단한 오디오 증강 (NumPy 2.x 호환)"""
    if np.random.random() < 0.3:  # 30% 확률로 노이즈 추가
        noise = np.random.normal(0, 0.005, audio.shape)
        audio = audio + noise
    
    if np.random.random() < 0.3:  # 30% 확률로 볼륨 조정
        volume_factor = np.random.uniform(0.8, 1.2)
        audio = audio * volume_factor
    
    return audio





def extract_number_from_filename(filename: str, type: Literal['content', 'emotion'] = 'emotion') -> Optional[int]:
    try:
        if type == "content":
            # 파일명에서 마지막 숫자 그룹 전체를 추출 (예: F2001_000123.wav -> 123)
            match = re.search(r'_(\d+)\.wav$', os.path.basename(filename))
            if match:
                return int(match.group(1))
            return None
        else:
            # F..._...xxxD.wav 에서 마지막 숫자 D를 추출
            match = re.search(r'_(\d+)\.wav$', os.path.basename(filename))
            if match:
                return int(match.group(1)) % 10
            return None
    except (ValueError, AttributeError):
        return None



def get_emotion_from_filename(filename: str) -> Optional[str]:
    """파일명에서 번호를 추출하여 감정 라벨 반환"""
    file_num = extract_number_from_filename(filename, type="content")
    if file_num is None:
        return None
        
    if 21 <= file_num <= 30:
        return "Anxious"
    elif 31 <= file_num <= 40:
        return "Kind"
    elif 141 <= file_num <= 150:
        return "Dry"
    else:
        return None




def split_data_by_last_digit(audio_paths: List[str], labels: List[str]) -> Tuple[
    Tuple[List[str], List[str]], 
    Tuple[List[str], List[str]], 
    Tuple[List[str], List[str]]
]:
    """파일명의 마지막 숫자를 기준으로 train/val/test 분할
    
    Args:
        audio_paths: 오디오 파일 경로 리스트
        labels: 해당하는 라벨 리스트
        
    Returns:
        ((train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels))
        - Train: 마지막 숫자가 1,2,3,4,5,6
        - Validation: 마지막 숫자가 7,8
        - Test: 마지막 숫자가 9,0
    """
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []
    
    for path, label in zip(audio_paths, labels):
        last_digit = extract_number_from_filename(path, type="emotion")
        
        if last_digit is None:
            print(f"⚠️ 파일명에서 마지막 숫자를 추출할 수 없습니다: {path}")
            continue
            
        if last_digit in [1, 2, 3, 4, 5, 6]:
            train_paths.append(path)
            train_labels.append(label)
        elif last_digit in [7, 8]:
            val_paths.append(path)
            val_labels.append(label)
        elif last_digit in [9, 0]:
            test_paths.append(path)
            test_labels.append(label)
    
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)



# (필수) 화자 ID 추출: data_dir 바로 아래 1단계 폴더명이 화자
def extract_speaker_id(audio_path: str, data_dir: str) -> str:
    rel = os.path.relpath(audio_path, data_dir)
    spk = rel.split(os.sep)[0]
    return spk



def build_speaker_mapping(train_paths, data_dir):
    train_speakers = sorted({extract_speaker_id(p, data_dir) for p in train_paths})
    spk2id = {spk: i for i, spk in enumerate(train_speakers)}
    return spk2id



# (선택) 경로에서 감정 라벨 추론 (폴더명에 Anxious/Kind/Dry가 있으면 그걸 사용)
def infer_emotion_from_path(audio_path: str) -> Optional[str]:
    parts = os.path.normpath(audio_path).split(os.sep)
    for p in reversed(parts):
        if p in EMOTION_LABELS:
            return p
    # 폴더명에 없으면 파일명 규칙으로 추론 (기존 함수)
    return get_emotion_from_filename(os.path.basename(audio_path))





    