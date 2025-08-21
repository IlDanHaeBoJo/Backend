"""
음성 감정 분석을 위한 데이터 로더
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from sklearn.model_selection import train_test_split

from .preprocessing import preprocessor
from .config import model_config, data_config, training_config

class SpeechEmotionDataset(Dataset):
    """음성 감정 분석 데이터셋"""
    
    def __init__(self, 
                 data_paths: List[str],
                 labels: List[str],
                 processor: Wav2Vec2Processor,
                 is_training: bool = True,
                 max_duration: float = 15.0):
        
        self.data_paths = data_paths
        self.labels = labels
        self.processor = processor
        self.is_training = is_training
        self.max_duration = max_duration
        self.sampling_rate = model_config.sampling_rate
        
        # 라벨을 숫자로 변환
        self.label_encoder = model_config.label2id
        self.encoded_labels = [self.label_encoder[label] for label in labels]
        
        print(f"데이터셋 생성 완료: {len(self.data_paths)}개 샘플")
        
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        try:
            # 오디오 파일 경로와 라벨
            audio_path = self.data_paths[idx]
            label = self.encoded_labels[idx]
            
            # 오디오 전처리
            audio = preprocessor.preprocess(
                audio_path, 
                apply_augmentation=self.is_training
            )
            
            if audio is None:
                # 전처리 실패 시 빈 오디오 반환
                audio = np.zeros(int(self.sampling_rate * self.max_duration))
            
            # Wav2Vec2 processor로 변환
            inputs = self.processor(
                audio,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
                padding=True,
                max_length=int(self.sampling_rate * self.max_duration),
                truncation=True
            )
            
            return {
                'input_values': inputs.input_values.squeeze(0),
                'attention_mask': inputs.attention_mask.squeeze(0) if 'attention_mask' in inputs else None,
                'labels': torch.tensor(label, dtype=torch.long),
                'audio_path': audio_path
            }
            
        except Exception as e:
            print(f"데이터 로드 오류 (idx: {idx}): {e}")
            # 오류 발생 시 더미 데이터 반환
            dummy_audio = np.zeros(int(self.sampling_rate * self.max_duration))
            inputs = self.processor(
                dummy_audio,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
                padding=True
            )
            
            return {
                'input_values': inputs.input_values.squeeze(0),
                'attention_mask': inputs.attention_mask.squeeze(0) if 'attention_mask' in inputs else None,
                'labels': torch.tensor(0, dtype=torch.long),  # 기본 라벨
                'audio_path': 'error'
            }

class DataCollator:
    """배치 데이터 정리를 위한 collator"""
    
    def __init__(self, processor: Wav2Vec2Processor, padding: bool = True):
        self.processor = processor
        self.padding = padding
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # input_values와 labels 분리
        input_values = [feature['input_values'] for feature in features]
        labels = [feature['labels'] for feature in features]
        attention_masks = [feature.get('attention_mask') for feature in features]
        
        # 배치 처리
        batch = {}
        
        # input_values 패딩
        if self.padding:
            input_values = torch.nn.utils.rnn.pad_sequence(
                input_values, batch_first=True, padding_value=0.0
            )
        else:
            input_values = torch.stack(input_values)
        
        batch['input_values'] = input_values
        
        # attention_mask 처리
        if attention_masks[0] is not None:
            if self.padding:
                attention_mask = torch.nn.utils.rnn.pad_sequence(
                    attention_masks, batch_first=True, padding_value=0
                )
            else:
                attention_mask = torch.stack(attention_masks)
            batch['attention_mask'] = attention_mask
        
        # labels
        batch['labels'] = torch.stack(labels)
        
        return batch

def load_dataset_from_directory(data_dir: str, 
                              emotion_mapping: Optional[Dict[str, str]] = None) -> Tuple[List[str], List[str]]:
    """디렉토리에서 데이터셋 로드
    
    예상 구조:
    data_dir/
    ├── emotion1/
    │   ├── file1.wav
    │   └── file2.wav
    ├── emotion2/
    │   ├── file3.wav
    │   └── file4.wav
    ...
    """
    
    audio_paths = []
    labels = []
    
    if not os.path.exists(data_dir):
        print(f"데이터 디렉토리를 찾을 수 없습니다: {data_dir}")
        return audio_paths, labels
    
    # 각 감정 폴더 탐색
    for emotion_folder in os.listdir(data_dir):
        emotion_path = os.path.join(data_dir, emotion_folder)
        
        if not os.path.isdir(emotion_path):
            continue
        
        # 감정 라벨 매핑
        emotion_label = emotion_mapping.get(emotion_folder, emotion_folder) if emotion_mapping else emotion_folder
        
        if emotion_label not in model_config.emotion_labels:
            print(f"알 수 없는 감정 라벨: {emotion_label}")
            continue
        
        # 오디오 파일 찾기
        for audio_file in os.listdir(emotion_path):
            if audio_file.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
                audio_path = os.path.join(emotion_path, audio_file)
                audio_paths.append(audio_path)
                labels.append(emotion_label)
    
    print(f"로드된 데이터: {len(audio_paths)}개 파일")
    return audio_paths, labels

def load_dataset_from_numbered_folders(data_dir: str) -> Tuple[List[str], List[str]]:
    """파일명 번호 기반으로 데이터셋 로드 (사용자 맞춤)
    
    데이터 구조:
    /data/ghdrnjs/SER/small/
    ├── F2001/wav_48000/F2001_000021.wav → Anxious
    ├── F2001/wav_48000/F2001_000031.wav → Kind
    ├── F2001/wav_48000/F2001_000091.wav → Dry
    └── ...
    
    파일명 번호에 따른 감정 매핑:
    - 000021 ~ 000030: Anxious
    - 000031 ~ 000040: Kind  
    - 000091 ~ 000100: Dry
    
    Args:
        data_dir: 데이터 디렉토리 경로 (예: /data/ghdrnjs/SER/small/)
    
    Returns:
        Tuple[List[str], List[str]]: (오디오 파일 경로 리스트, 라벨 리스트)
    """
    
    audio_paths = []
    labels = []
    
    if not os.path.exists(data_dir):
        print(f"데이터 디렉토리를 찾을 수 없습니다: {data_dir}")
        return audio_paths, labels
    
    # 파일명 번호에 따른 감정 매핑
    def get_emotion_from_filename(filename: str) -> Optional[str]:
        """
        파일명에서 번호를 추출하여 감정 라벨 반환
        예: F2001_000021.wav → 21 → Anxious
        """
        try:
            # 파일명에서 6자리 번호 추출 (F2001_000021.wav → 000021)
            import re
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
    
    # 각 person 폴더 탐색 (F2001, F2002, M2001 등)
    person_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    print(f"발견된 person 폴더: {len(person_folders)}개")
    
    total_files_processed = 0
    emotion_counts = {"Anxious": 0, "Kind": 0, "Dry": 0}
    
    for person_folder in sorted(person_folders):
        person_path = os.path.join(data_dir, person_folder)
        wav_path = os.path.join(person_path, "wav_48000")
        
        # wav_48000 폴더가 존재하는지 확인
        if not os.path.exists(wav_path):
            print(f"⚠️  {person_folder}: wav_48000 폴더를 찾을 수 없습니다")
            continue
        
        person_file_count = {"Anxious": 0, "Kind": 0, "Dry": 0, "Other": 0}
        
        # wav_48000 폴더 내의 모든 오디오 파일 확인
        for audio_file in os.listdir(wav_path):
            if not audio_file.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
                continue
            
            # 파일명으로 감정 라벨 결정
            emotion_label = get_emotion_from_filename(audio_file)
            
            if emotion_label is not None:
                audio_path = os.path.join(wav_path, audio_file)
                audio_paths.append(audio_path)
                labels.append(emotion_label)
                person_file_count[emotion_label] += 1
                emotion_counts[emotion_label] += 1
            else:
                person_file_count["Other"] += 1
            
            total_files_processed += 1
        
        # 각 person별 통계 출력
        if sum(person_file_count[e] for e in ["Anxious", "Kind", "Dry"]) > 0:
            print(f"📁 {person_folder}: "
                  f"Anxious={person_file_count['Anxious']}, "
                  f"Kind={person_file_count['Kind']}, "
                  f"Dry={person_file_count['Dry']}, "
                  f"기타={person_file_count['Other']}")
    
    # 전체 통계 출력
    print(f"\n{'='*50}")
    print(f"📊 전체 데이터 로딩 완료")
    print(f"{'='*50}")
    print(f"처리된 person 폴더: {len(person_folders)}개")
    print(f"전체 파일 확인: {total_files_processed}개")
    print(f"사용된 파일: {len(audio_paths)}개")
    print(f"\n클래스별 분포:")
    for emotion, count in emotion_counts.items():
        percentage = (count / len(audio_paths) * 100) if len(audio_paths) > 0 else 0
        print(f"  {emotion}: {count}개 ({percentage:.1f}%)")
    
    if len(audio_paths) == 0:
        print("❌ 조건에 맞는 오디오 파일을 찾을 수 없습니다.")
        print("   파일명 패턴을 확인해주세요: [PREFIX]_[6자리숫자].wav")
        print("   예상 번호 범위: 000021-000030(Anxious), 000031-000040(Kind), 000091-000100(Dry)")
    
    return audio_paths, labels

def load_dataset_from_csv(csv_path: str, 
                         audio_column: str = 'file_path',
                         label_column: str = 'emotion') -> Tuple[List[str], List[str]]:
    """CSV 파일에서 데이터셋 로드"""
    
    if not os.path.exists(csv_path):
        print(f"CSV 파일을 찾을 수 없습니다: {csv_path}")
        return [], []
    
    df = pd.read_csv(csv_path)
    audio_paths = df[audio_column].tolist()
    labels = df[label_column].tolist()
    
    # 존재하는 파일만 필터링
    valid_paths = []
    valid_labels = []
    
    for path, label in zip(audio_paths, labels):
        if os.path.exists(path) and label in model_config.emotion_labels:
            valid_paths.append(path)
            valid_labels.append(label)
    
    print(f"CSV에서 로드된 유효한 데이터: {len(valid_paths)}개 파일")
    return valid_paths, labels

def create_data_splits(audio_paths: List[str], 
                      labels: List[str],
                      train_ratio: float = 0.8,
                      val_ratio: float = 0.1,
                      test_ratio: float = 0.1,
                      random_state: int = 42) -> Tuple[Tuple[List[str], List[str]], ...]:
    """데이터를 train/val/test로 분할"""
    
    # train과 temp(val+test) 분할
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        audio_paths, labels,
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        stratify=labels
    )
    
    # temp를 val과 test로 분할
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=random_state,
        stratify=temp_labels
    )
    
    print(f"데이터 분할 완료:")
    print(f"  Train: {len(train_paths)}개")
    print(f"  Validation: {len(val_paths)}개")
    print(f"  Test: {len(test_paths)}개")
    
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)

def create_dataloaders(train_data: Tuple[List[str], List[str]],
                      val_data: Tuple[List[str], List[str]],
                      test_data: Optional[Tuple[List[str], List[str]]] = None,
                      processor: Optional[Wav2Vec2Processor] = None) -> Tuple[DataLoader, ...]:
    """데이터로더 생성"""
    
    if processor is None:
        from transformers import Wav2Vec2Processor
        processor = Wav2Vec2Processor.from_pretrained(model_config.model_name)
    
    # 데이터셋 생성
    train_dataset = SpeechEmotionDataset(
        train_data[0], train_data[1], processor, is_training=True
    )
    
    val_dataset = SpeechEmotionDataset(
        val_data[0], val_data[1], processor, is_training=False
    )
    
    # 데이터 collator
    data_collator = DataCollator(processor)
    
    # 데이터로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.dataloader_num_workers,
        collate_fn=data_collator,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.dataloader_num_workers,
        collate_fn=data_collator,
        pin_memory=True
    )
    
    if test_data is not None:
        test_dataset = SpeechEmotionDataset(
            test_data[0], test_data[1], processor, is_training=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=training_config.batch_size,
            shuffle=False,
            num_workers=training_config.dataloader_num_workers,
            collate_fn=data_collator,
            pin_memory=True
        )
        return train_loader, val_loader, test_loader
    
    return train_loader, val_loader

# 예시 사용법을 위한 함수
def prepare_sample_data():
    """샘플 데이터 준비 함수 (실제 데이터가 준비되면 수정 필요)"""
    
    # 현재는 더미 데이터 생성
    print("주의: 실제 데이터가 아닌 더미 데이터입니다.")
    print("실제 데이터 경로를 수정해주세요.")
    
    # 실제 사용 시 아래와 같이 데이터 로드
    # audio_paths, labels = load_dataset_from_directory("./data/emotions/")
    # 또는
    # audio_paths, labels = load_dataset_from_csv("./data/emotions.csv")
    
    # 더미 데이터
    audio_paths = []
    labels = []
    
    return audio_paths, labels