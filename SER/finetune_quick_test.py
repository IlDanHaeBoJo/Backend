#!/usr/bin/env python3
"""
빠른 테스트를 위한 작은 서브셋 파인튜닝
"""

import os
import sys
import re
import torch
import torch.nn as nn
import numpy as np
import librosa
from typing import List, Tuple, Optional, Dict, Any
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 진행 상황 표시
from tqdm import tqdm

# 기본 라이브러리
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report

# PyTorch 관련
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Transformers
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Processor,
    Wav2Vec2Config,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

# 감정 라벨 정의
EMOTION_LABELS = ["Anxious", "Dry", "Kind"]
LABEL2ID = {label: i for i, label in enumerate(EMOTION_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(EMOTION_LABELS)}

# 모델 및 훈련 설정
MODEL_NAME = "kresnik/wav2vec2-large-xlsr-korean"
SAMPLE_RATE = 16000
MAX_DURATION = 10.0

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

def load_dataset_subset(data_dir: str, max_per_class: int = 50) -> Tuple[List[str], List[str]]:
    """데이터셋의 작은 서브셋 로드 (빠른 테스트용)"""
    audio_paths = []
    labels = []
    
    if not os.path.exists(data_dir):
        print(f"❌ 데이터 디렉토리를 찾을 수 없습니다: {data_dir}")
        return audio_paths, labels
    
    person_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    print(f"📁 발견된 person 폴더: {len(person_folders)}개")
    
    emotion_counts = {"Anxious": 0, "Kind": 0, "Dry": 0}
    
    # 처음 몇 개 폴더만 사용하여 빠른 테스트
    selected_folders = sorted(person_folders)[:10]  # 처음 10개 폴더만
    
    for person_folder in tqdm(selected_folders, desc="폴더 처리 중"):
        person_path = os.path.join(data_dir, person_folder)
        wav_path = os.path.join(person_path, "wav_48000")
        
        if not os.path.exists(wav_path):
            continue
        
        for audio_file in os.listdir(wav_path):
            if not audio_file.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
                continue
            
            emotion_label = get_emotion_from_filename(audio_file)
            
            if emotion_label is not None and emotion_counts[emotion_label] < max_per_class:
                audio_path = os.path.join(wav_path, audio_file)
                audio_paths.append(audio_path)
                labels.append(emotion_label)
                emotion_counts[emotion_label] += 1
                
                # 모든 클래스가 충분히 모이면 중단
                if all(count >= max_per_class for count in emotion_counts.values()):
                    break
        
        if all(count >= max_per_class for count in emotion_counts.values()):
            break
    
    print(f"\n📊 로드된 데이터 분포 (빠른 테스트용):")
    for emotion, count in emotion_counts.items():
        percentage = (count / len(audio_paths) * 100) if len(audio_paths) > 0 else 0
        print(f"  {emotion}: {count}개 ({percentage:.1f}%)")
    
    return audio_paths, labels

def preprocess_audio(file_path: str, processor: Wav2Vec2Processor) -> Optional[torch.Tensor]:
    """오디오 전처리"""
    try:
        # 오디오 로드
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # 길이 조정
        target_length = int(SAMPLE_RATE * MAX_DURATION)
        if len(audio) > target_length:
            start_idx = np.random.randint(0, len(audio) - target_length + 1)
            audio = audio[start_idx:start_idx + target_length]
        elif len(audio) < target_length:
            pad_length = target_length - len(audio)
            audio = np.pad(audio, (0, pad_length), mode='constant', constant_values=0)
        
        # 정규화
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.8
        
        # Wav2Vec2 processor로 변환
        inputs = processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )
        
        return inputs.input_values.squeeze(0)
        
    except Exception as e:
        print(f"오디오 전처리 오류: {file_path}, {e}")
        return None

class EmotionDataset(Dataset):
    """음성 감정 분석 데이터셋"""
    
    def __init__(self, audio_paths: List[str], labels: List[str], processor: Wav2Vec2Processor):
        self.audio_paths = audio_paths
        self.labels = labels
        self.processor = processor
        self.encoded_labels = [LABEL2ID[label] for label in labels]
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.encoded_labels[idx]
        
        # 오디오 전처리
        input_values = preprocess_audio(audio_path, self.processor)
        
        if input_values is None:
            # 실패 시 더미 데이터
            input_values = torch.zeros(int(SAMPLE_RATE * MAX_DURATION))
        
        return {
            'input_values': input_values,
            'labels': torch.tensor(label, dtype=torch.long)
        }

def data_collator(features):
    """배치 데이터 정리"""
    input_values = [feature['input_values'] for feature in features]
    labels = [feature['labels'] for feature in features]
    
    # 패딩
    input_values = pad_sequence(input_values, batch_first=True, padding_value=0.0)
    labels = torch.stack(labels)
    
    return {
        'input_values': input_values,
        'labels': labels
    }

def compute_metrics(eval_pred):
    """평가 지표 계산"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1
    }

def create_model_and_processor():
    """모델과 프로세서 생성"""
    print(f"🤖 모델 로딩: {MODEL_NAME}")
    
    # 설정 수정
    config = Wav2Vec2Config.from_pretrained(
        MODEL_NAME,
        num_labels=len(EMOTION_LABELS),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        finetuning_task="emotion_classification"
    )
    
    # 모델 로드
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_NAME,
        config=config,
        ignore_mismatched_sizes=True
    )
    
    # 프로세서 로드
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    
    # Feature extractor 동결
    model.wav2vec2.feature_extractor._freeze_parameters()
    
    print(f"✅ 모델 로딩 완료")
    return model, processor

def main():
    """메인 함수"""
    
    print("🧪 wav2vec2-large-xlsr-korean 빠른 테스트 (작은 서브셋)")
    print("="*70)
    
    # GPU 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  사용 디바이스: {device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
    
    # 데이터 로드 (작은 서브셋)
    data_dir = "/data/ghdrnjs/SER/small/"
    print(f"\n📂 데이터 로딩 중: {data_dir}")
    print(f"   빠른 테스트를 위해 각 클래스당 50개씩만 사용")
    
    audio_paths, labels = load_dataset_subset(data_dir, max_per_class=50)
    
    if len(audio_paths) == 0:
        print("❌ 사용 가능한 데이터를 찾을 수 없습니다.")
        return
    
    print(f"✅ 총 {len(audio_paths)}개 파일 로드 완료")
    
    # 데이터 분할
    print(f"\n🔀 데이터 분할 중...")
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        audio_paths, labels, test_size=0.4, random_state=42, stratify=labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    print(f"  Train: {len(train_paths)}개")
    print(f"  Validation: {len(val_paths)}개")
    print(f"  Test: {len(test_paths)}개")
    
    # 모델과 프로세서 생성
    model, processor = create_model_and_processor()
    model.to(device)
    
    # 데이터셋 생성
    print(f"\n🔄 데이터셋 생성 중...")
    train_dataset = EmotionDataset(train_paths, train_labels, processor)
    val_dataset = EmotionDataset(val_paths, val_labels, processor)
    test_dataset = EmotionDataset(test_paths, test_labels, processor)
    
    # 훈련 설정 (빠른 테스트용)
    output_dir = "./results_quick_test"
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,  # 빠른 테스트를 위해 3 에포크만
        per_device_train_batch_size=2,  # 작은 배치
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=20,  # 적은 웜업
        learning_rate=3e-5,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=5,
        evaluation_strategy="steps",
        eval_steps=10,
        save_steps=20,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=True,
        dataloader_num_workers=1,
        report_to=None,
        push_to_hub=False,
    )
    
    # 트레이너 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # 훈련 시작
    print(f"\n🚀 빠른 테스트 훈련 시작!")
    print(f"⚙️  훈련 설정:")
    print(f"   - 에포크: {training_args.num_train_epochs}")
    print(f"   - 배치 크기: {training_args.per_device_train_batch_size}")
    print(f"   - 학습률: {training_args.learning_rate}")
    print(f"   - 데이터 크기: {len(audio_paths)}개 (작은 서브셋)")
    
    try:
        # 훈련 실행
        trainer.train()
        
        # 최종 평가
        print(f"\n📊 최종 평가 중...")
        eval_results = trainer.evaluate()
        
        print(f"\n🎯 검증 결과:")
        print(f"   - 정확도: {eval_results.get('eval_accuracy', 'N/A'):.4f}")
        print(f"   - F1 스코어: {eval_results.get('eval_f1', 'N/A'):.4f}")
        print(f"   - 손실: {eval_results.get('eval_loss', 'N/A'):.4f}")
        
        # 테스트 평가
        test_results = trainer.evaluate(test_dataset)
        print(f"\n🧪 테스트 결과:")
        print(f"   - 정확도: {test_results.get('eval_accuracy', 'N/A'):.4f}")
        print(f"   - F1 스코어: {test_results.get('eval_f1', 'N/A'):.4f}")
        print(f"   - 손실: {test_results.get('eval_loss', 'N/A'):.4f}")
        
        print(f"\n🎉 빠른 테스트 완료!")
        print(f"   이제 전체 데이터로 본격적인 훈련을 진행할 수 있습니다.")
        print(f"   전체 훈련: python finetune_simple.py")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  훈련이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 훈련 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()