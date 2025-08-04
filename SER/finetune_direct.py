#!/usr/bin/env python3
"""
직접 훈련 루프 구현 버전 (Trainer 사용 안함)
라이브러리 충돌 문제 해결
"""

import os
import sys
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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

# Transformers (Trainer 제외)
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Processor,
    Wav2Vec2Config
)

# 감정 라벨 정의
"""
"Anxious" : 0
"Kind" : 1
"Dry" : 2
"""
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

def load_dataset_subset(data_dir: str, max_per_class: int = 30) -> Tuple[List[str], List[str]]:
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
    # selected_folders = sorted(person_folders)[:5]  # 처음 5개 폴더만
    selected_folders = sorted(person_folders)
    
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

def collate_fn(batch):
    """배치 데이터 정리"""
    input_values = [item['input_values'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # 패딩 (가장 긴 것에 맞춤)
    max_len = max(x.size(0) for x in input_values)
    padded_inputs = []
    
    for x in input_values:
        if x.size(0) < max_len:
            pad_len = max_len - x.size(0)
            x = torch.cat([x, torch.zeros(pad_len)], dim=0)
        padded_inputs.append(x)
    
    input_values = torch.stack(padded_inputs)
    labels = torch.stack(labels)
    
    return {
        'input_values': input_values,
        'labels': labels
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

def evaluate_model(model, dataloader, device, criterion):
    """모델 평가"""
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="평가 중"):
            input_values = batch['input_values'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_values=input_values, labels=labels)
            loss = outputs.loss
            
            total_loss += loss.item()
            
            # 예측 결과 저장
            preds = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1,
        'predictions': predictions,
        'true_labels': true_labels
    }

def train_model(model, train_loader, val_loader, device, num_epochs=3, learning_rate=3e-5):
    """직접 훈련 루프"""
    
    # 옵티마이저 설정
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # 스케줄러 설정
    total_steps = len(train_loader) * num_epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # 손실 함수
    criterion = nn.CrossEntropyLoss()
    
    best_f1 = 0
    best_model_state = None
    
    print(f"\n🚀 훈련 시작!")
    print(f"   총 에포크: {num_epochs}")
    print(f"   배치 수: {len(train_loader)}")
    print(f"   총 스텝: {total_steps}")
    
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        
        # 훈련
        model.train()
        train_loss = 0
        train_predictions = []
        train_true_labels = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1} 훈련")
        for batch_idx, batch in enumerate(progress_bar):
            input_values = batch['input_values'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_values=input_values, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            # 예측 결과 저장
            preds = torch.argmax(outputs.logits, dim=-1)
            train_predictions.extend(preds.cpu().numpy())
            train_true_labels.extend(labels.cpu().numpy())
            
            # 진행 상황 업데이트
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })
        
        # 훈련 결과
        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_true_labels, train_predictions)
        train_f1 = f1_score(train_true_labels, train_predictions, average='weighted')
        
        print(f"🎯 훈련 결과 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        
        # 검증
        print("📊 검증 중...")
        val_results = evaluate_model(model, val_loader, device, criterion)
        
        print(f"🔍 검증 결과 - Loss: {val_results['loss']:.4f}, Acc: {val_results['accuracy']:.4f}, F1: {val_results['f1']:.4f}")
        
        # 최고 성능 모델 저장
        if val_results['f1'] > best_f1:
            best_f1 = val_results['f1']
            best_model_state = model.state_dict().copy()
            print(f"✨ 새로운 최고 F1 점수: {best_f1:.4f}")
    
    # 최고 성능 모델 로드
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n💫 최고 성능 모델 로드 완료 (F1: {best_f1:.4f})")
    
    return model

def main():
    """메인 함수"""
    
    print("🧪 wav2vec2-large-xlsr-korean 직접 훈련 (빠른 테스트)")
    print("="*70)
    
    # GPU 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  사용 디바이스: {device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
    
    # 데이터 로드 (작은 서브셋)
    data_dir = "/data/ghdrnjs/SER/small/"
    print(f"\n📂 데이터 로딩 중: {data_dir}")
    print(f"   빠른 테스트를 위해 각 클래스당 30개씩만 사용")
    
    audio_paths, labels = load_dataset_subset(data_dir, max_per_class=3000)
    
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
    
    # 데이터로더 생성
    batch_size = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    try:
        # 훈련 실행 (빠른 테스트용 1에포크)
        model = train_model(model, train_loader, val_loader, device, num_epochs=1, learning_rate=3e-5)
        
        # 최종 테스트 평가
        print(f"\n🧪 최종 테스트 평가...")
        test_results = evaluate_model(model, test_loader, device, nn.CrossEntropyLoss())
        
        print(f"\n🎯 최종 테스트 결과:")
        print(f"   - 정확도: {test_results['accuracy']:.4f}")
        print(f"   - F1 스코어: {test_results['f1']:.4f}")
        print(f"   - 손실: {test_results['loss']:.4f}")
        
        # 상세 분류 리포트
        print(f"\n📋 상세 분류 리포트:")
        report = classification_report(
            test_results['true_labels'], 
            test_results['predictions'],
            target_names=EMOTION_LABELS,
            digits=4
        )
        print(report)
        
        # 학습된 모델 저장
        print(f"\n💾 모델 저장 중...")
        output_dir = "./results_quick_test"
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 모델 상태 저장 (transformers 방식)
        model_save_path = os.path.join(output_dir, "final_model")
        os.makedirs(model_save_path, exist_ok=True)
        
        # 모델 저장
        model.save_pretrained(model_save_path)
        processor.save_pretrained(model_save_path)
        
        print(f"✅ 모델 저장 완료: {model_save_path}")
        print(f"\n🎉 빠른 테스트 완료!")
        print(f"   모든 파이프라인이 정상 작동합니다.")
        print(f"   저장된 모델로 테스트: python test_my_voice.py your_audio.wav --model_path {model_save_path}")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  훈련이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 훈련 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()