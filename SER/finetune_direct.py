#!/usr/bin/env python3
"""
직접 훈련 루프 구현 버전 (Trainer 사용 안함)
+ 적대적 학습 추가
"""

import os
import sys
import json
import re
import random
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
from audiomentations import Compose, PitchShift, TimeStretch, AddGaussianNoise, Shift, Gain, RoomSimulator, HighPassFilter, LowPassFilter
from typing import List, Tuple, Optional, Dict, Any, Literal
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report

from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Processor,
    Wav2Vec2Config
)
# 적대적 학습을 위한 custom model 임포트
from Wav2Vec2_seq_clf import custom_Wav2Vec2ForEmotionClassification
from data_utils import *
from datasets import EmotionDataset, collate_fn
from config import Config


# 감정 라벨 정의 -> config.py로 옮김
"""
"Anxious" : 0
"Kind" : 1
"Dry" : 2
"""
EMOTION_LABELS = ["Anxious", "Dry", "Kind"]
LABEL2ID = {label: i for i, label in enumerate(EMOTION_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(EMOTION_LABELS)}

MODEL_NAME = "kresnik/wav2vec2-large-xlsr-korean"
SAMPLE_RATE = 16000
MAX_DURATION = 10.0


# Character Vocabulary 로드
try:
    with open('char_to_id.json', 'r', encoding='utf-8') as f:
        CHAR2ID = json.load(f)
    CHAR_VOCAB = list(CHAR2ID.keys())
    ID2CHAR = {i: char for char, i in CHAR2ID.items()}
    print(f"✅ Character Vocabulary 로드 완료 ({len(CHAR_VOCAB)}개)")
except FileNotFoundError:
    print("❌ 'char_to_id.json'을 찾을 수 없습니다. 먼저 build_vocab.py를 실행하세요.")
    sys.exit(1)



def build_augment(epoch, total_epochs):
    scale = 0.6 if epoch < 0.2 * total_epochs else 1.0

    return Compose([
        RoomSimulator(p=0.20 * scale),
        HighPassFilter(min_cutoff_freq=60, max_cutoff_freq=120, p=0.15 * scale),
        LowPassFilter(min_cutoff_freq=3500, max_cutoff_freq=6000, p=0.15 * scale),

        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.006 if scale<1 else 0.008, p=0.35 * scale),
        Gain(min_gain_in_db=-2.0 if scale<1 else -3.0,
             max_gain_in_db= 2.0 if scale<1 else  3.0, p=0.35 * scale),

        Shift(min_fraction=-0.03 if scale<1 else -0.05,
              max_fraction= 0.03 if scale<1 else  0.05, p=0.35 * scale),

        PitchShift(min_semitones=-1, max_semitones=1, p=0.20 * scale),
        TimeStretch(min_rate=0.98 if scale<1 else 0.97,
                    max_rate=1.02 if scale<1 else 1.03, p=0.15 * scale),
    ])





def create_model_and_processor(freeze_base_model: bool = True, num_speakers: int = 500):
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
    config.char_vocab_size = len(CHAR_VOCAB) # 적대자 모델을 위한 설정
    config.num_speakers = num_speakers
    
    model = custom_Wav2Vec2ForEmotionClassification.from_pretrained(
        MODEL_NAME,
        config=config,
        ignore_mismatched_sizes=True
    )
    
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    
    if freeze_base_model:
        for param in model.wav2vec2.parameters():
            param.requires_grad = False

    return model, processor

def enable_last_k_blocks(model, last_k: int = 4):
    for p in model.wav2vec2.parameters():
        p.requires_grad = False
    
    layers = model.wav2vec2.encoder.layers
    num_layers = len(layers)
    for i in range(num_layers - last_k, num_layers):
        for p in layers[i].parameters():
            p.requires_grad = True
    
    for name, module in model.named_modules():
        if any(k in name for k in ["classifier", "adversary", "speaker_adversary", "pooler", "stats_projector", "projector"]):
            for p in module.parameters():
                p.requires_grad = True

def evaluate_model(model, dataloader, device):
    """모델 평가"""
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="평가 중"):
            
            outputs = model(
                input_values=batch['input_values'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['labels'].to(device),
                content_labels=batch['content_labels'].to(device),
                content_labels_lengths=batch['content_labels_lengths'].to(device),
                adv_lambda=0.0, # 평가 시에는 적대적 손실 반영 안함,
                speaker_ids = None,
            )
            
            loss = outputs['loss']
            
            # 훈련 중이 아닐 때도 loss가 계산되도록 adv_lambda=0.0으로 호출
            # 만약 loss가 None이면 (test set 등에서 content_label이 없을 경우), 감정 손실만 계산
            if loss is None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(outputs['emotion_logits'].view(-1, model.config.num_labels), batch['labels'].to(device).view(-1))

            total_loss += loss.item()
            
            # 예측 결과 저장
            preds = torch.argmax(outputs['emotion_logits'], dim=-1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch['labels'].cpu().numpy())
    
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
    
    # 옵티마이저 설정 (차등 학습률 적용)
    print("🚀 옵티마이저 설정 (차등 학습률 적용)")
    backbone_params = []
    head_params = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("wav2vec2."):
            backbone_params.append(p)
        else:
            # pooler, stats_projector, projector(부모), classifier, adversaries 등
            head_params.append(p)

    for n, p in model.adversary.named_parameters():
        p.requires_grad = False
    model.adversary.eval()

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": 5e-6, "weight_decay": 0.01},
            {"params": head_params,     "lr": 1e-4, "weight_decay": 0.01},
        ]
    )
    
    # 스케줄러 설정
    total_steps = len(train_loader) * num_epochs
    accumulation_steps = 16
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    best_f1 = 0
    best_model_state = None
    
    print(f"\n🚀 훈련 시작!")
    print(f"   총 에포크: {num_epochs}")
    print(f"   배치 수: {len(train_loader)}")
    print(f"   총 스텝: {total_steps}")
    
    max_adv = 0.05
    warmup_epochs = 1.0
    class_weights = torch.tensor([2.0, 1.5, 0.7], dtype=torch.float32).to(device)


    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        
        # 훈련
        model.train()
        train_loss = 0
        optimizer.zero_grad()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
        for step, batch in enumerate(progress_bar):
            current_step = epoch * len(train_loader) + step
            total_warmup_steps = warmup_epochs * len(train_loader)
            progress = min(1.0, current_step / total_warmup_steps)
            current_adv_lambda = max_adv * progress
            
            outputs = model(
                input_values=batch['input_values'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['labels'].to(device),
                content_labels=batch['content_labels'].to(device),
                content_labels_lengths=batch['content_labels_lengths'].to(device),
                adv_lambda=current_adv_lambda,
                speaker_ids=batch['speaker_ids'].to(device),
                class_weights = class_weights
            )
            
            loss = outputs['loss']
            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                continue

            loss /= accumulation_steps
            loss.backward()

            if (step + 1) % accumulation_steps == 0:
              torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
              optimizer.step()
              scheduler.step()
              optimizer.zero_grad()
            
            train_loss += loss.item()
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr_backbone': f'{scheduler.get_last_lr()[0]:.6f}',
                'lr_head': f'{scheduler.get_last_lr()[1]:.6f}',
                "adv": f'{adv_lambda:.3f}'
            })
        
        train_loss /= len(train_loader)
        
        # 검증
        print("📊 검증 중...")
        val_results = evaluate_model(model, val_loader, device)
        
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

# --- 메인 실행 함수 ---
def main():
    """메인 함수"""
    
    print("🧪 wav2vec2-large-xlsr-korean 직접 훈련 (빠른 테스트)")
    print("="*70)
    
    # GPU 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  사용 디바이스: {device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
    
    data_dir = "/data/ghdrnjs/SER/small/"
    print(f"\n📂 데이터 로딩 중: {data_dir}")
    index = build_corpus_index(data_dir, require_emotion=True)
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = \
        split_speaker_and_content(
            index,
            val_content_ratio=0.2,
            test_content_ratio=0.2,
            val_speaker_ratio=0.2,
            test_speaker_ratio=0.2,
            seed=42,
            # 예시) 특정 대화 ID를 고정하고 싶다면 주석 해제
            # fixed_val_content_ids=[22, 35],
            # fixed_test_content_ids=[27, 41],
        )
    print(f"\n📊 분할 결과:")
    print(f"  Train: {len(train_paths)}개")
    print(f"  Validation: {len(val_paths)}개")
    print(f"  Test: {len(test_paths)}개")

    train_speakers = sorted({extract_speaker_id(p, data_dir) for p in train_paths})
    num_speakers = len(train_speakers)
    print(f"🔍 화자 수: {num_speakers}")

    # audio_paths, labels = load_dataset_subset(data_dir, max_per_class=3000)
    # print(f"✅ 총 {len(audio_paths)}개 파일 로드 완료")
    
    # 데이터 분할 (파일명 마지막 숫자 기준)
    
    # (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = split_data_by_last_digit(audio_paths, labels)
    
    # 각 세트의 감정별 분포 확인
    from collections import Counter
    train_emotion_dist = Counter(train_labels)
    val_emotion_dist = Counter(val_labels)
    test_emotion_dist = Counter(test_labels)
    
    print(f"\n📈 감정별 분포:")
    print(f"  Train: {dict(train_emotion_dist)}")
    print(f"  Validation: {dict(val_emotion_dist)}")
    print(f"  Test: {dict(test_emotion_dist)}")
    
    # 모델과 프로세서 생성
    model, processor = create_model_and_processor(num_speakers=num_speakers)
    enable_last_k_blocks(model, last_k=4)
    model.to(device)
    
    # 데이터셋 생성
    print(f"\n🔄 데이터셋 생성 중...")
    train_dataset = EmotionDataset(train_paths, train_labels, processor)
    val_dataset = EmotionDataset(val_paths, val_labels, processor, is_training=False)
    test_dataset = EmotionDataset(test_paths, test_labels, processor, is_training=False)
    
    # 데이터로더 생성
    batch_size = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    try:
        # 훈련 실행 (빠른 테스트용 1에포크)
        model = train_model(model, train_loader, val_loader, device, num_epochs=5, learning_rate=3e-5)
        
        # 최종 테스트 평가
        print(f"\n🧪 최종 테스트 평가...")
        test_results = evaluate_model(model, test_loader, device)
        
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
        model_save_path = os.path.join(output_dir, "adversary_no_content_speaker_model_augment_v3_epoch_5_last_k_4")
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
