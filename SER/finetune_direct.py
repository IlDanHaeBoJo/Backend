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
from audiomentations import Compose, PitchShift, TimeStretch, AddGaussianNoise, Shift, Gain, ClippingDistortion
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

# 감정 라벨 정의
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

AUGMENTATION = Compose([
    PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
    TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    Gain(min_gain_in_db=-6.0, max_gain_in_db=6.0, p=0.5),
    ClippingDistortion(0, 2.0, p=0.3)

])

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

# --- 유틸리티 함수 ---

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
                # 숫자가 여러 자리일 경우 마지막 한 자리만 사용
                return int(match.group(1)) % 10
            return None
    except (ValueError, AttributeError):
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

def load_dataset_subset(data_dir: str, max_per_class: int) -> Tuple[List[str], List[str]]:
    audio_paths = []
    labels = []
    emotion_counts = {label: 0 for label in EMOTION_LABELS}
    
    person_folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
    print(f"📁 발견된 person 폴더: {len(person_folders)}개")
    
    for person_folder in tqdm(person_folders, desc="데이터셋 로딩 중"):
        wav_path = os.path.join(data_dir, person_folder, "wav_48000")
        if not os.path.exists(wav_path):
            continue
        
        for audio_file in os.listdir(wav_path):
            if not audio_file.lower().endswith('.wav'):
                continue
            
            emotion_label = get_emotion_from_filename(audio_file)
            if emotion_label and emotion_counts[emotion_label] < max_per_class:
                audio_paths.append(os.path.join(wav_path, audio_file))
                labels.append(emotion_label)
                emotion_counts[emotion_label] += 1
        
        if all(count >= max_per_class for count in emotion_counts.values()):
            break

    print(f"\n📊 로드된 데이터 분포 (빠른 테스트용):")
    for emotion, count in emotion_counts.items():
        percentage = (count / len(audio_paths) * 100) if len(audio_paths) > 0 else 0
        print(f"  {emotion}: {count}개 ({percentage:.1f}%)")
            
    return audio_paths, labels

def preprocess_audio(file_path: str, processor: Wav2Vec2Processor, is_training: bool = False) -> Optional[torch.Tensor]:
    """오디오 전처리"""
    try:
        # 오디오 로드
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        if is_training:
            audio = AUGMENTATION(samples=audio, sample_rate=sr)
       
        # 길이 조정
        target_length = int(SAMPLE_RATE * MAX_DURATION)
        if len(audio) > target_length:
            start_idx = np.random.randint(0, len(audio) - target_length + 1)
            audio = audio[start_idx:start_idx + target_length]
        elif len(audio) < target_length:
            pad_length = target_length - len(audio)
            audio = np.pad(audio, (0, pad_length), mode='constant')
        
        inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        return inputs.input_values.squeeze(0)
    except Exception as e:
        print(f"오디오 전처리 오류: {file_path}, {e}")
        return None

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

# 데이터 전체를 스캔해서 (경로, 감정, 화자, 스크립트ID) 인덱스 생성
def build_corpus_index(data_dir: str,
                       accept_exts={'.wav', '.flac'},
                       require_emotion=True) -> List[Dict[str, Any]]:
    """
    return: [{"path": p, "emotion": e, "speaker": s, "content_id": c}, ...]
    """
    index = []
    speakers = sorted([d for d in os.listdir(data_dir)
                       if os.path.isdir(os.path.join(data_dir, d))])
    print(f"📁 화자 폴더 수: {len(speakers)}")

    for spk in tqdm(speakers, desc="인덱스 구축"):
        spk_dir = os.path.join(data_dir, spk)
        # 하위 디렉토리를 재귀적으로 탐색 (감정별 폴더/단일 폴더 둘 다 대응)
        for root, _, files in os.walk(spk_dir):
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                if ext not in accept_exts:
                    continue
                path = os.path.join(root, fn)

                # 감정 라벨
                emo = infer_emotion_from_path(path)
                if require_emotion and emo not in EMOTION_LABELS:
                    # 감정 미매칭 샘플은 제외
                    continue

                # 스크립트(대화) ID: 파일명에서 추출 (기존 규칙 그대로)
                cid = extract_number_from_filename(fn, type="content")
                if cid is None:
                    # 스크립트 ID 없으면 제외(불교차 조건을 보장하기 위해)
                    continue

                index.append({
                    "path": path,
                    "emotion": emo,
                    "speaker": spk,
                    "content_id": cid
                })
    print(f"✅ 인덱스 샘플 수: {len(index)}")
    return index

def split_speaker_and_content(
    index: List[Dict[str, Any]],
    val_content_ratio: float = 0.2,
    test_content_ratio: float = 0.2,
    val_speaker_ratio: float = 0.2,
    test_speaker_ratio: float = 0.2,
    seed: int = 42,
    fixed_val_content_ids: Optional[List[int]] = None,
    fixed_test_content_ids: Optional[List[int]] = None,
) -> Tuple[Tuple[List[str], List[str]],
           Tuple[List[str], List[str]],
           Tuple[List[str], List[str]]]:
    """
    index: build_corpus_index() 반환 리스트
    반환: ((train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels))
    """
    rng = random.Random(seed)

    # 전체 스크립트 ID, 화자 목록
    all_contents = sorted(set([it["content_id"] for it in index]))
    all_speakers = sorted(set([it["speaker"] for it in index]))

    # --- 2-1) 스크립트(대화) 불교차 세트 만들기
    if fixed_val_content_ids is not None and fixed_test_content_ids is not None:
        val_contents = set(fixed_val_content_ids)
        test_contents = set(fixed_test_content_ids)
        train_contents = set(all_contents) - val_contents - test_contents
    else:
        contents = all_contents[:]
        rng.shuffle(contents)
        n_val = max(1, int(len(contents) * val_content_ratio))
        n_test = max(1, int(len(contents) * test_content_ratio))
        val_contents = set(contents[:n_val])
        test_contents = set(contents[n_val:n_val+n_test])
        train_contents = set(contents[n_val+n_test:])

    # --- 2-2) 화자 불교차 세트 만들기
    speakers = all_speakers[:]
    rng.shuffle(speakers)
    n_val_spk = max(1, int(len(speakers) * val_speaker_ratio))
    n_test_spk = max(1, int(len(speakers) * test_speaker_ratio))
    val_speakers = set(speakers[:n_val_spk])
    test_speakers = set(speakers[n_val_spk:n_val_spk+n_test_spk])
    train_speakers = set(speakers[n_val_spk+n_test_spk:])

    # --- 2-3) 교집합 제거: 두 조건(화자 세트, 스크립트 세트)을 동시에 만족하는 샘플만 채택
    train_items = [it for it in index
                   if it["speaker"] in train_speakers and it["content_id"] in train_contents]
    val_items   = [it for it in index
                   if it["speaker"] in val_speakers and it["content_id"] in val_contents]
    test_items  = [it for it in index
                   if it["speaker"] in test_speakers and it["content_id"] in test_contents]

    # --- 2-4) 점검 출력
    def summarize(name, items):
        spks = sorted(set([it["speaker"] for it in items]))
        cids = sorted(set([it["content_id"] for it in items]))
        emo_cnt = Counter([it["emotion"] for it in items])
        print(f"\n[{name}] 샘플: {len(items)}, 화자: {len(spks)}, 스크립트ID: {len(cids)}")
        print(f"  감정분포: {dict(emo_cnt)}")
        print(f"  예시 화자(최대 10): {spks[:10]}")
        print(f"  예시 스크립트ID(최대 20): {cids[:20]}")

    summarize("TRAIN", train_items)
    summarize("VAL",   val_items)
    summarize("TEST",  test_items)

    # --- 2-5) 교차 검증: 화자/스크립트 불교차 여부 확인
    assert set([it["speaker"] for it in train_items]).isdisjoint(set([it["speaker"] for it in val_items + test_items])), \
        "Train 화자가 Val/Test와 겹칩니다."
    assert set([it["speaker"] for it in val_items]).isdisjoint(set([it["speaker"] for it in test_items])), \
        "Val 화자가 Test와 겹칩니다."
    assert set([it["content_id"] for it in train_items]).isdisjoint(set([it["content_id"] for it in val_items + test_items])), \
        "Train 스크립트ID가 Val/Test와 겹칩니다."
    assert set([it["content_id"] for it in val_items]).isdisjoint(set([it["content_id"] for it in test_items])), \
        "Val 스크립트ID가 Test와 겹칩니다."

    # --- 2-6) 최종 리스트 변환
    def to_xy(items):
        return [it["path"] for it in items], [it["emotion"] for it in items]

    return to_xy(train_items), to_xy(val_items), to_xy(test_items)


class EmotionDataset(Dataset):
    def __init__(self, audio_paths: List[str], labels: List[str], processor: Wav2Vec2Processor, is_training: bool = True):
        self.data_dir = "/data/ghdrnjs/SER/small/"
        self.audio_paths = audio_paths
        self.labels = labels
        self.processor = processor
        self.encoded_labels = [LABEL2ID[label] for label in labels]
        self.is_training = is_training

        with open("script.json", "r", encoding="utf-8") as f:
            self.text_json = json.load(f)

        self.spk2id = build_speaker_mapping(audio_paths, self.data_dir)        
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        emotion_label = self.encoded_labels[idx]
        file_number = extract_number_from_filename(audio_path, type="content")
        
        content_text = ""
        if file_number is not None and str(file_number) in self.text_json:
            content_text = self.text_json[str(file_number)]
        
        input_values = preprocess_audio(audio_path, self.processor, self.is_training)
        if input_values is None:
            input_values = torch.zeros(int(SAMPLE_RATE * MAX_DURATION))
        
        spk_idx_tensor = None
        if self.spk2id is not None:
            spk_str = extract_speaker_id(audio_path, self.data_dir)
            if spk_str in self.spk2id:
                spk_idx = self.spk2id[spk_str]
                spk_idx_tensor = torch.tensor(spk_idx, dtype=torch.long)

        return {
            'input_values': input_values,
            'emotion_labels': torch.tensor(emotion_label, dtype=torch.long),
            'content_text': content_text,
            'speaker_id': spk_idx_tensor
        }

def collate_fn(batch: List[Dict[str, any]]) -> Dict[str, torch.Tensor]:
    input_values = [item['input_values'] for item in batch]
    emotion_labels = [item['emotion_labels'] for item in batch]
    content_texts = [item['content_text'] for item in batch]
    spk_list = [item.get('speaker_id', None) for item in batch]
    
    padded_input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True, padding_value=0.0)
    
    tokenized_contents = []
    content_lengths = []
    for text in content_texts:
        ids = [CHAR2ID.get(char, CHAR2ID['<unk>']) for char in text]
        tokenized_contents.append(torch.tensor(ids, dtype=torch.long))
        content_lengths.append(len(ids))

    padded_content_labels = torch.nn.utils.rnn.pad_sequence(
        tokenized_contents, 
        batch_first=True, 
        padding_value=CHAR2ID['<pad>']
    )

    # attention_mask for audio
    attention_mask = torch.ones_like(padded_input_values, dtype=torch.long)
    for i, seq in enumerate(input_values):
        attention_mask[i, len(seq):] = 0

    if all((s is not None) and isinstance(s, torch.Tensor) for s in spk_list):
        # 각 요소가 0-dim long tensor라면 stack -> (B,)
        speaker_ids = torch.stack(spk_list)            # shape: (B,)
        speaker_ids = speaker_ids.view(-1).long()      # 보정
    else:
        speaker_ids = None

    return {
        'input_values': padded_input_values,
        'attention_mask': attention_mask,
        'labels': torch.stack(emotion_labels),
        'content_labels': padded_content_labels,
        'content_labels_lengths': torch.tensor(content_lengths, dtype=torch.long),
        'speaker_ids': speaker_ids,
    }

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
    # 1) 전체 freeze
    for p in model.wav2vec2.parameters():
        p.requires_grad = False
    # 2) 마지막 K개 block만 unfreeze
    layers = model.wav2vec2.encoder.layers
    num_layers = len(layers)
    for i in range(num_layers - last_k, num_layers):
        for p in layers[i].parameters():
            p.requires_grad = True
    # 3) 헤드/적대자/풀러/프로젝터는 항상 학습
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
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [p for n, p in model.named_parameters() if "wav2vec2" in n],
    #         "lr": 1e-5,  # 사전 훈련된 Backbone은 낮은 학습률
    #     },
    #     {
    #         "params": [p for n, p in model.named_parameters() if "wav2vec2" not in n],
    #         "lr": 1e-4,  # 새로 추가된 Classifier와 Adversary는 높은 학습률
    #     },
    # ]
    # optimizer = optim.AdamW(optimizer_grouped_parameters, weight_decay=0.01)

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

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": 5e-6, "weight_decay": 0.01},
            {"params": head_params,     "lr": 1e-4, "weight_decay": 0.01},
        ]
    )
    
    # 스케줄러 설정
    total_steps = len(train_loader) * num_epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    best_f1 = 0
    best_model_state = None
    
    print(f"\n🚀 훈련 시작!")
    print(f"   총 에포크: {num_epochs}")
    print(f"   배치 수: {len(train_loader)}")
    print(f"   총 스텝: {total_steps}")
    
    max_adv = 0.1
    warmup_epochs = 1.0


    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        
        # 훈련
        model.train()
        train_loss = 0
        for step, batch in enumerate(train_loader):
            progress = min(1.0, (epoch + step/len(train_loader)) / warmup_epochs)
            adv_lambda = max_adv * progress
        train_predictions = []
        train_true_labels = []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
        for batch in tqdm(progress_bar):
            optimizer.zero_grad()
            
            
            # speaker_ids 넘겨줘야 함
            outputs = model(
                input_values=batch['input_values'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['labels'].to(device),
                content_labels=batch['content_labels'].to(device),
                content_labels_lengths=batch['content_labels_lengths'].to(device),
                adv_lambda=0.1,
                speaker_ids=batch['speaker_ids'].to(device)
            )
            
            loss = outputs['loss']
            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })
        
        # 훈련 결과
        train_loss /= len(train_loader)
        # train_acc = accuracy_score(train_true_labels, train_predictions)
        # train_f1 = f1_score(train_true_labels, train_predictions, average='weighted')
        
        # print(f"🎯 훈련 결과 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        
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
    
    print(f"\n📊 분할 결과:")
    print(f"  Train: {len(train_paths)}개")
    print(f"  Validation: {len(val_paths)}개") 
    print(f"  Test: {len(test_paths)}개")
    
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
        model_save_path = os.path.join(output_dir, "adversary_content_speaker_model_augment_v2_epoch_5_last_k_4")
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
