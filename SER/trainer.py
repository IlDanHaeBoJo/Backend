import torch
import torch.nn as nn
import torch.optim as optim
from transformers import Wav2Vec2Processor, Wav2Vec2Config
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

from . import config
from .datasets import EmotionDataset, collate_fn, preprocess_audio
from .utils import setup_logging, CheckpointManager, validate_audio_files
from .data_utils import split_speaker_and_content, build_corpus_index, extract_speaker_id, extract_number_from_filename
from .Wav2Vec2_seq_clf import custom_Wav2Vec2ForEmotionClassification
from .model_utils import enable_last_k_blocks

from collections import Counter
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score

import wandb
import random

def create_model_and_processor(freeze_base_model: bool = True):
    """모델과 프로세서 생성"""
    print(f"🤖 모델 로딩: {config.model.model_name}")
    
    # 설정 수정
    model_config = Wav2Vec2Config.from_pretrained(
        config.model.model_name,
        num_labels=config.model.num_labels,
        label2id=config.model.label2id,
        id2label=config.model.id2label,
        finetuning_task="emotion_classification"
    )
    
    if config.char_vocab:
        model_config.char_vocab_size = len(config.char_vocab)
    
    model = custom_Wav2Vec2ForEmotionClassification.from_pretrained(
        config.model.model_name,
        config=model_config,
        ignore_mismatched_sizes=True
    )
    
    processor = Wav2Vec2Processor.from_pretrained(config.model.model_name)
    
    if freeze_base_model:
        for param in model.wav2vec2.parameters():
            param.requires_grad = False
        print("✅ Base model 파라미터 고정")
    
    return model, processor

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
    
    max_adv = 0.1
    warmup_epochs = 1.0
    class_weights = torch.tensor([2.0, 1.5, 0.7], dtype=torch.float32).to(device)

    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        
        # 훈련
        model.train()
        train_loss = 0
        train_predictions = []
        train_true_labels = []
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
            
            # 예측 결과 저장
            preds = torch.argmax(outputs['emotion_logits'], dim=-1)
            train_predictions.extend(preds.cpu().numpy())
            train_true_labels.extend(batch['labels'].cpu().numpy())
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })
        
        train_loss /= len(train_loader)
        
        # 훈련 정확도 및 F1 계산
        train_accuracy = accuracy_score(train_true_labels, train_predictions)
        train_f1 = f1_score(train_true_labels, train_predictions, average='weighted')

        # Weight & Biases 로깅
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_accuracy,
            "train/f1": train_f1
        })

        # 검증
        print("📊 검증 중...")
        val_results = evaluate_model(model, val_loader, device, epoch=epoch, is_training=True)
        
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

def evaluate_model(model, dataloader, device, epoch=None, is_training=False):
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

    if is_training:
        wandb.log({
            "epoch" : epoch,
            "val/loss": loss,
            "val/accuracy": accuracy,
            "val/f1": f1,
        })
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1,
        'predictions': predictions,
        'true_labels': true_labels
    }

def main():
    """메인 훈련 함수"""
    logger = setup_logging("INFO", "training.log")
    logger.info("🚀 SER 모델 훈련 시작")
    
    logger.info(f"모델: {config.model.model_name}")
    logger.info(f"감정 라벨: {config.model.emotion_labels}")
    logger.info(f"배치 크기: {config.training.batch_size}")
    logger.info(f"학습률: {config.training.learning_rate}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"사용 장치: {device}")
    model, processor = create_model_and_processor()
    enable_last_k_blocks(model, last_k=4)
    model.to(device)

    # Weight & Biases 초기화
    wandb.init(
    project="Speech_Emotion_Recognition",
    name="3 Class Classification",

    config={
        "learning_rate": config.training.learning_rate,
        "epochs": config.training.num_epochs,
        "batch_size": config.training.batch_size,
        "architecture": "custom Wav2Vec2",
    }
    )

    
    # 데이터 로드
    index = build_corpus_index(config.paths.data_dir)
    valid_paths, invalid_paths = validate_audio_files([item["path"] for item in index])
    
    if invalid_paths:
        logger.warning(f"유효하지 않은 파일 {len(invalid_paths)}개 제외")
    
    # 데이터 분할
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = \
        split_speaker_and_content(
        index,
        val_content_ratio=0.2,
        test_content_ratio=0.2,
        val_speaker_ratio=0.2,
        test_speaker_ratio=0.2,
    )
    train_speakers = sorted({extract_speaker_id(p, config.paths.data_dir) for p in train_paths})
    num_speakers = len(train_speakers)
    print(f"🔍 화자 수: {num_speakers}")

    print(f"\n📊 분할 결과:")
    print(f"  Train: {len(train_paths)}개")
    print(f"  Validation: {len(val_paths)}개")
    print(f"  Test: {len(test_paths)}개")

    train_emotion_dist = Counter(train_labels)
    val_emotion_dist = Counter(val_labels)
    test_emotion_dist = Counter(test_labels)
    
    print(f"\n📈 감정별 분포:")
    print(f"  Train: {dict(train_emotion_dist)}")
    print(f"  Validation: {dict(val_emotion_dist)}")
    print(f"  Test: {dict(test_emotion_dist)}")
    
    # 데이터셋 생성
    train_dataset = EmotionDataset(train_paths, train_labels, processor, is_training=True)
    val_dataset = EmotionDataset(val_paths, val_labels, processor, is_training=False)
    test_dataset = EmotionDataset(test_paths, test_labels, processor, is_training=False)
    
    # 데이터로더 생성
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # 체크포인트 매니저
    checkpoint_manager = CheckpointManager()
    
    # 훈련 실행
    try:
        # 훈련 실행 (빠른 테스트용 1에포크)
        model = train_model(model, 
                            train_loader, 
                            val_loader, 
                            device, 
                            num_epochs=config.training.num_epochs, 
                            learning_rate=config.training.learning_rate)
        
        # 최종 테스트 평가
        print(f"\n🧪 최종 테스트 평가...")
        test_results = evaluate_model(model, test_loader, device, is_training=False)
        
        print(f"\n🎯 최종 테스트 결과:")
        print(f"   - 정확도: {test_results['accuracy']:.4f}")
        print(f"   - F1 스코어: {test_results['f1']:.4f}")
        print(f"   - 손실: {test_results['loss']:.4f}")
        
        # 상세 분류 리포트
        print(f"\n📋 상세 분류 리포트:")
        report = classification_report(
            test_results['true_labels'], 
            test_results['predictions'],
            target_names=config.model.emotion_labels,
            digits=4
        )
        print(report)
        
        # 학습된 모델 저장
        print(f"\n💾 모델 저장 중...")
        output_dir = config.paths.output_dir
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
    
    logger.info("✅ 훈련 완료")





if __name__ == "__main__":
    main()