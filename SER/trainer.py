"""
음성 감정 분석 모델 훈련 모듈
"""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    Wav2Vec2Processor
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
import json
import matplotlib.pyplot as plt
import seaborn as sns

from .model import SpeechEmotionClassifier
from .data_loader import create_dataloaders, prepare_sample_data
from .config import model_config, training_config, data_config

class EmotionTrainer:
    """음성 감정 분석 모델 훈련 클래스"""
    
    def __init__(self, 
                 model: Optional[SpeechEmotionClassifier] = None,
                 processor: Optional[Wav2Vec2Processor] = None):
        
        self.model = model or SpeechEmotionClassifier()
        self.processor = processor or Wav2Vec2Processor.from_pretrained(model_config.model_name)
        self.training_args = None
        self.trainer = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # 훈련 기록
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }
    
    def prepare_data(self, 
                    train_data: Optional[Tuple[List[str], List[str]]] = None,
                    val_data: Optional[Tuple[List[str], List[str]]] = None,
                    test_data: Optional[Tuple[List[str], List[str]]] = None):
        """데이터 준비"""
        
        if train_data is None or val_data is None:
            print("샘플 데이터를 사용합니다. 실제 데이터로 교체해주세요.")
            audio_paths, labels = prepare_sample_data()
            
            if len(audio_paths) == 0:
                print("데이터가 없습니다. 실제 데이터를 준비해주세요.")
                return
            
            from .data_loader import create_data_splits
            train_data, val_data, test_data = create_data_splits(
                audio_paths, labels,
                train_ratio=data_config.train_ratio,
                val_ratio=data_config.val_ratio,
                test_ratio=data_config.test_ratio
            )
        
        # 데이터로더 생성
        if test_data is not None:
            self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
                train_data, val_data, test_data, self.processor
            )
        else:
            self.train_loader, self.val_loader = create_dataloaders(
                train_data, val_data, processor=self.processor
            )
    
    def setup_training_args(self, output_dir: str = None):
        """훈련 인자 설정"""
        
        if output_dir is None:
            output_dir = training_config.output_dir
        
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=training_config.batch_size,
            per_device_eval_batch_size=training_config.batch_size,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            
            num_train_epochs=training_config.num_epochs,
            learning_rate=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            warmup_steps=training_config.warmup_steps,
            
            # 평가 및 저장 설정
            evaluation_strategy="steps",
            eval_steps=training_config.eval_steps,
            save_strategy="steps",
            save_steps=training_config.save_steps,
            save_total_limit=training_config.save_total_limit,
            
            # 로깅 설정
            logging_strategy="steps",
            logging_steps=training_config.logging_steps,
            
            # kresnik 모델에 최적화된 설정
            fp16=training_config.fp16,
            dataloader_num_workers=training_config.dataloader_num_workers,
            max_grad_norm=training_config.max_grad_norm,
            adam_epsilon=training_config.adam_epsilon,
            adam_beta1=training_config.adam_beta1,
            adam_beta2=training_config.adam_beta2,
            
            # 기타 설정
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            remove_unused_columns=False,
            push_to_hub=False,
            
            # 메모리 최적화
            dataloader_pin_memory=True,
            group_by_length=True,  # 길이별 그룹화로 효율성 향상
        )
    
    def compute_metrics(self, eval_pred):
        """평가 메트릭 계산"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # 정확도
        accuracy = accuracy_score(labels, predictions)
        
        # F1 스코어
        f1 = f1_score(labels, predictions, average='weighted')
        
        # 클래스별 F1 스코어
        f1_per_class = f1_score(labels, predictions, average=None)
        
        metrics = {
            'accuracy': accuracy,
            'f1': f1,
        }
        
        # 각 감정별 F1 스코어 추가
        for i, emotion in enumerate(model_config.emotion_labels):
            if i < len(f1_per_class):
                metrics[f'f1_{emotion.lower()}'] = f1_per_class[i]
        
        return metrics
    
    def create_trainer(self):
        """Trainer 객체 생성"""
        
        # Early stopping callback
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=training_config.early_stopping_patience,
            early_stopping_threshold=training_config.early_stopping_threshold
        )
        
        self.trainer = Trainer(
            model=self.model.model,
            args=self.training_args,
            train_dataset=self.train_loader.dataset,
            eval_dataset=self.val_loader.dataset,
            data_collator=self.train_loader.collate_fn,
            compute_metrics=self.compute_metrics,
            callbacks=[early_stopping],
        )
    
    def train(self, 
             train_data: Optional[Tuple[List[str], List[str]]] = None,
             val_data: Optional[Tuple[List[str], List[str]]] = None,
             test_data: Optional[Tuple[List[str], List[str]]] = None,
             output_dir: str = None):
        """모델 훈련"""
        
        print("=" * 50)
        print("음성 감정 분석 모델 훈련 시작")
        print("=" * 50)
        
        # 데이터 준비
        self.prepare_data(train_data, val_data, test_data)
        
        if self.train_loader is None:
            print("훈련 데이터가 준비되지 않았습니다.")
            return
        
        # 훈련 인자 설정
        self.setup_training_args(output_dir)
        
        # Trainer 생성
        self.create_trainer()
        
        # 훈련 시작
        print(f"훈련 데이터: {len(self.train_loader.dataset)}개")
        print(f"검증 데이터: {len(self.val_loader.dataset)}개")
        print(f"감정 클래스: {model_config.emotion_labels}")
        
        try:
            train_result = self.trainer.train()
            
            # 훈련 완료 후 최고 모델 저장
            self.trainer.save_model()
            self.processor.save_pretrained(self.training_args.output_dir)
            
            # 훈련 결과 저장
            self.save_training_results(train_result)
            
            print("훈련이 완료되었습니다!")
            print(f"모델이 저장되었습니다: {self.training_args.output_dir}")
            
        except Exception as e:
            print(f"훈련 중 오류 발생: {e}")
            raise e
    
    def evaluate(self, test_loader=None):
        """모델 평가"""
        
        if self.trainer is None:
            print("훈련된 모델이 없습니다.")
            return None
        
        eval_loader = test_loader or self.test_loader or self.val_loader
        
        if eval_loader is None:
            print("평가할 데이터가 없습니다.")
            return None
        
        # 평가 실행
        eval_results = self.trainer.evaluate(eval_dataset=eval_loader.dataset)
        
        # 상세 분석
        predictions = self.trainer.predict(eval_loader.dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        
        # 분류 보고서
        report = classification_report(
            y_true, y_pred,
            target_names=model_config.emotion_labels,
            output_dict=True
        )
        
        # 혼동 행렬
        cm = confusion_matrix(y_true, y_pred)
        
        # 결과 저장
        eval_results.update({
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': y_pred.tolist(),
            'true_labels': y_true.tolist()
        })
        
        # 혼동 행렬 시각화
        self.plot_confusion_matrix(cm)
        
        return eval_results
    
    def plot_confusion_matrix(self, cm):
        """혼동 행렬 시각화"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=model_config.emotion_labels,
            yticklabels=model_config.emotion_labels
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        # 저장
        if self.training_args:
            save_path = os.path.join(self.training_args.output_dir, 'confusion_matrix.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"혼동 행렬이 저장되었습니다: {save_path}")
        
        plt.show()
    
    def save_training_results(self, train_result):
        """훈련 결과 저장"""
        
        results = {
            'model_config': {
                'model_name': model_config.model_name,
                'num_labels': model_config.num_labels,
                'emotion_labels': model_config.emotion_labels
            },
            'training_config': {
                'learning_rate': training_config.learning_rate,
                'batch_size': training_config.batch_size,
                'num_epochs': training_config.num_epochs,
            },
            'training_results': {
                'train_runtime': train_result.metrics.get('train_runtime'),
                'train_samples_per_second': train_result.metrics.get('train_samples_per_second'),
                'final_train_loss': train_result.metrics.get('train_loss'),
            }
        }
        
        # 결과 파일 저장
        results_path = os.path.join(self.training_args.output_dir, 'training_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"훈련 결과가 저장되었습니다: {results_path}")
    
    def predict_single(self, audio_path: str):
        """단일 오디오 파일 예측"""
        if self.trainer is None:
            print("훈련된 모델이 없습니다.")
            return None
        
        try:
            # 예측 실행
            prediction = self.model.predict(audio_path, return_probabilities=True)
            
            # 결과 정리
            predicted_emotion = model_config.emotion_labels[np.argmax(prediction)]
            confidence = np.max(prediction)
            
            # 모든 감정별 확률
            emotion_probs = {
                emotion: float(prob) 
                for emotion, prob in zip(model_config.emotion_labels, prediction[0])
            }
            
            return {
                'predicted_emotion': predicted_emotion,
                'confidence': float(confidence),
                'all_probabilities': emotion_probs,
                'audio_path': audio_path
            }
            
        except Exception as e:
            print(f"예측 중 오류 발생: {e}")
            return None

def create_trainer(model=None, processor=None) -> EmotionTrainer:
    """Trainer 인스턴스 생성 팩토리 함수"""
    return EmotionTrainer(model, processor)