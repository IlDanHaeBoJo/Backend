"""
Wav2Vec2 기반 음성 감정 분석 모델
"""

import torch
import torch.nn as nn
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Config,
    Wav2Vec2Processor
)
from typing import Dict, Any, Optional
from .config import model_config

class SpeechEmotionClassifier(nn.Module):
    """음성 감정 분석을 위한 Wav2Vec2 모델 (kresnik/wav2vec2-large-xlsr-korean 기반)"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        if config is None:
            config = {}
            
        self.model_name = config.get('model_name', model_config.model_name)
        self.num_labels = config.get('num_labels', model_config.num_labels)
        
        print(f"🤖 모델 로딩: {self.model_name}")
        print(f"📊 감정 클래스 수: {self.num_labels}")
        
        # Wav2Vec2 설정 로드 및 수정
        self.wav2vec2_config = Wav2Vec2Config.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            label2id=model_config.label2id,
            id2label=model_config.id2label,
            finetuning_task="emotion_classification",
            # kresnik/wav2vec2-large-xlsr-korean 모델에 최적화된 설정
            attention_dropout=0.1,
            hidden_dropout=0.1,
            feat_proj_dropout=0.0,
            mask_time_prob=0.05,
            layerdrop=0.1,
        )
        
        # 모델 로드 - ASR 모델을 감정 분석용으로 변경
        try:
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                self.model_name,
                config=self.wav2vec2_config,
                ignore_mismatched_sizes=True
            )
            print("✅ 모델 로딩 성공")
        except Exception as e:
            print(f"⚠️ 사전 훈련된 분류 헤드 로딩 실패, 새로운 헤드 생성: {e}")
            # ASR 모델에서 특성 추출기만 로드하고 새로운 분류 헤드 추가
            from transformers import Wav2Vec2Model
            base_model = Wav2Vec2Model.from_pretrained(self.model_name)
            self.model = Wav2Vec2ForSequenceClassification(self.wav2vec2_config)
            self.model.wav2vec2 = base_model
        
        # Processor 로드
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        
        # Feature extractor 동결 (선택적)
        self._freeze_feature_extractor()
        
        # 모델 파라미터 정보 출력
        self._print_model_info()
    
    def _freeze_feature_extractor(self):
        """Feature extractor 가중치 동결 (한국어 ASR 가중치 보존)"""
        self.model.wav2vec2.feature_extractor._freeze_parameters()
        print("🔒 Feature extractor 가중치 동결 완료")
    
    def _print_model_info(self):
        """모델 정보 출력"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"📈 모델 파라미터 정보:")
        print(f"  전체 파라미터: {total_params:,}")
        print(f"  훈련 가능한 파라미터: {trainable_params:,}")
        print(f"  동결된 파라미터: {total_params - trainable_params:,}")
        print(f"  훈련 가능 비율: {100 * trainable_params / total_params:.1f}%")
    
    def forward(self, input_values, attention_mask=None, labels=None):
        """순전파"""
        outputs = self.model(
            input_values=input_values,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
    def predict(self, audio_input, return_probabilities=False):
        """단일 오디오에 대한 예측 (kresnik/wav2vec2-large-xlsr-korean 최적화)"""
        self.eval()
        with torch.no_grad():
            # 오디오 전처리
            if isinstance(audio_input, str):
                # 파일 경로인 경우 오디오 로드
                from .preprocessing import preprocessor
                audio_array = preprocessor.preprocess(audio_input, apply_augmentation=False)
                if audio_array is None:
                    raise ValueError(f"오디오 파일 로드 실패: {audio_input}")
                audio_input = audio_array
            
            if isinstance(audio_input, torch.Tensor):
                input_values = audio_input.unsqueeze(0) if audio_input.dim() == 1 else audio_input
                # 디바이스 설정
                input_values = input_values.to(next(self.model.parameters()).device)
            else:
                # numpy array인 경우 - kresnik 모델에 최적화
                inputs = self.processor(
                    audio_input,
                    sampling_rate=model_config.sampling_rate,
                    return_tensors="pt",
                    padding=True,
                    max_length=model_config.sampling_rate * int(model_config.max_duration),
                    truncation=True
                )
                input_values = inputs.input_values.to(next(self.model.parameters()).device)
            
            # 예측
            outputs = self.forward(input_values)
            logits = outputs.logits
            
            if return_probabilities:
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                return probabilities.cpu().numpy()
            else:
                predicted_ids = torch.argmax(logits, dim=-1)
                predicted_labels = [model_config.id2label[pred_id.item()] 
                                  for pred_id in predicted_ids]
                return predicted_labels[0] if len(predicted_labels) == 1 else predicted_labels
    
    def get_embeddings(self, input_values, attention_mask=None):
        """중간 표현 추출"""
        with torch.no_grad():
            outputs = self.model.wav2vec2(
                input_values=input_values,
                attention_mask=attention_mask
            )
            # Last hidden state의 평균을 사용
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings
    
    def save_model(self, save_path: str):
        """모델 저장"""
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)
    
    @classmethod
    def from_pretrained(cls, model_path: str):
        """저장된 모델 로드"""
        instance = cls()
        instance.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
        instance.processor = Wav2Vec2Processor.from_pretrained(model_path)
        return instance

class EmotionHead(nn.Module):
    """커스텀 감정 분류 헤드"""
    
    def __init__(self, input_dim: int, num_emotions: int, dropout_rate: float = 0.1):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim // 2, num_emotions)
        )
    
    def forward(self, hidden_states):
        # Global average pooling
        pooled_output = hidden_states.mean(dim=1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def create_model(config: Optional[Dict[str, Any]] = None) -> SpeechEmotionClassifier:
    """모델 생성 팩토리 함수"""
    return SpeechEmotionClassifier(config)