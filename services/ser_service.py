"""
SER (Speech Emotion Recognition) 서비스
음성 감정 분석 전담 서비스
"""

from typing import Dict, Optional
from pathlib import Path
import torch
import numpy as np
import librosa
import logging

logger = logging.getLogger(__name__)

class SERService:
    def __init__(self):
        """SER 서비스 초기화"""
        self.emotion_model = None
        self.emotion_processor = None
        self.emotion_labels = ["Anxious", "Dry", "Kind"]
        self.ser_model_path = Path("SER/results_quick_test/adversary_model_augment_v1_epoch_5")
        
        print("🎭 SER 서비스 초기화 완료")
    
    async def load_model(self):
        """감정 분석 모델 로드 (지연 로딩)"""
        if self.emotion_model is not None:
            return True
        
        try:
            print("🎭 감정 분석 모델 로드 중...")
            
            # 커스텀 모델이 있으면 사용
            if self.ser_model_path.exists():
                from transformers import Wav2Vec2Processor
                from SER.finetune_direct import custom_Wav2Vec2ForEmotionClassification
                
                self.emotion_model = custom_Wav2Vec2ForEmotionClassification.from_pretrained(
                    str(self.ser_model_path)
                )
                self.emotion_processor = Wav2Vec2Processor.from_pretrained(
                    str(self.ser_model_path)
                )
                
                print("✅ 커스텀 감정 분석 모델 로드 완료")
            else:
                # 기본 모델 사용
                from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor, Wav2Vec2Config
                
                model_name = "kresnik/wav2vec2-large-xlsr-korean"
                label2id = {label: i for i, label in enumerate(self.emotion_labels)}
                id2label = {i: label for i, label in enumerate(self.emotion_labels)}
                
                config = Wav2Vec2Config.from_pretrained(
                    model_name,
                    num_labels=len(self.emotion_labels),
                    label2id=label2id,
                    id2label=id2label,
                    finetuning_task="emotion_classification"
                )
                
                self.emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained(
                    model_name,
                    config=config,
                    ignore_mismatched_sizes=True
                )
                self.emotion_processor = Wav2Vec2Processor.from_pretrained(model_name)
                
                print("✅ 기본 감정 분석 모델 로드 완료")
            
            # 모델을 평가 모드로 설정
            self.emotion_model.eval()
            return True
            
        except Exception as e:
            logger.error(f"감정 분석 모델 로드 실패: {e}")
            self.emotion_model = None
            self.emotion_processor = None
            return False
    
    async def analyze_emotion_from_buffer(self, audio_buffer: bytearray) -> Dict:
        """
        오디오 버퍼에서 직접 감정 분석
        
        Args:
            audio_buffer: 분석할 오디오 버퍼 (16-bit PCM)
            
        Returns:
            감정 분석 결과 딕셔너리
        """
        # 모델이 로드되지 않았으면 로드 시도
        if self.emotion_model is None:
            model_loaded = await self.load_model()
            if not model_loaded:
                return {"error": "감정 분석 모델을 로드할 수 없습니다"}
        
        try:
            if not audio_buffer or len(audio_buffer) == 0:
                return {"error": "오디오 버퍼가 비어있습니다"}
            
            # 오디오 전처리 (버퍼에서 직접)
            audio_data = await self._preprocess_audio_from_buffer(audio_buffer)
            if audio_data is None:
                return {"error": "오디오 전처리 실패"}
            
            # 감정 분석 수행
            with torch.no_grad():
                inputs = {
                    "input_values": audio_data,
                    "attention_mask": None
                }
                
                outputs = self.emotion_model(**inputs)
                logits = outputs['emotion_logits']
                
                # 예측 결과 계산
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                predicted_id = torch.argmax(logits, dim=-1).item()
                confidence = probabilities[0][predicted_id].item()
                
                # 예측 감정 결과
                predicted_emotion = self.emotion_labels[predicted_id]
                
                # 모든 감정별 확률
                emotion_scores = {
                    emotion: probabilities[0][i].item() 
                    for i, emotion in enumerate(self.emotion_labels)
                }
                
                return {
                    "success": True,
                    "predicted_emotion": predicted_emotion,
                    "confidence": confidence,
                    "emotion_scores": emotion_scores,
                    "source": "audio_buffer"
                }
                
        except Exception as e:
            logger.error(f"감정 분석 중 오류 발생: {e}")
            return {"error": f"감정 분석 중 오류 발생: {e}"}
    
    async def analyze_emotion(self, audio_file_path: str) -> Dict:
        """
        음성 파일의 감정 분석 (호환성을 위한 레거시 메서드)
        
        Args:
            audio_file_path: 분석할 음성 파일 경로
            
        Returns:
            감정 분석 결과 딕셔너리
        """
        # 모델이 로드되지 않았으면 로드 시도
        if self.emotion_model is None:
            model_loaded = await self.load_model()
            if not model_loaded:
                return {"error": "감정 분석 모델을 로드할 수 없습니다"}
        
        try:
            # 오디오 파일 존재 확인
            audio_path = Path(audio_file_path)
            if not audio_path.exists():
                return {"error": f"오디오 파일을 찾을 수 없습니다: {audio_file_path}"}
            
            # 오디오 전처리
            audio_data = await self._preprocess_audio(str(audio_path))
            if audio_data is None:
                return {"error": "오디오 전처리 실패"}
            
            # 감정 분석 수행
            with torch.no_grad():
                inputs = {
                    "input_values": audio_data,
                    "attention_mask": None
                }
                
                outputs = self.emotion_model(**inputs)
                logits = outputs['emotion_logits']
                
                # 예측 결과 계산
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                predicted_id = torch.argmax(logits, dim=-1).item()
                confidence = probabilities[0][predicted_id].item()
                
                # 예측 감정 결과
                predicted_emotion = self.emotion_labels[predicted_id]
                
                # 모든 감정별 확률
                emotion_scores = {
                    emotion: probabilities[0][i].item() 
                    for i, emotion in enumerate(self.emotion_labels)
                }
                
                return {
                    "success": True,
                    "predicted_emotion": predicted_emotion,
                    "confidence": confidence,
                    "emotion_scores": emotion_scores,
                    "file_path": str(audio_path)
                }
                
        except Exception as e:
            logger.error(f"감정 분석 중 오류 발생: {e}")
            return {"error": f"감정 분석 중 오류 발생: {e}"}
    
    async def _preprocess_audio_from_buffer(self, audio_buffer: bytearray) -> Optional[torch.Tensor]:
        """오디오 버퍼 전처리 (Wav2Vec2용)"""
        try:
            # 16-bit PCM 버퍼를 numpy 배열로 변환
            audio_data = np.frombuffer(audio_buffer, dtype=np.int16)
            
            # float32로 정규화 (-1.0 ~ 1.0)
            audio = audio_data.astype(np.float32) / 32768.0
            
            # 길이 제한 (최대 15초)
            max_duration = 15.0
            target_length = int(16000 * max_duration)
            
            if len(audio) > target_length:
                # 가운데 부분 사용
                start_idx = np.random.randint(0, len(audio) - target_length + 1)
                audio = audio[start_idx:start_idx + target_length]
            elif len(audio) < target_length:
                # 패딩 추가
                pad_length = target_length - len(audio)
                audio = np.pad(audio, (0, pad_length), mode='constant', constant_values=0)
            
            # Wav2Vec2 processor로 변환
            inputs = self.emotion_processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            return inputs.input_values.squeeze(0)
            
        except Exception as e:
            logger.error(f"오디오 버퍼 전처리 오류: {e}")
            return None
    
    async def _preprocess_audio(self, file_path: str) -> Optional[torch.Tensor]:
        """오디오 전처리 (Wav2Vec2용)"""
        try:
            # 오디오 로드 (16kHz로 리샘플링)
            audio, sr = librosa.load(file_path, sr=16000, res_type='kaiser_fast')
            
            # 길이 제한 (최대 15초)
            max_duration = 15.0
            target_length = int(16000 * max_duration)
            
            if len(audio) > target_length:
                # 가운데 부분 사용
                start_idx = np.random.randint(0, len(audio) - target_length + 1)
                audio = audio[start_idx:start_idx + target_length]
            elif len(audio) < target_length:
                # 패딩 추가
                pad_length = target_length - len(audio)
                audio = np.pad(audio, (0, pad_length), mode='constant', constant_values=0)
            
            # Wav2Vec2 processor로 변환
            inputs = self.emotion_processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            return inputs.input_values.squeeze(0)
            
        except Exception as e:
            logger.error(f"오디오 전처리 오류: {e}")
            return None
    
    def is_model_loaded(self) -> bool:
        """모델 로드 상태 확인"""
        return self.emotion_model is not None and self.emotion_processor is not None
    
    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        return {
            "model_loaded": self.is_model_loaded(),
            "model_path": str(self.ser_model_path),
            "emotion_labels": self.emotion_labels,
            "model_exists": self.ser_model_path.exists()
        }


