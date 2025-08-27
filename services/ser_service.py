"""
SER (Speech Emotion Recognition) 서비스
음성 감정 분석 전담 서비스
"""

from typing import Dict, Optional
import os
import torch
import numpy as np
import librosa
import logging
import boto3
import json
import base64
from core.config import settings

logger = logging.getLogger(__name__)

class SERService:
    def __init__(self):
        """SER 서비스 초기화"""
        self.emotion_model = None
        self.emotion_processor = None
        self.emotion_labels = ["Anxious", "Dry", "Kind"]
        
        # SageMaker 클라이언트 초기화
        self.sagemaker_runtime = None
        self.endpoint_name = settings.SAGEMAKER_SER_ENDPOINT
        
        if self.endpoint_name:
            try:
                # SageMaker 전용 자격증명 사용
                access_key = settings.AWS_ACCESS_KEY_ID_SAGE or settings.AWS_ACCESS_KEY_ID
                secret_key = settings.AWS_SECRET_ACCESS_KEY_SAGE or settings.AWS_SECRET_ACCESS_KEY
                
                self.sagemaker_runtime = boto3.client(
                    'sagemaker-runtime',
                    region_name=settings.SAGEMAKER_REGION,
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key
                )
                print(f"🎭 SageMaker SER 서비스 초기화 완료 - 엔드포인트: {self.endpoint_name}")
            except Exception as e:
                logger.error(f"SageMaker 클라이언트 초기화 실패: {e}")
                self.sagemaker_runtime = None
        else:
            print("⚠️ SageMaker 엔드포인트가 설정되지 않았습니다. 환경변수 SAGEMAKER_SER_ENDPOINT를 확인하세요.")
    
    async def load_model(self):
        """감정 분석 모델 로드 (SageMaker 사용으로 인해 불필요)"""
        if self.sagemaker_runtime and self.endpoint_name:
            print("🎭 SageMaker 엔드포인트 준비 완료")
            print("💡 엔드포인트 상태 확인은 권한이 필요하므로 실제 호출로 테스트합니다")
            return True
        else:
            print("⚠️ SageMaker 클라이언트가 초기화되지 않았습니다")
            return False
    
    async def _invoke_sagemaker_endpoint(self, audio_buffer: bytearray) -> Dict:
        """
        SageMaker 엔드포인트를 호출하여 감정 분석 수행
        
        Args:
            audio_buffer: raw 오디오 버퍼 (bytearray)
            
        Returns:
            감정 분석 결과 딕셔너리
        """
        if not self.sagemaker_runtime or not self.endpoint_name:
            return {"error": "SageMaker 엔드포인트가 설정되지 않았습니다"}
        
        try:
            # 오디오 버퍼를 base64로 인코딩 (SageMaker 전송용)
            audio_base64 = base64.b64encode(audio_buffer).decode('utf-8')
            
            # SageMaker 엔드포인트 호출을 위한 페이로드 구성 (raw 오디오 버퍼)
            payload = {
                "instances": [{
                    "audio_data": audio_base64
                }]
            }
            
            # SageMaker 엔드포인트 호출
            response = self.sagemaker_runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='application/json',
                Body=json.dumps(payload)
            )
            
            # 응답 파싱
            result = json.loads(response['Body'].read().decode())
            
            # 결과 형식 통일 (SageMaker 응답을 기존 형식에 맞게 변환)
            if 'predictions' in result and len(result['predictions']) > 0:
                prediction = result['predictions'][0]
                
                # 감정 라벨과 확률 추출
                if 'emotion_scores' in prediction:
                    emotion_scores = prediction['emotion_scores']
                    predicted_emotion = max(emotion_scores.keys(), key=lambda k: emotion_scores[k])
                    confidence = emotion_scores[predicted_emotion]
                else:
                    # 기본 형식이 다를 경우 대비
                    predicted_emotion = prediction.get('predicted_emotion', 'Unknown')
                    confidence = prediction.get('confidence', 0.0)
                    emotion_scores = prediction.get('emotion_scores', {})
                
                return {
                    "success": True,
                    "predicted_emotion": predicted_emotion,
                    "confidence": confidence,
                    "emotion_scores": emotion_scores,
                    "source": "sagemaker_endpoint"
                }
            else:
                return {"error": "SageMaker 엔드포인트에서 유효하지 않은 응답을 받았습니다"}
                
        except Exception as e:
            logger.error(f"SageMaker 엔드포인트 호출 중 오류 발생: {e}")
            return {"error": f"SageMaker 엔드포인트 호출 중 오류 발생: {e}"}
    
    async def analyze_emotion_from_buffer(self, audio_buffer: bytearray) -> Dict:
        """
        오디오 버퍼에서 직접 감정 분석 (SageMaker 엔드포인트 사용)
        
        Args:
            audio_buffer: 분석할 오디오 버퍼 (16-bit PCM)
            
        Returns:
            감정 분석 결과 딕셔너리
        """
        try:
            if not audio_buffer or len(audio_buffer) == 0:
                return {"error": "오디오 버퍼가 비어있습니다"}
            
            # SageMaker 엔드포인트 호출 (raw 버퍼 직접 전송)
            result = await self._invoke_sagemaker_endpoint(audio_buffer)
            
            if result.get("success"):
                result["source"] = "audio_buffer_sagemaker"
            
            return result
                
        except Exception as e:
            logger.error(f"감정 분석 중 오류 발생: {e}")
            return {"error": f"감정 분석 중 오류 발생: {e}"}
    
    async def analyze_emotion(self, audio_file_path: str) -> Dict:
        """
        음성 파일의 감정 분석 (SageMaker 엔드포인트 사용)
        
        Args:
            audio_file_path: 분석할 음성 파일 경로
            
        Returns:
            감정 분석 결과 딕셔너리
        """
        try:
            # 오디오 파일 존재 확인
            if not os.path.exists(audio_file_path):
                return {"error": f"오디오 파일을 찾을 수 없습니다: {audio_file_path}"}
            
            # 파일을 바이너리로 읽기
            with open(audio_file_path, 'rb') as f:
                audio_buffer = bytearray(f.read())
            
            # SageMaker 엔드포인트 호출 (raw 파일 데이터 직접 전송)
            result = await self._invoke_sagemaker_endpoint(audio_buffer)
            
            if result.get("success"):
                result["file_path"] = audio_file_path
                result["source"] = "audio_file_sagemaker"
            
            return result
                
        except Exception as e:
            logger.error(f"감정 분석 중 오류 발생: {e}")
            return {"error": f"감정 분석 중 오류 발생: {e}"}
    
    # SageMaker용 전처리 메서드들 (현재 사용하지 않음 - raw 버퍼 직접 전송)
    # async def _preprocess_audio_from_buffer_for_sagemaker(self, audio_buffer: bytearray) -> Optional[np.ndarray]:
    #     """오디오 버퍼 전처리 (SageMaker용) - 사용하지 않음"""
    #     pass
    
    # async def _preprocess_audio_for_sagemaker(self, file_path: str) -> Optional[np.ndarray]:
    #     """오디오 파일 전처리 (SageMaker용) - 사용하지 않음"""
    #     pass
    
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
        """모델 로드 상태 확인 (SageMaker 엔드포인트 연결 상태)"""
        return self.sagemaker_runtime is not None and self.endpoint_name is not None
    
    def get_model_info(self) -> Dict:
        """모델 정보 반환 (SageMaker 엔드포인트 정보)"""
        return {
            "model_loaded": self.is_model_loaded(),
            "sagemaker_endpoint": self.endpoint_name,
            "sagemaker_region": settings.SAGEMAKER_REGION,
            "emotion_labels": self.emotion_labels,
            "using_sagemaker": True
        }


