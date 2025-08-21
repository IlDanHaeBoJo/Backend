"""
STT (Speech-to-Text) 서비스
Google Cloud Speech API 기반 음성 인식 서비스
"""

import logging
import numpy as np
from typing import Optional
from google.cloud import speech
from core.config import settings

logger = logging.getLogger(__name__)

class STTService:
    def __init__(self):
        """STT 서비스 초기화 - Google Cloud Speech 사용"""
        self.speech_client = None
        self.speech_config = None
        logger.info("🎤 STT 서비스 초기화 완료")
    
    async def initialize(self):
        """Google Cloud Speech API 초기화"""
        if self.speech_client is not None:
            return True
        
        try:
            logger.info("🎤 Google Cloud Speech API 초기화 중...")
            
            self.speech_client = speech.SpeechClient()
            
            # 한국어 실시간 인식 설정
            self.speech_config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=settings.AUDIO_SAMPLE_RATE,
                language_code="ko-KR",  # 한국어
                model="latest_long",    # 긴 대화용 모델
                enable_automatic_punctuation=True,  # 자동 문장부호
                use_enhanced=True,      # 향상된 모델 사용
            )
            
            logger.info("✅ Google Cloud Speech API 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌ Google Cloud Speech API 초기화 실패: {e}")
            return False
    
    async def transcribe_from_buffer(self, audio_numpy: np.ndarray) -> str:
        """numpy 배열에서 직접 STT 처리 (Google Cloud Speech)"""
        try:
            if len(audio_numpy) == 0:
                return ""
            
            # 클라이언트가 초기화되지 않았으면 초기화
            if self.speech_client is None:
                success = await self.initialize()
                if not success:
                    return ""
            
            # numpy 배열을 bytes로 변환 (16-bit PCM)
            audio_bytes = (audio_numpy * 32767).astype(np.int16).tobytes()
            
            # Google Cloud Speech API 호출
            audio = speech.RecognitionAudio(content=audio_bytes)
            response = self.speech_client.recognize(
                config=self.speech_config, 
                audio=audio
            )
            
            # 결과 처리
            if response.results:
                transcribed_text = response.results[0].alternatives[0].transcript.strip()
                logger.info(f"🎤 STT 완료: '{transcribed_text}'")
                return transcribed_text
            else:
                logger.info("🎤 STT 결과 없음")
                return ""
            
        except Exception as e:
            logger.error(f"❌ STT 처리 오류: {e}")
            return ""
    
    def get_service_info(self) -> dict:
        """서비스 정보 반환"""
        return {
            "provider": "Google Cloud Speech",
            "language": "ko-KR",
            "model": "latest_long",
            "client_initialized": self.speech_client is not None,
            "status": "ready" if self.speech_client else "not_initialized"
        }
