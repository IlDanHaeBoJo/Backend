import logging
from google.cloud import speech
from core.config import settings, Base, engine # Base와 engine 임포트
import core.models # 모델 정의를 로드하여 Base.metadata에 등록

from services.llm_service import LLMService
from services.tts_service import TTSService
from services.stt_service import STTService
from services.evaluation_service import EvaluationService
from services.ser_service import SERService

logger = logging.getLogger(__name__)

class ServiceManager:
    """서비스 매니저 - 모든 서비스들의 초기화 및 관리"""
    
    def __init__(self):
        self.speech_client = None
        self.speech_config = None
        self.llm_service = None
        self.tts_service = None
        self.evaluation_service = None
        self.vector_service = None
        self.ser_service = None
        self._initialized = False
    
    async def initialize_services(self):
        """모든 서비스 초기화"""
        if self._initialized:
            logger.warning("서비스가 이미 초기화되었습니다.")
            return
            
        try:
            logger.info("🚀 서비스 초기화 시작...")
            
            # 데이터베이스 테이블 생성
            logger.info("📊 데이터베이스 테이블 생성 중...")
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("✅ 데이터베이스 테이블 생성 완료!")
            
            # Google Cloud Speech-to-Text API 초기화
            await self._initialize_speech_service()
            
            # 다른 서비스들 초기화
            await self._initialize_ai_services()
            
            self._initialized = True
            logger.info("✅ 모든 서비스 초기화 완료!")
            
        except Exception as e:
            logger.error(f"❌ 서비스 초기화 실패: {e}")
            raise
    
    async def _initialize_speech_service(self):
        """Google Cloud Speech API 초기화 - STTService에서 처리됨"""
        logger.info("🎤 Google Cloud Speech API 초기화는 STTService에서 처리됩니다")
        # STTService에서 초기화하므로 여기서는 스킵
        self.speech_client = None  # 호환성을 위해 유지
        self.speech_config = None  # 호환성을 위해 유지
    
    async def _initialize_ai_services(self):
        """AI 서비스들 초기화"""
        logger.info("🧠 AI 서비스들 초기화 중...")
        
        # LLM 서비스 초기화
        self.llm_service = LLMService()
        logger.info("✅ LLM 서비스 초기화 완료")
        
        # STT 서비스 초기화
        self.stt_service = STTService()
        await self.stt_service.initialize()
        logger.info("✅ STT 서비스 초기화 완료")
        
        # TTS 서비스 초기화
        self.tts_service = TTSService()
        logger.info("✅ TTS 서비스 초기화 완료")

        # SER 서비스 초기화 (감정 분석 전담)
        self.ser_service = SERService()
        logger.info("✅ SER 서비스 초기화 완료")

        # Evaluation 서비스 초기화 (SER 기능 제거됨)
        self.evaluation_service = EvaluationService()
        logger.info("✅ Evaluation 서비스 초기화 완료")
    
    def get_health_status(self) -> dict:
        """서비스 상태 확인"""
        return {
            "status": "healthy" if self._initialized else "initializing",
            "llm": self.llm_service is not None,
            "stt": self.stt_service is not None,
            "tts": self.tts_service is not None,
            "ser": self.ser_service is not None,
            "evaluation": self.evaluation_service is not None,
            "initialized": self._initialized
        }
    
    async def shutdown(self):
        """서비스 종료"""
        logger.info("🛑 서비스 종료 중...")
        
        # 필요한 경우 여기에 정리 로직 추가
        self._initialized = False
        
        logger.info("✅ 서비스 종료 완료")

# 전역 서비스 매니저 인스턴스
service_manager = ServiceManager() 