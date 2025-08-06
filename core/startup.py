import logging
from google.cloud import speech
from core.config import settings, Base, engine # Base와 engine 임포트
import core.models # 모델 정의를 로드하여 Base.metadata에 등록

# SQLite 버전 업그레이드를 위해 pysqlite3를 sqlite3로 대체
try:
    import pysqlite3 as sqlite3
    import sys
    sys.modules['sqlite3'] = sqlite3
    logging.getLogger(__name__).info("✅ pysqlite3를 사용하여 SQLite 버전 업그레이드")
except ImportError:
    import sqlite3
    logging.getLogger(__name__).warning("⚠️  pysqlite3 없음, 기본 sqlite3 사용")

from services.llm_service import LLMService
from services.tts_service import TTSService
from services.vector_service import VectorService
from services.evaluation_service import EvaluationService

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
        """Google Cloud Speech API 초기화"""
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
    
    async def _initialize_ai_services(self):
        """AI 서비스들 초기화"""
        logger.info("🧠 AI 서비스들 초기화 중...")
        
        # LLM 서비스 초기화
        self.llm_service = LLMService()
        logger.info("✅ LLM 서비스 초기화 완료")
        
        # TTS 서비스 초기화
        self.tts_service = TTSService()
        logger.info("✅ TTS 서비스 초기화 완료")

        # Evaluatuin 서비스 초기화
        self.evaluation_service = EvaluationService()
        logger.info("✅ Evaluation 서비스 초기화 완료")

        
        # 벡터 서비스 초기화 (SQLite 이슈 처리)
        try:
            self.vector_service = VectorService()
            logger.info("✅ Vector 서비스 초기화 완료")
        except Exception as e:
            logger.warning(f"⚠️  Vector 서비스 초기화 부분 실패, 기본 모드로 계속: {e}")
            if "sqlite" in str(e).lower():
                logger.info("💡 SQLite 버전 문제인 경우 'pip install pysqlite3-binary' 실행 후 재시작하세요")
            # 기본 벡터 서비스 객체 생성 (검색 없이)
            self.vector_service = type('VectorService', (), {
                'vectorstore': None,
                '_use_fallback_knowledge': True,
                'search': self._fallback_search
            })()
            
    async def _fallback_search(self, query: str, k: int = 3):
        """fallback 검색 함수"""
        return ["기본 CPX 케이스 - 환자는 증상에 대해 자연스럽게 응답합니다."]
    
    def get_health_status(self) -> dict:
        """서비스 상태 확인"""
        return {
            "status": "healthy" if self._initialized else "initializing",
            "speech": self.speech_client is not None,
            "llm": self.llm_service is not None,
            "tts": self.tts_service is not None,
            "vector": self.vector_service is not None,
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