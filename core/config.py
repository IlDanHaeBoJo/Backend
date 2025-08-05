import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from pathlib import Path

# 환경변수 로드
load_dotenv()

class Settings:
    """애플리케이션 설정"""
    
    # 서버 설정
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # API 키들
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # JWT 설정
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-super-secret-key") # 실제 배포 시에는 반드시 변경하세요!
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))
    REFRESH_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("REFRESH_TOKEN_EXPIRE_MINUTES", 60 * 24 * 7))
    
    # Google Cloud 설정
    GOOGLE_APPLICATION_CREDENTIALS: str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    GOOGLE_CLOUD_PROJECT: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    
    # ChromaDB 설정
    CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    # CHROMA_ANALYTICS: bool = os.getenv("CHROMA_ANALYTICS", "false").lower() == "true" # 텔레메트리 비활성화
    
    # 로깅 설정
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # 음성 처리 설정
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_CHANNELS: int = 1
    AUDIO_SAMPLE_WIDTH: int = 2  # 16-bit
    
    # 디렉토리 설정
    TEMP_AUDIO_DIR: Path = Path("temp_audio")
    STATIC_AUDIO_DIR: Path = Path("static/audio")
    CACHE_DIR: Path = Path("cache")

    # PostgreSQL 설정
    DATABASE_URL: str = os.getenv("DATABASE_URL","")
    
    def __init__(self):
        """설정 초기화 및 검증"""
        # 필수 디렉토리 생성
        self.TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        self.STATIC_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        # 필수 API 키 검증
        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        
        # 필수 데이터베이스 URL 검증
        if not self.DATABASE_URL:
            raise ValueError("DATABASE_URL이 설정되지 않았습니다.")
    @property
    def is_production(self) -> bool:
        """프로덕션 환경 여부"""
        return not self.DEBUG

def setup_logging(settings: Settings):
    """로깅 설정"""
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# 전역 설정 인스턴스
settings = Settings()
setup_logging(settings)

# 데이터베이스 설정
engine = create_async_engine(settings.DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=AsyncSession)
Base = declarative_base()

async def get_db():
    """데이터베이스 세션 의존성"""
    async with SessionLocal() as session:
        yield session
