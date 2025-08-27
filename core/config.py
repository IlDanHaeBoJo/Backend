import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
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
    VERIFICATION_CODE_EXPIRE_MINUTES: int = int(os.getenv("VERIFICATION_CODE_EXPIRE_MINUTES", 5)) # 본인 확인 코드 만료 시간 (분)
    
    # Google Cloud 설정
    GOOGLE_APPLICATION_CREDENTIALS: str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    GOOGLE_CLOUD_PROJECT: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")

    #Amazon S3설정  
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "ap-northeast-2")
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "medicpx")
    
    # SageMaker 설정
    SAGEMAKER_SER_ENDPOINT: str = os.getenv("SAGEMAKER_SER_ENDPOINT", "ser-model-public")
    SAGEMAKER_REGION: str = os.getenv("SAGEMAKER_REGION", "ap-northeast-2")
    
    # SageMaker 전용 AWS 자격증명
    AWS_ACCESS_KEY_ID_SAGE: str = os.getenv("AWS_ACCESS_KEY_ID_SAGE", "")
    AWS_SECRET_ACCESS_KEY_SAGE: str = os.getenv("AWS_SECRET_ACCESS_KEY_SAGE", "")    
    
    # ChromaDB 설정
    CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    # CHROMA_ANALYTICS: bool = os.getenv("CHROMA_ANALYTICS", "false").lower() == "true" # 텔레메트리 비활성화
    
    # 로깅 설정
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # 음성 처리 설정
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_CHANNELS: int = 1
    AUDIO_SAMPLE_WIDTH: int = 2  # 16-bit
    
    # PostgreSQL 설정
    DATABASE_URL: str = os.getenv("DATABASE_URL","")

    # CORS 설정
    FRONTEND_ORIGINS: list[str] = os.getenv("FRONTEND_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")

    # 이메일 설정
    MAIL_USERNAME: str = os.getenv("MAIL_USERNAME", "")
    MAIL_PASSWORD: str = os.getenv("MAIL_PASSWORD", "")
    MAIL_FROM: str = os.getenv("MAIL_FROM", "")
    MAIL_PORT: int = int(os.getenv("MAIL_PORT", 587))
    MAIL_SERVER: str = os.getenv("MAIL_SERVER", "")
    MAIL_FROM_NAME: str = os.getenv("MAIL_FROM_NAME", "Your App Name")
    MAIL_STARTTLS: bool = os.getenv("MAIL_STARTTLS", "true").lower() == "true"
    MAIL_SSL_TLS: bool = os.getenv("MAIL_SSL_TLS", "false").lower() == "true"
    USE_CREDENTIALS: bool = os.getenv("USE_CREDENTIALS", "true").lower() == "true"
    VALIDATE_CERTS: bool = os.getenv("VALIDATE_CERTS", "true").lower() == "true"
    
    def __init__(self):
        """설정 초기화 및 검증"""
        # 필수 API 키 검증
        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        
        # 필수 데이터베이스 URL 검증
        if not self.DATABASE_URL:
            raise ValueError("DATABASE_URL이 설정되지 않았습니다.")
        
        # 이메일 설정 검증 (필요에 따라 추가)
        if self.USE_CREDENTIALS and (not self.MAIL_USERNAME or not self.MAIL_PASSWORD or not self.MAIL_SERVER or not self.MAIL_FROM):
            print("경고: 이메일 전송을 위한 MAIL_USERNAME, MAIL_PASSWORD, MAIL_SERVER, MAIL_FROM 환경 변수가 설정되지 않았습니다.")

        print(self.FRONTEND_ORIGINS)
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
