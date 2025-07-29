"""
🏥 CPX Virtual Standardized Patient System
의과대학 CPX 실기시험용 가상 표준화 환자 시스템

FastAPI 기반 실시간 음성 AI 서버
- Google Cloud Speech-to-Text (STT)
- Google Cloud Text-to-Speech (TTS)
- GPT-4o 기반 환자 역할 연기
- ChromaDB 벡터 검색 (RAG)
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from core.config import settings
from core.startup import service_manager
from routes import api, websocket

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    # 시작 시
    logger.info("🚀 CPX 시스템 시작 중...")
    await service_manager.initialize_services()
    logger.info("✅ CPX 시스템 준비 완료!")
    
    yield
    
    # 종료 시
    logger.info("🛑 CPX 시스템 종료 중...")
    await service_manager.shutdown()
    logger.info("✅ CPX 시스템 종료 완료")

# FastAPI 앱 생성
app = FastAPI(
    title="CPX Virtual Standardized Patient",
    description="""
    🏥 **의과대학 CPX 실기시험용 가상 표준화 환자 시스템**
    
    ## 🎯 주요 기능
    - 실시간 음성 인식 및 응답 (한국어)
    - 54종 CPX 케이스별 환자 역할 연기
    - AI 기반 자연스러운 환자 반응
    - 의학 용어 특화 STT/TTS
    - 평가 및 피드백 시스템
    
    ## 🔧 기술 스택
    - **STT**: Google Cloud Speech-to-Text
    - **TTS**: Google Cloud Text-to-Speech (한국어 Neural2-A)
    - **LLM**: GPT-4o (환자 역할 연기)
    - **Vector DB**: ChromaDB (의학 지식 RAG)
    - **Framework**: FastAPI + WebSocket
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발환경용 - 프로덕션에서는 제한 필요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(api.router, tags=["API"])
app.include_router(websocket.router, tags=["WebSocket"])

# 정적 파일 서빙 (TTS 오디오 파일)
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"🌐 서버 시작: http://{settings.HOST}:{settings.PORT}")
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    ) 