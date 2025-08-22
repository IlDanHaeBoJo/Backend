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
from dotenv import load_dotenv # load_dotenv 임포트

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from core.config import settings
from core.startup import service_manager
from routes import admin_notices, api, student_notices, websocket, auth, cpx, attachments, user_management, patient_images 

logger = logging.getLogger(__name__)

# 환경변수 로드 (uvicorn이 별도 프로세스로 실행될 때를 대비)
load_dotenv()

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
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],  # React 개발 서버 허용
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# 라우터 등록
app.include_router(api.router, tags=["API"])
app.include_router(websocket.router, tags=["WebSocket"])
app.include_router(auth.router, prefix="/auth", tags=["Auth"]) # Auth 라우터 추가
app.include_router(cpx.router) # CPX 학생용 라우터 포함
app.include_router(cpx.admin_router) # CPX 관리자용 라우터 포함
app.include_router(admin_notices.router, tags=["관리자용 공지사항"])
app.include_router(student_notices.router, tags=["학생용 공지사항"])
app.include_router(attachments.router, tags=["첨부파일"])
app.include_router(user_management.router)  # 사용자 관리 라우터 추가
app.include_router(patient_images.router, tags=["환자 이미지"]) # 환자 이미지 라우터 추가

# TTS 오디오 파일 서빙 (캐시 디렉토리에서)
app.mount("/cache", StaticFiles(directory="cache"), name="cache")

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
