from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from core.startup import service_manager

# API 라우터 생성
router = APIRouter()

class VoiceResponse(BaseModel):
    """음성 응답 모델"""
    text: str
    audio_url: Optional[str] = None
    avatar_action: Optional[str] = None

@router.get("/")
async def root():
    """API 루트 - 상태 확인"""
    return {
        "message": "CPX 가상 표준화 환자 시스템이 실행 중입니다!",
        "status": "online",
        "version": "1.0.0",
        "features": {
            "stt_engine": "Google Cloud Speech-to-Text",
            "tts_engine": "Google Cloud Text-to-Speech",
            "llm_engine": "GPT-4o",
            "vector_db": "ChromaDB"
        }
    }

@router.get("/health")
async def health_check():
    """헬스체크 - 모든 서비스 상태 확인"""
    return service_manager.get_health_status()

@router.get("/info")
async def system_info():
    """시스템 정보"""
    return {
        "system": "CPX Virtual Standardized Patient",
        "description": "의과대학 CPX 실기시험용 가상 표준화 환자 시스템",
        "supported_features": [
            "실시간 음성 인식 (한국어)",
            "CPX 케이스별 환자 역할 연기",
            "자연스러운 환자 응답 생성",
            "AI 기반 평가 및 피드백",
            "54종 실기 항목 지원"
        ],
        "languages": ["ko-KR"],
        "departments": [
            "내과", "정신과", "외과", "산부인과", 
            "소아과", "응급의학과", "가정의학과"
        ]
    }

@router.get("/voices")
async def get_available_voices():
    """사용 가능한 음성 목록"""
    if service_manager.tts_service:
        voices = await service_manager.tts_service.get_available_voices()
        return voices
    else:
        return {"error": "TTS 서비스가 초기화되지 않았습니다."} 