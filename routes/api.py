from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict

from core.startup import service_manager
from routes.auth import get_current_user
from core.models import User

# API 라우터 생성
router = APIRouter()

class VoiceResponse(BaseModel):
    """음성 응답 모델"""
    text: str
    audio_url: Optional[str] = None
    avatar_action: Optional[str] = None

class ScenarioInfo(BaseModel):
    """시나리오 정보 모델"""
    id: str
    name: str
    description: str
    patient_info: Dict
    has_patient_image: bool = False

class ScenarioListResponse(BaseModel):
    """시나리오 목록 응답 모델"""
    scenarios: List[ScenarioInfo]
    total_count: int
    message: str

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

@router.get("/scenarios", summary="사용 가능한 시나리오 목록 조회", response_model=ScenarioListResponse)
async def get_scenarios(current_user: User = Depends(get_current_user)):
    """모든 사용 가능한 CPX 시나리오 목록을 조회합니다."""
    try:
        llm_service = service_manager.llm_service
        available_scenarios = llm_service.get_available_scenarios()
        scenarios = []
        
        for scenario_id, scenario_name in available_scenarios.items():
            scenario_info = llm_service.get_scenario_info(scenario_id)
            if scenario_info:
                # 환자 이미지 존재 여부 확인
                has_patient_image = False
                try:
                    representative_image = service_manager.patient_image_service.get_scenario_representative_image(scenario_id)
                    has_patient_image = representative_image is not None
                except:
                    pass
                
                # 환자 정보 추출 (간단한 버전)
                patient_info = {
                    "age": "45세" if "45세" in scenario_name else "32세" if "32세" in scenario_name else "63세",
                    "gender": "남성" if "남성" in scenario_name else "여성",
                    "main_symptom": "흉통" if "흉통" in scenario_name else "복통" if "복통" in scenario_name else "기억력 저하"
                }
                
                scenarios.append(ScenarioInfo(
                    id=scenario_id,
                    name=scenario_name,
                    description=f"의과대학 CPX 실기시험용 {scenario_name}",
                    patient_info=patient_info,
                    has_patient_image=has_patient_image
                ))
        
        return ScenarioListResponse(
            scenarios=scenarios,
            total_count=len(scenarios),
            message=f"총 {len(scenarios)}개의 시나리오를 조회했습니다."
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"시나리오 목록 조회 중 오류가 발생했습니다: {str(e)}"
        ) 