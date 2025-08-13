from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Optional
from pydantic import BaseModel
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from core.config import get_db
from core.models import User
from routes.auth import get_current_user
from services.evaluation_service import EvaluationService

# API 라우터 생성
router = APIRouter(prefix="/evaluation", tags=["CPX 평가"])

# Pydantic 모델들
class EvaluationSummary(BaseModel):
    user_id: str
    scenario_id: str
    total_score: float
    grade: str
    evaluation_date: str
    conversation_duration_minutes: float

class DetailedEvaluationResponse(BaseModel):
    evaluation_metadata: Dict
    scores: Dict
    checklist_results: Dict
    question_analysis: Dict
    feedback: Dict
    conversation_summary: Dict

class EvaluationListResponse(BaseModel):
    evaluations: List[EvaluationSummary]
    total_count: int
    average_score: float

# 평가 서비스 인스턴스
evaluation_service = EvaluationService()

@router.get("/", summary="사용자의 평가 목록 조회", response_model=EvaluationListResponse)
async def get_user_evaluations(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    limit: int = 10,
    offset: int = 0
):
    """현재 사용자의 CPX 평가 목록을 조회합니다."""
    
    try:
        # TODO: 실제 데이터베이스에서 조회 구현
        # 현재는 임시 데이터 반환
        evaluations = []
        
        return EvaluationListResponse(
            evaluations=evaluations,
            total_count=len(evaluations),
            average_score=0.0
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"평가 목록 조회 중 오류가 발생했습니다: {str(e)}")

@router.get("/{evaluation_id}", summary="특정 평가 상세 조회", response_model=DetailedEvaluationResponse)
async def get_evaluation_detail(
    evaluation_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """특정 평가의 상세 내용을 조회합니다."""
    
    try:
        # TODO: 실제 데이터베이스에서 조회 구현
        # 현재는 임시 응답 반환
        
        raise HTTPException(status_code=404, detail="해당 평가를 찾을 수 없습니다.")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"평가 상세 조회 중 오류가 발생했습니다: {str(e)}")

@router.get("/stats/summary", summary="평가 통계 요약")
async def get_evaluation_stats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """사용자의 평가 통계를 요약해서 제공합니다."""
    
    try:
        # TODO: 실제 통계 계산 구현
        stats = {
            "total_evaluations": 0,
            "average_score": 0.0,
            "highest_score": 0.0,
            "recent_improvement": 0.0,
            "category_averages": {
                "checklist_score": 0.0,
                "technique_score": 0.0,
                "communication_score": 0.0
            },
            "scenario_performance": {},
            "recent_evaluations": []
        }
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"평가 통계 조회 중 오류가 발생했습니다: {str(e)}")

@router.post("/test", summary="평가 시스템 테스트 (개발용)")
async def test_evaluation_system(
    test_data: Dict,
    current_user: User = Depends(get_current_user)
):
    """평가 시스템을 테스트하기 위한 엔드포인트입니다."""
    
    try:
        # 테스트용 대화 로그
        test_conversation = test_data.get("conversation_log", [
            {
                "type": "student",
                "content": "안녕하세요. 어떻게 오셨나요?",
                "timestamp": datetime.now().isoformat()
            },
            {
                "type": "patient",
                "content": "자꾸 깜빡깜빡하는 것 같아요.",
                "timestamp": datetime.now().isoformat()
            }
        ])
        
        scenario_id = test_data.get("scenario_id", "3")
        user_id = str(current_user.id)
        
        # 평가 실행
        evaluation_result = await evaluation_service.evaluate_conversation(
            user_id=user_id,
            scenario_id=scenario_id,
            conversation_log=test_conversation
        )
        
        return {
            "message": "평가 테스트 완료",
            "evaluation_result": evaluation_result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"평가 테스트 중 오류가 발생했습니다: {str(e)}")

# 평가 기준 조회 API
@router.get("/criteria/checklist", summary="병력청취 체크리스트 조회")
async def get_checklist_criteria():
    """병력청취 체크리스트 기준을 조회합니다."""
    
    return {
        "checklist": evaluation_service.history_taking_checklist,
        "question_classification": evaluation_service.question_classification,
        "description": "CPX 병력청취 평가 기준"
    }

@router.get("/criteria/scoring", summary="점수 계산 기준 조회")
async def get_scoring_criteria():
    """점수 계산 기준을 조회합니다."""
    
    return {
        "scoring_weights": {
            "checklist_score": 0.7,
            "technique_score": 0.2,
            "communication_score": 0.1
        },
        "grading_scale": {
            "A+": "90-100점",
            "A": "85-89점", 
            "B+": "80-84점",
            "B": "75-79점",
            "C+": "70-74점",
            "C": "65-69점",
            "F": "65점 미만"
        },
        "description": "CPX 평가 점수 계산 및 등급 기준"
    }