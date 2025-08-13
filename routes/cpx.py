from fastapi import APIRouter, Depends, HTTPException, status
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession # AsyncSession 임포트

from core.config import get_db
from core.models import User
from services.cpx_service import CpxService
from schemas.cpx_schemas import (
    CpxResultsCreate, CpxEvaluationUpdate, CpxDetailsUpdate,
    CpxResultsResponse, CpxFullDetailsResponse, CpxEvaluationResponse, CpxDetailsResponse
)
from routes.auth import get_current_user # get_current_user 임포트

# CPX 라우터 생성
router = APIRouter(prefix="/cpx", tags=["CPX"])

# 권한 확인을 위한 의존성 주입 헬퍼
class RoleChecker:
    def __init__(self, allowed_roles: List[str]):
        self.allowed_roles = allowed_roles

    def __call__(self, current_user: User = Depends(get_current_user)):
        if current_user.role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"권한이 없습니다. 허용된 역할: {', '.join(self.allowed_roles)}"
            )
        return current_user

# CPX 서비스 의존성 주입
def get_cpx_service(db: AsyncSession = Depends(get_db)) -> CpxService: # AsyncSession으로 타입 힌트 변경
    return CpxService(db)

@router.post("/", summary="새로운 CPX 실습 결과 생성 (학생용)", response_model=CpxResultsResponse)
async def create_new_cpx_result(
    cpx_data: CpxResultsCreate,
    current_user: User = Depends(RoleChecker(["student"])), # student만 생성 가능
    cpx_service: CpxService = Depends(get_cpx_service)
):
    """
    새로운 CPX 실습 결과를 생성합니다.
    학생 역할의 사용자만 이 엔드포인트를 사용할 수 있습니다.
    """
    if current_user.id != cpx_data.student_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="본인의 CPX 결과만 생성할 수 있습니다."
        )
    
    new_cpx = await cpx_service.create_cpx_result(
        student_id=cpx_data.student_id,
        patient_name=cpx_data.patient_name,
        evaluation_status=cpx_data.evaluation_status
    )
    return CpxResultsResponse.model_validate(new_cpx) # 명시적 변환

@router.get("/me", summary="현재 사용자의 CPX 실습 결과 목록 조회 (학생용)", response_model=List[CpxResultsResponse])
async def get_my_cpx_results(
    current_user: User = Depends(RoleChecker(["student"])), # student만 조회 가능
    cpx_service: CpxService = Depends(get_cpx_service)
):
    """
    현재 로그인한 학생 사용자의 CPX 실습 결과 목록을 조회합니다.
    """
    cpx_results = await cpx_service.get_cpx_results_for_user(current_user.id)
    return [CpxResultsResponse.model_validate(cpx) for cpx in cpx_results] # 명시적 변환

@router.get("/{result_id}", summary="특정 CPX 실습 결과 상세 조회 (학생용)", response_model=CpxFullDetailsResponse)
async def get_single_cpx_details(
    result_id: int,
    current_user: User = Depends(RoleChecker(["student"])), # student만 조회 가능
    cpx_service: CpxService = Depends(get_cpx_service)
):
    """
    특정 CPX 실습 결과의 상세 정보(CpxDetails, CpxEvaluations 포함)를 조회합니다.
    학생 역할의 사용자는 본인의 결과만 조회할 수 있습니다.
    """
    cpx_result = await cpx_service.get_cpx_details_with_evaluations(result_id, current_user.id)
    if not cpx_result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="CPX 결과를 찾을 수 없거나 접근 권한이 없습니다."
        )
    return CpxFullDetailsResponse.model_validate(cpx_result) # 명시적 변환

@router.put("/{result_id}/details", summary="CPX 실습 상세 정보 업데이트 (학생용)", response_model=CpxDetailsResponse)
async def update_cpx_details_api(
    result_id: int,
    details_data: CpxDetailsUpdate,
    current_user: User = Depends(RoleChecker(["student"])), # student만 업데이트 가능
    cpx_service: CpxService = Depends(get_cpx_service)
):
    """
    특정 CPX 실습 결과의 상세 정보(CpxDetails)를 업데이트합니다.
    학생 역할의 사용자는 본인의 결과만 업데이트할 수 있습니다.
    """
    updated_details = await cpx_service.update_cpx_details(
        result_id=result_id,
        user_id=current_user.id,
        memo=details_data.memo,
        system_evaluation_data=details_data.system_evaluation_data
    )
    if not updated_details:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="CPX 상세 정보를 찾을 수 없거나 접근 권한이 없습니다."
        )
    return CpxDetailsResponse.model_validate(updated_details) # 명시적 변환

# 관리자용 CPX 라우터
admin_router = APIRouter(prefix="/admin/cpx", tags=["Admin CPX"])

@admin_router.get("/", summary="모든 CPX 실습 결과 목록 조회 (관리자용)", response_model=List[CpxFullDetailsResponse])
async def get_all_cpx_results_for_admin(
    current_user: User = Depends(RoleChecker(["admin"])), # admin만 조회 가능
    cpx_service: CpxService = Depends(get_cpx_service)
):
    """
    모든 CPX 실습 결과 목록을 조회합니다.
    관리자 역할의 사용자만 이 엔드포인트를 사용할 수 있습니다.
    """
    cpx_results = await cpx_service.get_all_cpx_results_admin()
    return [CpxFullDetailsResponse.model_validate(cpx) for cpx in cpx_results] # 명시적 변환

@admin_router.get("/students/{student_id}/results", summary="특정 학생의 CPX 실습 결과 목록 조회 (관리자용)", response_model=List[CpxFullDetailsResponse])
async def get_cpx_results_by_student_id_admin(
    student_id: int,
    current_user: User = Depends(RoleChecker(["admin"])), # admin만 조회 가능
    cpx_service: CpxService = Depends(get_cpx_service)
):
    """
    특정 학생 ID에 해당하는 CPX 실습 결과 목록을 조회합니다.
    관리자 역할의 사용자만 이 엔드포인트를 사용할 수 있습니다.
    """
    cpx_results = await cpx_service.get_cpx_results_by_student_id(student_id)
    if not cpx_results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"학생 ID {student_id}에 대한 CPX 결과를 찾을 수 없습니다."
        )
    return [CpxFullDetailsResponse.model_validate(cpx) for cpx in cpx_results]

@admin_router.get("/{result_id}", summary="특정 CPX 실습 결과 상세 조회 (관리자용)", response_model=CpxFullDetailsResponse)
async def get_single_cpx_result_admin(
    result_id: int,
    current_user: User = Depends(RoleChecker(["admin"])), # admin만 조회 가능
    cpx_service: CpxService = Depends(get_cpx_service)
):
    """
    특정 result_id에 해당하는 CPX 실습 결과의 상세 정보(CpxDetails, CpxEvaluations 포함)를 조회합니다.
    관리자 역할의 사용자만 이 엔드포인트를 사용할 수 있습니다.
    """
    cpx_result = await cpx_service.get_cpx_result_by_id(result_id)
    if not cpx_result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="CPX 결과를 찾을 수 없습니다."
        )
    return CpxFullDetailsResponse.model_validate(cpx_result)

@admin_router.put("/{result_id}/evaluate", summary="CPX 평가 업데이트 (관리자용)", response_model=CpxEvaluationResponse)
async def update_cpx_evaluation_api(
    result_id: int,
    evaluation_data: CpxEvaluationUpdate,
    current_user: User = Depends(RoleChecker(["admin"])), # admin만 평가 가능
    cpx_service: CpxService = Depends(get_cpx_service)
):
    """
    특정 CPX 실습 결과에 대한 평가를 업데이트합니다.
    관리자 역할의 사용자만 이 엔드포인트를 사용할 수 있습니다.
    """
    updated_evaluation = await cpx_service.update_cpx_evaluation(
        result_id=result_id,
        evaluator_id=current_user.id,
        overall_score=evaluation_data.overall_score,
        detailed_feedback=evaluation_data.detailed_feedback,
        evaluation_status=evaluation_data.evaluation_status
    )
    if not updated_evaluation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="평가 정보를 찾을 수 없습니다."
        )
    
    # CpxResults의 evaluation_status도 함께 업데이트
    # CpxEvaluation의 evaluation_status가 None이 아닐 경우에만 업데이트
    if evaluation_data.evaluation_status is not None:
        await cpx_service.update_cpx_result_status(
            result_id=result_id,
            new_status=evaluation_data.evaluation_status
        )
        
    return CpxEvaluationResponse.model_validate(updated_evaluation) # 명시적 변환
