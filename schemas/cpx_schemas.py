from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from datetime import datetime

# CpxEvaluations 스키마
class CpxEvaluationBase(BaseModel):
    overall_score: Optional[int] = Field(None, description="전체 점수 (예: 100점 만점)")
    detailed_feedback: Optional[str] = Field(None, description="학생에게 전달될 종합 피드백 및 상세 평가 코멘트")
    evaluation_status: str = Field("평가대기", description="평가 처리 상태 (예: 임시저장, 완료, 수정필요)")

class CpxEvaluationCreate(CpxEvaluationBase):
    pass

class CpxEvaluationUpdate(CpxEvaluationBase):
    pass

class CpxEvaluationResponse(CpxEvaluationBase):
    evaluation_id: int
    result_id: int
    evaluator_id: int
    evaluation_date: datetime
    created_at: datetime
    updated_at: datetime
    markdown_feedback: Optional[str] = Field(None, description="평가 결과를 마크다운 형식으로 변환한 피드백")

    class Config:
        from_attributes = True

# CpxDetails 스키마
class CpxDetailsBase(BaseModel):
    memo: Optional[str] = Field(None, description="실습 중 작성한 메모")
    system_evaluation_data: Optional[dict] = Field(None, description="CPX 실습에 대한 AI 평가 결과 및 상세 데이터")
    
    @field_validator('system_evaluation_data', mode='before')
    @classmethod
    def parse_json_string(cls, v):
        if isinstance(v, str):
            import json
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return v
        return v

class CpxDetailsUpdate(CpxDetailsBase):
    pass

class CpxDetailsResponse(CpxDetailsBase):
    detail_id: int
    result_id: int
    last_updated_at: datetime

    class Config:
        from_attributes = True

# CpxResults 스키마
class CpxResultsBase(BaseModel):
    student_id: int
    patient_name: str = Field(..., description="환자 이름 또는 역할 (가상 환자 이름)")
    evaluation_status: str = Field("진행중", description="평가 상태 (예: 진행중, 완료, 확인대기, 확인완료)")

class CpxResultsCreate(CpxResultsBase):
    pass

class CpxResultsResponse(CpxResultsBase):
    result_id: int
    practice_date: datetime
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# CPX 상세 정보와 평가를 포함하는 응답 스키마 (사용자 및 관리자 상세 조회용)
class CpxFullDetailsResponse(CpxResultsResponse):
    cpx_detail: Optional[CpxDetailsResponse] = None
    cpx_evaluation: Optional[CpxEvaluationResponse] = None
