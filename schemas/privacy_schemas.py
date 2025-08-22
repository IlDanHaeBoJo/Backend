from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

# TermsAndConditions 스키마
class TermsAndConditionsBase(BaseModel):
    title: str = Field(..., max_length=255, description="약관 제목 (예: 개인정보 처리방침, 서비스 이용약관)")
    content: str = Field(..., description="약관 내용")
    effective_date: datetime = Field(..., description="약관 발효일")

class TermsAndConditionsCreate(TermsAndConditionsBase):
    pass

class TermsAndConditionsResponse(TermsAndConditionsBase):
    id: int = Field(..., description="약관 고유 식별자")
    created_at: datetime = Field(..., description="약관 생성 시간")
    updated_at: datetime = Field(..., description="약관 마지막 업데이트 시간")

    class Config:
        from_attributes = True

# PrivacyConsent 스키마
class PrivacyConsentCreate(BaseModel):
    user_id: int = Field(..., description="동의한 사용자 ID")
    terms_id: int = Field(..., description="동의한 약관 ID")

class PrivacyConsentResponse(BaseModel):
    id: int = Field(..., description="개인정보 동의 기록 고유 식별자")
    user_id: int = Field(..., description="동의한 사용자 ID")
    terms_id: int = Field(..., description="동의한 약관 ID")
    consent_date: datetime = Field(..., description="동의한 날짜 및 시간")

    class Config:
        from_attributes = True
