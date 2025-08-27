from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession # AsyncSession 임포트
from typing import List, Optional

from core.config import get_db
from schemas import privacy_schemas
from services.privacy_service import PrivacyService

router = APIRouter(
    prefix="/privacy",
    tags=["Privacy & Terms"],
)

@router.get(
    "/terms",
    response_model=List[privacy_schemas.TermsAndConditionsResponse],
    summary="모든 약관 또는 특정 제목의 약관 조회",
    description="쿼리 파라미터로 `title`을 제공하여 특정 제목의 약관을 조회하거나, 제공하지 않으면 모든 약관을 조회합니다."
)
async def get_terms(title: Optional[str] = None, db: AsyncSession = Depends(get_db)): # async def 및 AsyncSession으로 변경
    """
    모든 약관 또는 특정 제목의 약관을 조회합니다.
    """
    privacy_service = PrivacyService(db)
    terms = await privacy_service.get_terms_and_conditions(title=title) # await 추가
    if not terms:
        if title:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Terms and conditions with title '{title}' not found."
            )
        else:
            return []
    return terms

@router.post(
    "/terms",
    response_model=privacy_schemas.TermsAndConditionsResponse,
    status_code=status.HTTP_201_CREATED,
    summary="새로운 약관 추가",
    description="새로운 약관 내용을 추가합니다."
)
async def create_terms(terms: privacy_schemas.TermsAndConditionsCreate, db: AsyncSession = Depends(get_db)): # async def 및 AsyncSession으로 변경
    """
    새로운 약관을 추가합니다.
    """
    privacy_service = PrivacyService(db)
    return await privacy_service.create_terms_and_conditions(terms=terms) # await 추가

@router.post(
    "/consent",
    response_model=privacy_schemas.PrivacyConsentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="개인정보 수집 동의 기록",
    description="사용자의 특정 약관에 대한 동의 기록을 생성합니다. 회원가입 시 사용됩니다."
)
async def create_consent(consent: privacy_schemas.PrivacyConsentCreate, db: AsyncSession = Depends(get_db)): # async def 및 AsyncSession으로 변경
    """
    사용자의 약관 동의 기록을 생성합니다.
    """
    privacy_service = PrivacyService(db)
    return await privacy_service.create_privacy_consent(consent=consent) # await 추가
