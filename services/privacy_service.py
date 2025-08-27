from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession # AsyncSession 임포트
from core.models import TermsAndConditions, PrivacyConsent # 필요한 모델만 임포트
from schemas import privacy_schemas
from fastapi import HTTPException, status
from typing import List, Optional

class PrivacyService:
    def __init__(self, db: AsyncSession): # AsyncSession으로 타입 힌트 변경
        self.db = db

    async def get_terms_and_conditions(self, title: Optional[str] = None) -> List[TermsAndConditions]:
        """
        모든 약관 또는 특정 제목의 약관을 조회합니다.
        """
        stmt = select(TermsAndConditions)
        if title:
            stmt = stmt.filter(TermsAndConditions.title == title) # 문자열로 필터링
        result = await self.db.execute(stmt)
        return result.scalars().all()

    async def create_terms_and_conditions(self, terms: privacy_schemas.TermsAndConditionsCreate) -> TermsAndConditions:
        """
        새로운 약관을 추가합니다.
        동일한 제목의 약관이 이미 존재하면 409 Conflict 에러를 발생시킵니다.
        """
        stmt = select(TermsAndConditions).filter(
            TermsAndConditions.title == terms.title # 문자열로 비교
        )
        result = await self.db.execute(stmt)
        db_terms = result.scalars().first()
        if db_terms:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Terms and conditions with title '{terms.title}' already exists."
            )
        
        db_terms = TermsAndConditions(**terms.model_dump())
        self.db.add(db_terms)
        await self.db.commit() # await 추가
        await self.db.refresh(db_terms) # await 추가
        return db_terms

    async def create_privacy_consent(self, consent: privacy_schemas.PrivacyConsentCreate) -> PrivacyConsent:
        """
        사용자의 약관 동의 기록을 생성합니다.
        """
        # 약관이 실제로 존재하는지 확인
        stmt = select(TermsAndConditions).filter(
            TermsAndConditions.id == consent.terms_id
        )
        result = await self.db.execute(stmt)
        terms_exists = result.scalars().first()
        if not terms_exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Terms and conditions with ID {consent.terms_id} not found."
            )

        # 사용자가 이미 해당 약관에 동의했는지 확인 (선택 사항, 필요에 따라 중복 동의 허용 여부 결정)
        # stmt_consent = select(PrivacyConsent).filter(
        #     PrivacyConsent.user_id == consent.user_id,
        #     PrivacyConsent.terms_id == consent.terms_id
        # )
        # result_consent = await self.db.execute(stmt_consent)
        # db_consent = result_consent.scalars().first()
        # if db_consent:
        #     raise HTTPException(
        #         status_code=status.HTTP_409_CONFLICT,
        #         detail="User has already consented to these terms."
        #     )

        db_consent = PrivacyConsent(**consent.model_dump())
        self.db.add(db_consent)
        await self.db.commit() # await 추가
        await self.db.refresh(db_consent) # await 추가
        return db_consent
