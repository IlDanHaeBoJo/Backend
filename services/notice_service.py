from typing import List, Optional, Dict
from datetime import datetime, timedelta
import logging
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession # AsyncSession 임포트
from sqlalchemy import or_
from sqlalchemy.future import select # 비동기 쿼리용 select 임포트
from core.models import Notices as DBNotice # DB 모델과 Pydantic 모델 이름 충돌 방지

logger = logging.getLogger(__name__)

# Pydantic 모델 정의
class NoticeBase(BaseModel):
    """공지사항 기본 모델"""
    title: str = Field(..., max_length=255, description="공지사항 제목")
    content: str = Field(..., description="공지사항 내용")
    important: bool = Field(default=False, description="공지사항 중요 여부")
    author_id: int = Field(..., description="작성자 ID")

class NoticeCreate(NoticeBase):
    """공지사항 생성 모델"""
    pass

class NoticeUpdate(BaseModel):
    """공지사항 수정 모델"""
    title: Optional[str] = Field(None, max_length=255, description="공지사항 제목")
    content: Optional[str] = Field(None, description="공지사항 내용")
    important: Optional[bool] = Field(None, description="공지사항 중요 여부")

class Notice(NoticeBase):
    """공지사항 응답 모델"""
    notice_id: int = Field(..., description="공지사항 고유 식별자")
    view_count: int = Field(default=0, description="조회수")
    created_at: datetime = Field(..., description="공지사항 생성 시간")
    updated_at: datetime = Field(..., description="공지사항 마지막 업데이트 시간")
    
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class NoticeStats(BaseModel):
    """공지사항 통계 모델"""
    total_notices: int = Field(..., description="총 공지사항 수")
    total_views: int = Field(..., description="총 조회수")
    important_notice_count: int = Field(..., description="중요 공지사항 수")
    recent_notices_count: int = Field(..., description="최근 7일 공지사항 수")

class NoticeService:
    # __init__ 메서드 제거
    
    async def get_all_notices(self, db: AsyncSession) -> List[Notice]:
        """모든 공지사항 조회"""
        result = await db.execute(select(DBNotice).order_by(DBNotice.created_at.desc()))
        db_notices = result.scalars().all()
        return [Notice.model_validate(notice) for notice in db_notices]
    
    async def get_notice_by_id(self, db: AsyncSession, notice_id: int) -> Optional[Notice]:
        """ID로 공지사항 조회"""
        result = await db.execute(select(DBNotice).filter(DBNotice.notice_id == notice_id))
        db_notice = result.scalars().first()
        if db_notice:
            return Notice.model_validate(db_notice)
        return None
    
    async def create_notice(self, db: AsyncSession, notice_data: NoticeCreate) -> Notice:
        """새 공지사항 생성"""
        db_notice = DBNotice(**notice_data.model_dump())
        
        db.add(db_notice)
        await db.commit()
        await db.refresh(db_notice)
        
        logger.info(f"새 공지사항 생성: {db_notice.title}")
        return Notice.model_validate(db_notice)
    
    async def update_notice(self, db: AsyncSession, notice_id: int, notice_data: NoticeUpdate) -> Optional[Notice]:
        """공지사항 수정"""
        result = await db.execute(select(DBNotice).filter(DBNotice.notice_id == notice_id))
        db_notice = result.scalars().first()
        if not db_notice:
            return None
        
        if notice_data.title is not None:
            db_notice.title = notice_data.title
        if notice_data.content is not None:
            db_notice.content = notice_data.content
        if notice_data.important is not None:
            db_notice.important = notice_data.important
        
        await db.commit()
        await db.refresh(db_notice)
        
        logger.info(f"공지사항 수정: {db_notice.title}")
        return Notice.model_validate(db_notice)
    
    async def delete_notice(self, db: AsyncSession, notice_id: int) -> bool:
        """공지사항 삭제"""
        result = await db.execute(select(DBNotice).filter(DBNotice.notice_id == notice_id))
        db_notice = result.scalars().first()
        if not db_notice:
            return False
        
        await db.delete(db_notice)
        await db.commit()
        logger.info(f"공지사항 삭제: {db_notice.title}")
        return True
    
    async def get_important_notices(self, db: AsyncSession) -> List[Notice]:
        """중요 공지사항만 조회"""
        result = await db.execute(select(DBNotice).filter(DBNotice.important == True).order_by(DBNotice.created_at.desc()))
        db_notices = result.scalars().all()
        return [Notice.model_validate(notice) for notice in db_notices]
    
    async def increment_view_count(self, db: AsyncSession, notice_id: int) -> bool:
        """공지사항 조회수 증가"""
        result = await db.execute(select(DBNotice).filter(DBNotice.notice_id == notice_id))
        db_notice = result.scalars().first()
        if not db_notice:
            return False
        
        db_notice.view_count += 1
        await db.commit()
        await db.refresh(db_notice)
        return True
    
    async def get_notice_statistics(self, db: AsyncSession) -> NoticeStats:
        """공지사항 통계 조회"""
        result = await db.execute(select(DBNotice))
        db_notices = result.scalars().all()
        if not db_notices:
            return NoticeStats(
                total_notices=0,
                total_views=0,
                important_notice_count=0,
                recent_notices_count=0
            )
        
        total_notices = len(db_notices)
        total_views = sum(notice.view_count for notice in db_notices)
        important_notice_count = len([n for n in db_notices if n.important])
        
        # 최근 7일 공지사항 수
        week_ago = datetime.now() - timedelta(days=7)
        recent_notices_count = len([n for n in db_notices if n.created_at >= week_ago])
        
        return NoticeStats(
            total_notices=total_notices,
            total_views=total_views,
            important_notice_count=important_notice_count,
            recent_notices_count=recent_notices_count
        )
    
    async def search_notices(self, db: AsyncSession, keyword: str) -> List[Notice]:
        """키워드로 공지사항 검색"""
        keyword_lower = f"%{keyword.lower()}%"
        result = await db.execute(
            select(DBNotice).filter(
                or_(
                    DBNotice.title.ilike(keyword_lower), 
                    DBNotice.content.ilike(keyword_lower)
                )
            ).order_by(DBNotice.created_at.desc())
        )
        db_notices = result.scalars().all()
        
        return [Notice.model_validate(notice) for notice in db_notices]
