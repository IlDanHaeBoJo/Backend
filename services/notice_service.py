from typing import List, Optional, Dict
from datetime import datetime, timedelta
import logging
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession # AsyncSession 임포트
from sqlalchemy import or_
from sqlalchemy.future import select # 비동기 쿼리용 select 임포트
from sqlalchemy.orm import selectinload # 관계 데이터 로딩용
from core.models import Notices as DBNotice # DB 모델과 Pydantic 모델 이름 충돌 방지
from services.attachment_service import Attachment, AttachmentService

logger = logging.getLogger(__name__)

# Pydantic 모델 정의
class NoticeBase(BaseModel):
    """공지사항 기본 모델"""
    title: str = Field(..., max_length=255, description="공지사항 제목")
    content: str = Field(..., description="공지사항 내용")
    priority: int = Field(default=0, description="공지사항 중요도 (높을수록 상단 노출 등, 0이 기본)")
    author_id: int = Field(..., description="작성자 ID")

class NoticeCreate(BaseModel):
    """공지사항 생성 모델"""
    title: str = Field(..., max_length=255, description="공지사항 제목")
    content: str = Field(..., description="공지사항 내용")
    priority: int = Field(default=0, description="공지사항 중요도 (높을수록 상단 노출 등, 0이 기본)")
    author_id: Optional[int] = Field(None, description="작성자 ID (자동 설정됨)")

class NoticeUpdate(BaseModel):
    """공지사항 수정 모델"""
    title: Optional[str] = Field(None, max_length=255, description="공지사항 제목")
    content: Optional[str] = Field(None, description="공지사항 내용")
    priority: Optional[int] = Field(None, description="공지사항 중요도 (높을수록 상단 노출 등, 0이 기본)")

class Notice(NoticeBase):
    """공지사항 응답 모델"""
    notice_id: int = Field(..., description="공지사항 고유 식별자")
    view_count: int = Field(default=0, description="조회수")
    created_at: datetime = Field(..., description="공지사항 생성 시간")
    updated_at: datetime = Field(..., description="공지사항 마지막 업데이트 시간")
    attachments: Optional[List[Attachment]] = Field(default=[], description="첨부파일 목록")
    
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class NoticeStats(BaseModel):
    """공지사항 통계 모델"""
    total_notices: int = Field(..., description="총 공지사항 수")
    total_views: int = Field(..., description="총 조회수")
    high_priority_notice_count: int = Field(..., description="높은 우선순위 공지사항 수 (priority > 0)")
    recent_notices_count: int = Field(..., description="최근 7일 공지사항 수")

class NoticeService:
    # __init__ 메서드 제거
    
    async def get_all_notices(self, db: AsyncSession) -> List[Notice]:
        """모든 공지사항 조회"""
        # selectinload를 사용하여 관계 데이터를 함께 로딩
        stmt = select(DBNotice).options(selectinload(DBNotice.attachments)).order_by(DBNotice.created_at.desc())
        result = await db.execute(stmt)
        db_notices = result.scalars().all()
        
        notices = []
        for notice in db_notices:
            # 관계 데이터가 이미 로딩되어 있으므로 직접 접근 가능
            notice_data = Notice.model_validate(notice)
            notices.append(notice_data)
        
        return notices
    
    async def get_notice_by_id(self, db: AsyncSession, notice_id: int) -> Optional[Notice]:
        """ID로 공지사항 조회"""
        stmt = select(DBNotice).options(selectinload(DBNotice.attachments)).filter(DBNotice.notice_id == notice_id)
        result = await db.execute(stmt)
        db_notice = result.scalars().first()
        if db_notice:
            # 관계 데이터가 이미 로딩되어 있으므로 직접 접근 가능
            notice_data = Notice.model_validate(db_notice)
            return notice_data
        return None
    
    async def create_notice(self, db: AsyncSession, notice_data: NoticeCreate) -> Notice:
        """새 공지사항 생성"""
        db_notice = DBNotice(**notice_data.model_dump())
        
        db.add(db_notice)
        await db.commit()
        await db.refresh(db_notice)
        
        logger.info(f"새 공지사항 생성: {db_notice.title}")
        # 빈 첨부파일 목록으로 딕셔너리 생성
        notice_dict = {
            'notice_id': db_notice.notice_id,
            'title': db_notice.title,
            'content': db_notice.content,
            'priority': db_notice.priority,
            'author_id': db_notice.author_id,
            'view_count': db_notice.view_count,
            'created_at': db_notice.created_at,
            'updated_at': db_notice.updated_at,
            'attachments': []
        }
        return Notice.model_validate(notice_dict)
    
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
        if notice_data.priority is not None:
            db_notice.priority = notice_data.priority
        
        await db.commit()
        await db.refresh(db_notice)
        
        logger.info(f"공지사항 수정: {db_notice.title}")
        # 수정된 공지사항을 다시 조회하여 관계 데이터 포함
        stmt = select(DBNotice).options(selectinload(DBNotice.attachments)).filter(DBNotice.notice_id == notice_id)
        result = await db.execute(stmt)
        updated_notice = result.scalars().first()
        return Notice.model_validate(updated_notice)
    
    async def delete_notice(self, db: AsyncSession, notice_id: int) -> bool:
        """공지사항 삭제"""
        result = await db.execute(select(DBNotice).filter(DBNotice.notice_id == notice_id))
        db_notice = result.scalars().first()
        if not db_notice:
            return False
        
        # 첨부파일 정보도 함께 삭제
        attachment_service = AttachmentService()
        await attachment_service.delete_attachments_by_notice_id(db, notice_id)
        
        await db.delete(db_notice)
        await db.commit()
        logger.info(f"공지사항 삭제: {db_notice.title}")
        return True
    
    async def get_high_priority_notices(self, db: AsyncSession) -> List[Notice]:
        """높은 우선순위 공지사항만 조회 (priority > 0)"""
        stmt = select(DBNotice).options(selectinload(DBNotice.attachments)).filter(DBNotice.priority > 0).order_by(DBNotice.priority.desc(), DBNotice.created_at.desc())
        result = await db.execute(stmt)
        db_notices = result.scalars().all()
        
        notices = []
        for notice in db_notices:
            # 관계 데이터가 이미 로딩되어 있으므로 직접 접근 가능
            notice_data = Notice.model_validate(notice)
            notices.append(notice_data)
        
        return notices
    
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
            high_priority_notice_count=0,
            recent_notices_count=0
        )
        
        total_notices = len(db_notices)
        total_views = sum(notice.view_count for notice in db_notices)
        high_priority_notice_count = len([n for n in db_notices if n.priority > 0])
        
        # 최근 7일 공지사항 수
        week_ago = datetime.now() - timedelta(days=7)
        recent_notices_count = len([n for n in db_notices if n.created_at >= week_ago])
        
        return NoticeStats(
            total_notices=total_notices,
            total_views=total_views,
            high_priority_notice_count=high_priority_notice_count,
            recent_notices_count=recent_notices_count
        )
    
    async def search_notices(self, db: AsyncSession, keyword: str) -> List[Notice]:
        """키워드로 공지사항 검색"""
        keyword_lower = f"%{keyword.lower()}%"
        stmt = select(DBNotice).options(selectinload(DBNotice.attachments)).filter(
            or_(
                DBNotice.title.ilike(keyword_lower), 
                DBNotice.content.ilike(keyword_lower)
            )
        ).order_by(DBNotice.created_at.desc())
        result = await db.execute(stmt)
        db_notices = result.scalars().all()
        
        notices = []
        for notice in db_notices:
            # 관계 데이터가 이미 로딩되어 있으므로 직접 접근 가능
            notice_data = Notice.model_validate(notice)
            notices.append(notice_data)
        
        return notices
