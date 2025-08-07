from typing import List, Optional
from datetime import datetime
import logging
from services.notice_service import Notice, NoticeService # NoticeService 임포트
from sqlalchemy.ext.asyncio import AsyncSession # AsyncSession 임포트

logger = logging.getLogger(__name__)

class StudentNoticeService:
    """학생용 공지사항 서비스 (읽기 전용)"""
    
    async def get_all_notices(self, db: AsyncSession, notice_service: NoticeService) -> List[Notice]:
        """모든 공지사항 조회 (학생용)"""
        logger.info("학생이 모든 공지사항을 조회했습니다.")
        return await notice_service.get_all_notices(db)
    
    async def get_notice_by_id(self, db: AsyncSession, notice_id: int, notice_service: NoticeService) -> Optional[Notice]:
        """ID로 특정 공지사항 조회 (학생용)"""
        logger.info(f"학생이 공지사항 ID {notice_id}를 조회했습니다.")
        # 조회수 증가
        await notice_service.increment_view_count(db, notice_id)
        return await notice_service.get_notice_by_id(db, notice_id)
    
    async def get_high_priority_notices(self, db: AsyncSession, notice_service: NoticeService) -> List[Notice]:
        """높은 우선순위 공지사항만 조회 (학생용)"""
        logger.info("학생이 높은 우선순위 공지사항을 조회했습니다.")
        return await notice_service.get_high_priority_notices(db)
    
    async def get_recent_notices(self, db: AsyncSession, notice_service: NoticeService, limit: int = 5) -> List[Notice]:
        """최근 공지사항 조회 (학생용)"""
        logger.info(f"학생이 최근 {limit}개 공지사항을 조회했습니다.")
        all_notices = await notice_service.get_all_notices(db)
        return all_notices[:limit]
    
    async def search_notices(self, db: AsyncSession, keyword: str, notice_service: NoticeService) -> List[Notice]:
        """키워드로 공지사항 검색 (학생용)"""
        logger.info(f"학생이 '{keyword}' 키워드로 공지사항을 검색했습니다.")
        return await notice_service.search_notices(db, keyword)
