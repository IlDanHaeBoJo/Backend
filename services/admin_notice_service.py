from typing import List, Optional
from datetime import datetime
import logging
from services.notice_service import Notice, NoticeCreate, NoticeUpdate, NoticeStats, NoticeService # NoticeService 임포트
from sqlalchemy.ext.asyncio import AsyncSession # AsyncSession 임포트

logger = logging.getLogger(__name__)

# AdminNoticeService 클래스 제거, 함수들로 직접 구성

async def get_all_notices(db: AsyncSession, notice_service: NoticeService) -> List[Notice]:
    """모든 공지사항 조회 (관리자용)"""
    logger.info("관리자가 모든 공지사항을 조회했습니다.")
    return await notice_service.get_all_notices(db)

async def get_notice_by_id(db: AsyncSession, notice_id: int, notice_service: NoticeService) -> Optional[Notice]:
    """ID로 특정 공지사항 조회 (관리자용)"""
    logger.info(f"관리자가 공지사항 ID {notice_id}를 조회했습니다.")
    return await notice_service.get_notice_by_id(db, notice_id)

async def create_notice(db: AsyncSession, notice_data: NoticeCreate, notice_service: NoticeService) -> Notice:
    """새 공지사항 생성 (관리자용)"""
    logger.info(f"관리자가 새 공지사항을 생성했습니다: {notice_data.title}")
    return await notice_service.create_notice(db, notice_data)

async def update_notice(db: AsyncSession, notice_id: int, notice_data: NoticeUpdate, notice_service: NoticeService) -> Optional[Notice]:
    """공지사항 수정 (관리자용)"""
    logger.info(f"관리자가 공지사항 ID {notice_id}를 수정했습니다.")
    return await notice_service.update_notice(db, notice_id, notice_data)

async def delete_notice(db: AsyncSession, notice_id: int, notice_service: NoticeService) -> bool:
    """공지사항 삭제 (관리자용)"""
    logger.info(f"관리자가 공지사항 ID {notice_id}를 삭제했습니다.")
    return await notice_service.delete_notice(db, notice_id)

async def get_important_notices(db: AsyncSession, notice_service: NoticeService) -> List[Notice]:
    """중요 공지사항만 조회 (관리자용)"""
    logger.info("관리자가 중요 공지사항을 조회했습니다.")
    return await notice_service.get_important_notices(db)

async def update_notice_important(db: AsyncSession, notice_id: int, important: bool, notice_service: NoticeService) -> Optional[Notice]:
    """공지사항 중요도 업데이트 (관리자용)"""
    logger.info(f"관리자가 공지사항 ID {notice_id}의 중요도를 {important}로 변경했습니다.")
    notice = await notice_service.get_notice_by_id(db, notice_id)
    if not notice:
        return None
    
    # important 필드를 업데이트
    updated_notice = await notice_service.update_notice(
        db, 
        notice_id, 
        NoticeUpdate(important=important)
    )
    return updated_notice

async def get_notice_statistics(db: AsyncSession, notice_service: NoticeService) -> NoticeStats:
    """공지사항 통계 조회 (관리자용)"""
    logger.info("관리자가 공지사항 통계를 조회했습니다.")
    return await notice_service.get_notice_statistics(db)

async def search_notices(db: AsyncSession, keyword: str, notice_service: NoticeService, search_type: str = "all") -> List[Notice]:
    """키워드로 공지사항 검색 (관리자용)"""
    logger.info(f"관리자가 '{keyword}' 키워드로 공지사항을 검색했습니다.")
    
    if search_type == "title":
        # 제목만 검색
        all_notices = await notice_service.get_all_notices(db) # 모든 공지사항을 가져와서 필터링
        filtered_notices = [
            notice for notice in all_notices
            if keyword.lower() in notice.title.lower()
        ]
    elif search_type == "content":
        # 내용만 검색
        all_notices = await notice_service.get_all_notices(db) # 모든 공지사항을 가져와서 필터링
        filtered_notices = [
            notice for notice in all_notices
            if keyword.lower() in notice.content.lower()
        ]
    else:
        # 전체 검색 (NoticeService의 search_notices는 이미 DB 검색을 수행)
        filtered_notices = await notice_service.search_notices(db, keyword)
    
    return filtered_notices
