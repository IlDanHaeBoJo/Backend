from typing import List, Optional
from datetime import datetime
import logging
from services.notice_service import notice_service, Notice, NoticeCreate, NoticeUpdate, NoticeStats

logger = logging.getLogger(__name__)

class AdminNoticeService:
    """관리자용 공지사항 서비스 (CRUD 모든 권한)"""
    
    def get_all_notices(self) -> List[Notice]:
        """모든 공지사항 조회 (관리자용)"""
        logger.info("관리자가 모든 공지사항을 조회했습니다.")
        return notice_service.get_all_notices()
    
    def get_notice_by_id(self, notice_id: int) -> Optional[Notice]:
        """ID로 특정 공지사항 조회 (관리자용)"""
        logger.info(f"관리자가 공지사항 ID {notice_id}를 조회했습니다.")
        return notice_service.get_notice_by_id(notice_id)
    
    def create_notice(self, notice_data: NoticeCreate) -> Notice:
        """새 공지사항 생성 (관리자용)"""
        logger.info(f"관리자가 새 공지사항을 생성했습니다: {notice_data.title}")
        return notice_service.create_notice(notice_data)
    
    def update_notice(self, notice_id: int, notice_data: NoticeUpdate) -> Optional[Notice]:
        """공지사항 수정 (관리자용)"""
        logger.info(f"관리자가 공지사항 ID {notice_id}를 수정했습니다.")
        return notice_service.update_notice(notice_id, notice_data)
    
    def delete_notice(self, notice_id: int) -> bool:
        """공지사항 삭제 (관리자용)"""
        logger.info(f"관리자가 공지사항 ID {notice_id}를 삭제했습니다.")
        return notice_service.delete_notice(notice_id)
    
    def get_important_notices(self) -> List[Notice]:
        """높은 우선순위 공지사항만 조회 (관리자용)"""
        logger.info("관리자가 중요 공지사항을 조회했습니다.")
        return notice_service.get_important_notices()
    
    def get_notices_by_priority(self, min_priority: int = 0) -> List[Notice]:
        """우선순위별 공지사항 조회 (관리자용)"""
        logger.info(f"관리자가 우선순위 {min_priority} 이상 공지사항을 조회했습니다.")
        return notice_service.get_notices_by_priority(min_priority)
    
    def toggle_notice_priority(self, notice_id: int) -> Optional[Notice]:
        """공지사항 우선순위 토글 (관리자용) - 높은 우선순위로 변경"""
        logger.info(f"관리자가 공지사항 ID {notice_id}의 우선순위를 토글했습니다.")
        notice = notice_service.get_notice_by_id(notice_id)
        if not notice:
            return None
        
        # 우선순위를 높은 값으로 토글 (50 이상이면 0으로, 아니면 80으로)
        new_priority = 0 if notice.priority >= 50 else 80
        updated_notice = notice_service.update_notice(
            notice_id, 
            NoticeUpdate(priority=new_priority)
        )
        return updated_notice
    
    def get_notice_statistics(self) -> NoticeStats:
        """공지사항 통계 조회 (관리자용)"""
        logger.info("관리자가 공지사항 통계를 조회했습니다.")
        return notice_service.get_notice_statistics()
    
    def search_notices(self, keyword: str, search_type: str = "all") -> List[Notice]:
        """키워드로 공지사항 검색 (관리자용)"""
        logger.info(f"관리자가 '{keyword}' 키워드로 공지사항을 검색했습니다.")
        all_notices = notice_service.get_all_notices()
        
        if search_type == "title":
            # 제목만 검색
            filtered_notices = [
                notice for notice in all_notices
                if keyword.lower() in notice.title.lower()
            ]
        elif search_type == "content":
            # 내용만 검색
            filtered_notices = [
                notice for notice in all_notices
                if keyword.lower() in notice.content.lower()
            ]
        else:
            # 전체 검색
            filtered_notices = notice_service.search_notices(keyword)
        
        return filtered_notices

# 전역 서비스 인스턴스
admin_notice_service = AdminNoticeService() 