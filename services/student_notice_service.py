from typing import List, Optional
from datetime import datetime
import logging
from services.notice_service import notice_service, Notice

logger = logging.getLogger(__name__)

class StudentNoticeService:
    """학생용 공지사항 서비스 (읽기 전용)"""
    
    def get_all_notices(self) -> List[Notice]:
        """모든 공지사항 조회 (학생용)"""
        logger.info("학생이 모든 공지사항을 조회했습니다.")
        return notice_service.get_all_notices()
    
    def get_notice_by_id(self, notice_id: int) -> Optional[Notice]:
        """ID로 특정 공지사항 조회 (학생용)"""
        logger.info(f"학생이 공지사항 ID {notice_id}를 조회했습니다.")
        # 조회수 증가
        notice_service.increment_view_count(notice_id)
        return notice_service.get_notice_by_id(notice_id)
    
    def get_important_notices(self) -> List[Notice]:
        """높은 우선순위 공지사항만 조회 (학생용)"""
        logger.info("학생이 중요 공지사항을 조회했습니다.")
        return notice_service.get_important_notices()
    
    def get_notices_by_priority(self, min_priority: int = 0) -> List[Notice]:
        """우선순위별 공지사항 조회 (학생용)"""
        logger.info(f"학생이 우선순위 {min_priority} 이상 공지사항을 조회했습니다.")
        return notice_service.get_notices_by_priority(min_priority)
    
    def get_recent_notices(self, limit: int = 5) -> List[Notice]:
        """최근 공지사항 조회 (학생용)"""
        logger.info(f"학생이 최근 {limit}개 공지사항을 조회했습니다.")
        all_notices = notice_service.get_all_notices()
        return all_notices[:limit]
    
    def search_notices(self, keyword: str) -> List[Notice]:
        """키워드로 공지사항 검색 (학생용)"""
        logger.info(f"학생이 '{keyword}' 키워드로 공지사항을 검색했습니다.")
        return notice_service.search_notices(keyword)

# 전역 서비스 인스턴스
student_notice_service = StudentNoticeService() 