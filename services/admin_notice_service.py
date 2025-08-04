from typing import List, Optional
from datetime import datetime
import logging
from models.notice import Notice, NoticeCreate, NoticeUpdate, NoticeType
from services.notice_service import notice_service

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
        """중요 공지사항만 조회 (관리자용)"""
        logger.info("관리자가 중요 공지사항을 조회했습니다.")
        return notice_service.get_important_notices()
    
    def get_notices_by_type(self, notice_type: NoticeType) -> List[Notice]:
        """타입별 공지사항 조회 (관리자용)"""
        logger.info(f"관리자가 {notice_type} 타입 공지사항을 조회했습니다.")
        return notice_service.get_notices_by_type(notice_type)
    
    def toggle_notice_importance(self, notice_id: int) -> Optional[Notice]:
        """공지사항 중요도 토글 (관리자용)"""
        logger.info(f"관리자가 공지사항 ID {notice_id}의 중요도를 토글했습니다.")
        notice = notice_service.get_notice_by_id(notice_id)
        if not notice:
            return None
        
        updated_notice = notice_service.update_notice(
            notice_id, 
            NoticeUpdate(is_important=not notice.is_important)
        )
        return updated_notice
    
    def get_notice_statistics(self) -> dict:
        """공지사항 통계 조회 (관리자용)"""
        logger.info("관리자가 공지사항 통계를 조회했습니다.")
        all_notices = notice_service.get_all_notices()
        
        stats = {
            "total_notices": len(all_notices),
            "important_notices": len([n for n in all_notices if n.is_important]),
            "notices_by_type": {},
            "recent_notices_count": len([n for n in all_notices 
                                       if (datetime.now() - n.created_at).days <= 7])
        }
        
        # 타입별 통계
        for notice_type in NoticeType:
            stats["notices_by_type"][notice_type] = len([
                n for n in all_notices if n.notice_type == notice_type
            ])
        
        return stats

# 전역 서비스 인스턴스
admin_notice_service = AdminNoticeService() 