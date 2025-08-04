from typing import List, Optional, Dict
from datetime import datetime
import logging
from models.notice import Notice, NoticeCreate, NoticeUpdate, NoticeType

logger = logging.getLogger(__name__)

class NoticeService:
    def __init__(self):
        # 메모리 기반 저장소 (실제로는 데이터베이스 사용)
        self.notices: Dict[int, Notice] = {}
        self._counter = 1
        self._initialize_sample_data()
    
    def _initialize_sample_data(self):
        """샘플 공지사항 데이터 초기화"""
        sample_notices = [
            {
                "title": "[필수] 2024년 1학기 CPX 실습 일정 변경 안내",
                "content": "코로나 상황으로 인해 기존 대면 실습에서 온라인 실습으로 변경됩니다. 자세한 내용은 본문을 확인해주세요.",
                "notice_type": NoticeType.REQUIRED,
                "is_important": True,
                "created_at": datetime(2024, 1, 15)
            },
            {
                "title": "CPX 평가 기준 업데이트 안내",
                "content": "2024학년도부터 새로운 CPX 평가 기준에 대해 안내드립니다.",
                "notice_type": NoticeType.GENERAL,
                "is_important": False,
                "created_at": datetime(2024, 1, 10)
            },
            {
                "title": "시스템 점검 안내 (1월 20일 02:00~04:00)",
                "content": "정기 시스템 점검으로 인해 해당 시간 동안 서비스 이용이 제한됩니다.",
                "notice_type": NoticeType.MAINTENANCE,
                "is_important": False,
                "created_at": datetime(2024, 1, 8)
            },
            {
                "title": "신규 업데이트: 음성인식 기능 개선",
                "content": "CPX 실습 음성 인식 정확도가 크게 향상되었습니다. 더욱 자연스러운 대화가 가능합니다.",
                "notice_type": NoticeType.UPDATE,
                "is_important": False,
                "created_at": datetime(2024, 1, 5)
            }
        ]
        
        for notice_data in sample_notices:
            self.create_notice(NoticeCreate(**notice_data))
    
    def get_all_notices(self) -> List[Notice]:
        """모든 공지사항 조회"""
        notices = list(self.notices.values())
        # 최신순으로 정렬
        notices.sort(key=lambda x: x.created_at, reverse=True)
        return notices
    
    def get_notice_by_id(self, notice_id: int) -> Optional[Notice]:
        """ID로 공지사항 조회"""
        return self.notices.get(notice_id)
    
    def create_notice(self, notice_data: NoticeCreate) -> Notice:
        """새 공지사항 생성"""
        now = datetime.now()
        notice = Notice(
            id=self._counter,
            title=notice_data.title,
            content=notice_data.content,
            notice_type=notice_data.notice_type,
            is_important=notice_data.is_important,
            created_at=now,
            updated_at=now
        )
        
        self.notices[self._counter] = notice
        self._counter += 1
        
        logger.info(f"새 공지사항 생성: {notice.title}")
        return notice
    
    def update_notice(self, notice_id: int, notice_data: NoticeUpdate) -> Optional[Notice]:
        """공지사항 수정"""
        if notice_id not in self.notices:
            return None
        
        notice = self.notices[notice_id]
        
        if notice_data.title is not None:
            notice.title = notice_data.title
        if notice_data.content is not None:
            notice.content = notice_data.content
        if notice_data.notice_type is not None:
            notice.notice_type = notice_data.notice_type
        if notice_data.is_important is not None:
            notice.is_important = notice_data.is_important
        
        notice.updated_at = datetime.now()
        
        logger.info(f"공지사항 수정: {notice.title}")
        return notice
    
    def delete_notice(self, notice_id: int) -> bool:
        """공지사항 삭제"""
        if notice_id not in self.notices:
            return False
        
        notice = self.notices.pop(notice_id)
        logger.info(f"공지사항 삭제: {notice.title}")
        return True
    
    def get_important_notices(self) -> List[Notice]:
        """중요 공지사항만 조회"""
        important_notices = [notice for notice in self.notices.values() if notice.is_important]
        important_notices.sort(key=lambda x: x.created_at, reverse=True)
        return important_notices
    
    def get_notices_by_type(self, notice_type: NoticeType) -> List[Notice]:
        """타입별 공지사항 조회"""
        filtered_notices = [notice for notice in self.notices.values() if notice.notice_type == notice_type]
        filtered_notices.sort(key=lambda x: x.created_at, reverse=True)
        return filtered_notices

# 전역 서비스 인스턴스
notice_service = NoticeService() 