from typing import List, Optional, Dict
from datetime import datetime
import logging
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# 임시 모델 정의 (실제로는 models/notice.py에 있어야 함)
class NoticeBase(BaseModel):
    """공지사항 기본 모델"""
    title: str = Field(..., max_length=255, description="공지사항 제목")
    content: str = Field(..., description="공지사항 내용")
    priority: int = Field(default=0, ge=0, le=100, description="공지사항 중요도 (0-100)")
    author_id: int = Field(..., description="작성자 ID")

class NoticeCreate(NoticeBase):
    """공지사항 생성 모델"""
    pass

class NoticeUpdate(BaseModel):
    """공지사항 수정 모델"""
    title: Optional[str] = Field(None, max_length=255, description="공지사항 제목")
    content: Optional[str] = Field(None, description="공지사항 내용")
    priority: Optional[int] = Field(None, ge=0, le=100, description="공지사항 중요도 (0-100)")

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
    average_priority: float = Field(..., description="평균 우선순위")
    high_priority_count: int = Field(..., description="높은 우선순위 공지사항 수")
    recent_notices_count: int = Field(..., description="최근 7일 공지사항 수")

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
                "priority": 90,
                "author_id": 1,
                "view_count": 150,
                "created_at": datetime(2024, 1, 15),
                "updated_at": datetime(2024, 1, 15)
            },
            {
                "title": "CPX 평가 기준 업데이트 안내",
                "content": "2024학년도부터 새로운 CPX 평가 기준에 대해 안내드립니다.",
                "priority": 50,
                "author_id": 1,
                "view_count": 89,
                "created_at": datetime(2024, 1, 10),
                "updated_at": datetime(2024, 1, 10)
            },
            {
                "title": "시스템 점검 안내 (1월 20일 02:00~04:00)",
                "content": "정기 시스템 점검으로 인해 해당 시간 동안 서비스 이용이 제한됩니다.",
                "priority": 30,
                "author_id": 2,
                "view_count": 45,
                "created_at": datetime(2024, 1, 8),
                "updated_at": datetime(2024, 1, 8)
            },
            {
                "title": "신규 업데이트: 음성인식 기능 개선",
                "content": "CPX 실습 음성 인식 정확도가 크게 향상되었습니다. 더욱 자연스러운 대화가 가능합니다.",
                "priority": 20,
                "author_id": 1,
                "view_count": 120,
                "created_at": datetime(2024, 1, 5),
                "updated_at": datetime(2024, 1, 5)
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
            notice_id=self._counter,
            title=notice_data.title,
            content=notice_data.content,
            priority=notice_data.priority,
            author_id=notice_data.author_id,
            view_count=0,
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
        if notice_data.priority is not None:
            notice.priority = notice_data.priority
        
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
        """높은 우선순위 공지사항만 조회 (priority >= 50)"""
        important_notices = [notice for notice in self.notices.values() if notice.priority >= 50]
        important_notices.sort(key=lambda x: x.priority, reverse=True)
        return important_notices
    
    def get_notices_by_priority(self, min_priority: int = 0) -> List[Notice]:
        """우선순위별 공지사항 조회"""
        filtered_notices = [notice for notice in self.notices.values() if notice.priority >= min_priority]
        filtered_notices.sort(key=lambda x: x.priority, reverse=True)
        return filtered_notices
    
    def increment_view_count(self, notice_id: int) -> bool:
        """공지사항 조회수 증가"""
        if notice_id not in self.notices:
            return False
        
        self.notices[notice_id].view_count += 1
        return True
    
    def get_notice_statistics(self) -> NoticeStats:
        """공지사항 통계 조회"""
        notices = list(self.notices.values())
        if not notices:
            return NoticeStats(
                total_notices=0,
                total_views=0,
                average_priority=0.0,
                high_priority_count=0,
                recent_notices_count=0
            )
        
        total_notices = len(notices)
        total_views = sum(notice.view_count for notice in notices)
        average_priority = sum(notice.priority for notice in notices) / total_notices
        high_priority_count = len([n for n in notices if n.priority >= 50])
        
        # 최근 7일 공지사항 수
        from datetime import timedelta
        week_ago = datetime.now() - timedelta(days=7)
        recent_notices_count = len([n for n in notices if n.created_at >= week_ago])
        
        return NoticeStats(
            total_notices=total_notices,
            total_views=total_views,
            average_priority=round(average_priority, 2),
            high_priority_count=high_priority_count,
            recent_notices_count=recent_notices_count
        )
    
    def search_notices(self, keyword: str) -> List[Notice]:
        """키워드로 공지사항 검색"""
        keyword_lower = keyword.lower()
        search_results = []
        
        for notice in self.notices.values():
            if (keyword_lower in notice.title.lower() or 
                keyword_lower in notice.content.lower()):
                search_results.append(notice)
        
        search_results.sort(key=lambda x: x.priority, reverse=True)
        return search_results

# 전역 서비스 인스턴스
notice_service = NoticeService() 