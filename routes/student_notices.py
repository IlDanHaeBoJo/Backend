from fastapi import APIRouter, HTTPException, status, Depends
from typing import List
from services.notice_service import Notice
from services.student_notice_service import student_notice_service
from routes.auth import get_current_user

router = APIRouter(prefix="/student/notices", tags=["학생용 공지사항"])

@router.get("/", summary="모든 공지사항 조회 (학생용)", response_model=List[Notice])
async def get_all_notices(current_user: str = Depends(get_current_user)):
    """학생이 모든 공지사항을 최신순으로 조회합니다."""
    return student_notice_service.get_all_notices()

@router.get("/{notice_id}", summary="특정 공지사항 조회 (학생용)", response_model=Notice)
async def get_notice(notice_id: int, current_user: str = Depends(get_current_user)):
    """학생이 ID로 특정 공지사항을 조회합니다."""
    notice = student_notice_service.get_notice_by_id(notice_id)
    if not notice:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="공지사항을 찾을 수 없습니다."
        )
    return notice

@router.get("/priority/", summary="높은 우선순위 공지사항 조회 (학생용)", response_model=List[Notice])
async def get_priority_notices(current_user: str = Depends(get_current_user)):
    """학생이 높은 우선순위 공지사항만 조회합니다."""
    return student_notice_service.get_important_notices()

@router.get("/priority/{min_priority}", summary="우선순위별 공지사항 조회 (학생용)", response_model=List[Notice])
async def get_notices_by_priority(min_priority: int = 0, current_user: str = Depends(get_current_user)):
    """학생이 특정 우선순위 이상의 공지사항을 조회합니다."""
    return student_notice_service.get_notices_by_priority(min_priority)

@router.get("/recent/", summary="최근 공지사항 조회 (학생용)", response_model=List[Notice])
async def get_recent_notices(limit: int = 5, current_user: str = Depends(get_current_user)):
    """학생이 최근 공지사항을 조회합니다 (기본 5개)."""
    return student_notice_service.get_recent_notices(limit)

@router.get("/search/", summary="공지사항 검색 (학생용)", response_model=List[Notice])
async def search_notices(keyword: str, current_user: str = Depends(get_current_user)):
    """학생이 키워드로 공지사항을 검색합니다."""
    return student_notice_service.search_notices(keyword)
