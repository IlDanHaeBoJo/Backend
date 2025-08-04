from fastapi import APIRouter, HTTPException, status, Depends
from typing import List
from models.notice import Notice, NoticeType
from services.student_notice_service import student_notice_service
from routes.auth import get_current_user

router = APIRouter(prefix="/student/notices", tags=["학생용 공지사항"])

@router.get("/", summary="모든 공지사항 조회 (학생용)", response_model=List[Notice])
async def get_all_notices():
    """학생이 모든 공지사항을 최신순으로 조회합니다."""
    return student_notice_service.get_all_notices()

@router.get("/{notice_id}", summary="특정 공지사항 조회 (학생용)", response_model=Notice)
async def get_notice(notice_id: int):
    """학생이 ID로 특정 공지사항을 조회합니다."""
    notice = student_notice_service.get_notice_by_id(notice_id)
    if not notice:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="공지사항을 찾을 수 없습니다."
        )
    return notice

@router.get("/important/", summary="중요 공지사항 조회 (학생용)", response_model=List[Notice])
async def get_important_notices():
    """학생이 중요한 공지사항만 조회합니다."""
    return student_notice_service.get_important_notices()

@router.get("/type/{notice_type}", summary="타입별 공지사항 조회 (학생용)", response_model=List[Notice])
async def get_notices_by_type(notice_type: NoticeType):
    """학생이 특정 타입의 공지사항만 조회합니다."""
    return student_notice_service.get_notices_by_type(notice_type)

@router.get("/recent/", summary="최근 공지사항 조회 (학생용)", response_model=List[Notice])
async def get_recent_notices(limit: int = 5):
    """학생이 최근 공지사항을 조회합니다 (기본 5개)."""
    return student_notice_service.get_recent_notices(limit)

@router.get("/search/", summary="공지사항 검색 (학생용)", response_model=List[Notice])
async def search_notices(keyword: str):
    """학생이 키워드로 공지사항을 검색합니다."""
    return student_notice_service.search_notices(keyword) 