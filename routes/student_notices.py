from fastapi import APIRouter, HTTPException, status, Depends
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession # AsyncSession 임포트
from core.config import get_db # get_db 임포트
from services.notice_service import Notice, NoticeService # NoticeService 임포트
from services.student_notice_service import StudentNoticeService # StudentNoticeService 클래스 임포트
from routes.auth import get_current_user
from core.models import User # User 모델 임포트

router = APIRouter(prefix="/student/notices", tags=["학생용 공지사항"])

@router.get("/", summary="모든 공지사항 조회 (학생용)", response_model=List[Notice])
async def get_all_notices(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user) # User 객체로 변경
):
    """학생이 모든 공지사항을 최신순으로 조회합니다."""
    notice_service = NoticeService()
    student_notice_service_instance = StudentNoticeService()
    return await student_notice_service_instance.get_all_notices(db, notice_service)

@router.get("/{notice_id}", summary="특정 공지사항 조회 (학생용)", response_model=Notice)
async def get_notice(
    notice_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user) # User 객체로 변경
):
    """학생이 ID로 특정 공지사항을 조회합니다."""
    notice_service = NoticeService()
    student_notice_service_instance = StudentNoticeService()
    notice = await student_notice_service_instance.get_notice_by_id(db, notice_id, notice_service)
    if not notice:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="공지사항을 찾을 수 없습니다."
        )
    return notice

@router.get("/high-priority/", summary="높은 우선순위 공지사항 조회 (학생용)", response_model=List[Notice])
async def get_high_priority_notices(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user) # User 객체로 변경
):
    """학생이 높은 우선순위 공지사항만 조회합니다."""
    notice_service = NoticeService()
    student_notice_service_instance = StudentNoticeService()
    return await student_notice_service_instance.get_high_priority_notices(db, notice_service)


@router.get("/recent/", summary="최근 공지사항 조회 (학생용)", response_model=List[Notice])
async def get_recent_notices(
    limit: int = 5,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user) # User 객체로 변경
):
    """학생이 최근 공지사항을 조회합니다 (기본 5개)."""
    notice_service = NoticeService()
    student_notice_service_instance = StudentNoticeService()
    return await student_notice_service_instance.get_recent_notices(db, notice_service, limit)

@router.get("/search/", summary="공지사항 검색 (학생용)", response_model=List[Notice])
async def search_notices(
    keyword: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user) # User 객체로 변경
):
    """학생이 키워드로 공지사항을 검색합니다."""
    notice_service = NoticeService()
    student_notice_service_instance = StudentNoticeService()
    return await student_notice_service_instance.search_notices(db, keyword, notice_service)
