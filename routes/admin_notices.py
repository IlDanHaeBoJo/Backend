from fastapi import APIRouter, HTTPException, status, Depends, Query
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession # AsyncSession 임포트
from core.config import get_db # get_db 임포트
from services.notice_service import Notice, NoticeCreate, NoticeUpdate, NoticeStats, NoticeService # NoticeService 임포트
from services.admin_notice_service import ( # AdminNoticeService 함수들 임포트
    get_all_notices, get_notice_by_id, create_notice, update_notice, delete_notice,
<<<<<<< HEAD
    get_high_priority_notices, update_notice_priority,
=======
    get_important_notices, toggle_notice_important,
>>>>>>> upstream/main
    get_notice_statistics, search_notices
)
from routes.auth import get_current_user
from utils.permissions import require_role
from core.models import User # User 모델 임포트

router = APIRouter(prefix="/admin/notices", tags=["관리자용 공지사항"])

@router.get("/", summary="모든 공지사항 조회 (관리자용)", response_model=List[Notice])
@require_role("admin")
async def get_all_notices_admin(
    db: AsyncSession = Depends(get_db),
<<<<<<< HEAD
    current_user: User = Depends(get_current_user) # User 객체로 변경
):
    """관리자가 모든 공지사항을 최신순으로 조회합니다."""
    notice_service = NoticeService()
=======
    notice_service: NoticeService = Depends(NoticeService),
    current_user: User = Depends(get_current_user) # User 객체로 변경
):
    """관리자가 모든 공지사항을 최신순으로 조회합니다."""
>>>>>>> upstream/main
    return await get_all_notices(db, notice_service)

@router.get("/{notice_id}", summary="특정 공지사항 조회 (관리자용)", response_model=Notice)
@require_role("admin")
async def get_notice_admin(
    notice_id: int,
    db: AsyncSession = Depends(get_db),
<<<<<<< HEAD
    current_user: User = Depends(get_current_user) # User 객체로 변경
):
    """관리자가 ID로 특정 공지사항을 조회합니다."""
    notice_service = NoticeService()
=======
    notice_service: NoticeService = Depends(NoticeService),
    current_user: User = Depends(get_current_user) # User 객체로 변경
):
    """관리자가 ID로 특정 공지사항을 조회합니다."""
>>>>>>> upstream/main
    notice = await get_notice_by_id(db, notice_id, notice_service)
    if not notice:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="공지사항을 찾을 수 없습니다."
        )
    return notice

@router.post("/", summary="새 공지사항 생성 (관리자용)", response_model=Notice)
@require_role("admin")
async def create_notice_admin(
    notice_data: NoticeCreate,
    db: AsyncSession = Depends(get_db),
<<<<<<< HEAD
=======
    notice_service: NoticeService = Depends(NoticeService),
>>>>>>> upstream/main
    current_user: User = Depends(get_current_user) # User 객체로 변경
):
    """관리자가 새 공지사항을 생성합니다."""
    # notice_data의 author_id를 현재 사용자의 ID로 설정
    notice_data.author_id = current_user.id
    
<<<<<<< HEAD
    notice_service = NoticeService()
=======
>>>>>>> upstream/main
    return await create_notice(db, notice_data, notice_service)

@router.put("/{notice_id}", summary="공지사항 수정 (관리자용)", response_model=Notice)
@require_role("admin")
async def update_notice_admin(
    notice_id: int,
    notice_data: NoticeUpdate,
    db: AsyncSession = Depends(get_db),
<<<<<<< HEAD
    current_user: User = Depends(get_current_user) # User 객체로 변경
):
    """관리자가 공지사항을 수정합니다."""
    notice_service = NoticeService()
=======
    notice_service: NoticeService = Depends(NoticeService),
    current_user: User = Depends(get_current_user) # User 객체로 변경
):
    """관리자가 공지사항을 수정합니다."""
>>>>>>> upstream/main
    notice = await update_notice(db, notice_id, notice_data, notice_service)
    if not notice:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="공지사항을 찾을 수 없습니다."
        )
    return notice

@router.delete("/{notice_id}", summary="공지사항 삭제 (관리자용)")
@require_role("admin")
async def delete_notice_admin(
    notice_id: int,
    db: AsyncSession = Depends(get_db),
<<<<<<< HEAD
    current_user: User = Depends(get_current_user) # User 객체로 변경
):
    """관리자가 공지사항을 삭제합니다."""
    notice_service = NoticeService()
=======
    notice_service: NoticeService = Depends(NoticeService),
    current_user: User = Depends(get_current_user) # User 객체로 변경
):
    """관리자가 공지사항을 삭제합니다."""
>>>>>>> upstream/main
    success = await delete_notice(db, notice_id, notice_service)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="공지사항을 찾을 수 없습니다."
        )
    return {"message": "공지사항이 삭제되었습니다."}

<<<<<<< HEAD
@router.get("/high-priority/", summary="높은 우선순위 공지사항 조회 (관리자용)", response_model=List[Notice])
@require_role("admin")
async def get_high_priority_notices_admin(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user) # User 객체로 변경
):
    """관리자가 높은 우선순위 공지사항만 조회합니다."""
    notice_service = NoticeService()
    return await get_high_priority_notices(db, notice_service)

@router.put("/{notice_id}/priority", summary="공지사항 우선순위 업데이트 (관리자용)", response_model=Notice)
@require_role("admin")
async def update_notice_priority_admin(
    notice_id: int,
    priority: int = Query(..., ge=0, description="새로운 우선순위 값 (0 이상)"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user) # User 객체로 변경
):
    """관리자가 공지사항의 우선순위를 업데이트합니다."""
    notice_service = NoticeService()
    updated_notice = await update_notice_priority(db, notice_id, priority, notice_service)
=======
@router.get("/important/", summary="중요 공지사항 조회 (관리자용)", response_model=List[Notice])
@require_role("admin")
async def get_important_notices_admin(
    db: AsyncSession = Depends(get_db),
    notice_service: NoticeService = Depends(NoticeService),
    current_user: User = Depends(get_current_user) # User 객체로 변경
):
    """관리자가 높은 우선순위 공지사항만 조회합니다."""
    return await get_important_notices(db, notice_service)

@router.post("/{notice_id}/toggle-important", summary="공지사항 중요 여부 토글 (관리자용)", response_model=Notice)
@require_role("admin")
async def toggle_notice_important_admin(
    notice_id: int,
    db: AsyncSession = Depends(get_db),
    notice_service: NoticeService = Depends(NoticeService),
    current_user: User = Depends(get_current_user) # User 객체로 변경
):
    """관리자가 공지사항의 중요 여부를 토글합니다."""
    updated_notice = await toggle_notice_important(db, notice_id, notice_service)
>>>>>>> upstream/main
    if not updated_notice:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="공지사항을 찾을 수 없습니다."
        )
    return updated_notice

@router.get("/stats/summary", summary="공지사항 통계 요약 (관리자용)", response_model=NoticeStats)
@require_role("admin")
async def get_notice_statistics_summary(
    db: AsyncSession = Depends(get_db),
<<<<<<< HEAD
    current_user: User = Depends(get_current_user) # User 객체로 변경
):
    """관리자가 공지사항 통계 요약을 조회합니다."""
    notice_service = NoticeService()
=======
    notice_service: NoticeService = Depends(NoticeService),
    current_user: User = Depends(get_current_user) # User 객체로 변경
):
    """관리자가 공지사항 통계 요약을 조회합니다."""
>>>>>>> upstream/main
    return await get_notice_statistics(db, notice_service)

@router.get("/search/", summary="공지사항 검색 (관리자용)", response_model=List[Notice])
@require_role("admin")
async def search_notices_admin(
    keyword: str = Query(..., description="검색 키워드"),
    search_type: str = Query("all", description="검색 타입 (all, title, content)"),
    db: AsyncSession = Depends(get_db),
<<<<<<< HEAD
    current_user: User = Depends(get_current_user) # User 객체로 변경
):
    """관리자가 키워드로 공지사항을 검색합니다."""
    notice_service = NoticeService()
=======
    notice_service: NoticeService = Depends(NoticeService),
    current_user: User = Depends(get_current_user) # User 객체로 변경
):
    """관리자가 키워드로 공지사항을 검색합니다."""
>>>>>>> upstream/main
    return await search_notices(db, keyword, notice_service, search_type)
