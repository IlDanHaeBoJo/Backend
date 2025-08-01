from fastapi import APIRouter, HTTPException, status, Depends
from typing import List
from models.notice import Notice, NoticeCreate, NoticeUpdate, NoticeType
from services.admin_notice_service import admin_notice_service
from routes.auth import get_current_user
from utils.permissions import require_role

router = APIRouter(prefix="/admin/notices", tags=["관리자용 공지사항"])

@router.get("/", summary="모든 공지사항 조회 (관리자용)", response_model=List[Notice])
@require_role("admin")
async def get_all_notices_admin(current_user: str = Depends(get_current_user)):
    """관리자가 모든 공지사항을 최신순으로 조회합니다."""
    return admin_notice_service.get_all_notices()

@router.get("/{notice_id}", summary="특정 공지사항 조회 (관리자용)", response_model=Notice)
@require_role("admin")
async def get_notice_admin(
    notice_id: int,
    current_user: str = Depends(get_current_user)
):
    """관리자가 ID로 특정 공지사항을 조회합니다."""
    notice = admin_notice_service.get_notice_by_id(notice_id)
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
    current_user: str = Depends(get_current_user)
):
    """관리자가 새 공지사항을 생성합니다."""
    return admin_notice_service.create_notice(notice_data)

@router.put("/{notice_id}", summary="공지사항 수정 (관리자용)", response_model=Notice)
@require_role("admin")
async def update_notice_admin(
    notice_id: int,
    notice_data: NoticeUpdate,
    current_user: str = Depends(get_current_user)
):
    """관리자가 공지사항을 수정합니다."""
    notice = admin_notice_service.update_notice(notice_id, notice_data)
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
    current_user: str = Depends(get_current_user)
):
    """관리자가 공지사항을 삭제합니다."""
    success = admin_notice_service.delete_notice(notice_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="공지사항을 찾을 수 없습니다."
        )
    return {"message": "공지사항이 삭제되었습니다."}

@router.get("/important/", summary="중요 공지사항 조회 (관리자용)", response_model=List[Notice])
@require_role("admin")
async def get_important_notices_admin(current_user: str = Depends(get_current_user)):
    """관리자가 중요한 공지사항만 조회합니다."""
    return admin_notice_service.get_important_notices()

@router.get("/type/{notice_type}", summary="타입별 공지사항 조회 (관리자용)", response_model=List[Notice])
@require_role("admin")
async def get_notices_by_type_admin(
    notice_type: NoticeType,
    current_user: str = Depends(get_current_user)
):
    """관리자가 특정 타입의 공지사항만 조회합니다."""
    return admin_notice_service.get_notices_by_type(notice_type)

@router.post("/{notice_id}/toggle-important", summary="공지사항 중요도 토글 (관리자용)", response_model=Notice)
@require_role("admin")
async def toggle_notice_importance(
    notice_id: int,
    current_user: str = Depends(get_current_user)
):
    """관리자가 공지사항의 중요도를 토글합니다."""
    updated_notice = admin_notice_service.toggle_notice_importance(notice_id)
    if not updated_notice:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="공지사항을 찾을 수 없습니다."
        )
    return updated_notice

@router.get("/statistics/", summary="공지사항 통계 조회 (관리자용)")
@require_role("admin")
async def get_notice_statistics(current_user: str = Depends(get_current_user)):
    """관리자가 공지사항 통계를 조회합니다."""
    return admin_notice_service.get_notice_statistics() 