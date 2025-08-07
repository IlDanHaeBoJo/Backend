from fastapi import APIRouter, HTTPException, status, Depends, Query
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from core.config import get_db
from services.attachment_service import Attachment, AttachmentCreate, AttachmentUpdate, AttachmentService
from routes.auth import get_current_user
from utils.permissions import require_role
from core.models import User

router = APIRouter(prefix="/attachments", tags=["첨부파일 관리"])

@router.get("/notice/{notice_id}", summary="공지사항의 첨부파일 목록 조회", response_model=List[Attachment])
async def get_attachments_by_notice_id(
    notice_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """특정 공지사항의 모든 첨부파일을 조회합니다."""
    attachment_service = AttachmentService()
    return await attachment_service.get_attachments_by_notice_id(db, notice_id)

@router.get("/{attachment_id}", summary="특정 첨부파일 조회", response_model=Attachment)
async def get_attachment(
    attachment_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """ID로 특정 첨부파일을 조회합니다."""
    attachment_service = AttachmentService()
    attachment = await attachment_service.get_attachment_by_id(db, attachment_id)
    if not attachment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="첨부파일을 찾을 수 없습니다."
        )
    return attachment

@router.post("/", summary="첨부파일 정보 생성", response_model=Attachment)
@require_role("admin")
async def create_attachment(
    attachment_data: AttachmentCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """새 첨부파일 정보를 생성합니다."""
    attachment_service = AttachmentService()
    return await attachment_service.create_attachment(db, attachment_data)

@router.put("/{attachment_id}", summary="첨부파일 정보 수정", response_model=Attachment)
@require_role("admin")
async def update_attachment(
    attachment_id: int,
    attachment_data: AttachmentUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """첨부파일 정보를 수정합니다."""
    attachment_service = AttachmentService()
    attachment = await attachment_service.update_attachment(db, attachment_id, attachment_data)
    if not attachment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="첨부파일을 찾을 수 없습니다."
        )
    return attachment

@router.delete("/{attachment_id}", summary="첨부파일 정보 삭제")
@require_role("admin")
async def delete_attachment(
    attachment_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """첨부파일 정보를 삭제합니다."""
    attachment_service = AttachmentService()
    success = await attachment_service.delete_attachment(db, attachment_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="첨부파일을 찾을 수 없습니다."
        )
    return {"message": "첨부파일 정보가 삭제되었습니다."}

@router.delete("/notice/{notice_id}/all", summary="공지사항의 모든 첨부파일 정보 삭제")
@require_role("admin")
async def delete_all_attachments_by_notice_id(
    notice_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """특정 공지사항의 모든 첨부파일 정보를 삭제합니다."""
    attachment_service = AttachmentService()
    success = await attachment_service.delete_attachments_by_notice_id(db, notice_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="공지사항을 찾을 수 없습니다."
        )
    return {"message": "모든 첨부파일 정보가 삭제되었습니다."} 