from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from core.config import get_db
from core.models import Attachments
from services.attachment_service import attachment_service, AttachmentCreate
from routes.auth import get_current_user
from core.models import User

router = APIRouter(prefix="/attachments", tags=["첨부파일"])

@router.post("/upload/{notice_id}")
async def upload_attachment(
    notice_id: int,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """파일을 S3에 업로드하고 첨부파일 생성"""
    
    # 권한 확인 (관리자 또는 교수만)
    if current_user.role not in ["admin", "교수"]:
        raise HTTPException(status_code=403, detail="첨부파일 업로드 권한이 없습니다.")
    
    try:
        attachment = await attachment_service.upload_attachment(db, notice_id, file)
        return {
            "message": "첨부파일이 성공적으로 업로드되었습니다.",
            "attachment": {
                "attachment_id": attachment.attachment_id,
                "original_filename": attachment.original_filename,
                "s3_url": attachment.file_path,
                "file_size": attachment.file_size,
                "file_type": attachment.file_type,
                "uploaded_at": attachment.uploaded_at
            }
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="첨부파일 업로드 중 오류가 발생했습니다.")

@router.post("/create/{notice_id}")
async def create_attachment(
    notice_id: int,
    attachment_data: AttachmentCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """S3에서 업로드된 파일 정보로 첨부파일 생성"""
    
    # 권한 확인 (관리자 또는 교수만)
    if current_user.role not in ["admin", "교수"]:
        raise HTTPException(status_code=403, detail="첨부파일 생성 권한이 없습니다.")
    
    try:
        attachment = await attachment_service.create_attachment(db, notice_id, attachment_data)
        return {
            "message": "첨부파일이 성공적으로 생성되었습니다.",
            "attachment": {
                "attachment_id": attachment.attachment_id,
                "original_filename": attachment.original_filename,
                "s3_url": attachment.file_path,
                "file_size": attachment.file_size,
                "file_type": attachment.file_type,
                "uploaded_at": attachment.uploaded_at
            }
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="첨부파일 생성 중 오류가 발생했습니다.")

@router.get("/notice/{notice_id}")
async def get_attachments_by_notice(
    notice_id: int,
    db: AsyncSession = Depends(get_db)
):
    """공지사항의 첨부파일 목록 조회"""
    
    try:
        attachments = await attachment_service.get_attachments_by_notice(db, notice_id)
        return {
            "notice_id": notice_id,
            "attachments": [
                {
                    "attachment_id": att.attachment_id,
                    "original_filename": att.original_filename,
                    "file_size": att.file_size,
                    "file_type": att.file_type,
                    "uploaded_at": att.uploaded_at
                }
                for att in attachments
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="첨부파일 목록 조회 중 오류가 발생했습니다.")

@router.get("/download/{attachment_id}")
async def get_attachment_download_url(
    attachment_id: int,
    db: AsyncSession = Depends(get_db)
):
    """첨부파일 다운로드 URL 반환"""
    
    try:
        attachment = await attachment_service.get_attachment(db, attachment_id)
        if not attachment:
            raise HTTPException(status_code=404, detail="첨부파일을 찾을 수 없습니다.")
        
        # S3에서 presigned URL 생성
        download_url = attachment_service.get_s3_download_url(attachment)
        if not download_url:
            raise HTTPException(status_code=500, detail="다운로드 URL 생성에 실패했습니다.")
        
        if not attachment_service.is_s3_file_exists(attachment):
            raise HTTPException(status_code=404, detail="파일이 존재하지 않습니다.")
        
        return {
            "attachment_id": attachment.attachment_id,
            "original_filename": attachment.original_filename,
            "download_url": download_url,
            "file_size": attachment.file_size,
            "file_type": attachment.file_type
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="파일 정보 조회 중 오류가 발생했습니다.")

@router.delete("/{attachment_id}")
async def delete_attachment(
    attachment_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """첨부파일 삭제"""
    
    # 권한 확인 (관리자 또는 교수만)
    if current_user.role not in ["admin", "교수"]:
        raise HTTPException(status_code=403, detail="첨부파일 삭제 권한이 없습니다.")
    
    try:
        await attachment_service.delete_attachment(db, attachment_id)
        return {"message": "첨부파일이 성공적으로 삭제되었습니다."}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="첨부파일 삭제 중 오류가 발생했습니다.")

@router.delete("/notice/{notice_id}/all")
async def delete_all_attachments_by_notice(
    notice_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """공지사항의 모든 첨부파일 삭제"""
    
    # 권한 확인 (관리자 또는 교수만)
    if current_user.role not in ["admin", "교수"]:
        raise HTTPException(status_code=403, detail="첨부파일 삭제 권한이 없습니다.")
    
    try:
        await attachment_service.delete_attachments_by_notice(db, notice_id)
        return {"message": "모든 첨부파일이 성공적으로 삭제되었습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail="첨부파일 삭제 중 오류가 발생했습니다.")

@router.get("/{attachment_id}/info")
async def get_attachment_info(
    attachment_id: int,
    db: AsyncSession = Depends(get_db)
):
    """첨부파일 정보 조회"""
    
    try:
        attachment = await attachment_service.get_attachment(db, attachment_id)
        if not attachment:
            raise HTTPException(status_code=404, detail="첨부파일을 찾을 수 없습니다.")
        
        return {
            "attachment_id": attachment.attachment_id,
            "notice_id": attachment.notice_id,
            "original_filename": attachment.original_filename,
            "stored_filename": attachment.stored_filename,
            "s3_url": attachment.file_path,
            "file_size": attachment.file_size,
            "file_type": attachment.file_type,
            "uploaded_at": attachment.uploaded_at,
            "file_exists": attachment_service.is_s3_file_exists(attachment)
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="첨부파일 정보 조회 중 오류가 발생했습니다.")
