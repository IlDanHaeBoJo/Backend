from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from core.config import get_db
from core.constants import ADMIN_ROLES, S3_PRESIGNED_URL_EXPIRES_IN
from services.attachment_service import AttachmentService, AttachmentCreate
from routes.auth import get_current_user
from core.models import User
from services.s3_service import s3_service
from utils.exceptions import (
    NoticeNotFoundException,
    AttachmentNotFoundException,
    FileSizeExceededException,
    UnsupportedFileTypeException,
    S3UploadFailedException,
    DownloadUrlGenerationFailedException,
    FileNotExistsException,
    PermissionDeniedException
)

router = APIRouter(prefix="/attachments", tags=["첨부파일"])

# 서비스 인스턴스 생성
attachment_service = AttachmentService()

# 백엔드 프록시 방식 제거 - Presigned URL 방식만 사용

@router.post("/create/{notice_id}")
async def create_attachment(
    notice_id: int,
    attachment_data: AttachmentCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """첨부파일 생성 (프론트엔드에서 S3 업로드 후)"""
    
    # 권한 확인 (관리자 또는 교수만)
    if current_user.role not in ADMIN_ROLES:
        raise PermissionDeniedException("첨부파일 생성")
    
    try:
        attachment = await attachment_service.create_attachment(db, notice_id, attachment_data)
        return {
            "message": "첨부파일이 성공적으로 생성되었습니다.",
            "attachment_id": attachment.attachment_id,
            "original_filename": attachment.original_filename,
            "s3_url": attachment.file_path
        }
    except (NoticeNotFoundException, FileSizeExceededException, UnsupportedFileTypeException) as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="첨부파일 생성 중 오류가 발생했습니다.")

@router.post("/upload-url/{notice_id}")
async def generate_upload_url(
    notice_id: int,
    filename: str,
    file_type: str,
    file_size: int,
    method: str = "PUT",  # PUT 또는 POST 선택
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """첨부파일 업로드용 presigned URL 생성 (PUT/POST 방식 선택 가능)"""
    
    # 권한 확인 (관리자 또는 교수만)
    if current_user.role not in ADMIN_ROLES:
        raise PermissionDeniedException("첨부파일 업로드")
    
    try:
        # 파일 검증
        attachment_service._validate_file_size(file_size)
        attachment_service._validate_file_type(file_type)
        
        # 공지사항 존재 확인
        from services.notice_service import NoticeService
        notice_service = NoticeService()
        notice = await notice_service.get_notice_by_id(db, notice_id)
        if not notice:
            raise NoticeNotFoundException()
        
        # 고유한 S3 키 생성
        s3_key = s3_service.generate_unique_key(filename)
        logger.info(f"Presigned URL 생성 - Notice ID: {notice_id}, 파일명: {filename}, S3 키: {s3_key}")
        
        # 업로드 방식에 따라 URL 생성
        if method.upper() == "POST":
            # Presigned POST URL 생성
            presigned_post = s3_service.generate_presigned_post(s3_key, file_type)
            if not presigned_post:
                raise S3UploadFailedException("Presigned POST URL 생성에 실패했습니다.")
            
            logger.info(f"Presigned POST URL 생성 완료 - S3 키: {s3_key}, 만료시간: {S3_PRESIGNED_URL_EXPIRES_IN}초")
            
            return {
                "notice_id": notice_id,
                "original_filename": filename,
                "stored_filename": s3_key,
                "upload_method": "POST",
                "upload_url": presigned_post['url'],
                "upload_fields": presigned_post['fields'],
                "file_type": file_type,
                "file_size": file_size,
                "expires_in": S3_PRESIGNED_URL_EXPIRES_IN,
                "s3_url": s3_service.get_file_url(s3_key),
                "message": "Presigned POST URL이 생성되었습니다. multipart/form-data로 파일을 업로드하세요."
            }
        else:
            # Presigned PUT URL 생성 (기본값)
            upload_url = s3_service.get_upload_url_with_cors(s3_key, file_type)
            if not upload_url:
                raise S3UploadFailedException("업로드 URL 생성에 실패했습니다.")
            
            logger.info(f"Presigned PUT URL 생성 완료 - S3 키: {s3_key}, 만료시간: {S3_PRESIGNED_URL_EXPIRES_IN}초")
            
            return {
                "notice_id": notice_id,
                "original_filename": filename,
                "stored_filename": s3_key,
                "upload_method": "PUT",
                "upload_url": upload_url,
                "file_type": file_type,
                "file_size": file_size,
                "expires_in": S3_PRESIGNED_URL_EXPIRES_IN,
                "s3_url": s3_service.get_file_url(s3_key),
                "message": "업로드 URL이 생성되었습니다. 이 URL로 PUT 요청을 보내 파일을 업로드하세요."
            }
        
    except (NoticeNotFoundException, FileSizeExceededException, UnsupportedFileTypeException, S3UploadFailedException) as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="업로드 URL 생성 중 오류가 발생했습니다.")

@router.post("/upload-complete/{notice_id}")
async def upload_complete(
    notice_id: int,
    upload_data: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """S3 업로드 완료 알림 및 DB 저장"""
    
    # 권한 확인 (관리자 또는 교수만)
    if current_user.role not in ADMIN_ROLES:
        raise PermissionDeniedException("첨부파일 업로드 완료 처리")
    
    try:
        # S3 파일 존재 확인 (HeadObject)
        s3_key = s3_service._extract_s3_key_from_url(upload_data["s3_url"])
        logger.info(f"업로드 완료 처리 시작 - Notice ID: {notice_id}, 파일명: {upload_data['original_filename']}, S3 키: {s3_key}")
        
        if not s3_service.file_exists(s3_key):
            logger.error(f"S3 파일 존재 확인 실패 - S3 키: {s3_key}")
            raise HTTPException(status_code=404, detail="S3에서 업로드된 파일을 찾을 수 없습니다.")
        
        logger.info(f"S3 파일 존재 확인 성공 - S3 키: {s3_key}")
        
        # ETag 확인 (선택사항)
        if "etag" in upload_data:
            logger.info(f"업로드 완료 - ETag: {upload_data['etag']}")
        
        # 첨부파일 정보 생성
        attachment = await attachment_service.create_attachment(
            db, 
            notice_id, 
            upload_data["original_filename"],
            upload_data["s3_url"],
            upload_data["file_size"],
            upload_data["file_type"]
        )
        
        logger.info(f"첨부파일 DB 저장 완료 - Attachment ID: {attachment.attachment_id}, S3 키: {s3_key}")
        
        return {
            "message": "업로드가 완료되었고 첨부파일이 성공적으로 저장되었습니다.",
            "attachment_id": attachment.attachment_id,
            "original_filename": attachment.original_filename,
            "s3_url": attachment.file_path,
            "s3_key": s3_key,
            "verified": True
        }
        
    except (NoticeNotFoundException, FileSizeExceededException, UnsupportedFileTypeException) as e:
        raise e
    except Exception as e:
        logger.error(f"업로드 완료 처리 중 오류: {e}")
        raise HTTPException(status_code=500, detail="업로드 완료 처리 중 오류가 발생했습니다.")

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
                    "s3_url": att.file_path,
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
            raise AttachmentNotFoundException()
        
        # S3에서 presigned URL 생성
        download_url = attachment_service.get_s3_download_url(attachment)
        if not download_url:
            raise DownloadUrlGenerationFailedException()
        
        if not attachment_service.is_s3_file_exists(attachment):
            raise FileNotExistsException()
        
        return {
            "attachment_id": attachment.attachment_id,
            "original_filename": attachment.original_filename,
            "download_url": download_url,
            "file_size": attachment.file_size,
            "file_type": attachment.file_type
        }
    except (AttachmentNotFoundException, DownloadUrlGenerationFailedException, FileNotExistsException) as e:
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
    if current_user.role not in ADMIN_ROLES:
        raise PermissionDeniedException("첨부파일 삭제")
    
    try:
        await attachment_service.delete_attachment(db, attachment_id)
        return {"message": "첨부파일이 성공적으로 삭제되었습니다."}
    except AttachmentNotFoundException as e:
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
    if current_user.role not in ADMIN_ROLES:
        raise PermissionDeniedException("첨부파일 삭제")
    
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
            raise AttachmentNotFoundException()
        
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
    except AttachmentNotFoundException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="첨부파일 정보 조회 중 오류가 발생했습니다.")
