from fastapi import APIRouter, HTTPException, status, Depends, Query
from typing import Optional
from pydantic import BaseModel, Field
from routes.auth import get_current_user
from utils.permissions import require_role
from utils.s3_utils import s3_uploader
from core.models import User

router = APIRouter(prefix="/s3", tags=["S3 파일 업로드"])

class S3UploadRequest(BaseModel):
    """S3 업로드 요청 모델"""
    filename: str = Field(..., description="원본 파일명")
    file_type: str = Field(..., description="파일 MIME 타입")
    notice_id: Optional[int] = Field(None, description="공지사항 ID (선택사항)")

class S3UploadResponse(BaseModel):
    """S3 업로드 응답 모델"""
    upload_url: str = Field(..., description="S3 업로드용 presigned URL")
    file_path: str = Field(..., description="S3 파일 경로")
    stored_filename: str = Field(..., description="저장될 파일명")

@router.post("/upload-url", summary="S3 업로드용 presigned URL 생성", response_model=S3UploadResponse)
@require_role("admin")
async def generate_upload_url(
    request: S3UploadRequest,
    current_user: User = Depends(get_current_user)
):
    """파일 업로드를 위한 S3 presigned URL을 생성합니다."""
    try:
        # 고유한 파일명 생성
        stored_filename = s3_uploader.generate_unique_filename(request.filename)
        
        # 파일 경로 생성
        notice_id = request.notice_id or 0  # notice_id가 없으면 0 사용
        file_path = s3_uploader.get_file_path(notice_id, stored_filename)
        
        # S3 업로드용 presigned URL 생성
        upload_url = s3_uploader.generate_upload_url(file_path, request.file_type)
        
        if not upload_url:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="S3 업로드 URL 생성에 실패했습니다."
            )
        
        return S3UploadResponse(
            upload_url=upload_url,
            file_path=file_path,
            stored_filename=stored_filename
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"S3 업로드 URL 생성 중 오류가 발생했습니다: {str(e)}"
        )

@router.get("/download-url/{file_path:path}", summary="파일 다운로드용 presigned URL 생성")
async def generate_download_url(
    file_path: str,
    expiration: int = Query(3600, description="URL 만료 시간 (초)", ge=60, le=86400),
    current_user: User = Depends(get_current_user)
):
    """파일 다운로드를 위한 S3 presigned URL을 생성합니다."""
    try:
        download_url = s3_uploader.generate_presigned_url(file_path, expiration)
        
        if not download_url:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="파일을 찾을 수 없습니다."
            )
        
        return {"download_url": download_url, "expires_in": expiration}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"다운로드 URL 생성 중 오류가 발생했습니다: {str(e)}"
        ) 