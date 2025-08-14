from pydantic import BaseModel, Field
from typing import Optional

class PresignedUrlRequest(BaseModel):
    """Presigned URL 생성 요청 모델"""
    filename: str = Field(..., description="원본 파일명")
    file_type: str = Field(..., alias="content_type", description="파일 MIME 타입")
    file_size: int = Field(..., alias="content_length", description="파일 크기 (bytes)")
    method: str = Field(default="PUT", description="업로드 방식 (PUT 또는 POST)")
    
    class Config:
        populate_by_name = True

class PresignedUrlResponse(BaseModel):
    """Presigned URL 생성 응답 모델"""
    notice_id: int
    original_filename: str
    stored_filename: str
    upload_method: str
    upload_url: str
    upload_fields: Optional[dict] = None
    file_type: str
    file_size: int
    expires_in: int
    s3_url: str
    message: str

class UploadCompleteRequest(BaseModel):
    """업로드 완료 알림 요청 모델"""
    original_filename: str = Field(..., description="원본 파일명")
    s3_key: str = Field(..., description="S3 키 (stored_filename)")
    file_size: int = Field(..., description="파일 크기")
    file_type: str = Field(..., description="파일 타입")
    s3_url: Optional[str] = Field(None, description="S3 파일 URL (선택사항)")
    etag: Optional[str] = Field(None, description="S3 업로드 ETag (선택사항)")
