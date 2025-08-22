from pydantic import BaseModel, Field
from typing import Optional

class PatientImageUploadRequest(BaseModel):
    """환자 이미지 업로드 요청 모델"""
    filename: str = Field(..., description="원본 파일명")
    file_type: str = Field(..., alias="content_type", description="파일 MIME 타입")
    file_size: int = Field(..., alias="content_length", description="파일 크기 (bytes)")
    method: str = Field(default="PUT", description="업로드 방식 (PUT 또는 POST)")
    
    class Config:
        populate_by_name = True

class PatientImageUploadResponse(BaseModel):
    """환자 이미지 업로드 URL 생성 응답 모델"""
    scenario_id: str
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
