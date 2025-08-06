from typing import List, Optional
from datetime import datetime
import logging
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from core.models import Attachments as DBAttachment
from utils.s3_utils import s3_uploader

logger = logging.getLogger(__name__)

# Pydantic 모델 정의
class AttachmentBase(BaseModel):
    """첨부파일 기본 모델"""
    notice_id: int = Field(..., description="공지사항 ID")
    original_filename: str = Field(..., max_length=255, description="원본 파일명")
    stored_filename: str = Field(..., max_length=255, description="저장된 파일명")
    file_path: str = Field(..., max_length=500, description="파일 저장 경로")
    file_size: int = Field(..., description="파일 크기 (바이트)")
    file_type: str = Field(..., max_length=100, description="파일 MIME 타입")

class AttachmentCreate(AttachmentBase):
    """첨부파일 생성 모델"""
    pass

class AttachmentUpdate(BaseModel):
    """첨부파일 수정 모델"""
    original_filename: Optional[str] = Field(None, max_length=255, description="원본 파일명")
    stored_filename: Optional[str] = Field(None, max_length=255, description="저장된 파일명")
    file_path: Optional[str] = Field(None, max_length=500, description="파일 저장 경로")
    file_size: Optional[int] = Field(None, description="파일 크기 (바이트)")
    file_type: Optional[str] = Field(None, max_length=100, description="파일 MIME 타입")

class Attachment(AttachmentBase):
    """첨부파일 응답 모델"""
    attachment_id: int = Field(..., description="첨부파일 고유 식별자")
    uploaded_at: datetime = Field(..., description="업로드 시간")
    download_url: Optional[str] = Field(None, description="파일 다운로드 URL")
    
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class AttachmentService:
    """첨부파일 서비스"""
    
    async def get_attachments_by_notice_id(self, db: AsyncSession, notice_id: int) -> List[Attachment]:
        """공지사항 ID로 첨부파일 목록 조회"""
        result = await db.execute(
            select(DBAttachment).filter(DBAttachment.notice_id == notice_id)
        )
        db_attachments = result.scalars().all()
        
        attachments = []
        for attachment in db_attachments:
            attachment_data = Attachment.model_validate(attachment)
            # S3 다운로드 URL 생성
            file_path = s3_uploader.get_file_path(notice_id, attachment.stored_filename)
            download_url = s3_uploader.generate_presigned_url(file_path)
            attachment_data.download_url = download_url
            attachments.append(attachment_data)
        
        return attachments
    
    async def get_attachment_by_id(self, db: AsyncSession, attachment_id: int) -> Optional[Attachment]:
        """ID로 첨부파일 조회"""
        result = await db.execute(
            select(DBAttachment).filter(DBAttachment.attachment_id == attachment_id)
        )
        db_attachment = result.scalars().first()
        if db_attachment:
            attachment_data = Attachment.model_validate(db_attachment)
            # S3 다운로드 URL 생성
            file_path = s3_uploader.get_file_path(db_attachment.notice_id, db_attachment.stored_filename)
            download_url = s3_uploader.generate_presigned_url(file_path)
            attachment_data.download_url = download_url
            return attachment_data
        return None
    
    async def create_attachment(self, db: AsyncSession, attachment_data: AttachmentCreate) -> Attachment:
        """새 첨부파일 정보 생성"""
        db_attachment = DBAttachment(**attachment_data.model_dump())
        
        db.add(db_attachment)
        await db.commit()
        await db.refresh(db_attachment)
        
        logger.info(f"새 첨부파일 정보 생성: {db_attachment.original_filename}")
        return Attachment.model_validate(db_attachment)
    
    async def update_attachment(self, db: AsyncSession, attachment_id: int, attachment_data: AttachmentUpdate) -> Optional[Attachment]:
        """첨부파일 정보 수정"""
        result = await db.execute(
            select(DBAttachment).filter(DBAttachment.attachment_id == attachment_id)
        )
        db_attachment = result.scalars().first()
        if not db_attachment:
            return None
        
        if attachment_data.original_filename is not None:
            db_attachment.original_filename = attachment_data.original_filename
        if attachment_data.stored_filename is not None:
            db_attachment.stored_filename = attachment_data.stored_filename
        if attachment_data.file_path is not None:
            db_attachment.file_path = attachment_data.file_path
        if attachment_data.file_size is not None:
            db_attachment.file_size = attachment_data.file_size
        if attachment_data.file_type is not None:
            db_attachment.file_type = attachment_data.file_type
        
        await db.commit()
        await db.refresh(db_attachment)
        
        logger.info(f"첨부파일 정보 수정: {db_attachment.original_filename}")
        return Attachment.model_validate(db_attachment)
    
    async def delete_attachment(self, db: AsyncSession, attachment_id: int) -> bool:
        """첨부파일 정보 삭제"""
        result = await db.execute(
            select(DBAttachment).filter(DBAttachment.attachment_id == attachment_id)
        )
        db_attachment = result.scalars().first()
        if not db_attachment:
            return False
        
        await db.delete(db_attachment)
        await db.commit()
        logger.info(f"첨부파일 정보 삭제: {db_attachment.original_filename}")
        return True
    
    async def delete_attachments_by_notice_id(self, db: AsyncSession, notice_id: int) -> bool:
        """공지사항 ID로 모든 첨부파일 정보 삭제"""
        result = await db.execute(
            select(DBAttachment).filter(DBAttachment.notice_id == notice_id)
        )
        db_attachments = result.scalars().all()
        
        for attachment in db_attachments:
            await db.delete(attachment)
        
        await db.commit()
        logger.info(f"공지사항 ID {notice_id}의 모든 첨부파일 정보 삭제")
        return True 