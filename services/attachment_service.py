import os
import uuid
from pathlib import Path
from typing import List, Optional
from fastapi import HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from sqlalchemy.orm import selectinload
from core.models import Attachments, Notices
from core.config import settings
from pydantic import BaseModel
from services.s3_service import s3_service

class AttachmentCreate(BaseModel):
    """첨부파일 생성 모델"""
    original_filename: str
    s3_url: str
    file_size: int
    file_type: str

class AttachmentUpload(BaseModel):
    """첨부파일 업로드 모델"""
    original_filename: str
    file_size: int
    file_type: str

class AttachmentService:
    """첨부파일 관리 서비스 (S3 기반)"""
    
    def __init__(self):
        # 허용된 파일 타입
        self.allowed_types = [
            'application/pdf',
            'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/vnd.ms-excel',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/vnd.ms-powerpoint',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            'image/jpeg',
            'image/png',
            'image/gif',
            'text/plain'
        ]
    
    async def create_attachment(
        self, 
        db: AsyncSession, 
        notice_id: int, 
        attachment_data: AttachmentCreate
    ) -> Attachments:
        """S3에서 업로드된 파일 정보로 첨부파일 생성"""
        
        # 공지사항 존재 확인
        notice = await db.get(Notices, notice_id)
        if not notice:
            raise HTTPException(status_code=404, detail="공지사항을 찾을 수 없습니다.")
        
        # 파일 크기 제한 (10MB)
        if attachment_data.file_size > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="파일 크기는 10MB를 초과할 수 없습니다.")
        
        # 허용된 파일 타입 확인
        if attachment_data.file_type not in self.allowed_types:
            raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다.")
        
        # S3 URL에서 S3 키 추출
        s3_url = attachment_data.s3_url
        s3_key = s3_url.split(f"{s3_service.bucket_name}.s3.{s3_service.region_name}.amazonaws.com/")[-1]
        
        # 데이터베이스에 기록
        attachment = Attachments(
            notice_id=notice_id,
            original_filename=attachment_data.original_filename,
            stored_filename=s3_key,  # S3 키를 stored_filename에 저장
            file_path=s3_url,  # S3 URL을 file_path에 저장
            file_size=attachment_data.file_size,
            file_type=attachment_data.file_type
        )
        
        db.add(attachment)
        await db.commit()
        await db.refresh(attachment)
        
        return attachment
    
    async def upload_attachment(
        self, 
        db: AsyncSession, 
        notice_id: int, 
        file: UploadFile
    ) -> Attachments:
        """파일을 S3에 업로드하고 첨부파일 생성"""
        
        # 공지사항 존재 확인
        notice = await db.get(Notices, notice_id)
        if not notice:
            raise HTTPException(status_code=404, detail="공지사항을 찾을 수 없습니다.")
        
        # 파일 크기 제한 (10MB)
        if file.size and file.size > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="파일 크기는 10MB를 초과할 수 없습니다.")
        
        # 허용된 파일 타입 확인
        if file.content_type not in self.allowed_types:
            raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다.")
        
        # 고유한 S3 키 생성
        s3_key = s3_service.generate_unique_key(file.filename)
        
        # S3에 파일 업로드
        try:
            s3_service.upload_fileobj(file.file, s3_key, file.content_type)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"S3 업로드 실패: {str(e)}")
        
        # S3 URL 생성
        s3_url = s3_service.get_file_url(s3_key)
        
        # 데이터베이스에 기록
        attachment = Attachments(
            notice_id=notice_id,
            original_filename=file.filename,
            stored_filename=s3_key,
            file_path=s3_url,
            file_size=file.size or 0,
            file_type=file.content_type
        )
        
        db.add(attachment)
        await db.commit()
        await db.refresh(attachment)
        
        return attachment
    
    async def get_attachments_by_notice(
        self, 
        db: AsyncSession, 
        notice_id: int
    ) -> List[Attachments]:
        """공지사항의 첨부파일 목록 조회"""
        
        stmt = select(Attachments).options(selectinload(Attachments.notice)).where(Attachments.notice_id == notice_id)
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def get_attachment(
        self, 
        db: AsyncSession, 
        attachment_id: int
    ) -> Optional[Attachments]:
        """첨부파일 상세 조회"""
        
        stmt = select(Attachments).options(selectinload(Attachments.notice)).where(Attachments.attachment_id == attachment_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def delete_attachment(
        self, 
        db: AsyncSession, 
        attachment_id: int
    ) -> bool:
        """첨부파일 삭제 (S3 파일도 함께 삭제)"""
        
        attachment = await db.get(Attachments, attachment_id)
        if not attachment:
            raise HTTPException(status_code=404, detail="첨부파일을 찾을 수 없습니다.")
        
        # S3에서 파일 삭제
        try:
            s3_service.delete_file(attachment.stored_filename)
        except Exception as e:
            # S3 삭제 실패 시에도 DB에서 삭제
            pass
        
        # 데이터베이스에서 삭제
        await db.delete(attachment)
        await db.commit()
        
        return True
    
    async def delete_attachments_by_notice(
        self, 
        db: AsyncSession, 
        notice_id: int
    ) -> bool:
        """공지사항의 모든 첨부파일 삭제"""
        
        attachments = await self.get_attachments_by_notice(db, notice_id)
        
        # S3에서 모든 파일 삭제
        for attachment in attachments:
            try:
                s3_service.delete_file(attachment.stored_filename)
            except Exception as e:
                # S3 삭제 실패 시에도 계속 진행
                pass
        
        # 데이터베이스에서 삭제
        stmt = delete(Attachments).where(Attachments.notice_id == notice_id)
        await db.execute(stmt)
        await db.commit()
        
        return True
    
    def get_s3_url(self, attachment: Attachments) -> str:
        """첨부파일의 S3 URL 반환"""
        return attachment.file_path
    
    def get_s3_download_url(self, attachment: Attachments, expires_in: int = 3600) -> Optional[str]:
        """첨부파일의 S3 다운로드 URL 생성"""
        return s3_service.get_download_url(attachment.stored_filename, expires_in)
    
    def is_s3_file_exists(self, attachment: Attachments) -> bool:
        """S3 파일 존재 여부 확인"""
        return s3_service.file_exists(attachment.stored_filename)

# 전역 인스턴스
attachment_service = AttachmentService()
