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
from core.constants import MAX_FILE_SIZE, ALLOWED_FILE_TYPES, S3_PRESIGNED_URL_EXPIRES_IN
from pydantic import BaseModel
from services.s3_service import s3_service
from utils.exceptions import (
    NoticeNotFoundException,
    FileSizeExceededException,
    UnsupportedFileTypeException,
    S3UploadFailedException,
    AttachmentNotFoundException
)
from utils.logging_config import attachment_logger

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
        self.allowed_types = ALLOWED_FILE_TYPES
        # 파일 크기 제한
        self.max_file_size = MAX_FILE_SIZE
    
    async def _validate_notice_exists(self, db: AsyncSession, notice_id: int) -> Notices:
        """공지사항 존재 확인"""
        notice = await db.get(Notices, notice_id)
        if not notice:
            raise NoticeNotFoundException()
        return notice
    
    def _validate_file_size(self, file_size: int):
        """파일 크기 검증"""
        if file_size > self.max_file_size:
            raise FileSizeExceededException()
    
    def _validate_file_type(self, file_type: str):
        """파일 타입 검증"""
        if file_type not in self.allowed_types:
            raise UnsupportedFileTypeException()
    
    def _extract_s3_key_from_url(self, s3_url: str) -> str:
        """S3 URL에서 S3 키 추출"""
        try:
            # S3 URL에서 키 부분만 추출
            if s3_service.bucket_name in s3_url and s3_service.region_name in s3_url:
                # 전체 URL에서 키 부분만 추출
                key_part = s3_url.split(f"{s3_service.bucket_name}.s3.{s3_service.region_name}.amazonaws.com/")[-1]
                return key_part
            else:
                # URL이 아닌 경우 그대로 반환 (이미 키인 경우)
                return s3_url
        except Exception as e:
            # 오류 발생 시 원본 URL 반환
            return s3_url
    
    async def create_attachment(
        self, 
        db: AsyncSession, 
        notice_id: int, 
        attachment_data: AttachmentCreate
    ) -> Attachments:
        """S3에서 업로드된 파일 정보로 첨부파일 생성"""
        
        # 공통 검증
        await self._validate_notice_exists(db, notice_id)
        self._validate_file_size(attachment_data.file_size)
        self._validate_file_type(attachment_data.file_type)
        
        # S3 URL에서 S3 키 추출
        s3_key = self._extract_s3_key_from_url(attachment_data.s3_url)
        
        # 데이터베이스에 기록
        attachment = Attachments(
            notice_id=notice_id,
            original_filename=attachment_data.original_filename,
            stored_filename=s3_key,  # S3 키를 stored_filename에 저장
            file_path=attachment_data.s3_url,  # S3 URL을 file_path에 저장
            file_size=attachment_data.file_size,
            file_type=attachment_data.file_type
        )
        
        db.add(attachment)
        await db.commit()
        await db.refresh(attachment)
        
        attachment_logger.info(f"첨부파일 생성 완료: {attachment.original_filename} (ID: {attachment.attachment_id})")
        return attachment
    
    # 백엔드 프록시 방식 제거 - Presigned URL 방식만 사용
    
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
        
        stmt = select(Attachments).options(selectinload(Attachments.notice)).where(Attachments.attachment_id == attachment_id)
        result = await db.execute(stmt)
        attachment = result.scalar_one_or_none()
        
        if not attachment:
            raise AttachmentNotFoundException()
        
        # S3에서 파일 삭제
        try:
            s3_service.delete_file(attachment.stored_filename)
            attachment_logger.info(f"S3 파일 삭제 완료: {attachment.stored_filename}")
        except Exception as e:
            # S3 삭제 실패 시에도 DB에서 삭제
            attachment_logger.warning(f"S3 파일 삭제 실패: {attachment.stored_filename} - {str(e)}")
        
        # 데이터베이스에서 삭제
        await db.delete(attachment)
        await db.commit()
        
        attachment_logger.info(f"첨부파일 삭제 완료: {attachment.original_filename} (ID: {attachment.attachment_id})")
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
