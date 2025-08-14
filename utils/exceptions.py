from fastapi import HTTPException
from typing import Optional

class AttachmentException(HTTPException):
    """첨부파일 관련 예외"""
    def __init__(self, status_code: int, detail: str):
        super().__init__(status_code=status_code, detail=detail)

class NoticeNotFoundException(AttachmentException):
    """공지사항을 찾을 수 없을 때"""
    def __init__(self):
        super().__init__(status_code=404, detail="공지사항을 찾을 수 없습니다.")

class AttachmentNotFoundException(AttachmentException):
    """첨부파일을 찾을 수 없을 때"""
    def __init__(self):
        super().__init__(status_code=404, detail="첨부파일을 찾을 수 없습니다.")

class FileSizeExceededException(AttachmentException):
    """파일 크기 초과"""
    def __init__(self, max_size_mb: int = 10):
        super().__init__(status_code=400, detail=f"파일 크기는 {max_size_mb}MB를 초과할 수 없습니다.")

class UnsupportedFileTypeException(AttachmentException):
    """지원하지 않는 파일 형식"""
    def __init__(self):
        super().__init__(status_code=400, detail="지원하지 않는 파일 형식입니다.")

class S3UploadFailedException(AttachmentException):
    """S3 업로드 실패"""
    def __init__(self, error_message: str):
        super().__init__(status_code=500, detail=f"S3 업로드 실패: {error_message}")

class DownloadUrlGenerationFailedException(AttachmentException):
    """다운로드 URL 생성 실패"""
    def __init__(self):
        super().__init__(status_code=500, detail="다운로드 URL 생성에 실패했습니다.")

class FileNotExistsException(AttachmentException):
    """파일이 존재하지 않음"""
    def __init__(self):
        super().__init__(status_code=404, detail="파일이 존재하지 않습니다.")

class PermissionDeniedException(AttachmentException):
    """권한 없음"""
    def __init__(self, operation: str = "첨부파일"):
        super().__init__(status_code=403, detail=f"{operation} 권한이 없습니다.")

