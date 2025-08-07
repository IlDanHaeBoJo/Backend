import boto3
import uuid
import os
from typing import Optional
from botocore.exceptions import ClientError
import logging

logger = logging.getLogger(__name__)

class S3Uploader:
    """S3 파일 업로드 유틸리티"""
    
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'ap-northeast-2')
        )
        self.bucket_name = os.getenv('S3_BUCKET_NAME')
    
    def generate_unique_filename(self, original_filename: str) -> str:
        """고유한 파일명 생성 (UUID + 원본 확장자)"""
        file_extension = os.path.splitext(original_filename)[1]
        unique_id = str(uuid.uuid4())
        return f"{unique_id}{file_extension}"
    
    def get_file_path(self, notice_id: int, stored_filename: str) -> str:
        """S3 파일 경로 생성"""
        return f"notices/{notice_id}/{stored_filename}"
    
    def generate_presigned_url(self, file_path: str, expiration: int = 3600) -> Optional[str]:
        """파일 다운로드를 위한 presigned URL 생성"""
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': file_path},
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            logger.error(f"Presigned URL 생성 실패: {e}")
            return None
    
    def get_file_info(self, file_path: str) -> Optional[dict]:
        """S3 파일 정보 조회"""
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=file_path)
            return {
                'file_size': response['ContentLength'],
                'file_type': response['ContentType'],
                'last_modified': response['LastModified']
            }
        except ClientError as e:
            logger.error(f"파일 정보 조회 실패: {e}")
            return None

# 전역 인스턴스
s3_uploader = S3Uploader() 