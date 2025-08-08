import boto3
import uuid
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import HTTPException
from botocore.exceptions import ClientError, NoCredentialsError
from core.config import settings
import logging

logger = logging.getLogger(__name__)

class S3Service:
    """AWS S3 파일 관리 서비스"""
    
    def __init__(self):
        self.s3_client = None
        self.bucket_name = getattr(settings, 'S3_BUCKET_NAME', 'cpx-attachments')
        self.region_name = getattr(settings, 'AWS_REGION', 'ap-northeast-2')
        
        # S3 클라이언트 초기화
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=getattr(settings, 'AWS_ACCESS_KEY_ID', None),
                aws_secret_access_key=getattr(settings, 'AWS_SECRET_ACCESS_KEY', None),
                region_name=self.region_name
            )
        except Exception as e:
            logger.warning(f"S3 클라이언트 초기화 실패: {e}")
    
    def generate_presigned_url(
        self, 
        operation: str, 
        key: str, 
        expires_in: int = 3600
    ) -> Optional[str]:
        """S3 파일에 대한 서명된 URL 생성"""
        if not self.s3_client:
            return None
        
        try:
            url = self.s3_client.generate_presigned_url(
                operation,
                Params={
                    'Bucket': self.bucket_name,
                    'Key': key
                },
                ExpiresIn=expires_in
            )
            return url
        except Exception as e:
            logger.error(f"Presigned URL 생성 실패: {e}")
            return None
    
    def upload_file(
        self, 
        file_path: str, 
        s3_key: str,
        content_type: Optional[str] = None
    ) -> bool:
        """파일을 S3에 업로드"""
        if not self.s3_client:
            raise HTTPException(status_code=500, detail="S3 클라이언트가 초기화되지 않았습니다.")
        
        try:
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type
            
            self.s3_client.upload_file(
                file_path,
                self.bucket_name,
                s3_key,
                ExtraArgs=extra_args
            )
            return True
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="업로드할 파일을 찾을 수 없습니다.")
        except NoCredentialsError:
            raise HTTPException(status_code=500, detail="AWS 인증 정보가 설정되지 않았습니다.")
        except ClientError as e:
            logger.error(f"S3 업로드 실패: {e}")
            raise HTTPException(status_code=500, detail="S3 업로드 중 오류가 발생했습니다.")
        except Exception as e:
            logger.error(f"S3 업로드 중 예상치 못한 오류: {e}")
            raise HTTPException(status_code=500, detail="파일 업로드 중 오류가 발생했습니다.")
    
    def upload_fileobj(
        self, 
        file_obj, 
        s3_key: str,
        content_type: Optional[str] = None
    ) -> bool:
        """파일 객체를 S3에 업로드"""
        if not self.s3_client:
            raise HTTPException(status_code=500, detail="S3 클라이언트가 초기화되지 않았습니다.")
        
        try:
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type
            
            self.s3_client.upload_fileobj(
                file_obj,
                self.bucket_name,
                s3_key,
                ExtraArgs=extra_args
            )
            return True
        except NoCredentialsError:
            raise HTTPException(status_code=500, detail="AWS 인증 정보가 설정되지 않았습니다.")
        except ClientError as e:
            logger.error(f"S3 업로드 실패: {e}")
            raise HTTPException(status_code=500, detail="S3 업로드 중 오류가 발생했습니다.")
        except Exception as e:
            logger.error(f"S3 업로드 중 예상치 못한 오류: {e}")
            raise HTTPException(status_code=500, detail="파일 업로드 중 오류가 발생했습니다.")
    
    def download_file(self, s3_key: str, local_path: str) -> bool:
        """S3에서 파일 다운로드"""
        if not self.s3_client:
            raise HTTPException(status_code=500, detail="S3 클라이언트가 초기화되지 않았습니다.")
        
        try:
            self.s3_client.download_file(
                self.bucket_name,
                s3_key,
                local_path
            )
            return True
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                raise HTTPException(status_code=404, detail="S3에서 파일을 찾을 수 없습니다.")
            else:
                logger.error(f"S3 다운로드 실패: {e}")
                raise HTTPException(status_code=500, detail="S3 다운로드 중 오류가 발생했습니다.")
        except Exception as e:
            logger.error(f"S3 다운로드 중 예상치 못한 오류: {e}")
            raise HTTPException(status_code=500, detail="파일 다운로드 중 오류가 발생했습니다.")
    
    def delete_file(self, s3_key: str) -> bool:
        """S3에서 파일 삭제"""
        if not self.s3_client:
            raise HTTPException(status_code=500, detail="S3 클라이언트가 초기화되지 않았습니다.")
        
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            return True
        except ClientError as e:
            logger.error(f"S3 삭제 실패: {e}")
            raise HTTPException(status_code=500, detail="S3 삭제 중 오류가 발생했습니다.")
        except Exception as e:
            logger.error(f"S3 삭제 중 예상치 못한 오류: {e}")
            raise HTTPException(status_code=500, detail="파일 삭제 중 오류가 발생했습니다.")
    
    def file_exists(self, s3_key: str) -> bool:
        """S3에 파일이 존재하는지 확인"""
        if not self.s3_client:
            return False
        
        try:
            self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            return True
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                return False
            else:
                logger.error(f"S3 파일 존재 확인 실패: {e}")
                return False
        except Exception as e:
            logger.error(f"S3 파일 존재 확인 중 예상치 못한 오류: {e}")
            return False
    
    def get_file_url(self, s3_key: str) -> str:
        """S3 파일의 공개 URL 반환"""
        return f"https://{self.bucket_name}.s3.{self.region_name}.amazonaws.com/{s3_key}"
    
    def get_download_url(self, s3_key: str, expires_in: int = 3600) -> Optional[str]:
        """S3 파일의 다운로드 URL 생성"""
        return self.generate_presigned_url('get_object', s3_key, expires_in)
    
    def get_upload_url(self, s3_key: str, expires_in: int = 3600) -> Optional[str]:
        """S3 파일의 업로드 URL 생성"""
        return self.generate_presigned_url('put_object', s3_key, expires_in)
    
    def generate_unique_key(self, original_filename: str) -> str:
        """고유한 S3 키 생성"""
        file_extension = Path(original_filename).suffix
        unique_id = str(uuid.uuid4())
        return f"attachments/{unique_id}{file_extension}"

# 전역 인스턴스
s3_service = S3Service()
