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
        self.bucket_name = getattr(settings, 'S3_BUCKET_NAME', 'medicpx')
        self.region_name = getattr(settings, 'AWS_REGION', 'ap-northeast-2')
        
        # S3 클라이언트 초기화
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=getattr(settings, 'AWS_ACCESS_KEY_ID', None),
                aws_secret_access_key=getattr(settings, 'AWS_SECRET_ACCESS_KEY', None),
                region_name=self.region_name
            )
            
            # S3 버킷 CORS 설정 확인 및 설정
            self._setup_cors()
        except Exception as e:
            logger.warning(f"S3 클라이언트 초기화 실패: {e}")
    
    def _setup_cors(self):
        """S3 버킷 CORS 설정"""
        try:
            cors_configuration = {
                'CORSRules': [
                    {
                        'AllowedHeaders': ['*'],
                        'AllowedMethods': ['GET', 'PUT', 'POST', 'DELETE', 'HEAD'],
                        'AllowedOrigins': ['*'],  # 프로덕션에서는 특정 도메인으로 제한
                        'ExposeHeaders': ['ETag', 'Content-Length'],
                        'MaxAgeSeconds': 3000
                    }
                ]
            }
            
            self.s3_client.put_bucket_cors(
                Bucket=self.bucket_name,
                CORSConfiguration=cors_configuration
            )
            logger.info(f"S3 버킷 CORS 설정 완료: {self.bucket_name}")
        except Exception as e:
            logger.warning(f"S3 CORS 설정 실패: {e}")
    
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

    def generate_download_url(self, s3_key: str, expires_in: int = 3600) -> str:
        """S3 파일 다운로드용 Presigned URL 생성"""
        if not self.s3_client:
            raise HTTPException(status_code=500, detail="S3 클라이언트가 초기화되지 않았습니다.")
        
        try:
            logger.info(f"다운로드 Presigned URL 생성 시작 - S3 키: {s3_key}, 만료시간: {expires_in}초")
            
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': s3_key
                },
                ExpiresIn=expires_in
            )
            
            logger.info(f"다운로드 Presigned URL 생성 완료 - S3 키: {s3_key}, URL 길이: {len(url)}")
            return url
            
        except Exception as e:
            logger.error(f"다운로드 Presigned URL 생성 실패 - S3 키: {s3_key}, 오류: {e}")
            raise HTTPException(status_code=500, detail=f"다운로드 URL 생성 중 오류가 발생했습니다: {str(e)}")
    
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
            logger.warning(f"S3 클라이언트가 초기화되지 않음 - S3 키: {s3_key}")
            return False
        
        try:
            logger.info(f"S3 파일 존재 확인 시작 - S3 키: {s3_key}")
            self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            logger.info(f"S3 파일 존재 확인 성공 - S3 키: {s3_key}")
            return True
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logger.info(f"S3 파일 존재하지 않음 - S3 키: {s3_key}")
                return False
            else:
                logger.error(f"S3 파일 존재 확인 실패 - S3 키: {s3_key}, 오류: {e}")
                return False
        except Exception as e:
            logger.error(f"S3 파일 존재 확인 중 예상치 못한 오류 - S3 키: {s3_key}, 오류: {e}")
            return False
    
    def get_file_url(self, s3_key: str) -> str:
        """S3 파일의 공개 URL 반환"""
        return f"https://{self.bucket_name}.s3.{self.region_name}.amazonaws.com/{s3_key}"
    
    def get_download_url(self, s3_key: str, expires_in: int = 3600) -> Optional[str]:
        """S3 파일의 다운로드 URL 생성"""
        return self.generate_presigned_url('get_object', s3_key, expires_in)
    
    def get_upload_url(self, s3_key: str, expires_in: int = 3600) -> Optional[str]:
        """S3 파일의 업로드 URL 생성 (PUT 요청용)"""
        try:
            url = self.s3_client.generate_presigned_url(
                'put_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': s3_key
                },
                ExpiresIn=expires_in
            )
            return url
        except Exception as e:
            logger.error(f"업로드 URL 생성 실패: {e}")
            return None
    
    def get_upload_url_with_cors(self, s3_key: str, content_type: str = None, expires_in: int = 3600) -> Optional[str]:
        """CORS 지원 업로드 URL 생성 (PUT 요청용)"""
        try:
            # Presigned URL 생성 시 Content-Type을 파라미터로 전달하지 않음
            # Content-Type은 실제 업로드 시 헤더로 전달해야 함
            logger.info(f"Presigned PUT URL 생성 시작 - S3 키: {s3_key}, 만료시간: {expires_in}초")
            
            url = self.s3_client.generate_presigned_url(
                'put_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': s3_key
                },
                ExpiresIn=expires_in
            )
            
            logger.info(f"Presigned PUT URL 생성 완료 - S3 키: {s3_key}, URL 길이: {len(url)}")
            return url
        except Exception as e:
            logger.error(f"CORS 업로드 URL 생성 실패 - S3 키: {s3_key}, 오류: {e}")
            return None
    
    def generate_presigned_post(self, s3_key: str, content_type: str = None, expires_in: int = 3600) -> Optional[Dict[str, Any]]:
        """Presigned POST URL 생성 (POST 요청용)"""
        try:
            logger.info(f"Presigned POST URL 생성 시작 - S3 키: {s3_key}, Content-Type: {content_type}, 만료시간: {expires_in}초")
            
            conditions = [
                {'bucket': self.bucket_name},
                {'key': s3_key}
            ]
            
            # Content-Type 조건 추가
            if content_type:
                conditions.append({'Content-Type': content_type})
            
            fields = {
                'key': s3_key
            }
            
            # Content-Type이 지정된 경우 필드에 추가
            if content_type:
                fields['Content-Type'] = content_type
            
            response = self.s3_client.generate_presigned_post(
                Bucket=self.bucket_name,
                Key=s3_key,
                Fields=fields,
                Conditions=conditions,
                ExpiresIn=expires_in
            )
            
            logger.info(f"Presigned POST URL 생성 완료 - S3 키: {s3_key}, URL 길이: {len(response['url'])}, 필드 수: {len(response['fields'])}")
            
            return {
                'url': response['url'],
                'fields': response['fields']
            }
        except Exception as e:
            logger.error(f"Presigned POST 생성 실패 - S3 키: {s3_key}, 오류: {e}")
            return None
    
    def generate_unique_key(self, original_filename: str) -> str:
        """고유한 S3 키 생성"""
        file_extension = Path(original_filename).suffix
        unique_id = str(uuid.uuid4())
        s3_key = f"attachments/{unique_id}{file_extension}"
        logger.info(f"S3 키 생성 - 원본파일명: {original_filename}, 생성된 키: {s3_key}")
        return s3_key
    
    def _extract_s3_key_from_url(self, s3_url: str) -> str:
        """S3 URL에서 S3 키 추출"""
        try:
            # S3 URL에서 키 부분만 추출
            if self.bucket_name in s3_url and self.region_name in s3_url:
                # 전체 URL에서 키 부분만 추출
                key_part = s3_url.split(f"{self.bucket_name}.s3.{self.region_name}.amazonaws.com/")[-1]
                logger.info(f"S3 키 추출 - URL: {s3_url}, 추출된 키: {key_part}")
                return key_part
            else:
                # URL이 아닌 경우 그대로 반환 (이미 키인 경우)
                logger.info(f"S3 키 추출 - 이미 키 형태: {s3_url}")
                return s3_url
        except Exception as e:
            # 오류 발생 시 원본 URL 반환
            logger.warning(f"S3 키 추출 실패 - URL: {s3_url}, 오류: {e}")
            return s3_url
    
    def list_objects(self, prefix: str = "") -> list:
        """S3 버킷에서 특정 접두사로 시작하는 객체 목록 조회"""
        if not self.s3_client:
            logger.error("S3 클라이언트가 초기화되지 않았습니다.")
            return []
        
        try:
            logger.info(f"S3 객체 목록 조회 시작 - 접두사: {prefix}")
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            objects = response.get('Contents', [])
            logger.info(f"S3 객체 목록 조회 완료 - 접두사: {prefix}, 객체 수: {len(objects)}")
            
            return objects
            
        except Exception as e:
            logger.error(f"S3 객체 목록 조회 실패 - 접두사: {prefix}, 오류: {e}")
            return []

# 전역 인스턴스
s3_service = S3Service()
