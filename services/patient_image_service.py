import logging
import uuid
from typing import Optional, List
from fastapi import HTTPException

from services.s3_service import s3_service
from utils.exceptions import (
    FileSizeExceededException,
    UnsupportedFileTypeException,
    S3UploadFailedException
)

logger = logging.getLogger(__name__)

class PatientImageService:
    """환자 이미지 관리 서비스 (S3만 사용)"""
    
    # 허용된 이미지 타입
    ALLOWED_IMAGE_TYPES = {
        "image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp"
    }
    
    # 최대 파일 크기 (10MB)
    MAX_FILE_SIZE = 10 * 1024 * 1024
    
    def _validate_file_size(self, file_size: int):
        """파일 크기 검증"""
        if file_size > self.MAX_FILE_SIZE:
            raise FileSizeExceededException(f"파일 크기가 {self.MAX_FILE_SIZE // (1024*1024)}MB를 초과합니다.")
    
    def _validate_file_type(self, file_type: str):
        """파일 타입 검증"""
        if file_type.lower() not in self.ALLOWED_IMAGE_TYPES:
            raise UnsupportedFileTypeException(f"지원하지 않는 이미지 형식입니다: {file_type}")
    
    def _generate_s3_key(self, scenario_id: str, filename: str) -> str:
        """S3 키 생성"""
        file_extension = filename.split('.')[-1] if '.' in filename else 'jpg'
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        return f"patient_images/scenario_{scenario_id}/{unique_filename}"
    
    def list_scenario_images(self, scenario_id: str) -> List[dict]:
        """시나리오별 환자 이미지 목록 조회"""
        try:
            # S3에서 해당 시나리오 폴더의 모든 이미지 조회
            prefix = f"patient_images/scenario_{scenario_id}/"
            objects = s3_service.list_objects(prefix)
            
            # 만약 폴더 구조에 이미지가 없으면, patient/ 폴더의 scenario_{id}.jpg 파일도 확인
            if not objects:
                patient_objects = s3_service.list_objects("patient/")
                for obj in patient_objects:
                    if obj['Key'] == f"patient/scenario_{scenario_id}.jpg":
                        objects = [obj]
                        break
            
            # 여전히 없으면 루트의 scenario_{id}.jpg 파일도 확인
            if not objects:
                root_objects = s3_service.list_objects("")
                for obj in root_objects:
                    if obj['Key'] == f"scenario_{scenario_id}.jpg":
                        objects = [obj]
                        break
            
            images = []
            for obj in objects:
                # 파일 확장자 확인
                if any(obj['Key'].lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                    image_info = {
                        "s3_key": obj['Key'],
                        "s3_url": f"https://{s3_service.bucket_name}.s3.{s3_service.region_name}.amazonaws.com/{obj['Key']}",
                        "filename": obj['Key'].split('/')[-1],
                        "file_size": obj.get('Size', 0),
                        "last_modified": obj.get('LastModified'),
                        "etag": obj.get('ETag', '').strip('"')
                    }
                    images.append(image_info)
            
            # 최신 업로드 순으로 정렬
            images.sort(key=lambda x: x['last_modified'], reverse=True)
            
            logger.info(f"시나리오 {scenario_id}의 환자 이미지 {len(images)}개 조회")
            return images
            
        except Exception as e:
            logger.error(f"시나리오 이미지 목록 조회 실패: {e}")
            return []
    
    def get_scenario_representative_image(self, scenario_id: str) -> Optional[dict]:
        """시나리오의 대표 환자 이미지 조회 (첫 번째 이미지)"""
        try:
            images = self.list_scenario_images(scenario_id)
            
            if not images:
                logger.info(f"시나리오 {scenario_id}의 환자 이미지가 없습니다.")
                return None
            
            # 첫 번째 이미지를 대표 이미지로 반환
            representative_image = images[0]
            logger.info(f"시나리오 {scenario_id}의 대표 이미지: {representative_image['filename']}")
            return representative_image
            
        except Exception as e:
            logger.error(f"시나리오 대표 이미지 조회 실패: {e}")
            return None
    
    async def generate_upload_url(
        self, 
        scenario_id: str, 
        filename: str, 
        file_type: str, 
        file_size: int, 
        method: str = "PUT"
    ) -> dict:
        """환자 이미지 업로드용 presigned URL 생성"""
        
        # 파일 검증
        self._validate_file_size(file_size)
        self._validate_file_type(file_type)
        
        # S3 키 생성
        s3_key = self._generate_s3_key(scenario_id, filename)
        stored_filename = s3_key.split('/')[-1]
        
        logger.info(f"환자 이미지 업로드 URL 생성: 시나리오 {scenario_id}, 파일 {filename}")
        
        try:
            if method.upper() == "PUT":
                # PUT 방식 presigned URL 생성
                upload_url = s3_service.get_upload_url_with_cors(
                    s3_key, 
                    file_type, 
                    method="PUT"
                )
                upload_fields = None
            else:
                # POST 방식 presigned URL 생성
                upload_url, upload_fields = s3_service.generate_presigned_post(
                    s3_key, 
                    file_type
                )
            
            s3_url = f"https://{s3_service.bucket_name}.s3.{s3_service.region_name}.amazonaws.com/{s3_key}"
            
            return {
                "scenario_id": scenario_id,
                "original_filename": filename,
                "stored_filename": stored_filename,
                "upload_method": method.upper(),
                "upload_url": upload_url,
                "upload_fields": upload_fields,
                "file_type": file_type,
                "file_size": file_size,
                "expires_in": s3_service.presigned_url_expires_in,
                "s3_url": s3_url,
                "message": f"업로드 URL이 생성되었습니다. 이 URL로 {method.upper()} 요청을 보내 파일을 업로드하세요."
            }
            
        except Exception as e:
            logger.error(f"Presigned URL 생성 실패: {e}")
            raise S3UploadFailedException(f"업로드 URL 생성 중 오류가 발생했습니다: {str(e)}")
    
    def get_s3_url_from_key(self, s3_key: str) -> str:
        """S3 키로부터 URL 생성"""
        return f"https://{s3_service.bucket_name}.s3.{s3_service.region_name}.amazonaws.com/{s3_key}"
    
    def generate_download_url(self, s3_key: str, expires_in: int = 3600) -> str:
        """다운로드용 presigned URL 생성"""
        try:
            return s3_service.generate_download_url(s3_key, expires_in)
        except Exception as e:
            logger.error(f"다운로드 URL 생성 실패: {e}")
            raise S3UploadFailedException(f"다운로드 URL 생성 중 오류가 발생했습니다: {str(e)}")
    
    def delete_file(self, s3_key: str) -> bool:
        """S3에서 파일 삭제"""
        try:
            s3_service.delete_file(s3_key)
            logger.info(f"환자 이미지 삭제 완료: {s3_key}")
            return True
        except Exception as e:
            logger.error(f"환자 이미지 삭제 실패: {e}")
            return False
