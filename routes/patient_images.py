import logging
from fastapi import APIRouter, HTTPException, Depends
from typing import List

from core.constants import ADMIN_ROLES
from services.patient_image_service import PatientImageService
from routes.auth import get_current_user
from core.models import User
from schemas.patient_image_schemas import (
    PatientImageUploadRequest,
    PatientImageUploadResponse
)
from utils.exceptions import (
    FileSizeExceededException,
    UnsupportedFileTypeException,
    S3UploadFailedException,
    PermissionDeniedException
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/patient-images", tags=["환자 이미지"])

# 서비스 인스턴스 생성
patient_image_service = PatientImageService()

@router.get("/scenarios/{scenario_id}/images")
async def get_scenario_images(
    scenario_id: str
):
    """시나리오별 환자 이미지 목록 조회"""
    
    try:
        logger.info(f"시나리오 {scenario_id}의 환자 이미지 목록 조회 요청")
        
        # S3에서 해당 시나리오의 이미지 목록 조회
        images = patient_image_service.list_scenario_images(scenario_id)
        
        return {
            "scenario_id": scenario_id,
            "images": images,
            "total_count": len(images),
            "message": f"시나리오 {scenario_id}의 환자 이미지 {len(images)}개를 조회했습니다."
        }
        
    except Exception as e:
        logger.error(f"시나리오 이미지 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail="이미지 목록 조회 중 오류가 발생했습니다.")

@router.get("/scenarios/{scenario_id}/image")
async def get_scenario_image(
    scenario_id: str
):
    """시나리오의 대표 환자 이미지 Presigned URL 조회 (첫 번째 이미지)"""
    
    try:
        logger.info(f"시나리오 {scenario_id}의 대표 환자 이미지 Presigned URL 조회 요청")
        
        # S3에서 해당 시나리오의 첫 번째 이미지 조회
        image = patient_image_service.get_scenario_representative_image(scenario_id)
        
        if not image:
            raise HTTPException(status_code=404, detail=f"시나리오 {scenario_id}의 환자 이미지를 찾을 수 없습니다.")
        
        # S3 Presigned URL 생성 (읽기용, 1시간 유효)
        presigned_url = patient_image_service.generate_download_url(image['s3_key'], expires_in=3600)
        
        return {
            "scenario_id": scenario_id,
            "image_info": {
                "s3_key": image['s3_key'],
                "filename": image['filename'],
                "file_size": image['file_size'],
                "file_type": image.get('file_type', 'image/jpeg'),
                "last_modified": image.get('last_modified'),
                "etag": image.get('etag')
            },
            "presigned_url": presigned_url,
            "expires_in": 3600,
            "message": "대표 환자 이미지 Presigned URL을 조회했습니다."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"시나리오 대표 이미지 Presigned URL 조회 실패: {e}")
        raise HTTPException(status_code=500, detail="이미지 Presigned URL 조회 중 오류가 발생했습니다.")

@router.post("/upload-url/{scenario_id}", response_model=PatientImageUploadResponse)
async def generate_patient_image_upload_url(
    scenario_id: str,
    request: PatientImageUploadRequest,
    current_user: User = Depends(get_current_user)
):
    """환자 이미지 업로드용 presigned URL 생성 (PUT/POST 방식 선택 가능)"""
    
    # 권한 확인 (관리자 또는 교수만)
    if current_user.role not in ADMIN_ROLES:
        raise PermissionDeniedException("환자 이미지 업로드")
    
    try:
        logger.info(f"환자 이미지 업로드 URL 생성 요청: 시나리오 {scenario_id}, 파일 {request.filename}")
        
        # Presigned URL 생성
        result = await patient_image_service.generate_upload_url(
            scenario_id=scenario_id,
            filename=request.filename,
            file_type=request.file_type,
            file_size=request.file_size,
            method=request.method
        )
        
        return PatientImageUploadResponse(**result)
        
    except (FileSizeExceededException, UnsupportedFileTypeException, S3UploadFailedException) as e:
        raise e
    except Exception as e:
        logger.error(f"환자 이미지 업로드 URL 생성 실패: {e}")
        raise HTTPException(status_code=500, detail="업로드 URL 생성 중 오류가 발생했습니다.")

@router.get("/scenarios/{scenario_id}/image/presigned")
async def get_scenario_image_presigned_url(
    scenario_id: str,
    expires_in: int = 3600
):
    """시나리오의 대표 환자 이미지 Presigned URL 생성"""
    
    try:
        logger.info(f"시나리오 {scenario_id}의 대표 환자 이미지 Presigned URL 생성 요청")
        
        # S3에서 해당 시나리오의 첫 번째 이미지 조회
        image = patient_image_service.get_scenario_representative_image(scenario_id)
        
        if not image:
            raise HTTPException(status_code=404, detail=f"시나리오 {scenario_id}의 환자 이미지를 찾을 수 없습니다.")
        
        # S3 Presigned URL 생성 (읽기용)
        presigned_url = patient_image_service.generate_download_url(image['s3_key'], expires_in)
        
        return {
            "scenario_id": scenario_id,
            "presigned_url": presigned_url,
            "expires_in": expires_in,
            "s3_key": image['s3_key'],
            "filename": image['filename'],
            "message": f"시나리오 {scenario_id}의 환자 이미지 Presigned URL이 생성되었습니다."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"시나리오 환자 이미지 Presigned URL 생성 실패: {e}")
        raise HTTPException(status_code=500, detail="Presigned URL 생성 중 오류가 발생했습니다.")

@router.post("/download-url/{scenario_id}")
async def generate_patient_image_download_url(
    scenario_id: str,
    s3_key: str,
    expires_in: int = 3600,
    current_user: User = Depends(get_current_user)
):
    """환자 이미지 다운로드용 presigned URL 생성"""
    
    try:
        logger.info(f"환자 이미지 다운로드 URL 생성 요청: 시나리오 {scenario_id}, S3 키 {s3_key}")
        
        # 다운로드 URL 생성
        download_url = patient_image_service.generate_download_url(s3_key, expires_in)
        
        return {
            "scenario_id": scenario_id,
            "s3_key": s3_key,
            "download_url": download_url,
            "expires_in": expires_in,
            "message": "다운로드 URL이 생성되었습니다."
        }
        
    except S3UploadFailedException as e:
        raise e
    except Exception as e:
        logger.error(f"환자 이미지 다운로드 URL 생성 실패: {e}")
        raise HTTPException(status_code=500, detail="다운로드 URL 생성 중 오류가 발생했습니다.")

@router.delete("/{scenario_id}")
async def delete_patient_image(
    scenario_id: str,
    s3_key: str,
    current_user: User = Depends(get_current_user)
):
    """환자 이미지 삭제"""
    
    # 권한 확인 (관리자 또는 교수만)
    if current_user.role not in ADMIN_ROLES:
        raise PermissionDeniedException("환자 이미지 삭제")
    
    try:
        logger.info(f"환자 이미지 삭제 요청: 시나리오 {scenario_id}, S3 키 {s3_key}")
        
        # S3에서 파일 삭제
        success = patient_image_service.delete_file(s3_key)
        
        if not success:
            raise HTTPException(status_code=404, detail="삭제할 환자 이미지를 찾을 수 없습니다.")
        
        return {"message": "환자 이미지가 성공적으로 삭제되었습니다."}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"환자 이미지 삭제 실패: {e}")
        raise HTTPException(status_code=500, detail="환자 이미지 삭제 중 오류가 발생했습니다.")
