#!/usr/bin/env python3
"""
환자 이미지 업로드 테스트 스크립트 (S3만 사용)
S3 Presigned URL을 사용한 환자 이미지 업로드 플로우 테스트
"""

import asyncio
import aiohttp
import json
import os
from pathlib import Path

class PatientImageUploadTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        self.access_token = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def login(self, username: str, password: str) -> bool:
        """로그인하여 액세스 토큰 획득"""
        try:
            login_data = {
                "username": username,
                "password": password
            }
            
            async with self.session.post(
                f"{self.base_url}/auth/login",
                json=login_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    self.access_token = result.get("access_token")
                    print(f"✅ 로그인 성공: {username}")
                    return True
                else:
                    print(f"❌ 로그인 실패: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ 로그인 오류: {e}")
            return False
    
    async def generate_upload_url(self, scenario_id: str, filename: str, file_type: str, file_size: int) -> dict:
        """환자 이미지 업로드 URL 생성"""
        try:
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            
            request_data = {
                "filename": filename,
                "content_type": file_type,
                "content_length": file_size,
                "method": "PUT"
            }
            
            async with self.session.post(
                f"{self.base_url}/patient-images/upload-url/{scenario_id}",
                json=request_data,
                headers=headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ 업로드 URL 생성 성공: {filename}")
                    return result
                else:
                    error_text = await response.text()
                    print(f"❌ 업로드 URL 생성 실패: {response.status} - {error_text}")
                    return None
        except Exception as e:
            print(f"❌ 업로드 URL 생성 오류: {e}")
            return None
    
    async def upload_to_s3(self, upload_url: str, file_path: str, content_type: str) -> bool:
        """S3에 파일 업로드"""
        try:
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            headers = {
                "Content-Type": content_type,
                "Content-Length": str(len(file_content))
            }
            
            async with self.session.put(
                upload_url,
                data=file_content,
                headers=headers
            ) as response:
                if response.status == 200:
                    etag = response.headers.get('ETag', '').strip('"')
                    print(f"✅ S3 업로드 성공: {file_path}")
                    return etag
                else:
                    error_text = await response.text()
                    print(f"❌ S3 업로드 실패: {response.status} - {error_text}")
                    return None
        except Exception as e:
            print(f"❌ S3 업로드 오류: {e}")
            return None
    
    async def generate_download_url(self, scenario_id: str, s3_key: str) -> dict:
        """다운로드 URL 생성"""
        try:
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            
            request_data = {
                "s3_key": s3_key,
                "expires_in": 3600
            }
            
            async with self.session.post(
                f"{self.base_url}/patient-images/download-url/{scenario_id}",
                json=request_data,
                headers=headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ 다운로드 URL 생성 성공: {s3_key}")
                    return result
                else:
                    error_text = await response.text()
                    print(f"❌ 다운로드 URL 생성 실패: {response.status} - {error_text}")
                    return None
        except Exception as e:
            print(f"❌ 다운로드 URL 생성 오류: {e}")
            return None
    
    async def delete_patient_image(self, scenario_id: str, s3_key: str) -> bool:
        """환자 이미지 삭제"""
        try:
            headers = {
                "Authorization": f"Bearer {self.access_token}"
            }
            
            params = {"s3_key": s3_key}
            
            async with self.session.delete(
                f"{self.base_url}/patient-images/{scenario_id}",
                params=params,
                headers=headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ 환자 이미지 삭제 성공: {s3_key}")
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ 환자 이미지 삭제 실패: {response.status} - {error_text}")
                    return False
        except Exception as e:
            print(f"❌ 환자 이미지 삭제 오류: {e}")
            return False

    async def get_scenario_images(self, scenario_id: str) -> list:
        """시나리오별 환자 이미지 목록 조회"""
        try:
            headers = {
                "Authorization": f"Bearer {self.access_token}"
            }
            
            async with self.session.get(
                f"{self.base_url}/patient-images/scenarios/{scenario_id}/images",
                headers=headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ 시나리오 {scenario_id}의 환자 이미지 목록 조회 성공: {len(result.get('images', []))}개")
                    return result.get('images', [])
                else:
                    error_text = await response.text()
                    print(f"❌ 시나리오 이미지 목록 조회 실패: {response.status} - {error_text}")
                    return []
        except Exception as e:
            print(f"❌ 시나리오 이미지 목록 조회 오류: {e}")
            return []
    
    async def get_scenario_image(self, scenario_id: str) -> dict:
        """시나리오의 대표 환자 이미지 조회"""
        try:
            headers = {
                "Authorization": f"Bearer {self.access_token}"
            }
            
            async with self.session.get(
                f"{self.base_url}/patient-images/scenarios/{scenario_id}/image",
                headers=headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ 시나리오 {scenario_id}의 대표 환자 이미지 조회 성공")
                    return result.get('image', {})
                else:
                    error_text = await response.text()
                    print(f"❌ 시나리오 대표 이미지 조회 실패: {response.status} - {error_text}")
                    return {}
        except Exception as e:
            print(f"❌ 시나리오 대표 이미지 조회 오류: {e}")
            return {}

async def create_test_image(file_path: str, content: str = "Test patient image content"):
    """테스트용 이미지 파일 생성"""
    try:
        # 간단한 텍스트 파일을 이미지로 가정 (실제 테스트에서는 실제 이미지 파일 사용)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ 테스트 파일 생성: {file_path}")
        return True
    except Exception as e:
        print(f"❌ 테스트 파일 생성 실패: {e}")
        return False

async def main():
    """메인 실행 함수"""
    print("🏥 환자 이미지 업로드 테스트 (S3만 사용)")
    print("=" * 50)
    
    # 테스트 파일 생성
    test_file_path = "test_patient_image.txt"
    if not await create_test_image(test_file_path):
        return
    
    # 파일 정보
    file_size = os.path.getsize(test_file_path)
    file_type = "image/jpeg"  # 실제로는 텍스트 파일이지만 이미지로 가정
    scenario_id = "1"  # 시나리오 1번
    
    async with PatientImageUploadTester() as tester:
        # 1. 로그인
        print("\n1️⃣ 로그인 중...")
        if not await tester.login("admin", "admin123"):
            print("❌ 로그인 실패로 테스트를 중단합니다.")
            return
        
        # 2. 업로드 URL 생성
        print(f"\n2️⃣ 환자 이미지 업로드 URL 생성 중...")
        upload_result = await tester.generate_upload_url(
            scenario_id=scenario_id,
            filename="test_patient.jpg",
            file_type=file_type,
            file_size=file_size
        )
        
        if not upload_result:
            print("❌ 업로드 URL 생성 실패로 테스트를 중단합니다.")
            return
        
        # 3. S3 업로드
        print(f"\n3️⃣ S3에 파일 업로드 중...")
        etag = await tester.upload_to_s3(
            upload_url=upload_result["upload_url"],
            file_path=test_file_path,
            content_type=file_type
        )
        
        if not etag:
            print("❌ S3 업로드 실패로 테스트를 중단합니다.")
            return
        
        # 4. 다운로드 URL 생성 (선택사항)
        print(f"\n4️⃣ 다운로드 URL 생성 중...")
        download_result = await tester.generate_download_url(
            scenario_id=scenario_id,
            s3_key=upload_result["stored_filename"]
        )
        
        if download_result:
            print(f"✅ 다운로드 URL 생성됨: {download_result['download_url']}")
        
        # 5. 시나리오 이미지 목록 조회
        print(f"\n5️⃣ 시나리오 이미지 목록 조회 중...")
        images = await tester.get_scenario_images(scenario_id)
        
        if images:
            print(f"✅ 시나리오 {scenario_id}의 환자 이미지 {len(images)}개 확인됨:")
            for img in images:
                print(f"  - {img['filename']} (크기: {img['file_size']} bytes)")
        
        # 6. 시나리오 대표 이미지 조회
        print(f"\n6️⃣ 시나리오 대표 이미지 조회 중...")
        representative_image = await tester.get_scenario_image(scenario_id)
        
        if representative_image:
            print(f"✅ 시나리오 {scenario_id}의 대표 이미지: {representative_image['filename']}")
            print(f"  - S3 URL: {representative_image['s3_url']}")
        
        # 7. 결과 출력
        print(f"\n7️⃣ 업로드 결과:")
        print(f"  - 시나리오 ID: {upload_result['scenario_id']}")
        print(f"  - 원본 파일명: {upload_result['original_filename']}")
        print(f"  - S3 키: {upload_result['stored_filename']}")
        print(f"  - S3 URL: {upload_result['s3_url']}")
        print(f"  - 파일 크기: {upload_result['file_size']} bytes")
        print(f"  - 파일 타입: {upload_result['file_type']}")
        
        # 8. 파일 삭제 (선택사항)
        print(f"\n8️⃣ 테스트 파일 삭제 중...")
        delete_success = await tester.delete_patient_image(
            scenario_id=scenario_id,
            s3_key=upload_result["stored_filename"]
        )
        
        if delete_success:
            print("✅ 테스트 파일 삭제 완료")
        else:
            print("❌ 테스트 파일 삭제 실패")
    
    # 테스트 파일 정리
    try:
        os.remove(test_file_path)
        print(f"\n🧹 로컬 테스트 파일 정리 완료: {test_file_path}")
    except:
        pass
    
    print("\n" + "=" * 50)
    print("✅ 환자 이미지 업로드 테스트 완료!")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())
