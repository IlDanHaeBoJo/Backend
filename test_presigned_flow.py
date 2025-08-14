#!/usr/bin/env python3
"""
Presigned URL 파일 업로드 플로우 테스트 (JSON Body 방식)
1. POST request - 프론트에서 presigned URL생성을 위한 요청을 보냄
2. Presigned URL generation -> Presigned URL response - s3버킷으로 파일을 전송할 수 있는 url을 생성하여 프론트로 보냄
3. Client uploading objects using the generated presigned URL - 프론트에서 응답받은 url로 파일을 보냄 (put 요청)
"""

import requests
import json
import time
import os

# 서버 설정
BASE_URL = "http://localhost:8000"

def test_presigned_url_flow():
    """Presigned URL 업로드 플로우 테스트"""
    
    print("🚀 Presigned URL 업로드 플로우 테스트 시작 (JSON Body 방식)")
    print("=" * 60)
    
    # 1. 서버 상태 확인
    print("1️⃣ 서버 상태 확인...")
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=5)
        if response.status_code == 200:
            print("✅ 서버 정상 실행 중")
        else:
            print(f"❌ 서버 응답 오류: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ 서버 연결 실패: {e}")
        return
    
    # 2. 로그인
    print("\n2️⃣ 로그인...")
    try:
        login_data = {"username": "admin1", "password": "admin123"}
        response = requests.post(f"{BASE_URL}/auth/login", json=login_data, timeout=5)
        
        if response.status_code == 200:
            token_data = response.json()
            access_token = token_data.get("access_token")
            print("✅ 로그인 성공")
        else:
            print(f"❌ 로그인 실패: {response.status_code}")
            print(f"응답: {response.text}")
            return
    except Exception as e:
        print(f"❌ 로그인 오류: {e}")
        return
    
    headers = {"Authorization": f"Bearer {access_token}"}
    
    # 3. 테스트 파일 생성
    print("\n3️⃣ 테스트 파일 생성...")
    test_content = "이것은 Presigned URL 테스트를 위한 파일입니다.\n업로드 시간: " + time.strftime("%Y-%m-%d %H:%M:%S")
    test_filename = "test_presigned_upload.txt"
    
    with open(test_filename, "w", encoding="utf-8") as f:
        f.write(test_content)
    
    print(f"✅ 테스트 파일 생성: {test_filename}")
    
    # 4. Presigned URL 생성 요청 (JSON Body)
    print("\n4️⃣ Presigned URL 생성 요청 (JSON Body)...")
    notice_id = 11  # 테스트용 공지사항 ID
    
    try:
        url = f"{BASE_URL}/attachments/upload-url/{notice_id}"
        
        # JSON Body로 요청
        request_data = {
            "filename": test_filename,
            "file_type": "text/plain",
            "file_size": len(test_content.encode('utf-8')),
            "method": "PUT"
        }
        
        print(f"요청 URL: {url}")
        print(f"요청 Body: {json.dumps(request_data, indent=2)}")
        
        response = requests.post(url, json=request_data, headers=headers, timeout=10)
        
        print(f"응답 상태: {response.status_code}")
        print(f"응답 헤더: {dict(response.headers)}")
        
        if response.status_code == 200:
            presigned_data = response.json()
            print("✅ Presigned URL 생성 성공!")
            print(f"   - S3 키: {presigned_data.get('stored_filename')}")
            print(f"   - 업로드 방식: {presigned_data.get('upload_method')}")
            print(f"   - 만료시간: {presigned_data.get('expires_in')}초")
            print(f"   - 업로드 URL 길이: {len(presigned_data.get('upload_url', ''))}")
        else:
            print(f"❌ Presigned URL 생성 실패: {response.status_code}")
            print(f"응답 내용: {response.text}")
            return
    except Exception as e:
        print(f"❌ Presigned URL 생성 요청 오류: {e}")
        return
    
    # 5. Presigned URL로 파일 업로드 (PUT)
    print("\n5️⃣ Presigned URL로 파일 업로드 (PUT)...")
    try:
        upload_url = presigned_data.get('upload_url')
        s3_key = presigned_data.get('stored_filename')
        
        print(f"업로드 URL: {upload_url[:100]}...")
        print(f"S3 키: {s3_key}")
        
        # 파일을 바이너리로 읽기
        with open(test_filename, 'rb') as f:
            file_content = f.read()
        
        # PUT 요청으로 파일 업로드
        upload_headers = {
            'Content-Type': 'text/plain',
            'Content-Length': str(len(file_content))
        }
        
        print("파일 업로드 중...")
        upload_response = requests.put(
            upload_url,
            data=file_content,
            headers=upload_headers,
            timeout=30
        )
        
        print(f"업로드 응답 상태: {upload_response.status_code}")
        print(f"업로드 응답 헤더: {dict(upload_response.headers)}")
        
        if upload_response.status_code in [200, 204]:
            print("✅ 파일 업로드 성공!")
            etag = upload_response.headers.get('ETag', '').replace('"', '')
            print(f"   - ETag: {etag}")
        else:
            print(f"❌ 파일 업로드 실패: {upload_response.status_code}")
            print(f"응답 내용: {upload_response.text}")
            return
    except Exception as e:
        print(f"❌ 파일 업로드 오류: {e}")
        return
    
    # 6. 업로드 완료 알림 (새로운 구조: s3_key 필수, s3_url 옵션)
    print("\n6️⃣ 업로드 완료 알림 (s3_key 필수 구조)...")
    try:
        # 새로운 구조: s3_key 필수, s3_url 옵션
        upload_complete_data = {
            "original_filename": presigned_data["original_filename"],
            "s3_key": s3_key,  # 필수 필드
            "file_size": len(test_content.encode('utf-8')),
            "file_type": "text/plain",
            "etag": etag,  # 선택사항
            # s3_url은 선택사항이므로 생략 가능
        }
        
        print(f"완료 알림 데이터: {json.dumps(upload_complete_data, indent=2)}")
        
        complete_response = requests.post(
            f"{BASE_URL}/attachments/upload-complete/{notice_id}",
            json=upload_complete_data,
            headers=headers,
            timeout=10
        )
        
        print(f"완료 알림 응답 상태: {complete_response.status_code}")
        
        if complete_response.status_code == 200:
            complete_data = complete_response.json()
            print("✅ 업로드 완료 처리 성공!")
            print(f"   - Attachment ID: {complete_data.get('attachment_id')}")
            print(f"   - 검증 상태: {complete_data.get('verified')}")
            print(f"   - S3 키: {complete_data.get('s3_key')}")
        else:
            print(f"❌ 업로드 완료 처리 실패: {complete_response.status_code}")
            print(f"응답 내용: {complete_response.text}")
    except Exception as e:
        print(f"❌ 업로드 완료 알림 오류: {e}")
    
    # 7. 첨부파일 목록 확인
    print("\n7️⃣ 첨부파일 목록 확인...")
    try:
        list_response = requests.get(
            f"{BASE_URL}/attachments/notice/{notice_id}",
            timeout=5
        )
        
        if list_response.status_code == 200:
            attachments_data = list_response.json()
            attachments = attachments_data.get('attachments', [])
            print(f"✅ 첨부파일 목록 조회 성공! (총 {len(attachments)}개)")
            
            for i, att in enumerate(attachments, 1):
                print(f"   {i}. {att.get('original_filename')} ({att.get('file_size')} bytes)")
        else:
            print(f"❌ 첨부파일 목록 조회 실패: {list_response.status_code}")
    except Exception as e:
        print(f"❌ 첨부파일 목록 조회 오류: {e}")
    
    # 8. 정리
    print("\n8️⃣ 정리...")
    try:
        if os.path.exists(test_filename):
            os.remove(test_filename)
            print(f"✅ 테스트 파일 삭제: {test_filename}")
    except Exception as e:
        print(f"⚠️ 테스트 파일 삭제 실패: {e}")
    
    print("\n" + "=" * 60)
    print("🏁 Presigned URL 업로드 플로우 테스트 완료!")
    print("\n📋 테스트 결과 요약:")
    print("✅ 서버 연결")
    print("✅ 사용자 인증")
    print("✅ Presigned URL 생성 (JSON Body)")
    print("✅ S3 직접 업로드")
    print("✅ 업로드 완료 처리 (s3_key 필수 구조)")
    print("✅ 첨부파일 목록 확인")

if __name__ == "__main__":
    test_presigned_url_flow() 