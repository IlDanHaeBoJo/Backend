#!/usr/bin/env python3
"""
Presigned URL 생성 및 사용 테스트 스크립트
"""

import requests
import json
import os
from datetime import datetime

# 서버 설정
BASE_URL = "http://localhost:8000"

def test_presigned_url_flow():
    """Presigned URL 플로우 테스트"""
    
    print("🧪 Presigned URL 테스트 시작")
    print("=" * 50)
    
    # 1. 로그인
    print("1️⃣ 로그인 중...")
    login_data = {
        "username": "admin1",
        "password": "admin123"
    }
    
    try:
        login_response = requests.post(f"{BASE_URL}/auth/login", json=login_data)
        if login_response.status_code != 200:
            print(f"❌ 로그인 실패: {login_response.status_code}")
            print(login_response.text)
            return
        
        token_data = login_response.json()
        access_token = token_data.get("access_token")
        print("✅ 로그인 성공")
        
    except Exception as e:
        print(f"❌ 로그인 중 오류: {e}")
        return
    
    headers = {"Authorization": f"Bearer {access_token}"}
    
    # 2. 공지사항 생성 (없는 경우)
    print("\n2️⃣ 공지사항 확인/생성...")
    try:
        notices_response = requests.get(f"{BASE_URL}/admin/notices/", headers=headers)
        if notices_response.status_code == 200:
            notices = notices_response.json()
            if len(notices) > 0:
                notice_id = notices[0]['notice_id']
                print(f"✅ 기존 공지사항 사용: ID {notice_id}")
            else:
                # 공지사항 생성
                new_notice = {
                    "title": "Presigned URL 테스트 공지사항",
                    "content": "Presigned URL 테스트를 위한 공지사항입니다.",
                    "important": False,
                    "author_id": 1
                }
                create_response = requests.post(f"{BASE_URL}/admin/notices/", json=new_notice, headers=headers)
                if create_response.status_code == 200:
                    created_notice = create_response.json()
                    notice_id = created_notice['notice_id']
                    print(f"✅ 새 공지사항 생성: ID {notice_id}")
                else:
                    print(f"❌ 공지사항 생성 실패: {create_response.status_code}")
                    return
        else:
            print(f"❌ 공지사항 조회 실패: {notices_response.status_code}")
            return
    except Exception as e:
        print(f"❌ 공지사항 처리 중 오류: {e}")
        return
    
    # 3. Presigned URL 생성 테스트 (PUT 방식)
    print(f"\n3️⃣ Presigned URL 생성 테스트 (PUT 방식)...")
    
    try:
        url_response = requests.post(
            f"{BASE_URL}/attachments/upload-url/{notice_id}?filename=test_document.txt&file_type=text/plain&file_size=1024&method=PUT",
            headers=headers
        )
        
        print(f"URL 생성 응답 상태: {url_response.status_code}")
        print(f"응답 내용: {url_response.text}")
        
        if url_response.status_code == 200:
            url_data = url_response.json()
            print("✅ Presigned URL 생성 성공")
            print(f"   - 원본 파일명: {url_data['original_filename']}")
            print(f"   - S3 키: {url_data['stored_filename']}")
            print(f"   - 만료 시간: {url_data['expires_in']}초")
            print(f"   - S3 URL: {url_data.get('s3_url', 'N/A')}")
            print(f"   - 업로드 방식: {url_data.get('upload_method', 'PUT')}")
            print(f"   - 업로드 URL: {url_data['upload_url'][:100]}...")
            
            # 4. S3 업로드 테스트 (PUT 방식)
            print("\n4️⃣ S3 업로드 테스트 (PUT 방식)...")
            test_content = f"이것은 Presigned URL 테스트 파일입니다.\n생성 시간: {datetime.now()}"
            
            try:
                upload_response = requests.put(
                    url_data["upload_url"],
                    data=test_content.encode('utf-8'),
                    headers={"Content-Type": "text/plain"}
                )
                
                print(f"S3 업로드 응답 상태: {upload_response.status_code}")
                if upload_response.status_code == 200:
                    print("✅ S3 업로드 성공")
                    
                    # 5. 업로드 완료 알림 (S3 파일 존재 확인 포함)
                    print("\n5️⃣ 업로드 완료 알림...")
                    upload_complete_data = {
                        "original_filename": url_data["original_filename"],
                        "s3_url": url_data.get("s3_url", f"https://medicpx.s3.ap-northeast-2.amazonaws.com/{url_data['stored_filename']}"),
                        "file_size": len(test_content.encode('utf-8')),
                        "file_type": "text/plain",
                        "etag": upload_response.headers.get('ETag', '').replace('"', '')  # ETag 포함
                    }
                    
                    complete_response = requests.post(
                        f"{BASE_URL}/attachments/upload-complete/{notice_id}",
                        json=upload_complete_data,
                        headers=headers
                    )
                    
                    print(f"업로드 완료 응답 상태: {complete_response.status_code}")
                    print(f"응답 내용: {complete_response.text}")
                    
                    if complete_response.status_code == 200:
                        print("✅ 업로드 완료 처리 성공")
                        
                        # 6. 최종 확인
                        print("\n6️⃣ 최종 확인...")
                        final_response = requests.get(f"{BASE_URL}/attachments/notice/{notice_id}")
                        if final_response.status_code == 200:
                            final_data = final_response.json()
                            attachments = final_data.get("attachments", [])
                            print(f"✅ 첨부파일 목록 조회 성공: {len(attachments)}개")
                            for att in attachments:
                                print(f"   - {att['original_filename']} ({att['file_size']} bytes)")
                        else:
                            print(f"❌ 첨부파일 목록 조회 실패: {final_response.status_code}")
                    else:
                        print("❌ 업로드 완료 처리 실패")
                else:
                    print("❌ S3 업로드 실패")
                    print(upload_response.text)
            except Exception as e:
                print(f"❌ S3 업로드 중 오류: {e}")
        else:
            print("❌ Presigned URL 생성 실패")
            
    except Exception as e:
        print(f"❌ Presigned URL 생성 중 오류: {e}")
    
    print("\n" + "=" * 50)
    print("🧪 Presigned URL 테스트 완료!")

if __name__ == "__main__":
    test_presigned_url_flow()
