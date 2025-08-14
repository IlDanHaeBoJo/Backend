#!/usr/bin/env python3
"""
JSON Body 방식 Presigned URL 생성 테스트
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_json_body():
    print("🔍 JSON Body 방식 테스트")
    print("=" * 40)
    
    # 1. 로그인
    print("1️⃣ 로그인...")
    login_data = {"username": "admin1", "password": "admin123"}
    response = requests.post(f"{BASE_URL}/auth/login", json=login_data)
    
    if response.status_code != 200:
        print(f"❌ 로그인 실패: {response.status_code}")
        print(response.text)
        return
    
    token_data = response.json()
    access_token = token_data.get("access_token")
    print("✅ 로그인 성공")
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    # 2. Presigned URL 생성 요청
    print("\n2️⃣ Presigned URL 생성 요청...")
    
    request_data = {
        "filename": "test.txt",
        "file_type": "text/plain",
        "file_size": 1024,
        "method": "PUT"
    }
    
    print(f"요청 URL: {BASE_URL}/attachments/upload-url/11")
    print(f"요청 Body: {json.dumps(request_data, indent=2)}")
    print(f"Headers: {headers}")
    
    response = requests.post(
        f"{BASE_URL}/attachments/upload-url/11",
        json=request_data,
        headers=headers
    )
    
    print(f"\n응답 상태: {response.status_code}")
    print(f"응답 헤더: {dict(response.headers)}")
    
    if response.status_code == 200:
        print("✅ 성공!")
        print(f"응답: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    else:
        print("❌ 실패!")
        print(f"응답 내용: {response.text}")
        
        # 422 오류인 경우 상세 정보 출력
        if response.status_code == 422:
            try:
                error_detail = response.json()
                print(f"검증 오류 상세: {json.dumps(error_detail, indent=2, ensure_ascii=False)}")
            except:
                pass

if __name__ == "__main__":
    test_json_body()
