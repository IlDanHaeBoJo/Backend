#!/usr/bin/env python3
"""
10초 자가 테스트 - Presigned URL 생성 테스트
"""

import requests
import time
import os

# 서버 설정
BASE_URL = "http://localhost:8000"

def quick_test():
    """10초 자가 테스트"""
    
    print("🚀 10초 자가 테스트 시작")
    print("=" * 40)
    
    start_time = time.time()
    
    # 1. 서버 상태 확인 (2초)
    print("1️⃣ 서버 상태 확인...")
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=2)
        if response.status_code == 200:
            print("✅ 서버 정상 실행 중")
        else:
            print(f"⚠️ 서버 응답: {response.status_code}")
    except Exception as e:
        print(f"❌ 서버 연결 실패: {e}")
        return
    
    # 2. 로그인 (3초)
    print("\n2️⃣ 로그인...")
    try:
        login_data = {"username": "admin1", "password": "admin123"}
        response = requests.post(f"{BASE_URL}/auth/login", json=login_data, timeout=3)
        
        if response.status_code == 200:
            token_data = response.json()
            access_token = token_data.get("access_token")
            print("✅ 로그인 성공")
        else:
            print(f"❌ 로그인 실패: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ 로그인 오류: {e}")
        return
    
    headers = {"Authorization": f"Bearer {access_token}"}
    
    # 3. Presigned URL 생성 테스트 (5초)
    print("\n3️⃣ Presigned URL 생성 테스트...")
    try:
        url = f"{BASE_URL}/attachments/upload-url/11?filename=test.txt&file_type=text/plain&file_size=1024&method=PUT"
        response = requests.post(url, headers=headers, timeout=5)
        
        print(f"응답 상태: {response.status_code}")
        print(f"응답 내용: {response.text[:200]}...")
        
        if response.status_code == 200:
            print("✅ Presigned URL 생성 성공!")
            data = response.json()
            print(f"   - S3 키: {data.get('stored_filename', 'N/A')}")
            print(f"   - 업로드 방식: {data.get('upload_method', 'N/A')}")
        else:
            print("❌ Presigned URL 생성 실패")
            
    except Exception as e:
        print(f"❌ Presigned URL 테스트 오류: {e}")
    
    # 총 소요 시간 계산
    elapsed_time = time.time() - start_time
    print(f"\n⏱️ 총 소요 시간: {elapsed_time:.1f}초")
    print("=" * 40)
    print("🏁 테스트 완료!")

if __name__ == "__main__":
    quick_test()

