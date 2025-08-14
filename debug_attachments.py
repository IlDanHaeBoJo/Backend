#!/usr/bin/env python3
"""
첨부파일 업로드 문제 디버깅 스크립트
"""

import requests
import json
import os

# 서버 설정
BASE_URL = "http://localhost:8000"

def debug_attachments():
    """첨부파일 업로드 문제 디버깅"""
    
    print("🔍 첨부파일 업로드 문제 디버깅 시작")
    print("=" * 50)
    
    # 1. 로그인
    print("1️⃣ 로그인 중...")
    login_data = {
        "username": "admin",  # 실제 계정으로 변경
        "password": "admin123"  # 실제 비밀번호로 변경
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
    
    # 2. 공지사항 목록 확인
    print("\n2️⃣ 공지사항 목록 확인...")
    try:
        notices_response = requests.get(f"{BASE_URL}/admin/notices/", headers=headers)
        if notices_response.status_code == 200:
            notices = notices_response.json()
            print(f"✅ 공지사항 개수: {len(notices)}")
            for notice in notices:
                print(f"   - ID: {notice['notice_id']}, 제목: {notice['title']}")
                if 'attachments' in notice:
                    print(f"     첨부파일: {len(notice['attachments'])}개")
        else:
            print(f"❌ 공지사항 조회 실패: {notices_response.status_code}")
            print(notices_response.text)
    except Exception as e:
        print(f"❌ 공지사항 조회 중 오류: {e}")
    
    # 3. 공지사항이 없으면 생성
    if len(notices) == 0:
        print("\n3️⃣ 테스트용 공지사항 생성...")
        new_notice = {
            "title": "첨부파일 테스트 공지사항",
            "content": "이 공지사항은 첨부파일 업로드 테스트를 위한 것입니다.",
            "important": False,
            "author_id": 1
        }
        
        try:
            create_response = requests.post(f"{BASE_URL}/admin/notices/", json=new_notice, headers=headers)
            if create_response.status_code == 200:
                created_notice = create_response.json()
                notice_id = created_notice['notice_id']
                print(f"✅ 공지사항 생성 성공: ID {notice_id}")
            else:
                print(f"❌ 공지사항 생성 실패: {create_response.status_code}")
                print(create_response.text)
                return
        except Exception as e:
            print(f"❌ 공지사항 생성 중 오류: {e}")
            return
    else:
        notice_id = notices[0]['notice_id']
        print(f"\n3️⃣ 기존 공지사항 사용: ID {notice_id}")
    
    # 4. 업로드 URL 생성 테스트
    print(f"\n4️⃣ 업로드 URL 생성 테스트 (공지사항 ID: {notice_id})...")
    upload_url_data = {
        "filename": "test_document.txt",
        "file_type": "text/plain",
        "file_size": 1024
    }
    
    try:
        url_response = requests.post(
            f"{BASE_URL}/attachments/upload-url/{notice_id}",
            json=upload_url_data,
            headers=headers
        )
        
        print(f"URL 생성 응답 상태: {url_response.status_code}")
        print(f"응답 내용: {url_response.text}")
        
        if url_response.status_code == 200:
            url_data = url_response.json()
            print("✅ 업로드 URL 생성 성공")
            print(f"   - 원본 파일명: {url_data['original_filename']}")
            print(f"   - S3 키: {url_data['stored_filename']}")
            print(f"   - 업로드 URL: {url_data['upload_url'][:100]}...")
            
            # 5. S3 업로드 테스트
            print("\n5️⃣ S3 업로드 테스트...")
            test_content = "이것은 테스트 파일입니다."
            
            try:
                upload_response = requests.put(
                    url_data["upload_url"],
                    data=test_content.encode('utf-8'),
                    headers={"Content-Type": "text/plain"}
                )
                
                print(f"S3 업로드 응답 상태: {upload_response.status_code}")
                if upload_response.status_code == 200:
                    print("✅ S3 업로드 성공")
                    
                    # 6. 첨부파일 정보 생성
                    print("\n6️⃣ 첨부파일 정보 생성...")
                    attachment_data = {
                        "original_filename": url_data["original_filename"],
                        "s3_url": f"https://cpx-attachments.s3.ap-northeast-2.amazonaws.com/{url_data['stored_filename']}",
                        "file_size": url_data["file_size"],
                        "file_type": url_data["file_type"]
                    }
                    
                    create_response = requests.post(
                        f"{BASE_URL}/attachments/create/{notice_id}",
                        json=attachment_data,
                        headers=headers
                    )
                    
                    print(f"첨부파일 생성 응답 상태: {create_response.status_code}")
                    print(f"응답 내용: {create_response.text}")
                    
                    if create_response.status_code == 200:
                        print("✅ 첨부파일 생성 성공")
                        
                        # 7. 최종 확인
                        print("\n7️⃣ 최종 확인...")
                        final_response = requests.get(f"{BASE_URL}/admin/notices/", headers=headers)
                        if final_response.status_code == 200:
                            final_notices = final_response.json()
                            for notice in final_notices:
                                if notice['notice_id'] == notice_id:
                                    print(f"공지사항 제목: {notice['title']}")
                                    if 'attachments' in notice:
                                        print(f"첨부파일 개수: {len(notice['attachments'])}")
                                        for att in notice['attachments']:
                                            print(f"  - {att['original_filename']} ({att['file_size']} bytes)")
                                    else:
                                        print("❌ 첨부파일 정보가 없습니다!")
                    else:
                        print("❌ 첨부파일 생성 실패")
                else:
                    print("❌ S3 업로드 실패")
                    print(upload_response.text)
            except Exception as e:
                print(f"❌ S3 업로드 중 오류: {e}")
        else:
            print("❌ 업로드 URL 생성 실패")
            
    except Exception as e:
        print(f"❌ 업로드 URL 생성 중 오류: {e}")
    
    print("\n" + "=" * 50)
    print("🔍 디버깅 완료!")

if __name__ == "__main__":
    debug_attachments()

