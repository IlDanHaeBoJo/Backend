"""
공지사항 API 테스트 스크립트
"""
import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"

def test_student_notices():
    """학생용 공지사항 API 테스트"""
    print("=== 학생용 공지사항 API 테스트 ===")
    
    # 1. 모든 공지사항 조회
    print("\n1. 모든 공지사항 조회")
    response = requests.get(f"{BASE_URL}/student/notices/")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        notices = response.json()
        print(f"공지사항 개수: {len(notices)}")
        for notice in notices:
            print(f"  - {notice['title']} (우선순위: {notice['priority']})")
    
    # 2. 높은 우선순위 공지사항 조회
    print("\n2. 높은 우선순위 공지사항 조회")
    response = requests.get(f"{BASE_URL}/student/notices/priority/")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        notices = response.json()
        print(f"높은 우선순위 공지사항 개수: {len(notices)}")
        for notice in notices:
            print(f"  - {notice['title']} (우선순위: {notice['priority']})")
    
    # 3. 최근 공지사항 조회
    print("\n3. 최근 공지사항 조회")
    response = requests.get(f"{BASE_URL}/student/notices/recent/")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        notices = response.json()
        print(f"최근 공지사항 개수: {len(notices)}")
        for notice in notices:
            print(f"  - {notice['title']} (생성일: {notice['created_at']})")

def test_admin_notices():
    """관리자용 공지사항 API 테스트"""
    print("\n=== 관리자용 공지사항 API 테스트 ===")
    
    # 1. 관리자 로그인
    print("\n1. 관리자 로그인")
    login_data = {
        "username": "admin",
        "password": "admin123"
    }
    response = requests.post(f"{BASE_URL}/auth/login", json=login_data)
    print(f"Login Status: {response.status_code}")
    
    if response.status_code == 200:
        token_data = response.json()
        access_token = token_data.get("access_token")
        print("로그인 성공")
        
        headers = {"Authorization": f"Bearer {access_token}"}
        
        # 2. 모든 공지사항 조회 (관리자용)
        print("\n2. 모든 공지사항 조회 (관리자용)")
        response = requests.get(f"{BASE_URL}/admin/notices/", headers=headers)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            notices = response.json()
            print(f"공지사항 개수: {len(notices)}")
            for notice in notices:
                print(f"  - {notice['title']} (우선순위: {notice['priority']})")
        
        # 3. 새 공지사항 생성
        print("\n3. 새 공지사항 생성")
        new_notice = {
            "title": "테스트 공지사항",
            "content": "이것은 테스트용 공지사항입니다.",
            "priority": 20,
            "author_id": 1
        }
        response = requests.post(f"{BASE_URL}/admin/notices/", json=new_notice, headers=headers)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            created_notice = response.json()
            print(f"생성된 공지사항: {created_notice['title']}")
            
            # 4. 공지사항 수정
            print("\n4. 공지사항 수정")
            notice_id = created_notice["notice_id"]
            update_data = {
                "title": "수정된 테스트 공지사항",
                "priority": 25
            }
            response = requests.put(f"{BASE_URL}/admin/notices/{notice_id}", json=update_data, headers=headers)
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                updated_notice = response.json()
                print(f"수정된 공지사항: {updated_notice['title']}")
            
            # 5. 공지사항 삭제
            print("\n5. 공지사항 삭제")
            response = requests.delete(f"{BASE_URL}/admin/notices/{notice_id}", headers=headers)
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"삭제 결과: {result['message']}")
        
        # 6. 통계 조회
        print("\n6. 공지사항 통계")
        response = requests.get(f"{BASE_URL}/admin/notices/stats/summary", headers=headers)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            stats = response.json()
            print(f"총 공지사항: {stats['total_notices']}")
            print(f"총 조회수: {stats['total_views']}")
            print(f"평균 우선순위: {stats['average_priority']}")
            print(f"높은 우선순위 공지사항: {stats['high_priority_count']}")
            print(f"최근 7일 공지사항: {stats['recent_notices_count']}")
        
        # 7. 검색 기능
        print("\n7. 공지사항 검색")
        response = requests.get(f"{BASE_URL}/admin/notices/search/?keyword=CPX&search_type=title", headers=headers)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            search_results = response.json()
            print(f"검색 결과 개수: {len(search_results)}")
            for result in search_results:
                print(f"  - {result['title']}")
    else:
        print("로그인 실패")

def main():
    """메인 테스트 함수"""
    print("공지사항 기능 테스트 시작...")
    
    try:
        test_student_notices()
        test_admin_notices()
        print("\n✅ 모든 테스트 완료!")
    except requests.exceptions.ConnectionError:
        print("❌ 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")

if __name__ == "__main__":
    main()