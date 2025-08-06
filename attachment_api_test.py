import requests
import json
import time

class AttachmentAPITest:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.student_token = None
        self.admin_token = None
        
    def test_1_student_login(self):
        """학생 로그인 테스트"""
        print("=" * 50)
        print("🧪 학생 로그인 테스트")
        print("=" * 50)
        
        url = f"{self.base_url}/auth/login"
        data = {
            "username": "student1",
            "password": "password123"
        }
        
        try:
            response = requests.post(url, json=data)
            print(f"상태코드: {response.status_code}")
            print(f"응답: {response.text}")
            
            if response.status_code == 200:
                result = response.json()
                self.student_token = result.get("access_token")
                print("✅ 학생 로그인 성공")
            else:
                print("❌ 학생 로그인 실패")
                
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
    
    def test_2_admin_login(self):
        """관리자 로그인 테스트"""
        print("=" * 50)
        print("🧪 관리자 로그인 테스트")
        print("=" * 50)
        
        url = f"{self.base_url}/auth/login"
        data = {
            "username": "admin1",
            "password": "password123"
        }
        
        try:
            response = requests.post(url, json=data)
            print(f"상태코드: {response.status_code}")
            print(f"응답: {response.text}")
            
            if response.status_code == 200:
                result = response.json()
                self.admin_token = result.get("access_token")
                print("✅ 관리자 로그인 성공")
            else:
                print("❌ 관리자 로그인 실패")
                
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
    
    def test_3_get_attachments_by_notice(self):
        """공지사항의 첨부파일 목록 조회 테스트"""
        print("=" * 50)
        print("🧪 공지사항의 첨부파일 목록 조회 테스트")
        print("=" * 50)
        
        notice_id = 1  # 첫 번째 공지사항
        url = f"{self.base_url}/attachments/notice/{notice_id}"
        headers = {"Authorization": f"Bearer {self.student_token}"}
        
        try:
            response = requests.get(url, headers=headers)
            print(f"상태코드: {response.status_code}")
            print(f"응답: {response.text}")
            
            if response.status_code == 200:
                print("✅ 첨부파일 목록 조회 성공")
            else:
                print("❌ 첨부파일 목록 조회 실패")
                
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
    
    def test_4_get_attachment_by_id(self):
        """특정 첨부파일 조회 테스트"""
        print("=" * 50)
        print("🧪 특정 첨부파일 조회 테스트")
        print("=" * 50)
        
        attachment_id = 1  # 첫 번째 첨부파일
        url = f"{self.base_url}/attachments/{attachment_id}"
        headers = {"Authorization": f"Bearer {self.student_token}"}
        
        try:
            response = requests.get(url, headers=headers)
            print(f"상태코드: {response.status_code}")
            print(f"응답: {response.text}")
            
            if response.status_code == 200:
                print("✅ 특정 첨부파일 조회 성공")
            else:
                print("❌ 특정 첨부파일 조회 실패")
                
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
    
    def test_5_create_attachment(self):
        """첨부파일 정보 생성 테스트"""
        print("=" * 50)
        print("🧪 첨부파일 정보 생성 테스트")
        print("=" * 50)
        
        url = f"{self.base_url}/attachments/"
        headers = {
            "Authorization": f"Bearer {self.admin_token}",
            "Content-Type": "application/json"
        }
        
        data = {
            "notice_id": 1,
            "original_filename": "test_document.pdf",
            "stored_filename": "test_doc_12345.pdf",
            "file_path": "/uploads/notices/1/test_doc_12345.pdf",
            "file_size": 1024000,
            "file_type": "application/pdf"
        }
        
        try:
            response = requests.post(url, json=data, headers=headers)
            print(f"상태코드: {response.status_code}")
            print(f"응답: {response.text}")
            
            if response.status_code == 200:
                print("✅ 첨부파일 정보 생성 성공")
            else:
                print("❌ 첨부파일 정보 생성 실패")
                
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
    
    def test_6_update_attachment(self):
        """첨부파일 정보 수정 테스트"""
        print("=" * 50)
        print("🧪 첨부파일 정보 수정 테스트")
        print("=" * 50)
        
        attachment_id = 1
        url = f"{self.base_url}/attachments/{attachment_id}"
        headers = {
            "Authorization": f"Bearer {self.admin_token}",
            "Content-Type": "application/json"
        }
        
        data = {
            "original_filename": "updated_document.pdf",
            "file_size": 2048000
        }
        
        try:
            response = requests.put(url, json=data, headers=headers)
            print(f"상태코드: {response.status_code}")
            print(f"응답: {response.text}")
            
            if response.status_code == 200:
                print("✅ 첨부파일 정보 수정 성공")
            else:
                print("❌ 첨부파일 정보 수정 실패")
                
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
    
    def test_7_delete_attachment(self):
        """첨부파일 정보 삭제 테스트"""
        print("=" * 50)
        print("🧪 첨부파일 정보 삭제 테스트")
        print("=" * 50)
        
        attachment_id = 1
        url = f"{self.base_url}/attachments/{attachment_id}"
        headers = {"Authorization": f"Bearer {self.admin_token}"}
        
        try:
            response = requests.delete(url, headers=headers)
            print(f"상태코드: {response.status_code}")
            print(f"응답: {response.text}")
            
            if response.status_code == 200:
                print("✅ 첨부파일 정보 삭제 성공")
            else:
                print("❌ 첨부파일 정보 삭제 실패")
                
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
    
    def test_8_delete_all_attachments_by_notice(self):
        """공지사항의 모든 첨부파일 삭제 테스트"""
        print("=" * 50)
        print("🧪 공지사항의 모든 첨부파일 삭제 테스트")
        print("=" * 50)
        
        notice_id = 1
        url = f"{self.base_url}/attachments/notice/{notice_id}/all"
        headers = {"Authorization": f"Bearer {self.admin_token}"}
        
        try:
            response = requests.delete(url, headers=headers)
            print(f"상태코드: {response.status_code}")
            print(f"응답: {response.text}")
            
            if response.status_code == 200:
                print("✅ 모든 첨부파일 삭제 성공")
            else:
                print("❌ 모든 첨부파일 삭제 실패")
                
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
    
    def run_all_tests(self):
        """모든 테스트 실행"""
        print("🎯 첨부파일 API 클릭 테스트 시작")
        print("=" * 60)
        
        # 로그인 테스트
        self.test_1_student_login()
        self.test_2_admin_login()
        
        # 첨부파일 조회 테스트
        self.test_3_get_attachments_by_notice()
        self.test_4_get_attachment_by_id()
        
        # 첨부파일 관리 테스트 (관리자만)
        self.test_5_create_attachment()
        self.test_6_update_attachment()
        self.test_7_delete_attachment()
        self.test_8_delete_all_attachments_by_notice()
        
        print("=" * 60)
        print("📊 첨부파일 API 테스트 완료")
        print("=" * 60)

if __name__ == "__main__":
    tester = AttachmentAPITest()
    tester.run_all_tests()
