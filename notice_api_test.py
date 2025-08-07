#!/usr/bin/env python3
import requests
import json
import time

class NoticeAPITest:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.access_token = None
        self.admin_token = None
        
    def print_test_header(self, test_name):
        print(f"\n{'='*50}")
        print(f"🧪 {test_name}")
        print(f"{'='*50}")
    
    def print_result(self, success, message, response=None):
        if success:
            print(f"✅ {message}")
            if response:
                print(f"📄 응답: {json.dumps(response, indent=2, ensure_ascii=False)}")
        else:
            print(f"❌ {message}")
            if response:
                print(f"📄 응답: {response}")
        print()
    
    def test_1_register_student(self):
        """1. 학생 회원가입 테스트"""
        self.print_test_header("학생 회원가입 테스트")
        
        url = f"{self.base_url}/auth/register"
        data = {
            "username": "student1",
            "password": "password123",
            "email": "student1@example.com",
            "name": "학생1",
            "role": "student",
            "student_id": "2024001",
            "major": "의학과"
        }
        
        try:
            response = requests.post(url, json=data)
            if response.status_code == 200:
                result = response.json()
                self.print_result(True, "학생 회원가입 성공", result)
                return True
            else:
                self.print_result(False, f"회원가입 실패 (상태코드: {response.status_code})", response.text)
                return False
        except Exception as e:
            self.print_result(False, f"회원가입 오류: {str(e)}")
            return False
    
    def test_2_login_student(self):
        """2. 학생 로그인 테스트"""
        self.print_test_header("학생 로그인 테스트")
        
        url = f"{self.base_url}/auth/login"
        data = {
            "username": "student1",
            "password": "password123"
        }
        
        try:
            response = requests.post(url, json=data)
            if response.status_code == 200:
                result = response.json()
                self.access_token = result.get("access_token")
                self.print_result(True, "학생 로그인 성공", {"access_token": "***"})
                return True
            else:
                self.print_result(False, f"로그인 실패 (상태코드: {response.status_code})", response.text)
                return False
        except Exception as e:
            self.print_result(False, f"로그인 오류: {str(e)}")
            return False
    
    def test_3_register_admin(self):
        """3. 관리자 회원가입 테스트"""
        self.print_test_header("관리자 회원가입 테스트")
        
        url = f"{self.base_url}/auth/register"
        data = {
            "username": "admin1",
            "password": "password123",
            "email": "admin1@example.com",
            "name": "관리자1",
            "role": "admin"
        }
        
        try:
            response = requests.post(url, json=data)
            if response.status_code == 200:
                result = response.json()
                self.print_result(True, "관리자 회원가입 성공", result)
                return True
            else:
                self.print_result(False, f"관리자 회원가입 실패 (상태코드: {response.status_code})", response.text)
                return False
        except Exception as e:
            self.print_result(False, f"관리자 회원가입 오류: {str(e)}")
            return False
    
    def test_4_login_admin(self):
        """4. 관리자 로그인 테스트"""
        self.print_test_header("관리자 로그인 테스트")
        
        url = f"{self.base_url}/auth/login"
        data = {
            "username": "admin1",
            "password": "password123"
        }
        
        try:
            response = requests.post(url, json=data)
            if response.status_code == 200:
                result = response.json()
                self.admin_token = result.get("access_token")
                self.print_result(True, "관리자 로그인 성공", {"access_token": "***"})
                return True
            else:
                self.print_result(False, f"관리자 로그인 실패 (상태코드: {response.status_code})", response.text)
                return False
        except Exception as e:
            self.print_result(False, f"관리자 로그인 오류: {str(e)}")
            return False
    
    def test_5_create_notice(self):
        """5. 공지사항 생성 테스트"""
        self.print_test_header("공지사항 생성 테스트")
        
        if not self.admin_token:
            self.print_result(False, "관리자 토큰이 없습니다. 먼저 로그인해주세요.")
            return False
        
        url = f"{self.base_url}/admin/notices/"
        headers = {"Authorization": f"Bearer {self.admin_token}"}
        data = {
            "title": "테스트 공지사항",
            "content": "이것은 테스트용 공지사항입니다. 클릭 테스트를 위해 생성되었습니다.",
            "priority": 1
        }
        
        try:
            response = requests.post(url, json=data, headers=headers)
            if response.status_code == 200:
                result = response.json()
                self.print_result(True, "공지사항 생성 성공", result)
                return True
            else:
                self.print_result(False, f"공지사항 생성 실패 (상태코드: {response.status_code})", response.text)
                return False
        except Exception as e:
            self.print_result(False, f"공지사항 생성 오류: {str(e)}")
            return False
    
    def test_6_get_notices_student(self):
        """6. 학생용 공지사항 목록 조회 테스트"""
        self.print_test_header("학생용 공지사항 목록 조회 테스트")
        
        if not self.access_token:
            self.print_result(False, "학생 토큰이 없습니다. 먼저 로그인해주세요.")
            return False
        
        url = f"{self.base_url}/student/notices/"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                result = response.json()
                self.print_result(True, "학생용 공지사항 목록 조회 성공", result)
                return True
            else:
                self.print_result(False, f"공지사항 목록 조회 실패 (상태코드: {response.status_code})", response.text)
                return False
        except Exception as e:
            self.print_result(False, f"공지사항 목록 조회 오류: {str(e)}")
            return False
    
    def test_7_get_notice_detail(self):
        """7. 특정 공지사항 조회 테스트"""
        self.print_test_header("특정 공지사항 조회 테스트")
        
        if not self.access_token:
            self.print_result(False, "학생 토큰이 없습니다. 먼저 로그인해주세요.")
            return False
        
        url = f"{self.base_url}/student/notices/1"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                result = response.json()
                self.print_result(True, "특정 공지사항 조회 성공", result)
                return True
            else:
                self.print_result(False, f"특정 공지사항 조회 실패 (상태코드: {response.status_code})", response.text)
                return False
        except Exception as e:
            self.print_result(False, f"특정 공지사항 조회 오류: {str(e)}")
            return False
    
    def test_8_increment_view_count(self):
        """8. 공지사항 조회수 증가 테스트"""
        self.print_test_header("공지사항 조회수 증가 테스트")
        
        if not self.access_token:
            self.print_result(False, "학생 토큰이 없습니다. 먼저 로그인해주세요.")
            return False
        
        url = f"{self.base_url}/student/notices/1/view"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        try:
            response = requests.post(url, headers=headers)
            if response.status_code == 200:
                result = response.json()
                self.print_result(True, "공지사항 조회수 증가 성공", result)
                return True
            else:
                self.print_result(False, f"조회수 증가 실패 (상태코드: {response.status_code})", response.text)
                return False
        except Exception as e:
            self.print_result(False, f"조회수 증가 오류: {str(e)}")
            return False
    
    def test_9_get_admin_notices(self):
        """9. 관리자용 공지사항 목록 조회 테스트"""
        self.print_test_header("관리자용 공지사항 목록 조회 테스트")
        
        if not self.admin_token:
            self.print_result(False, "관리자 토큰이 없습니다. 먼저 로그인해주세요.")
            return False
        
        url = f"{self.base_url}/admin/notices/"
        headers = {"Authorization": f"Bearer {self.admin_token}"}
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                result = response.json()
                self.print_result(True, "관리자용 공지사항 목록 조회 성공", result)
                return True
            else:
                self.print_result(False, f"관리자용 공지사항 목록 조회 실패 (상태코드: {response.status_code})", response.text)
                return False
        except Exception as e:
            self.print_result(False, f"관리자용 공지사항 목록 조회 오류: {str(e)}")
            return False
    
    def run_all_tests(self):
        """모든 테스트 실행"""
        print("🎯 공지사항 API 클릭 테스트 시작")
        print("=" * 60)
        
        tests = [
            ("학생 회원가입", self.test_1_register_student),
            ("학생 로그인", self.test_2_login_student),
            ("관리자 회원가입", self.test_3_register_admin),
            ("관리자 로그인", self.test_4_login_admin),
            ("공지사항 생성", self.test_5_create_notice),
            ("학생용 공지사항 목록 조회", self.test_6_get_notices_student),
            ("특정 공지사항 조회", self.test_7_get_notice_detail),
            ("공지사항 조회수 증가", self.test_8_increment_view_count),
            ("관리자용 공지사항 목록 조회", self.test_9_get_admin_notices)
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                success = test_func()
                results.append((test_name, success))
                time.sleep(1)  # 테스트 간 간격
            except Exception as e:
                print(f"❌ {test_name} 테스트 중 오류 발생: {str(e)}")
                results.append((test_name, False))
        
        # 결과 요약
        print("\n" + "=" * 60)
        print("📊 테스트 결과 요약")
        print("=" * 60)
        
        passed = 0
        for test_name, success in results:
            status = "✅ 성공" if success else "❌ 실패"
            print(f"{test_name}: {status}")
            if success:
                passed += 1
        
        print(f"\n총 {len(results)}개 테스트 중 {passed}개 성공, {len(results) - passed}개 실패")
        
        if passed == len(results):
            print("🎉 모든 테스트가 성공했습니다!")
        else:
            print("⚠️  일부 테스트가 실패했습니다.")

if __name__ == "__main__":
    tester = NoticeAPITest()
    tester.run_all_tests()
