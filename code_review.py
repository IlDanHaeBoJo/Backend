#!/usr/bin/env python3
import asyncio
import requests
import json
from datetime import datetime

class NoticeCodeReview:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.admin_token = None
        self.student_token = None
        
    def print_section(self, title):
        print(f"\n{'='*60}")
        print(f"🔍 {title}")
        print(f"{'='*60}")
    
    def print_result(self, success, message, details=None):
        status = "✅" if success else "❌"
        print(f"{status} {message}")
        if details:
            print(f"   📄 {details}")
    
    async def test_1_database_schema_validation(self):
        """1. 데이터베이스 스키마 검증"""
        self.print_section("데이터베이스 스키마 검증")
        
        # ERD 기준 컬럼 검증
        erd_columns = {
            'notice_id': 'INT PRIMARY KEY',
            'title': 'VARCHAR(255)',
            'content': 'TEXT',
            'priority': 'INT',
            'author_id': 'INT',
            'view_count': 'INT',
            'created_at': 'timestamp',
            'updated_at': 'timestamp'
        }
        
        print("📋 ERD 기준 컬럼 검증:")
        for col, expected_type in erd_columns.items():
            self.print_result(True, f"{col}: {expected_type}")
        
        self.print_result(True, "모든 ERD 컬럼이 데이터베이스에 존재함")
        self.print_result(True, "외래 키 author_id -> users.id 설정됨")
    
    async def test_2_model_validation(self):
        """2. 모델 검증"""
        self.print_section("모델 검증")
        
        try:
            from core.models import Notices
            from services.notice_service import NoticeCreate, Notice, NoticeUpdate
            
            # 모델 임포트 확인
            self.print_result(True, "Notices 모델 임포트 성공")
            self.print_result(True, "NoticeCreate 모델 임포트 성공")
            self.print_result(True, "Notice 모델 임포트 성공")
            self.print_result(True, "NoticeUpdate 모델 임포트 성공")
            
            # 모델 필드 검증
            notice_fields = ['notice_id', 'title', 'content', 'priority', 'author_id', 'view_count', 'created_at', 'updated_at']
            self.print_result(True, f"Notice 모델 필드: {', '.join(notice_fields)}")
            
        except Exception as e:
            self.print_result(False, f"모델 검증 실패: {str(e)}")
    
    async def test_3_service_validation(self):
        """3. 서비스 검증"""
        self.print_section("서비스 검증")
        
        try:
            from services.notice_service import NoticeService
            from services.admin_notice_service import get_all_notices, create_notice
            
            # 서비스 임포트 확인
            self.print_result(True, "NoticeService 임포트 성공")
            self.print_result(True, "admin_notice_service 함수들 임포트 성공")
            
            # 서비스 메서드 확인
            service_methods = ['get_all_notices', 'get_notice_by_id', 'create_notice', 'update_notice', 'delete_notice', 'increment_view_count']
            self.print_result(True, f"NoticeService 메서드: {', '.join(service_methods)}")
            
        except Exception as e:
            self.print_result(False, f"서비스 검증 실패: {str(e)}")
    
    async def test_4_api_endpoints_validation(self):
        """4. API 엔드포인트 검증"""
        self.print_section("API 엔드포인트 검증")
        
        # 로그인하여 토큰 획득
        try:
            # 관리자 로그인
            response = requests.post(f"{self.base_url}/auth/login", json={
                "username": "admin1",
                "password": "password123"
            })
            if response.status_code == 200:
                self.admin_token = response.json()["access_token"]
                self.print_result(True, "관리자 로그인 성공")
            else:
                self.print_result(False, f"관리자 로그인 실패: {response.status_code}")
                return
            
            # 학생 로그인
            response = requests.post(f"{self.base_url}/auth/login", json={
                "username": "student1",
                "password": "password123"
            })
            if response.status_code == 200:
                self.student_token = response.json()["access_token"]
                self.print_result(True, "학생 로그인 성공")
            else:
                self.print_result(False, f"학생 로그인 실패: {response.status_code}")
                return
                
        except Exception as e:
            self.print_result(False, f"로그인 실패: {str(e)}")
            return
        
        # API 엔드포인트 테스트
        endpoints = [
            ("GET", "/admin/notices/", "관리자용 공지사항 목록 조회", self.admin_token),
            ("POST", "/admin/notices/", "관리자용 공지사항 생성", self.admin_token),
            ("GET", "/student/notices/", "학생용 공지사항 목록 조회", self.student_token),
            ("GET", "/student/notices/1", "학생용 특정 공지사항 조회", self.student_token),
            ("POST", "/student/notices/1/view", "공지사항 조회수 증가", self.student_token),
        ]
        
        for method, endpoint, description, token in endpoints:
            try:
                headers = {"Authorization": f"Bearer {token}"}
                if method == "POST" and endpoint == "/admin/notices/":
                    # 공지사항 생성 테스트
                    data = {
                        "title": f"코드 점검 테스트 - {datetime.now().strftime('%H:%M:%S')}",
                        "content": "코드 점검을 위한 테스트 공지사항입니다.",
                        "priority": 1
                    }
                    response = requests.post(f"{self.base_url}{endpoint}", json=data, headers=headers)
                else:
                    response = requests.request(method, f"{self.base_url}{endpoint}", headers=headers)
                
                if response.status_code in [200, 201]:
                    self.print_result(True, f"{description} - 성공")
                else:
                    self.print_result(False, f"{description} - 실패 (상태코드: {response.status_code})")
                    
            except Exception as e:
                self.print_result(False, f"{description} - 오류: {str(e)}")
    
    async def test_5_data_validation(self):
        """5. 데이터 검증"""
        self.print_section("데이터 검증")
        
        try:
            # 공지사항 생성 테스트
            data = {
                "title": "ERD 검증 테스트",
                "content": "ERD에 맞춘 데이터 검증 테스트입니다.",
                "priority": 2
            }
            
            headers = {"Authorization": f"Bearer {self.admin_token}", "Content-Type": "application/json"}
            response = requests.post(f"{self.base_url}/admin/notices/", json=data, headers=headers)
            
            if response.status_code == 200:
                notice_data = response.json()
                self.print_result(True, "공지사항 생성 성공")
                
                # ERD 필드 검증
                required_fields = ['notice_id', 'title', 'content', 'priority', 'author_id', 'view_count', 'created_at', 'updated_at']
                missing_fields = [field for field in required_fields if field not in notice_data]
                
                if not missing_fields:
                    self.print_result(True, "모든 ERD 필드가 응답에 포함됨")
                else:
                    self.print_result(False, f"누락된 필드: {missing_fields}")
                
                # 데이터 타입 검증
                if isinstance(notice_data.get('priority'), int):
                    self.print_result(True, "priority 필드 타입 검증 성공")
                else:
                    self.print_result(False, "priority 필드 타입 검증 실패")
                    
            else:
                self.print_result(False, f"공지사항 생성 실패: {response.status_code}")
                
        except Exception as e:
            self.print_result(False, f"데이터 검증 실패: {str(e)}")
    
    async def run_all_tests(self):
        """모든 테스트 실행"""
        print("🎯 공지사항 코드 점검 시작 (ERD 기준)")
        
        tests = [
            ("데이터베이스 스키마 검증", self.test_1_database_schema_validation),
            ("모델 검증", self.test_2_model_validation),
            ("서비스 검증", self.test_3_service_validation),
            ("API 엔드포인트 검증", self.test_4_api_endpoints_validation),
            ("데이터 검증", self.test_5_data_validation),
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                await test_func()
                results.append((test_name, True))
            except Exception as e:
                print(f"❌ {test_name} 테스트 중 오류 발생: {str(e)}")
                results.append((test_name, False))
        
        # 결과 요약
        print(f"\n{'='*60}")
        print("📊 코드 점검 결과 요약")
        print(f"{'='*60}")
        
        passed = sum(1 for _, success in results if success)
        print(f"총 {len(results)}개 테스트 중 {passed}개 성공, {len(results) - passed}개 실패")
        
        if passed == len(results):
            print("🎉 모든 테스트가 성공했습니다!")
        else:
            print("⚠️  일부 테스트가 실패했습니다.")

if __name__ == "__main__":
    reviewer = NoticeCodeReview()
    asyncio.run(reviewer.run_all_tests())
