# Postman API 테스트 가이드

## 서버 정보
- **Base URL**: `http://localhost:8000`
- **서버 상태**: 실행 중

## 1. 사용자 인증 API

### 1.1 사용자 등록
```
POST /auth/register
Content-Type: application/json

{
  "username": "professor@test.com",
  "password": "testpassword123",
  "email": "professor@test.com",
  "name": "테스트 교수",
  "role": "admin",
  "major": "컴퓨터공학과"
}
```

### 1.2 로그인
```
POST /auth/login
Content-Type: application/json

{
  "username": "professor@test.com",
  "password": "testpassword123"
}
```

**응답 예시:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "refresh_token": "refresh_token_here"
}
```

## 2. 공지사항 API (관리자용)

### 2.1 공지사항 생성
```
POST /admin/notices
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "title": "첨부파일 테스트 공지사항",
  "content": "이 공지사항은 첨부파일 API 테스트를 위한 것입니다.",
  "important": false,
  "author_id": 1
}
```

### 2.2 공지사항 조회
```
GET /admin/notices/{notice_id}
Authorization: Bearer {access_token}
```

### 2.3 공지사항 목록 조회
```
GET /admin/notices/
Authorization: Bearer {access_token}
```

### 2.4 공지사항 삭제
```
DELETE /admin/notices/{notice_id}
Authorization: Bearer {access_token}
```

## 3. 첨부파일 API

### 3.1 파일 업로드 (S3)
```
POST /attachments/upload/{notice_id}
Authorization: Bearer {access_token}
Content-Type: multipart/form-data

Body (form-data):
- file: [파일 선택]
```

### 3.2 S3 URL로 첨부파일 생성
```
POST /attachments/create/{notice_id}
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "original_filename": "test_document.pdf",
  "s3_url": "https://cpx-attachments.s3.ap-northeast-2.amazonaws.com/attachments/uuid.pdf",
  "file_size": 1024,
  "file_type": "application/pdf"
}
```

### 3.3 첨부파일 목록 조회
```
GET /attachments/notice/{notice_id}
```

### 3.4 첨부파일 정보 조회
```
GET /attachments/{attachment_id}/info
```

### 3.5 첨부파일 다운로드 URL 조회
```
GET /attachments/download/{attachment_id}
```

### 3.6 첨부파일 삭제
```
DELETE /attachments/{attachment_id}
Authorization: Bearer {access_token}
```

### 3.7 공지사항의 모든 첨부파일 삭제
```
DELETE /attachments/notice/{notice_id}/all
Authorization: Bearer {access_token}
```

## 4. 테스트 시나리오

### 시나리오 1: 기본 첨부파일 업로드
1. 사용자 등록/로그인
2. 공지사항 생성
3. 파일 업로드
4. 첨부파일 목록 조회
5. 첨부파일 정보 조회
6. 다운로드 URL 조회
7. 첨부파일 삭제
8. 공지사항 삭제

### 시나리오 2: S3 URL 기반 첨부파일 생성
1. 사용자 등록/로그인
2. 공지사항 생성
3. S3 URL로 첨부파일 생성
4. 첨부파일 목록 조회
5. 첨부파일 삭제
6. 공지사항 삭제

## 5. 환경 변수 설정

Postman에서 다음 환경 변수를 설정하세요:

```
BASE_URL: http://localhost:8000
ACCESS_TOKEN: (로그인 후 받은 토큰)
NOTICE_ID: (생성된 공지사항 ID)
ATTACHMENT_ID: (생성된 첨부파일 ID)
```

## 6. 주의사항

1. **권한**: 첨부파일 업로드/삭제는 `admin` 또는 `교수` 역할이 필요합니다.
2. **파일 크기**: 최대 10MB까지 업로드 가능합니다.
3. **파일 타입**: PDF, Word, Excel, PowerPoint, 이미지 파일 등이 지원됩니다.
4. **S3 설정**: AWS S3 인증 정보가 필요합니다 (환경 변수에서 설정).

## 7. 지원되는 파일 타입

- `application/pdf`
- `application/msword`
- `application/vnd.openxmlformats-officedocument.wordprocessingml.document`
- `application/vnd.ms-excel`
- `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`
- `application/vnd.ms-powerpoint`
- `application/vnd.openxmlformats-officedocument.presentationml.presentation`
- `image/jpeg`
- `image/png`
- `image/gif`
- `text/plain`

## 8. 에러 코드

- `400`: 잘못된 요청 (파일 크기 초과, 지원하지 않는 파일 타입)
- `401`: 인증 실패
- `403`: 권한 없음
- `404`: 리소스를 찾을 수 없음
- `500`: 서버 내부 오류
