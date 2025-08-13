# Postman API 테스트 가이드

## Presigned URL 파일 업로드 플로우

### 1단계: Presigned URL 생성

**Method:** `POST`  
**URL:** `http://localhost:8000/attachments/upload-url/{notice_id}`

**Headers:**
```
Authorization: Bearer {your_access_token}
Content-Type: application/json
```

**Body (JSON):**
```json
{
  "filename": "test.txt",
  "file_type": "text/plain",
  "file_size": 1024,
  "method": "PUT"
}
```

**응답 예시:**
```json
{
  "notice_id": 11,
  "original_filename": "test.txt",
  "stored_filename": "attachments/uuid.txt",
  "upload_method": "PUT",
  "upload_url": "https://medicpx.s3.ap-northeast-2.amazonaws.com/attachments/uuid.txt?presigned_params...",
  "file_type": "text/plain",
  "file_size": 1024,
  "expires_in": 3600,
  "s3_url": "https://medicpx.s3.ap-northeast-2.amazonaws.com/attachments/uuid.txt",
  "message": "업로드 URL이 생성되었습니다. 이 URL로 PUT 요청을 보내 파일을 업로드하세요."
}
```

### 2단계: S3 직접 업로드

**Method:** `PUT`  
**URL:** `{upload_url_from_step_1}`

**Headers:**
```
Content-Type: text/plain
Content-Length: {file_size}
```

**Body:** 파일 내용 (raw)

**응답:** 200 OK (ETag 헤더 포함)

### 3단계: 업로드 완료 알림 (새로운 구조)

**Method:** `POST`  
**URL:** `http://localhost:8000/attachments/upload-complete/{notice_id}`

**Headers:**
```
Authorization: Bearer {your_access_token}
Content-Type: application/json
```

**Body (JSON) - 새로운 구조:**
```json
{
  "original_filename": "test.txt",
  "s3_key": "attachments/uuid.txt",
  "file_size": 1024,
  "file_type": "text/plain",
  "etag": "your-etag-from-s3",
  "s3_url": "https://medicpx.s3.ap-northeast-2.amazonaws.com/attachments/uuid.txt"
}
```

**또는 s3_url 생략 가능:**
```json
{
  "original_filename": "test.txt",
  "s3_key": "attachments/uuid.txt",
  "file_size": 1024,
  "file_type": "text/plain",
  "etag": "your-etag-from-s3"
}
```

**응답 예시:**
```json
{
  "message": "업로드가 완료되었고 첨부파일이 성공적으로 저장되었습니다.",
  "attachment_id": 1,
  "original_filename": "test.txt",
  "s3_url": "https://medicpx.s3.ap-northeast-2.amazonaws.com/attachments/uuid.txt",
  "s3_key": "attachments/uuid.txt",
  "verified": true
}
```

## 주요 변경사항

### 이전 구조:
- `s3_url` 필수
- URL에서 S3 키 추출

### 새로운 구조:
- `s3_key` 필수 (stored_filename)
- `s3_url` 선택사항 (제공되지 않으면 자동 생성)
- 더 명확하고 안전한 구조

## Postman 설정 순서

1. **1단계 요청 생성**
   - Method: POST
   - URL: `http://localhost:8000/attachments/upload-url/11`
   - Headers: Authorization, Content-Type
   - Body: JSON (위 예시 참조)

2. **2단계 요청 생성**
   - Method: PUT
   - URL: 1단계에서 받은 `upload_url`
   - Headers: Content-Type, Content-Length
   - Body: Raw (파일 내용)

3. **3단계 요청 생성**
   - Method: POST
   - URL: `http://localhost:8000/attachments/upload-complete/11`
   - Headers: Authorization, Content-Type
   - Body: JSON (새로운 구조)

## 테스트 시나리오

### 시나리오 1: s3_url 포함
```json
{
  "original_filename": "test.txt",
  "s3_key": "attachments/uuid.txt",
  "file_size": 1024,
  "file_type": "text/plain",
  "etag": "abc123",
  "s3_url": "https://medicpx.s3.ap-northeast-2.amazonaws.com/attachments/uuid.txt"
}
```

### 시나리오 2: s3_url 생략 (자동 생성)
```json
{
  "original_filename": "test.txt",
  "s3_key": "attachments/uuid.txt",
  "file_size": 1024,
  "file_type": "text/plain",
  "etag": "abc123"
}
```

## 오류 처리

### 422 Unprocessable Content
- 필수 필드 누락 확인
- JSON 형식 확인
- Content-Type 헤더 확인

### 404 Not Found
- notice_id 존재 확인
- S3 파일 존재 확인

### 500 Internal Server Error
- 서버 로그 확인
- AWS 인증 정보 확인
