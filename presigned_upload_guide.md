# Presigned URL 방식 S3 업로드 가이드

## 개요

이 시스템은 **Presigned URL 방식**을 사용하여 S3에 파일을 업로드합니다. 백엔드 프록시 방식을 제거하고, 프론트엔드에서 직접 S3로 업로드하는 방식을 채택했습니다.

## 업로드 플로우

### 1. 업로드 URL 생성
```
POST /attachments/upload-url/{notice_id}
```

**요청 본문:**
```json
{
  "filename": "document.pdf",
  "file_type": "application/pdf",
  "file_size": 1048576
}
```

**응답:**
```json
{
  "notice_id": 1,
  "original_filename": "document.pdf",
  "stored_filename": "attachments/uuid-filename.pdf",
  "upload_url": "https://s3.amazonaws.com/bucket/attachments/uuid-filename.pdf?...",
  "file_type": "application/pdf",
  "file_size": 1048576,
  "expires_in": 3600,
  "message": "업로드 URL이 생성되었습니다. 이 URL로 PUT 요청을 보내 파일을 업로드하세요."
}
```

### 2. S3에 직접 업로드 (PUT 요청)
```javascript
// 프론트엔드에서 실행
const response = await fetch(upload_url, {
  method: 'PUT',
  body: file,
  headers: {
    'Content-Type': file_type
  }
});

if (response.ok) {
  console.log('S3 업로드 성공');
} else {
  console.error('S3 업로드 실패');
}
```

### 3. 첨부파일 정보 생성
```
POST /attachments/create/{notice_id}
```

**요청 본문:**
```json
{
  "original_filename": "document.pdf",
  "s3_url": "https://bucket.s3.region.amazonaws.com/attachments/uuid-filename.pdf",
  "file_size": 1048576,
  "file_type": "application/pdf"
}
```

## 장점

1. **성능**: 서버를 거치지 않고 직접 S3로 업로드
2. **확장성**: 서버 리소스 사용량 감소
3. **안정성**: 대용량 파일 업로드 시 타임아웃 위험 감소

## 보안

- 업로드 URL은 **1시간 후 만료**
- 파일 크기 및 타입 검증
- 관리자 권한 확인
- 고유한 S3 키 생성 (UUID 사용)

## 에러 처리

### 일반적인 오류

1. **401 Unauthorized**: JWT 토큰이 유효하지 않음
2. **403 Forbidden**: 관리자 권한 없음
3. **404 Not Found**: 공지사항이 존재하지 않음
4. **413 Payload Too Large**: 파일 크기 초과
5. **415 Unsupported Media Type**: 지원하지 않는 파일 타입

### S3 관련 오류

1. **AWS 인증 실패**: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY 설정 확인
2. **버킷 접근 실패**: S3_BUCKET_NAME, AWS_REGION 설정 확인
3. **업로드 URL 만료**: 1시간 내에 업로드 완료 필요

## 테스트

테스트 스크립트 실행:
```bash
python test_presigned_upload.py
```

## 환경 변수 설정

```bash
# AWS S3 설정
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=ap-northeast-2
S3_BUCKET_NAME=cpx-attachments

# 서버 설정
HOST=0.0.0.0
PORT=8000
DEBUG=true
```

## 프론트엔드 구현 예시

```javascript
async function uploadFile(noticeId, file) {
  try {
    // 1. 업로드 URL 생성
    const urlResponse = await fetch(`/attachments/upload-url/${noticeId}`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        filename: file.name,
        file_type: file.type,
        file_size: file.size
      })
    });
    
    const urlData = await urlResponse.json();
    
    // 2. S3에 직접 업로드
    const uploadResponse = await fetch(urlData.upload_url, {
      method: 'PUT',
      body: file,
      headers: {
        'Content-Type': file.type
      }
    });
    
    if (!uploadResponse.ok) {
      throw new Error('S3 업로드 실패');
    }
    
    // 3. 첨부파일 정보 생성
    const createResponse = await fetch(`/attachments/create/${noticeId}`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        original_filename: file.name,
        s3_url: urlData.s3_url,
        file_size: file.size,
        file_type: file.type
      })
    });
    
    const result = await createResponse.json();
    console.log('업로드 완료:', result);
    
  } catch (error) {
    console.error('업로드 실패:', error);
  }
}
```
