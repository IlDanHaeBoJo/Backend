# S3 CORS 설정 가이드

## 개요

Presigned URL 방식으로 S3에 직접 업로드할 때 CORS(Cross-Origin Resource Sharing) 설정이 필요합니다. 프론트엔드에서 브라우저를 통해 S3에 직접 요청할 때 CORS 정책 위반으로 인한 오류를 방지합니다.

## 자동 CORS 설정

현재 시스템에서는 S3 서비스 초기화 시 자동으로 CORS 설정을 적용합니다:

```python
def _setup_cors(self):
    """S3 버킷 CORS 설정"""
    cors_configuration = {
        'CORSRules': [
            {
                'AllowedHeaders': ['*'],
                'AllowedMethods': ['GET', 'PUT', 'POST', 'DELETE', 'HEAD'],
                'AllowedOrigins': ['*'],  # 프로덕션에서는 특정 도메인으로 제한
                'ExposeHeaders': ['ETag', 'Content-Length'],
                'MaxAgeSeconds': 3000
            }
        ]
    }
```

## 수동 CORS 설정 (AWS 콘솔)

### 1. AWS S3 콘솔 접속
- AWS Management Console → S3
- 해당 버킷 선택

### 2. 권한 탭에서 CORS 설정
```json
[
    {
        "AllowedHeaders": [
            "*"
        ],
        "AllowedMethods": [
            "GET",
            "PUT",
            "POST",
            "DELETE",
            "HEAD"
        ],
        "AllowedOrigins": [
            "*"
        ],
        "ExposeHeaders": [
            "ETag",
            "Content-Length"
        ],
        "MaxAgeSeconds": 3000
    }
]
```

### 3. AWS CLI를 통한 설정
```bash
aws s3api put-bucket-cors --bucket cpx-attachments --cors-configuration file://cors.json
```

**cors.json 파일:**
```json
{
    "CORSRules": [
        {
            "AllowedHeaders": ["*"],
            "AllowedMethods": ["GET", "PUT", "POST", "DELETE", "HEAD"],
            "AllowedOrigins": ["*"],
            "ExposeHeaders": ["ETag", "Content-Length"],
            "MaxAgeSeconds": 3000
        }
    ]
}
```

## 보안 고려사항

### 개발 환경
- `AllowedOrigins: ["*"]` - 모든 도메인 허용
- 테스트 및 개발 목적으로만 사용

### 프로덕션 환경
```json
{
    "AllowedOrigins": [
        "https://yourdomain.com",
        "https://www.yourdomain.com"
    ]
}
```

## CORS 오류 해결

### 1. 일반적인 CORS 오류
```
Access to fetch at 'https://bucket.s3.region.amazonaws.com/...' 
from origin 'http://localhost:3000' has been blocked by CORS policy
```

### 2. 해결 방법
1. **S3 버킷 CORS 설정 확인**
2. **Presigned URL에 올바른 헤더 포함**
3. **프론트엔드에서 올바른 Content-Type 설정**

### 3. 프론트엔드 업로드 예시
```javascript
async function uploadToS3(presignedUrl, file) {
    try {
        const response = await fetch(presignedUrl, {
            method: 'PUT',
            body: file,
            headers: {
                'Content-Type': file.type,
                // CORS 관련 헤더는 브라우저가 자동으로 처리
            }
        });
        
        if (response.ok) {
            console.log('업로드 성공');
        } else {
            console.error('업로드 실패:', response.status);
        }
    } catch (error) {
        console.error('CORS 오류:', error);
    }
}
```

## 테스트 방법

### 1. CORS 설정 확인
```bash
aws s3api get-bucket-cors --bucket cpx-attachments
```

### 2. 브라우저 개발자 도구에서 확인
- Network 탭에서 CORS 오류 확인
- Console에서 CORS 관련 오류 메시지 확인

### 3. 테스트 스크립트 실행
```bash
python test_presigned_upload.py
```

## 주의사항

1. **CORS 설정 변경 후 즉시 적용되지 않을 수 있음** (최대 몇 분 소요)
2. **프로덕션에서는 특정 도메인만 허용**
3. **HTTPS 사용 권장** (보안상)
4. **정기적인 CORS 설정 검토 필요**

