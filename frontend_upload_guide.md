# 프론트엔드 S3 업로드 가이드

## 개요

이 가이드는 프론트엔드에서 Presigned URL/POST를 사용하여 S3에 직접 파일을 업로드하는 방법을 설명합니다.

## 1. Presigned URL 생성 (백엔드)

### PUT 방식 (기본)
```javascript
// 1. 업로드 URL 생성 요청
const response = await fetch('/attachments/upload-url/1', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    filename: file.name,
    file_type: file.type,
    file_size: file.size,
    method: 'PUT'  // 기본값
  })
});

const data = await response.json();
// 응답:
// {
//   "upload_method": "PUT",
//   "upload_url": "https://s3.amazonaws.com/bucket/...",
//   "s3_url": "https://bucket.s3.region.amazonaws.com/...",
//   ...
// }
```

### POST 방식
```javascript
const response = await fetch('/attachments/upload-url/1', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    filename: file.name,
    file_type: file.type,
    file_size: file.size,
    method: 'POST'
  })
});

const data = await response.json();
// 응답:
// {
//   "upload_method": "POST",
//   "upload_url": "https://s3.amazonaws.com/bucket/...",
//   "upload_fields": {
//     "key": "attachments/uuid-filename.pdf",
//     "Content-Type": "application/pdf",
//     "policy": "...",
//     "x-amz-algorithm": "AWS4-HMAC-SHA256",
//     "x-amz-credential": "...",
//     "x-amz-date": "...",
//     "x-amz-signature": "..."
//   },
//   ...
// }
```

## 2. S3 업로드 (프론트엔드)

### PUT 방식 업로드
```javascript
async function uploadWithPUT(presignedUrl, file) {
  try {
    const response = await fetch(presignedUrl, {
      method: 'PUT',
      body: file,
      headers: {
        'Content-Type': file.type
      }
    });
    
    if (response.ok) {
      console.log('PUT 업로드 성공');
      return {
        success: true,
        etag: response.headers.get('ETag')?.replace(/"/g, '')  // ETag에서 따옴표 제거
      };
    } else {
      console.error('PUT 업로드 실패:', response.status);
      return { success: false };
    }
  } catch (error) {
    console.error('PUT 업로드 오류:', error);
    return { success: false };
  }
}

// 사용 예시
const result = await uploadWithPUT(data.upload_url, file);
```

### POST 방식 업로드
```javascript
async function uploadWithPOST(presignedPost, file) {
  try {
    const formData = new FormData();
    
    // Presigned POST 필드 추가
    Object.keys(presignedPost.upload_fields).forEach(key => {
      formData.append(key, presignedPost.upload_fields[key]);
    });
    
    // 파일 추가 (반드시 마지막에 추가)
    formData.append('file', file);
    
    const response = await fetch(presignedPost.upload_url, {
      method: 'POST',
      body: formData
    });
    
    if (response.ok) {
      console.log('POST 업로드 성공');
      return {
        success: true,
        etag: response.headers.get('ETag')?.replace(/"/g, '')  // ETag에서 따옴표 제거
      };
    } else {
      console.error('POST 업로드 실패:', response.status);
      return { success: false };
    }
  } catch (error) {
    console.error('POST 업로드 오류:', error);
    return { success: false };
  }
}

// 사용 예시
const result = await uploadWithPOST(data, file);
```

## 3. 완전한 업로드 플로우

```javascript
async function uploadFile(noticeId, file) {
  try {
    // 1. Presigned URL 생성
    const urlResponse = await fetch(`/attachments/upload-url/${noticeId}`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        filename: file.name,
        file_type: file.type,
        file_size: file.size,
        method: 'PUT'  // 또는 'POST'
      })
    });
    
    if (!urlResponse.ok) {
      throw new Error('Presigned URL 생성 실패');
    }
    
    const urlData = await urlResponse.json();
    
    // 2. S3 업로드
    let uploadResult = null;
    
    if (urlData.upload_method === 'POST') {
      uploadResult = await uploadWithPOST(urlData, file);
    } else {
      uploadResult = await uploadWithPUT(urlData.upload_url, file);
    }
    
    if (!uploadResult.success) {
      throw new Error('S3 업로드 실패');
    }
    
    // 3. 업로드 완료 알림 (S3 파일 존재 확인 포함)
    const completeResponse = await fetch(`/attachments/upload-complete/${noticeId}`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        original_filename: file.name,
        s3_url: urlData.s3_url,
        file_size: file.size,
        file_type: file.type,
        etag: uploadResult.etag  // ETag 포함 (선택사항)
      })
    });
    
    if (!completeResponse.ok) {
      throw new Error('업로드 완료 처리 실패');
    }
    
    const result = await completeResponse.json();
    console.log('업로드 완료:', result);
    return result;
    
  } catch (error) {
    console.error('업로드 실패:', error);
    throw error;
  }
}
```

## 4. React 컴포넌트 예시

```jsx
import React, { useState } from 'react';

function FileUpload({ noticeId, onUploadComplete }) {
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);

  const handleFileUpload = async (file) => {
    setUploading(true);
    setProgress(0);
    
    try {
      const result = await uploadFile(noticeId, file);
      onUploadComplete(result);
      setProgress(100);
    } catch (error) {
      console.error('업로드 실패:', error);
      alert('파일 업로드에 실패했습니다.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div>
      <input
        type="file"
        onChange={(e) => {
          const file = e.target.files[0];
          if (file) {
            handleFileUpload(file);
          }
        }}
        disabled={uploading}
      />
      {uploading && (
        <div>
          <p>업로드 중... {progress}%</p>
          <progress value={progress} max="100" />
        </div>
      )}
    </div>
  );
}

export default FileUpload;
```

## 5. 에러 처리

### 일반적인 오류
```javascript
// CORS 오류
if (error.message.includes('CORS')) {
  console.error('CORS 정책 위반. S3 CORS 설정을 확인하세요.');
}

// 인증 오류
if (error.message.includes('403')) {
  console.error('인증 실패. 토큰을 확인하세요.');
}

// 파일 크기 오류
if (error.message.includes('413')) {
  console.error('파일 크기가 너무 큽니다.');
}

// 파일 타입 오류
if (error.message.includes('415')) {
  console.error('지원하지 않는 파일 타입입니다.');
}
```

## 6. 보안 고려사항

1. **토큰 관리**: JWT 토큰을 안전하게 저장하고 관리
2. **파일 검증**: 프론트엔드에서도 파일 크기와 타입 검증
3. **HTTPS 사용**: 프로덕션에서는 반드시 HTTPS 사용
4. **에러 메시지**: 민감한 정보가 포함되지 않도록 주의

## 7. 성능 최적화

1. **청크 업로드**: 대용량 파일은 청크 단위로 분할 업로드
2. **진행률 표시**: 업로드 진행률을 사용자에게 표시
3. **재시도 로직**: 네트워크 오류 시 자동 재시도
4. **캐싱**: Presigned URL을 적절히 캐싱하여 재사용
