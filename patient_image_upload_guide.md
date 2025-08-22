# 환자 이미지 업로드 가이드 (S3만 사용)

## 🏥 개요

CPX 시나리오별 환자 이미지를 S3에 업로드하는 단순화된 시스템입니다. 데이터베이스 없이 S3만 사용하여 이미지를 관리합니다.

## �� API 엔드포인트

### 1. 시나리오별 환자 이미지 목록 조회
**GET** `/patient-images/scenarios/{scenario_id}/images`

**Headers:**
```
Authorization: Bearer {access_token}
```

**Response:**
```json
{
  "scenario_id": "1",
  "images": [
    {
      "s3_key": "patient_images/scenario_1/uuid1.jpg",
      "s3_url": "https://medicpx.s3.ap-northeast-2.amazonaws.com/patient_images/scenario_1/uuid1.jpg",
      "filename": "uuid1.jpg",
      "file_size": 1024000,
      "last_modified": "2024-01-15T10:30:00Z",
      "etag": "abc123"
    }
  ],
  "total_count": 1,
  "message": "시나리오 1의 환자 이미지 1개를 조회했습니다."
}
```

### 2. 시나리오 대표 환자 이미지 조회
**GET** `/patient-images/scenarios/{scenario_id}/image`

**Headers:**
```
Authorization: Bearer {access_token}
```

**Response:**
```json
{
  "scenario_id": "1",
  "image": {
    "s3_key": "patient_images/scenario_1/uuid1.jpg",
    "s3_url": "https://medicpx.s3.ap-northeast-2.amazonaws.com/patient_images/scenario_1/uuid1.jpg",
    "filename": "uuid1.jpg",
    "file_size": 1024000,
    "last_modified": "2024-01-15T10:30:00Z",
    "etag": "abc123"
  },
  "message": "대표 환자 이미지를 조회했습니다."
}
```

### 3. 환자 이미지 업로드 URL 생성
**POST** `/patient-images/upload-url/{scenario_id}`

**Headers:**
```
Authorization: Bearer {access_token}
Content-Type: application/json
```

**Request Body:**
```json
{
  "filename": "patient_photo.jpg",
  "content_type": "image/jpeg",
  "content_length": 1024000,
  "method": "PUT"
}
```

**Response:**
```json
{
  "scenario_id": "1",
  "original_filename": "patient_photo.jpg",
  "stored_filename": "uuid.jpg",
  "upload_method": "PUT",
  "upload_url": "https://medicpx.s3.ap-northeast-2.amazonaws.com/patient_images/scenario_1/uuid.jpg?presigned_params...",
  "file_type": "image/jpeg",
  "file_size": 1024000,
  "expires_in": 3600,
  "s3_url": "https://medicpx.s3.ap-northeast-2.amazonaws.com/patient_images/scenario_1/uuid.jpg",
  "message": "업로드 URL이 생성되었습니다. 이 URL로 PUT 요청을 보내 파일을 업로드하세요."
}
```

### 4. S3 직접 업로드
**PUT** `{upload_url_from_step_1}`

**Headers:**
```
Content-Type: image/jpeg
Content-Length: {file_size}
```

**Body:** 이미지 파일 바이너리 데이터

### 5. 다운로드 URL 생성 (선택사항)
**POST** `/patient-images/download-url/{scenario_id}`

**Headers:**
```
Authorization: Bearer {access_token}
Content-Type: application/json
```

**Request Body:**
```json
{
  "s3_key": "patient_images/scenario_1/uuid.jpg",
  "expires_in": 3600
}
```

**Response:**
```json
{
  "scenario_id": "1",
  "s3_key": "patient_images/scenario_1/uuid.jpg",
  "download_url": "https://medicpx.s3.ap-northeast-2.amazonaws.com/patient_images/scenario_1/uuid.jpg?presigned_params...",
  "expires_in": 3600,
  "message": "다운로드 URL이 생성되었습니다."
}
```

### 6. 환자 이미지 삭제
**DELETE** `/patient-images/{scenario_id}?s3_key={s3_key}`

**Headers:**
```
Authorization: Bearer {access_token}
```

**Response:**
```json
{
  "message": "환자 이미지가 성공적으로 삭제되었습니다."
}
```

## 🔧 프론트엔드 구현 예시

### JavaScript (React/Axios)

```javascript
// 1. 시나리오별 환자 이미지 목록 조회
const getScenarioImages = async (scenarioId) => {
  const response = await axios.get(
    `/patient-images/scenarios/${scenarioId}/images`,
    {
      headers: {
        Authorization: `Bearer ${accessToken}`
      }
    }
  );
  return response.data;
};

// 2. 시나리오 대표 환자 이미지 조회
const getScenarioImage = async (scenarioId) => {
  const response = await axios.get(
    `/patient-images/scenarios/${scenarioId}/image`,
    {
      headers: {
        Authorization: `Bearer ${accessToken}`
      }
    }
  );
  return response.data;
};

// 3. 업로드 URL 생성
const generateUploadUrl = async (scenarioId, file) => {
  const response = await axios.post(
    `/patient-images/upload-url/${scenarioId}`,
    {
      filename: file.name,
      content_type: file.type,
      content_length: file.size,
      method: "PUT"
    },
    {
      headers: {
        Authorization: `Bearer ${accessToken}`
      }
    }
  );
  return response.data;
};

// 4. S3 업로드
const uploadToS3 = async (uploadUrl, file) => {
  const response = await axios.put(uploadUrl, file, {
    headers: {
      'Content-Type': file.type,
      'Content-Length': file.size
    }
  });
  return response.headers.etag;
};

// 5. 환자 이미지 업로드 전체 플로우
const uploadPatientImage = async (scenarioId, file) => {
  try {
    // 1단계: 업로드 URL 생성
    const uploadData = await generateUploadUrl(scenarioId, file);
    
    // 2단계: S3 업로드
    await uploadToS3(uploadData.upload_url, file);
    
    // 3단계: S3 URL 반환 (DB 저장 없음)
    console.log('환자 이미지 업로드 완료:', uploadData.s3_url);
    return uploadData.s3_url;
    
  } catch (error) {
    console.error('환자 이미지 업로드 실패:', error);
    throw error;
  }
};

// 6. 다운로드 URL 생성
const generateDownloadUrl = async (scenarioId, s3Key) => {
  const response = await axios.post(
    `/patient-images/download-url/${scenarioId}`,
    {
      s3_key: s3Key,
      expires_in: 3600
    },
    {
      headers: {
        Authorization: `Bearer ${accessToken}`
      }
    }
  );
  return response.data;
};

// 7. 이미지 삭제
const deletePatientImage = async (scenarioId, s3Key) => {
  const response = await axios.delete(
    `/patient-images/${scenarioId}?s3_key=${s3Key}`,
    {
      headers: {
        Authorization: `Bearer ${accessToken}`
      }
    }
  );
  return response.data;
};

// 8. 실습에서 환자 이미지 표시 예시
const displayPatientImage = async (scenarioId) => {
  try {
    // 시나리오의 대표 환자 이미지 조회
    const result = await getScenarioImage(scenarioId);
    
    if (result.image) {
      // 이미지 표시
      const imgElement = document.getElementById('patient-image');
      imgElement.src = result.image.s3_url;
      imgElement.alt = `시나리오 ${scenarioId} 환자 이미지`;
    } else {
      console.log(`시나리오 ${scenarioId}의 환자 이미지가 없습니다.`);
    }
  } catch (error) {
    console.error('환자 이미지 표시 실패:', error);
  }
};
```

## 📁 파일 구조

### S3 저장 구조
```
medicpx/
└── patient_images/
    ├── scenario_1/
    │   ├── uuid1.jpg
    │   └── uuid2.png
    ├── scenario_2/
    │   └── uuid3.jpeg
    └── scenario_3/
        └── uuid4.webp
```

## 🔒 권한 요구사항

- **업로드/삭제**: 관리자 또는 교수 역할 필요
- **다운로드**: 모든 인증된 사용자 가능

## 📏 제한사항

- **파일 크기**: 최대 10MB
- **지원 형식**: JPEG, JPG, PNG, GIF, WebP
- **시나리오 ID**: 문자열 (1, 2, 3 등)

## 🧪 테스트

테스트 스크립트 실행:
```bash
python test_patient_image_upload.py
```

## 🚨 주의사항

1. **파일 검증**: 업로드 전 파일 크기와 형식을 확인하세요
2. **권한 확인**: 관리자/교수 권한이 필요합니다
3. **S3 비용**: 이미지 저장에 따른 S3 비용이 발생할 수 있습니다
4. **보안**: 민감한 환자 정보가 포함된 이미지는 업로드하지 마세요
5. **메타데이터 관리**: 데이터베이스가 없으므로 이미지 메타데이터는 프론트엔드에서 관리해야 합니다

## 💡 사용 시나리오

### 시나리오 1: 교수가 환자 이미지 업로드
1. 교수가 환자 이미지 선택
2. 업로드 URL 생성 요청
3. S3에 직접 업로드
4. S3 URL을 시나리오 정보에 저장

### 시나리오 2: 학생이 환자 이미지 조회
1. 시나리오 정보에서 S3 URL 확인
2. 이미지 직접 표시 (S3 URL 사용)

### 시나리오 3: 교수가 이미지 삭제
1. S3 키로 삭제 요청
2. S3에서 파일 삭제
