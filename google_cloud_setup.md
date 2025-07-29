# 🌐 Google Cloud APIs 설정 가이드

## 🚀 **1단계: Google Cloud 프로젝트 설정**

### **1.1 Google Cloud Console 접속**
1. [Google Cloud Console](https://console.cloud.google.com/)에 접속
2. 새 프로젝트 생성 또는 기존 프로젝트 선택

### **1.2 필요한 API들 활성화**
```bash
# 1. API 라이브러리에서 다음 API들 검색 및 활성화:
# - "Speech-to-Text API" (STT용)
# - "Text-to-Speech API" (TTS용)  
# 2. 각각 "사용" 버튼 클릭
# 3. 몇 분 후 활성화 완료
```

## 🔑 **2단계: 인증 설정**

### **방법 A: 서비스 계정 키 (권장)**
```bash
# 1. IAM 및 관리자 > 서비스 계정
# 2. "서비스 계정 만들기" 클릭
# 3. 이름: cpx-speech-service
# 4. 역할 추가:
#    - "Cloud Speech Client" 
#    - "Cloud Text-to-Speech Client"
# 5. "키 만들기" > JSON 선택
# 6. JSON 파일 다운로드
```

### **방법 B: gcloud CLI (로컬 테스트용)**
```bash
# gcloud CLI 설치 후
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

## 🛠️ **3단계: 환경 변수 설정**

### **.env 파일 생성:**
```bash
# OpenAI API 키
OPENAI_API_KEY=your_openai_api_key

# Google Cloud 인증 (방법 A 사용 시)
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json

# Google Cloud 프로젝트 ID
GOOGLE_CLOUD_PROJECT=your-project-id

# 서버 및 ChromaDB 설정
HOST=0.0.0.0
PORT=8000
CHROMA_PERSIST_DIRECTORY=./chroma_db
LOG_LEVEL=INFO
```

## 💰 **4단계: 비용 설정**

### **무료 할당량:**
- **월 60분 무료** (표준 모델)
- 이후 **분당 $0.006** (약 8원)

### **예산 알림 설정:**
```bash
# Google Cloud Console > 결제 > 예산 및 알림
# 월 $10 (약 13,000원) 예산 설정 권장
```

## 🧪 **5단계: 테스트**

### **5.1 패키지 설치:**
```bash
pip install -r requirements.txt
```

### **5.2 서버 실행:**
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### **5.3 Health Check:**
```bash
curl http://localhost:8000/health
```

**성공 응답:**
```json
{
  "status": "healthy",
  "speech": true,
  "llm": true,
  "tts": true,
  "vector": true
}
```

## 🎤 **6단계: 실제 테스트**

### **WebSocket 연결 테스트:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/test_student');

ws.onopen = function() {
    console.log('🟢 연결 성공!');
    
    // 세션 시작
    ws.send(JSON.stringify({
        type: "start_session",
        case_id: "IM_001"
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('📨 서버 응답:', data);
};
```

## ⚡ **장점 (vs faster-whisper):**

| 항목 | Google Cloud | faster-whisper |
|------|-------------|----------------|
| **🖥️ GPU 필요** | ❌ 불필요 | ✅ 필수 |
| **⚡ 응답 속도** | **0.2초** | 1-2초 |
| **🌐 확장성** | **무제한** | GPU 메모리 제한 |
| **💰 비용** | 분당 8원 | GPU 서버비 |
| **🔧 설정** | **간단** | 복잡 |

## 🎯 **CPX 시스템에 완벽한 이유:**

1. **⚡ 실시간 대화**: 0.2초 지연으로 자연스러운 대화
2. **🔧 쉬운 설정**: GPU 없이 바로 테스트 가능
3. **📈 확장성**: 동시 사용자 수 제한 없음
4. **💰 합리적 비용**: 테스트는 무료, 실사용도 저렴
5. **🇰🇷 한국어 완벽 지원**: 의료 용어도 정확하게 인식

## 🚨 **문제 해결:**

### **인증 오류:**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"
```

### **API 할당량 초과:**
```bash
# Google Cloud Console > API 및 서비스 > 할당량
# Speech-to-Text API 할당량 확인
```

### **권한 오류:**
```bash
# 서비스 계정에 "Cloud Speech Client" 역할 추가
```

---

**🎉 이제 GPU 없이도 실시간 CPX 대화 테스트가 가능합니다!** 