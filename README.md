# CPX 가상 표준화 환자 시스템 Backend

의과대학 CPX(Clinical Performance Examination) 실기시험을 위한 가상 표준화 환자 시스템입니다.

## 🏗 아키텍처

```
🩺 의대생 (React 클라이언트)
  └─ 🎤 환자 문진 질문 → WebSocket 송신
      ↕
📦 CPX 가상환자 서버 (FastAPI + 클라우드)
  ├─ 🎧 Google Cloud Speech STT (질문 → 텍스트)
  ├─ 🧠 LangChain + VectorDB (54종 CPX 케이스 + GPT-4o)
  ├─ 🔊 TTS (표준화환자 음성 합성)
  └─ 📤 WebSocket 응답 (환자답변 + 음성 + 아바타)
      ↕
🩺 의대생 브라우저
  └─ 🧍‍♀️ 가상 표준화 환자 + 실시간 대화
```

## 🚀 주요 기능

- **가상 표준화 환자**: 54종 CPX 실기 항목별 케이스 시뮬레이션
- **STT (Speech-to-Text)**: Google Cloud Speech API 기반 실시간 한국어 음성 인식
- **LLM**: GPT-4o 기반 환자 역할 연기 (LangChain)
- **RAG**: CPX 케이스 데이터베이스 기반 일관된 환자 정보 제공
- **TTS**: 자연스러운 환자 음성 응답 생성
- **WebSocket**: 실시간 문진 대화 지원
- **평가 시스템**: AI 기반 학습 피드백 (추후 확장)

## 📋 필수 요구사항

- Python 3.9+
- Google Cloud 프로젝트 및 Speech-to-Text API 활성화
- OpenAI API 키
- Google Cloud Text-to-Speech API

## 🛠 설치 및 설정

### 1. 가상환경 생성 및 활성화
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate     # Windows
```

### 2. 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. Google Cloud 설정
Google Cloud Speech-to-Text API 설정:

```bash
# Google Cloud CLI 설치 및 로그인
gcloud auth application-default login

# 또는 서비스 계정 키 파일 사용
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account-key.json"
```

자세한 설정 가이드: [google_cloud_setup.md](google_cloud_setup.md)

### 4. 환경변수 설정
`.env` 파일을 생성하고 다음과 같이 설정:

```env
# OpenAI API 설정
OPENAI_API_KEY=your_openai_api_key_here

# Google Cloud 설정
GOOGLE_CLOUD_PROJECT=your-google-cloud-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json

# 서버 설정
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO

# 서버 설정
HOST=0.0.0.0
PORT=8000

# ChromaDB 설정
CHROMA_PERSIST_DIRECTORY=./chroma_db
```

환경변수 템플릿: [.env.example](.env.example)

### 5. 환경 검증
Google Cloud 설정이 올바른지 확인:

```bash
python test_setup.py
```

### 6. 서버 실행
```bash
python main.py
```

또는
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 🐘 PostgreSQL 설정 (Docker Compose)

이 프로젝트는 Docker Compose를 사용하여 PostgreSQL 데이터베이스를 관리합니다.

### 1. Docker Compose 파일 확인
`docker-compose.yml` 파일은 PostgreSQL 서비스를 정의합니다.

```yaml
version: '3.8'

services:
  db:
    image: postgres:13-alpine
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    ports:
      - "5432:5432"
    volumes:
      - db-data:/var/lib/postgresql/data

volumes:
  db-data:
```

### 2. 환경 변수 설정 (`.env`)
PostgreSQL 연결 정보는 `.env` 파일에서 관리됩니다. 다음 변수들이 `.env` 파일에 정의되어 있는지 확인하거나 추가하세요:

```env
# ===== PostgreSQL 설정 =====
POSTGRES_USER=
POSTGRES_PASSWORD=
POSTGRES_DB=
DATABASE_URL=postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@localhost:5432/{POSTGRES_DB}
```

- `POSTGRES_USER`: PostgreSQL 사용자 이름
- `POSTGRES_PASSWORD`: PostgreSQL 비밀번호
- `POSTGRES_DB`: PostgreSQL 데이터베이스 이름
- `DATABASE_URL`: 애플리케이션에서 데이터베이스에 연결하는 데 사용되는 URL

### 3. PostgreSQL 컨테이너 실행
프로젝트 루트 디렉토리에서 다음 명령어를 실행하여 PostgreSQL 컨테이너를 시작합니다:

```bash
docker-compose up -d
```

- `-d` 옵션은 컨테이너를 백그라운드에서 실행합니다.
- 컨테이너를 중지하려면 `docker-compose down` 명령어를 사용합니다.

### 4. PostgreSQL 데이터베이스 접근
실행 중인 PostgreSQL 컨테이너에 접근하여 데이터베이스를 관리할 수 있습니다.

1. **컨테이너 이름 확인**:
   다음 명령어를 사용하여 실행 중인 Docker 컨테이너 목록에서 PostgreSQL 컨테이너의 이름을 확인합니다. 일반적으로 `docker-compose.yml`에 정의된 서비스 이름(`db`)과 프로젝트 디렉토리 이름이 조합된 형태입니다 (예: `backend-db-1`).

   ```bash
   docker ps
   ```

2. **데이터베이스 접근**:
   확인된 컨테이너 이름을 사용하여 `psql` 클라이언트로 데이터베이스에 접속합니다.

   ```bash
   docker exec -it [컨테이너 이름] psql -U ${POSTGRES_USER} -d ${POSTGRES_DB}
   ```

   예시:
   ```bash
   docker exec -it backend-db-1 psql -U postgre -d IlDanHaeBoJo
   ```

   접속 후에는 PostgreSQL 명령어를 직접 실행할 수 있습니다.

## 🌐 API 엔드포인트

### 기본 정보
- `GET /` - API 상태 확인
- `GET /health` - 헬스체크

### 실시간 음성 대화
- `WS /ws/{user_id}` - WebSocket 실시간 음성 대화

### API 문서
서버 실행 후 [http://localhost:8000/docs](http://localhost:8000/docs)에서 Swagger UI 확인

## 💬 사용 예시

### WebSocket 실시간 CPX 문진
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/student123');

// 의대생 질문 음성 데이터 전송
ws.send(questionAudioBlob);

// 가상 환자 응답 수신
ws.onmessage = (event) => {
  const response = JSON.parse(event.data);
  console.log('학생 질문:', response.user_text);
  console.log('환자 응답:', response.ai_text);
  console.log('환자 음성:', response.audio_url);
  console.log('아바타 상태:', response.avatar_action);
};
```

## 🏥 CPX 케이스 데이터베이스

시스템은 다음과 같은 초기 CPX 케이스를 포함합니다:
- **내과**: 급성 위염 (김민수, 35세 남성)
- **정신과**: 우울증 (이지은, 28세 여성) 
- **외과**: 충수염 의심 (박준호, 22세 남성)

**54종 실기 항목 지원 예정**:
- 내과, 정신과, 외과, 산부인과, 소아과, 가정의학과 등
- 각 케이스별 표준화된 환자 정보, 병력, 증상, 반응 패턴

새로운 CPX 케이스는 Vector Service를 통해 추가할 수 있습니다.

## 🔧 설정 옵션

### Google Cloud Speech API 설정
- **language_code**: `ko-KR` (한국어, 기본값)
- **model**: `latest_long` (긴 대화용, 권장) / `latest_short` (짧은 명령용)
- **encoding**: `LINEAR16` (16-bit PCM)
- **sample_rate**: `16000` Hz
- **enhanced 모델**: 더 높은 정확도 (비용 증가)

### TTS 설정
- Google Cloud TTS로 고품질 한국어 음성 합성
- 한국어 Neural2-A 모델 사용 (자연스러운 발음 및 억양)
- 크로스 플랫폼 호환성 (Linux/Windows/macOS)

## 🚨 주의사항

1. **교육 목적**: 이 시스템은 의과대학 CPX 교육 전용이며, 실제 환자 진료를 대체할 수 없습니다.
2. **API 키 보안**: `.env` 파일을 Git에 커밋하지 마세요.
3. **Google Cloud 비용**: Speech API는 사용량에 따라 과금됩니다 (월 60분 무료).
4. **파일 정리**: TTS 생성 파일들이 누적되므로 주기적으로 정리하세요.
5. **케이스 일관성**: CPX 케이스 정보는 실제 의료진 검토 후 사용을 권장합니다.

## 📁 프로젝트 구조

```
Backend/
├── main.py                 # FastAPI CPX 서버
├── requirements.txt        # Python 패키지 목록
├── services/              # 서비스 모듈들
│   ├── __init__.py
│   ├── llm_service.py     # GPT-4o 표준화 환자 역할 서비스
│   ├── tts_service.py     # 환자 음성 합성 서비스
│   └── vector_service.py  # CPX 케이스 DB/RAG 서비스
├── static/               # 정적 파일 (생성된 환자 음성)
│   └── audio/
├── temp_audio/           # 임시 음성 파일 (학생 질문)
├── cache/               # TTS 캐시 (환자 응답)
│   └── tts/
└── chroma_db/           # CPX 케이스 벡터 저장소
```

## 🔗 관련 기술

- **FastAPI**: 웹 프레임워크
- **Google Cloud Speech**: 실시간 음성 인식
- **LangChain**: LLM 프레임워크
- **ChromaDB**: 벡터 데이터베이스
- **Google Cloud TTS**: 한국어 고품질 음성 합성
- **WebSocket**: 실시간 통신

## 📈 성능 최적화

- Google Cloud Speech 스트리밍 최적화
- TTS 결과 캐싱
- 벡터 검색 최적화
- 네트워크 지연시간 최소화

## 🔧 트러블슈팅

### 일반적인 문제들

1. **CUDA/GPU 인식 안됨**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

2. **ChromaDB 초기화 실패**
   ```bash
   rm -rf chroma_db/
   ```

3. **TTS 서비스 에러**
   - API 키 확인
   - 잔액 확인
   - 네트워크 연결 확인

## 📧 문의

프로젝트 관련 문의사항이 있으시면 이슈를 생성해 주세요.
