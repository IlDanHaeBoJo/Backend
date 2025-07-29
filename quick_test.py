#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 CPX 시스템 빠른 테스트 스크립트
Google Cloud Speech API 설정 확인용
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

def check_environment():
    """환경 설정 확인"""
    print("🔍 환경 설정 확인 중...")
    
    # OpenAI API 키 확인
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        print("✅ OpenAI API 키: 설정됨")
    else:
        print("❌ OpenAI API 키: 설정 필요")
        return False
    
    # Google Cloud 인증 확인
    gcp_credentials = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if gcp_credentials and Path(gcp_credentials).exists():
        print("✅ Google Cloud 인증: 서비스 계정 키 파일 확인됨")
    else:
        print("⚠️  Google Cloud 인증: gcloud CLI 또는 서비스 계정 키 필요")
    
    return True

def test_google_speech():
    """Google Cloud Speech API 테스트"""
    print("\n🎤 Google Cloud Speech API 테스트...")
    
    try:
        from google.cloud import speech
        
        # 클라이언트 초기화
        client = speech.SpeechClient()
        
        # 설정 생성
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="ko-KR",
            model="latest_long",
            enable_automatic_punctuation=True,
        )
        
        print("✅ Google Cloud Speech API 클라이언트 초기화 성공")
        print("✅ 한국어 설정 완료")
        return True
        
    except Exception as e:
        print(f"❌ Google Cloud Speech API 오류: {e}")
        print("💡 해결방법:")
        print("   1. 'pip install google-cloud-speech' 실행")
        print("   2. Google Cloud 프로젝트에서 Speech-to-Text API 활성화")
        print("   3. 인증 정보 설정 확인")
        return False

def test_google_tts():
    """Google Cloud Text-to-Speech API 테스트"""
    print("\n🔊 Google Cloud Text-to-Speech API 테스트...")
    
    try:
        from google.cloud import texttospeech
        
        # 클라이언트 초기화
        client = texttospeech.TextToSpeechClient()
        
        # 한국어 음성 설정 테스트
        voice = texttospeech.VoiceSelectionParams(
            language_code="ko-KR",
            name="ko-KR-Neural2-A",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        
        print("✅ Google Cloud TTS API 클라이언트 초기화 성공")
        print("✅ 한국어 고품질 음성 설정 완료")
        return True
        
    except Exception as e:
        print(f"❌ Google Cloud TTS API 오류: {e}")
        print("💡 해결방법:")
        print("   1. 'pip install google-cloud-texttospeech' 실행")
        print("   2. Google Cloud 프로젝트에서 Text-to-Speech API 활성화")
        print("   3. 인증 정보 설정 확인")
        return False

def test_other_services():
    """다른 서비스들 테스트"""
    print("\n🔧 다른 서비스들 테스트...")
    
    services = {
        "openai": "OpenAI GPT-4",
        "langchain": "LangChain", 
        "chromadb": "ChromaDB"
    }
    
    for package, name in services.items():
        try:
            __import__(package)
            print(f"✅ {name}: 설치됨")
        except ImportError:
            print(f"❌ {name}: 설치 필요")
        except RuntimeError as e:
            if "sqlite3" in str(e).lower():
                print(f"⚠️  {name}: SQLite 버전 이슈 (기본 모드로 동작)")
            else:
                print(f"❌ {name}: 런타임 오류 - {e}")
        except Exception as e:
            print(f"❌ {name}: 오류 - {e}")

def main():
    """메인 테스트 실행"""
    print("🏥 CPX 가상 표준화 환자 시스템 - 설정 테스트")
    print("=" * 50)
    
    # 환경 변수 확인
    if not check_environment():
        print("\n❌ 환경 설정을 완료한 후 다시 실행해주세요.")
        print("📖 자세한 설정 방법: google_cloud_setup.md 참조")
        sys.exit(1)
    
    # Google Cloud APIs 테스트
    speech_ok = test_google_speech()
    tts_ok = test_google_tts()
    
    # 다른 서비스들 테스트
    test_other_services()
    
    print("\n" + "=" * 50)
    print("📋 필수 서비스 결과:")
    print(f"   🎤 Google Speech API: {'✅' if speech_ok else '❌'}")
    print(f"   🔊 Google TTS API: {'✅' if tts_ok else '❌'}")
    
    if speech_ok and tts_ok:
        print("\n🎉 모든 필수 설정 완료! 서버 실행 가능합니다.")
        print("   Google Cloud TTS로 고품질 한국어 음성 생성!")
        print("\n🚀 서버 실행 명령:")
        print("   python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
        print("\n🌐 접속 URL:")
        print("   Health Check: http://localhost:8000/health")
        print("   WebSocket: ws://localhost:8000/ws/test_student")
    else:
        print("\n❌ Google Cloud APIs 설정을 완료해주세요.")
        print("📖 설정 가이드: google_cloud_setup.md")

if __name__ == "__main__":
    main() 