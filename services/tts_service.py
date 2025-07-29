import os
import logging
import hashlib
from typing import Optional
from pathlib import Path
import aiofiles
import asyncio

try:
    from google.cloud import texttospeech
    GOOGLE_TTS_AVAILABLE = True
except ImportError:
    GOOGLE_TTS_AVAILABLE = False

logger = logging.getLogger(__name__)

class TTSService:
    def __init__(self):
        """TTS 서비스 초기화 (Google Cloud TTS 전용)"""
        self.output_dir = Path("static/audio")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 캐시 디렉토리
        self.cache_dir = Path("cache/tts")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Google Cloud TTS 초기화
        self.google_tts_available = False
        if GOOGLE_TTS_AVAILABLE:
            try:
                self.google_tts_client = texttospeech.TextToSpeechClient()
                
                # 한국어 여성 음성 설정 (고품질)
                self.google_voice = texttospeech.VoiceSelectionParams(
                    language_code="ko-KR",
                    name="ko-KR-Neural2-A",  # 한국어 고품질 여성 음성
                    ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
                )
                
                self.google_audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.MP3
                )
                
                self.google_tts_available = True
                logger.info("✅ Google Cloud TTS 서비스 활성화 (한국어 Neural2-A)")
                
            except Exception as e:
                logger.error(f"❌ Google Cloud TTS 초기화 실패: {e}")
                self.google_tts_available = False
        else:
            logger.error("❌ google-cloud-texttospeech 패키지가 설치되지 않음")
        
        if not self.google_tts_available:
            logger.error("❌ 사용 가능한 TTS 서비스가 없습니다!")

    def get_cache_key(self, text: str, service: str = "google_tts") -> str:
        """텍스트와 서비스에 대한 캐시 키 생성"""
        cache_text = f"{text}_{service}"
        return hashlib.md5(cache_text.encode()).hexdigest()

    async def generate_speech_google_tts(self, text: str) -> Optional[str]:
        """Google Cloud TTS를 사용한 음성 생성"""
        try:
            cache_key = self.get_cache_key(text, "google_tts")
            output_path = self.output_dir / f"tts_{cache_key}.mp3"
            cache_path = self.cache_dir / f"tts_{cache_key}.mp3"
            
            # 캐시 확인
            if cache_path.exists():
                async with aiofiles.open(cache_path, 'rb') as src:
                    content = await src.read()
                    async with aiofiles.open(output_path, 'wb') as dst:
                        await dst.write(content)
                logger.info(f"Google TTS 캐시 사용: {output_path}")
                return str(output_path)
            
            # Google TTS 요청 준비
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # API 호출
            response = self.google_tts_client.synthesize_speech(
                input=synthesis_input,
                voice=self.google_voice,
                audio_config=self.google_audio_config,
            )
            
            # 캐시에 저장
            with open(cache_path, 'wb') as f:
                f.write(response.audio_content)
            
            # output 디렉토리에 복사
            async with aiofiles.open(output_path, 'wb') as f:
                await f.write(response.audio_content)
            
            logger.info(f"Google TTS 생성 완료: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Google TTS 생성 실패: {e}")
            return None

    async def generate_speech(self, text: str) -> Optional[str]:
        """음성 생성 (Google Cloud TTS 전용)"""
        if not text or not text.strip():
            logger.warning("빈 텍스트 입력")
            return None
        
        text = text.strip()
        logger.info(f"TTS 요청: '{text[:50]}...' (Google Cloud TTS)")
        
        if not self.google_tts_available:
            logger.error("❌ Google Cloud TTS 서비스가 사용 불가합니다")
            return None
        
        try:
            result = await self.generate_speech_google_tts(text)
            if result:
                return result
            else:
                logger.error("❌ Google Cloud TTS 음성 생성 실패")
                return None
                
        except Exception as e:
            logger.error(f"TTS 처리 오류: {e}")
            return None

    async def get_available_voices(self) -> dict:
        """사용 가능한 음성 목록 반환"""
        if not self.google_tts_available:
            return {
                "available": False,
                "error": "Google Cloud TTS 서비스가 사용 불가합니다"
            }
        
        return {
            "available": True,
            "service": "Google Cloud TTS",
            "current_voice": "ko-KR-Neural2-A",
            "available_voices": [
                {
                    "name": "ko-KR-Neural2-A",
                    "gender": "여성",
                    "type": "Neural",
                    "description": "한국어 고품질 여성 음성"
                },
                {
                    "name": "ko-KR-Neural2-B", 
                    "gender": "남성",
                    "type": "Neural",
                    "description": "한국어 고품질 남성 음성"
                },
                {
                    "name": "ko-KR-Neural2-C",
                    "gender": "여성", 
                    "type": "Neural",
                    "description": "한국어 고품질 여성 음성 (대안)"
                },
                {
                    "name": "ko-KR-Standard-A",
                    "gender": "여성",
                    "type": "Standard", 
                    "description": "한국어 표준 여성 음성"
                },
                {
                    "name": "ko-KR-Standard-B",
                    "gender": "남성",
                    "type": "Standard",
                    "description": "한국어 표준 남성 음성"
                }
            ]
        }

    def clear_cache(self):
        """TTS 캐시 초기화"""
        try:
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("TTS 캐시 초기화 완료")
        except Exception as e:
            logger.error(f"캐시 초기화 실패: {e}")

    def get_service_status(self) -> dict:
        """TTS 서비스 상태 반환"""
        return {
            "google_tts_available": self.google_tts_available,
            "service_name": "Google Cloud Text-to-Speech",
            "voice_model": "ko-KR-Neural2-A",
            "language": "Korean (한국어)"
        }

# 사용 예시
async def test_tts():
    """TTS 서비스 테스트"""
    tts = TTSService()
    
    test_text = "안녕하세요, 저는 CPX 가상 환자입니다. 속이 아파서 왔어요."
    
    result = await tts.generate_speech(test_text)
    if result:
        print(f"✅ TTS 성공: {result}")
    else:
        print("❌ TTS 실패")
    
    voices = await tts.get_available_voices()
    print(f"🎤 사용 가능한 음성: {voices}")

if __name__ == "__main__":
    asyncio.run(test_tts()) 