from google.cloud import texttospeech
from typing import Optional


class TTSService:
    def __init__(self):
        """TTS 서비스 초기화 - 캐시 없는 실시간 생성"""
        # Google TTS 바로 초기화
        self.client = texttospeech.TextToSpeechClient()
        self.voice = texttospeech.VoiceSelectionParams(
            language_code="ko-KR",
            name="ko-KR-Neural2-A",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

    async def generate_speech(self, text: str) -> Optional[bytes]:
        """텍스트를 음성으로 변환 - 메모리 버퍼 반환"""
        if not text:
            return None

        try:
            # Google TTS 실시간 생성
            synthesis_input = texttospeech.SynthesisInput(text=text)
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=self.voice,
                audio_config=self.audio_config
            )

            # 메모리 버퍼로 직접 반환 (파일 저장 없음)
            return response.audio_content
            
        except Exception as e:
            print(f"❌ TTS 생성 실패: {e}")
            return None
