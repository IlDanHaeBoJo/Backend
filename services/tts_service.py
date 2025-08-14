import hashlib
from pathlib import Path
from google.cloud import texttospeech


class TTSService:
    def __init__(self):
        """TTS 서비스 초기화"""
        self.cache_dir = Path("cache/tts")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

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

    async def generate_speech(self, text: str) -> str:
        """텍스트를 음성으로 변환"""
        if not text:
            return None

        # 캐시 확인
        cache_key = hashlib.md5(text.encode()).hexdigest()
        cache_path = self.cache_dir / f"tts_{cache_key}.mp3"

        if cache_path.exists():
            return str(cache_path)

        # Google TTS 생성
        synthesis_input = texttospeech.SynthesisInput(text=text)
        response = self.client.synthesize_speech(
            input=synthesis_input,
            voice=self.voice,
            audio_config=self.audio_config
        )

        # 캐시에만 저장
        with open(cache_path, 'wb') as f:
            f.write(response.audio_content)

        return str(cache_path)
