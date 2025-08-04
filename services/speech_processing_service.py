import asyncio
from typing import Dict
from pathlib import Path
import wave
import os


class SpeechProcessingService:
    def __init__(self, whisper_model, llm_service, tts_service, evaluation_service):
        """음성 처리 서비스 초기화"""
        self.whisper_model = whisper_model
        self.llm_service = llm_service
        self.tts_service = tts_service
        self.evaluation_service = evaluation_service
        self.user_buffers = {}

        # temp_audio 디렉토리 생성
        Path("temp_audio").mkdir(exist_ok=True)

    async def process_audio_chunk(self, user_id: str, audio_chunk: bytes, is_silent: bool):
        """음성 청크 처리"""
        if user_id not in self.user_buffers:
            self.user_buffers[user_id] = {"audio": bytearray(), "speaking": False, "silence": 0}

        buffer = self.user_buffers[user_id]
        buffer["audio"].extend(audio_chunk)

        if not is_silent:
            buffer["speaking"] = True
            buffer["silence"] = 0
        else:
            buffer["silence"] += 1
            if buffer["speaking"] and self.should_process_speech(user_id):
                await self.process_speech(user_id)

    def should_process_speech(self, user_id: str) -> bool:
        """실시간 버퍼 분석으로 적응적 처리"""
        buffer = self.user_buffers[user_id]
        silence_time = buffer["silence"]
        audio_length = len(buffer["audio"])

        # 너무 짧은 건 무시 (0.3초)
        if silence_time < 3:
            return False

        # 너무 짧은 오디오도 무시 (0.5초)
        if audio_length < 8000:  # 16kHz * 0.5초
            return False

        # 🎯 핵심: 실시간 부분 전사로 의도 파악
        if audio_length > 16000:  # 1초 이상 음성 있으면
            partial_text = self.get_partial_transcript(user_id)
            if partial_text:
                # 질문이면 빠르게 처리 (0.5초)
                if self.is_question(partial_text):
                    return silence_time >= 5

                # 망설임이면 충분히 기다리기 (2.0초)
                if self.is_hesitation(partial_text):
                    return silence_time >= 20

                # 일반 문장이면 중간 속도 (0.7초)
                return silence_time >= 7

        # 기본 규칙: 0.5초
        return silence_time >= 5

    async def process_speech(self, user_id: str):
        """완성된 음성 처리"""
        buffer = self.user_buffers[user_id]

        if not buffer["audio"]:
            return

            # 1. STT 처리
        audio_path = f"temp_audio/speech_{user_id}.wav"
        self.save_audio(buffer["audio"], audio_path)

        result = self.whisper_model.transcribe(audio_path)
        user_text = result["text"].strip()

        if not user_text:
            return

        # 입력 로깅
        print(f"\n🎤 사용자 입력: '{user_text}'")

        # 2. LLM 응답 생성 (고정된 시나리오 사용)
        ai_response = await self.llm_service.generate_response(user_text, user_id)

        # 출력 로깅
        print(f"🤖 AI 응답: '{ai_response}'")

        # 3. TTS 생성
        audio_path = await self.tts_service.generate_speech(ai_response)

        # 4. 평가 기록
        await self.evaluation_service.record_interaction(user_id, user_text, ai_response)

        # 5. 버퍼 초기화
        buffer["audio"].clear()
        buffer["speaking"] = False
        buffer["silence"] = 0

        # 6. 임시 파일 정리
        if os.path.exists(f"temp_audio/speech_{user_id}.wav"):
            os.remove(f"temp_audio/speech_{user_id}.wav")

        return {
            "user_text": user_text,
            "ai_text": ai_response,
            "audio_url": f"/static/audio/{Path(audio_path).name}" if audio_path else None
        }

    def get_partial_transcript(self, user_id: str) -> str:
        """실시간 부분 전사 (버퍼 분석용)"""
        try:
            buffer = self.user_buffers[user_id]
            temp_path = f"temp_audio/partial_{user_id}.wav"
            self.save_audio(buffer["audio"], temp_path)

            result = self.whisper_model.transcribe(temp_path)
            partial_text = result["text"].strip()

            # 임시 파일 정리
            if os.path.exists(temp_path):
                os.remove(temp_path)

            return partial_text
        except:
            return ""  # 실패하면 빈 문자열

    def is_question(self, text: str) -> bool:
        """질문인지 판단 - 빠른 처리"""
        text = text.strip().lower()

        # 명확한 질문 패턴들
        if text.endswith(('?', '요?', '세요?', '까요?', '나요?')):
            return True

        # 질문 시작 단어들
        question_starts = ['언제', '어디', '어떻게', '왜', '뭐', '무엇', '어떤', '몇', '혹시']
        if any(text.startswith(word) for word in question_starts):
            return True

        # 의료 질문 패턴들
        medical_questions = ['아프', '통증', '증상', '언제부터', '얼마나']
        question_endings = ['세요', '나요', '어요', '까요']

        for med in medical_questions:
            if med in text:
                for ending in question_endings:
                    if text.endswith(ending):
                        return True

        return False

    def is_hesitation(self, text: str) -> bool:
        """망설임인지 판단 - 더 기다리기"""
        text = text.strip().lower()

        # 망설임 패턴들
        hesitation_patterns = [
            '음', '어', '그', '그런', '아', '어떻게', '잠깐', '잠시',
            '어떻게 말하지', '그게', '그러니까', '아니', '어디보자'
        ]

        # 짧고 망설이는 표현들
        if len(text) < 10 and any(pattern in text for pattern in hesitation_patterns):
            return True

        # 미완성 문장들
        incomplete_endings = ['그런', '그게', '어떻게', '그러니까', '음']
        if any(text.endswith(pattern) for pattern in incomplete_endings):
            return True

        return False

    def save_audio(self, audio_buffer: bytearray, file_path: str):
        """오디오 버퍼를 WAV 파일로 저장"""
        with wave.open(file_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(bytes(audio_buffer))

    def clear_user_buffer(self, user_id: str):
        """사용자 버퍼 초기화"""
        if user_id in self.user_buffers:
            del self.user_buffers[user_id]
