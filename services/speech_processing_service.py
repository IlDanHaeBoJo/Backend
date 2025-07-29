import os
import logging
import asyncio
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json
import re
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class ConversationIntent(Enum):
    """대화 의도 분류"""
    QUESTION = "question"           # 질문 (환자가 답변해야 함)
    ACKNOWLEDGMENT = "acknowledgment"  # 호응 (환자가 계속 말함)
    CLOSING = "closing"             # 마무리 (대화 종료)
    GREETING = "greeting"           # 인사 (환자가 인사)
    UNCLEAR = "unclear"             # 불분명 (재질문 유도)

class SpeechProcessingService:
    def __init__(self, whisper_model, llm_service, tts_service, evaluation_service):
        """음성 처리 서비스 초기화"""
        self.whisper_model = whisper_model
        self.llm_service = llm_service
        self.tts_service = tts_service  
        self.evaluation_service = evaluation_service
        
        # 사용자별 음성 버퍼
        self.user_buffers = {}
        
        # 대화 의도 분석을 위한 패턴들
        self.intent_patterns = {
            ConversationIntent.QUESTION: [
                # 직접 질문
                r"어디가?\?|언제부터?\?|어떤?\?|왜?\?|무엇을?\?|어떻게?\?",
                r"아프신가요?\?|있나요?\?|없나요?\?|하시나요?\?",
                r"증상|통증|병력|가족력|알레르기|약물|수술",
                # 진료 관련 질문
                r"검사|진단|치료|처방|병원|의사"
            ],
            ConversationIntent.ACKNOWLEDGMENT: [
                # 호응어  
                r"^(네|네네|응|어|음|아|아하|그렇군요|알겠습니다)$",
                r"^(이해했습니다|그러셨군요|힘드셨겠어요)$",
                # 공감 표현
                r"걱정|이해|공감|마음|힘들",
            ],
            ConversationIntent.CLOSING: [
                # 마무리 표현
                r"마지막으로|마무리|끝|이상|수고|감사|안녕",
                r"다른.*없|더.*없|괜찮|충분|알겠습니다.*감사",
                r"검사.*수고|진료.*끝|오늘.*감사"
            ],
            ConversationIntent.GREETING: [
                # 인사
                r"안녕하세요|처음|만나서|반갑|안녕",
                r"저는|의사|선생님|진료"
            ]
        }
        
        logger.info("음성 처리 서비스 초기화 완료")

    async def process_audio_chunk(self, user_id: str, audio_chunk: bytes, is_silent: bool):
        """음성 청크 처리"""
        # 사용자별 버퍼 초기화
        if user_id not in self.user_buffers:
            self.user_buffers[user_id] = {
                "audio_buffer": bytearray(),
                "is_speaking": False,
                "silence_duration": 0,
                "speech_duration": 0,
                "last_processing_time": 0
            }
        
        user_buffer = self.user_buffers[user_id]
        
        # 음성 데이터 누적
        user_buffer["audio_buffer"].extend(audio_chunk)
        
        if not is_silent:
            # 발화 중
            if not user_buffer["is_speaking"]:
                logger.info(f"[{user_id}] 발화 시작 감지")
                user_buffer["is_speaking"] = True
                user_buffer["silence_duration"] = 0
            
            user_buffer["speech_duration"] += 0.1  # 청크당 약 100ms
            user_buffer["silence_duration"] = 0
            
        else:
            # 침묵 중
            if user_buffer["is_speaking"]:
                user_buffer["silence_duration"] += 0.1
                
                # 발화 완료 판단
                if await self.is_utterance_complete(user_id):
                    await self.process_complete_utterance(user_id)
    
    async def is_utterance_complete(self, user_id: str) -> bool:
        """발화 완료 여부 판단"""
        user_buffer = self.user_buffers[user_id]
        
        # 기본 침묵 시간 체크
        basic_silence_threshold = 1.2
        
        # 너무 짧은 발화는 더 기다림 (호응어 가능성)
        if user_buffer["speech_duration"] < 0.5:
            return user_buffer["silence_duration"] >= 0.8
        
        # 일반적인 발화
        return user_buffer["silence_duration"] >= basic_silence_threshold

    async def process_complete_utterance(self, user_id: str):
        """완성된 발화 처리"""
        user_buffer = self.user_buffers[user_id]
        
        if len(user_buffer["audio_buffer"]) == 0:
            return
        
        try:
            logger.info(f"[{user_id}] 발화 완료 - 분석 시작")
            
            # 1. STT 처리
            temp_path = f"temp_audio/speech_{user_id}_{int(datetime.now().timestamp())}.wav"
            await self.save_audio_as_wav(user_buffer["audio_buffer"], temp_path)
            
            stt_result = self.whisper_model.transcribe(temp_path)
            user_text = stt_result["text"].strip()
            
            if not user_text:
                logger.warning(f"[{user_id}] STT 결과 없음")
                return
            
            logger.info(f"[{user_id}] STT 결과: {user_text}")
            
            # 2. 대화 의도 분석
            intent = await self.analyze_conversation_intent(user_text)
            logger.info(f"[{user_id}] 감지된 의도: {intent.value}")
            
            # 3. 평가 시스템에 기록 (호응어는 긍정 점수)
            await self.record_for_evaluation(user_id, user_text, intent)
            
            # 4. 의도에 따른 응답 생성
            response = await self.generate_contextual_response(user_id, user_text, intent)
            
            # 5. 응답 반환 (WebSocket으로 전송하기 위해)
            return response
            
        except Exception as e:
            logger.error(f"[{user_id}] 발화 처리 오류: {e}")
            return None
            
        finally:
            # 버퍼 초기화
            user_buffer["audio_buffer"].clear()
            user_buffer["is_speaking"] = False
            user_buffer["silence_duration"] = 0
            user_buffer["speech_duration"] = 0
            
            # 임시 파일 정리
            if os.path.exists(temp_path):
                os.remove(temp_path)

    async def analyze_conversation_intent(self, text: str) -> ConversationIntent:
        """대화 의도 분석"""
        text_lower = text.lower().strip()
        
        # 각 의도별 패턴 매칭
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return intent
        
        # 패턴으로 판단 안되면 길이와 구조로 추가 판단
        if len(text) < 10 and not text.endswith('?'):
            # 짧고 질문이 아니면 호응으로 판단
            return ConversationIntent.ACKNOWLEDGMENT
        elif text.endswith('?'):
            # 물음표로 끝나면 질문
            return ConversationIntent.QUESTION
        elif len(text) > 30:
            # 긴 발화는 보통 질문이나 설명
            return ConversationIntent.QUESTION
        
        return ConversationIntent.UNCLEAR

    async def analyze_patient_conversation_state(self, user_id: str, user_text: str) -> Dict:
        """환자의 현재 대화 상태 분석"""
        # 대화 히스토리 분석
        conversation_summary = self.llm_service.get_conversation_summary(user_id)
        
        # 호응 유형 분석
        acknowledgment_type = "neutral"
        if any(word in user_text.lower() for word in ["힘드", "걱정", "이해", "마음", "공감"]):
            acknowledgment_type = "empathetic"  # 공감적 호응 → 환자가 감정적으로 반응할 가능성
        elif any(word in user_text.lower() for word in ["네네", "아하", "그렇군요", "알겠"]):  
            acknowledgment_type = "understanding"  # 이해 표현 → 환자가 추가 설명할 가능성
        elif user_text.strip() in ["네", "음", "어", "응"]:
            acknowledgment_type = "minimal"  # 최소 호응 → 환자가 간단히 대답할 가능성
        
        # 대화 단계 추측
        conversation_stage = "middle"
        total_interactions = len(conversation_summary.split()) if conversation_summary else 0
        if total_interactions < 3:
            conversation_stage = "early"
        elif total_interactions > 15:
            conversation_stage = "late"
        
        return {
            "acknowledgment_type": acknowledgment_type,
            "conversation_stage": conversation_stage,
            "total_interactions": total_interactions
        }

    async def generate_contextual_response(self, user_id: str, user_text: str, intent: ConversationIntent) -> Dict:
        """의도에 따른 맞춤 응답 생성"""
        
        if intent == ConversationIntent.QUESTION:
            # 질문 → 환자가 답변
            response_text = await self.llm_service.generate_response(
                user_text + "\n\n[환자로서 이 질문에 자세히 답변해주세요]",
                [], user_id
            )
            avatar_action = "talking"
            
        elif intent == ConversationIntent.ACKNOWLEDGMENT:
            # 호응 → 환자 상태 분석 후 자연스러운 반응
            patient_state = await self.analyze_patient_conversation_state(user_id, user_text)
            
            # 상황별 맞춤 프롬프트 생성
            if patient_state["acknowledgment_type"] == "minimal":
                # 최소 호응 ("네", "음") → 환자도 간단히 대답
                context_prompt = f"""
학생이 '{user_text}'라고 간단히 호응했습니다.

환자로서 자연스럽게 반응하세요:
- 만약 더 말할 내용이 없다면: "네" 또는 "음" 정도로 간단히 대답
- 학생의 다음 질문을 자연스럽게 기다리는 상태
- 억지로 길게 말하지 마세요

현재 대화 단계: {patient_state["conversation_stage"]}
"""
                avatar_action = "waiting"
                
            elif patient_state["acknowledgment_type"] == "empathetic":
                # 공감적 호응 → 환자가 감정적으로 반응 가능
                context_prompt = f"""
학생이 '{user_text}'라고 공감을 표현했습니다.

환자로서 감정적으로 반응하되 자연스럽게:
- 학생의 공감에 감사함을 표현하거나
- 본인의 감정 상태를 짧게 표현하거나  
- 더 말하고 싶은 게 있다면 추가로 설명

하지만 무리해서 길게 말하지 마세요.
현재 대화 단계: {patient_state["conversation_stage"]}
"""
                avatar_action = "emotional"
                
            else:  # understanding
                # 이해 표현 → 환자가 추가 설명할 수 있음
                context_prompt = f"""
학생이 '{user_text}'라고 이해를 표현했습니다.

환자로서 상황에 맞게 반응하세요:
- 학생이 이해했다면: "네, 맞아요" 정도로 확인
- 추가로 설명할 관련 증상이 있다면: 자연스럽게 덧붙이기
- 특별히 더 말할 게 없다면: 간단히 대답하고 대기

억지로 내용을 만들어내지 마세요.
현재 대화 단계: {patient_state["conversation_stage"]}
총 대화 횟수: {patient_state["total_interactions"]}
"""
                avatar_action = "considering"
            
            response_text = await self.llm_service.generate_response(
                context_prompt, [], user_id
            )
            
        elif intent == ConversationIntent.CLOSING:
            # 마무리 → 정중한 종료
            response_text = await self.llm_service.generate_response(
                user_text + "\n\n[학생이 마무리 인사를 했으니, 환자로서 감사 인사를 간단히 해주세요]",
                [], user_id
            )
            avatar_action = "greeting"
            
        elif intent == ConversationIntent.GREETING:
            # 인사 → 환자 인사 및 준비
            response_text = await self.llm_service.generate_response(
                user_text + "\n\n[학생이 인사했으니, 환자로서 인사하고 진료 준비가 되었음을 표현해주세요]",
                [], user_id
            )
            avatar_action = "greeting"
            
        else:
            # 불분명 → 재질문 유도
            response_text = "죄송한데, 잘 못 들었어요. 다시 한 번 말씀해 주시겠어요?"
            avatar_action = "confused"
        
        # TTS 생성
        audio_path = await self.tts_service.generate_speech(response_text)
        
        return {
            "type": "voice_response",
            "user_text": user_text,
            "ai_text": response_text,
            "audio_url": f"/static/audio/{Path(audio_path).name}" if audio_path else None,
            "avatar_action": avatar_action,
            "detected_intent": intent.value,
            "processing_time": "실시간"
        }

    async def record_for_evaluation(self, user_id: str, user_text: str, intent: ConversationIntent):
        """평가 시스템에 기록"""
        try:
            # 호응어는 의사소통 점수에 긍정적 영향
            if intent == ConversationIntent.ACKNOWLEDGMENT:
                logger.info(f"[{user_id}] 호응 표현 감지 - 의사소통 점수 향상")
                # evaluation_service에 긍정적 피드백 추가
                await self.evaluation_service.record_positive_communication(
                    user_id, "appropriate_acknowledgment", user_text
                )
            
            # 모든 상호작용 기록
            await self.evaluation_service.record_interaction(
                user_id, user_text, "", intent.value
            )
            
        except Exception as e:
            logger.error(f"평가 기록 오류: {e}")

    async def save_audio_as_wav(self, audio_buffer: bytearray, file_path: str):
        """오디오 버퍼를 WAV 파일로 저장"""
        try:
            import wave
            
            # WAV 파일 설정
            sample_rate = 16000
            channels = 1
            sample_width = 2
            
            with wave.open(file_path, 'wb') as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(sample_width)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(bytes(audio_buffer))
                
        except Exception as e:
            logger.error(f"WAV 파일 저장 오류: {e}")
            raise

    def clear_user_buffer(self, user_id: str):
        """사용자 버퍼 초기화"""
        if user_id in self.user_buffers:
            del self.user_buffers[user_id]
            logger.info(f"[{user_id}] 버퍼 초기화 완료") 