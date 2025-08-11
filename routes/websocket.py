import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any
import wave
import numpy as np
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from google.cloud import speech

from core.startup import service_manager
from core.config import settings
from infra.inmemory_queue import (
    enqueue_user_utterance,
    enqueue_ai_utterance,
    enqueue_conversation_ended,
    start_worker_once,
)

logger = logging.getLogger(__name__)

# WebSocket 라우터 생성
router = APIRouter()

class AudioProcessor:
    """실시간 오디오 처리 클래스"""
    
    def __init__(self):
        # 사용자별 세션 관리
        self.user_sessions: Dict[str, Dict] = {}
        # 평가 세션 ID 관리
        self.user_evaluation_sessions: Dict[str, str] = {}  # user_id -> session_id
    
    def get_user_session(self, user_id: str) -> Dict[str, Any]:
        """사용자 세션 가져오기 또는 생성"""
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                "audio_buffer": bytearray(),
                "is_speaking": False,
                "silence_duration": 0,
                "min_speech_duration": 1.0,
                "max_silence_duration": 1.0,  # 빠른 응답
                "is_processing": False,  # STT 처리 중 플래그
                "should_cancel": False,  # 처리 취소 플래그
                "conversation_ended": False,  # 대화 종료 플래그
                # "conversation_log": [],  # 대화 로그 저장
                "scenario_id": None,  # 선택된 시나리오
                "session_start_time": None,  # 세션 시작 시간
            }
        return self.user_sessions[user_id]
    
    def clear_user_session(self, user_id: str):
        """사용자 세션 정리"""
        if user_id in self.user_sessions:
            del self.user_sessions[user_id]
        if user_id in self.user_evaluation_sessions:
            del self.user_evaluation_sessions[user_id]
    
    async def detect_voice_activity(self, audio_chunk: bytes) -> bool:
        """음성 활동 감지 (VAD)"""
        try:
            if len(audio_chunk) == 0:
                return False
            
            # 16-bit PCM으로 변환
            audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
            
            if len(audio_data) == 0:
                return False
            
            # RMS 계산으로 음성 레벨 측정
            rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
            
            # 동적 임계값 (환경에 따라 조정 가능)
            voice_threshold = 300  # 조정 가능
            
            return rms > voice_threshold
            
        except Exception as e:
            logger.error(f"VAD 처리 오류: {e}")
            return False
    
    async def save_audio_buffer_as_wav(self, audio_buffer: bytearray, file_path: str):
        """오디오 버퍼를 WAV 파일로 저장"""
        try:
            with wave.open(file_path, 'wb') as wav_file:
                wav_file.setnchannels(settings.AUDIO_CHANNELS)
                wav_file.setsampwidth(settings.AUDIO_SAMPLE_WIDTH)
                wav_file.setframerate(settings.AUDIO_SAMPLE_RATE)
                wav_file.writeframes(bytes(audio_buffer))
                
        except Exception as e:
            logger.error(f"WAV 파일 저장 오류: {e}")
            raise
    
    async def process_complete_utterance(self, websocket: WebSocket, user_id: str, audio_buffer: bytearray, session: dict):
        """완전한 발화 처리 (취소 가능)"""
        if len(audio_buffer) == 0:
            return
        
        try:
            logger.info(f"[{user_id}] 발화 완료, STT 처리 시작")
            
            # 취소 확인
            if session["should_cancel"]:
                logger.info(f"[{user_id}] ⏹️  처리 취소됨 (새 발화 감지)")
                return
            
            # 처리 중 상태 전송
            await websocket.send_text(json.dumps({
                "type": "processing",
                "message": "음성을 분석하고 있습니다...",
                "avatar_action": "thinking"
            }, ensure_ascii=False))
            
            # 취소 확인
            if session["should_cancel"]:
                logger.info(f"[{user_id}] ⏹️  처리 취소됨 (새 발화 감지)")
                return
            
            # 임시 WAV 파일 생성
            timestamp = int(asyncio.get_event_loop().time()) # ??

            # 사용자별 하위 디렉터리 생성 (세션별)
            user_audio_dir = settings.TEMP_AUDIO_DIR / str(user_id) / settings.RUN_ID # "temp_audio/user_id/run_id(250807_151053)"
            user_audio_dir.mkdir(parents=True, exist_ok=True)
            temp_path = user_audio_dir / f"stream_{timestamp}.wav"
            
            # 오디오 저장
            await self.save_audio_buffer_as_wav(audio_buffer, str(temp_path))
            
            # 취소 확인
            if session["should_cancel"]:
                logger.info(f"[{user_id}] ⏹️  처리 취소됨 (새 발화 감지)")
                if temp_path.exists():
                    temp_path.unlink()
                return
            
            # STT 처리
            user_text = await self._perform_stt(temp_path)
            
            # 취소 확인
            if session["should_cancel"]:
                logger.info(f"[{user_id}] ⏹️  처리 취소됨 (새 발화 감지)")
                if temp_path.exists():
                    temp_path.unlink()
                return
            
            if user_text:
                logger.info(f"[{user_id}] STT 결과: {user_text}")
                
                # 사용자 발화 큐 적재 (비동기, 백오프 재시도)
                if user_id in self.user_evaluation_sessions:
                    session_id = self.user_evaluation_sessions[user_id]
                    session.setdefault("seq", 0)
                    session["seq"] += 1
                    asyncio.create_task(
                        self._enqueue_with_retry(
                            enqueue_user_utterance,
                            session_id,
                            user_id,
                            session["seq"],
                            str(temp_path),
                            user_text,
                        )
                    )
                
                # AI 응답 생성 (음성 파일 경로 포함)
                response_data = await self._generate_ai_response(user_id, user_text, str(temp_path))
                
            # 취소 확인 (마지막 체크)
                if session["should_cancel"]:
                    logger.info(f"[{user_id}] ⏹️  처리 취소됨 (새 발화 감지)")
                    if temp_path.exists():
                        temp_path.unlink()
                    return
                
                # 대화 종료 확인 및 세션에 플래그 설정
                if response_data.get("conversation_ended", False):
                    session["conversation_ended"] = True
                    logger.info(f"🏁 [{user_id}] 세션에 대화 종료 플래그 설정 - 이후 음성 처리 차단")
                
                # WebSocket으로 응답 전송
                await websocket.send_text(json.dumps(response_data, ensure_ascii=False))

                # 백그라운드 큐 적재 (AI 발화 / 종료)
                if user_id in self.user_evaluation_sessions:
                    session_id = self.user_evaluation_sessions[user_id]
                    if not response_data.get("conversation_ended", False):
                        # AI 발화 큐 적재 (비동기, 백오프 재시도)
                        session.setdefault("seq", 0)
                        session["seq"] += 1
                        asyncio.create_task(
                            self._enqueue_with_retry(
                                enqueue_ai_utterance,
                                session_id,
                                user_id,
                                session["seq"],
                                response_data.get("audio_path"),
                                response_data.get("ai_text", ""),
                            )
                        )
                    else:
                        # 종료 신호 큐 적재 후 소켓 종료 (비동기, 백오프 재시도)
                        session.setdefault("seq", 0)
                        session["seq"] += 1
                        asyncio.create_task(
                            self._enqueue_with_retry(
                                enqueue_conversation_ended,
                                session_id,
                                user_id,
                                session["seq"],
                            )
                        )
                        try:
                            await websocket.close(code=1000)
                        except Exception:
                            pass
            else:
                # 음성 인식 실패
                if not session["should_cancel"]:
                    await websocket.send_text(json.dumps({
                        "type": "no_speech",
                        "message": "음성을 인식하지 못했습니다. 다시 말씀해 주세요.",
                        "avatar_action": "listening"
                    }, ensure_ascii=False))
            
            # 임시 파일 정리 - 평가 서비스에서 관리하므로 여기서는 삭제하지 않음
            # 평가 완료 시 _cleanup_audio_files()에서 일괄 삭제
            # if temp_path.exists():
            #     temp_path.unlink()
                
        except Exception as e:
            logger.error(f"발화 처리 오류: {e}")
            if not session["should_cancel"]:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "음성 처리 중 오류가 발생했습니다.",
                    "avatar_action": "error"
                }, ensure_ascii=False))
    
    async def _perform_stt(self, audio_file_path: Path) -> str:
        """STT 수행"""
        try:
            with open(audio_file_path, "rb") as audio_file:
                audio_content = audio_file.read()
            
            audio = speech.RecognitionAudio(content=audio_content)
            response = service_manager.speech_client.recognize(
                config=service_manager.speech_config, 
                audio=audio
            )
            
            # 인식 결과 수집
            user_text = ""
            for result in response.results:
                user_text += result.alternatives[0].transcript
            
            # 한국어 의료 용어 후처리
            user_text = self._correct_medical_terms(user_text.strip())
            
            return user_text
            
        except Exception as e:
            logger.error(f"STT 처리 오류: {e}")
            return ""
    
    def _correct_medical_terms(self, text: str) -> str:
        """한국어 의료 용어 오인식 교정"""
        medical_corrections = {
            "위염": ["위엽", "위열"],
            "복통": ["복퉁", "복튼", "복톤"],
            "두통": ["두튼", "두톤", "투통"],
            "어지럼": ["어지럼증", "어지러움"],
            "구토": ["굴토", "쿠토"],
            "설사": ["셜사", "설싸"]
        }
        
        for correct, variants in medical_corrections.items():
            for variant in variants:
                if variant in text and variant != correct:
                    text = text.replace(variant, correct)
                    logger.info(f"의료 용어 교정: {variant} → {correct}")
        
        return text
    
    async def _perform_automatic_evaluation(self, user_id: str, session: Dict) -> Dict:
        """대화 종료 시 자동 CPX 평가 실행"""
        try:
            print(f"🎯 [{user_id}] CPX 자동 평가 시작")
            
            # 평가에 필요한 정보 수집
            conversation_log = session.get("conversation_log", [])
            scenario_id = session.get("scenario_id", "unknown")
            
            if not conversation_log:
                print(f"⚠️ [{user_id}] 대화 로그가 비어있음")
                return {
                    "error": "대화 내용이 없어 평가할 수 없습니다.",
                    "scores": {"total_score": 0}
                }
            
            # 대화 로그를 평가 서비스에 맞는 형식으로 변환
            formatted_conversation = []
            for entry in conversation_log:
                formatted_conversation.extend([
                    {
                        "type": "student",
                        "content": entry.get("student_input", ""),
                        "timestamp": entry.get("timestamp", "")
                    },
                    {
                        "type": "patient", 
                        "content": entry.get("patient_response", ""),
                        "timestamp": entry.get("timestamp", "")
                    }
                ])
            
            print(f"📋 [{user_id}] 평가 데이터: {len(formatted_conversation)}개 메시지, 시나리오: {scenario_id}")
            
            # 기존 평가 세션 ID 확인 (이미 실시간으로 데이터가 수집되고 있음)
            if user_id in audio_processor.user_evaluation_sessions:
                session_id = audio_processor.user_evaluation_sessions[user_id]
                print(f"🎯 [{user_id}] 기존 평가 세션 사용: {session_id}")
                
                # 평가 세션 종료 및 종합 평가 수행 (실시간 수집된 데이터 사용)
                evaluation_result = await service_manager.evaluation_service.end_evaluation_session(session_id)
                
            else:
                # 백업: 평가 세션이 없으면 새로 생성 (기존 방식)
                print(f"⚠️ [{user_id}] 기존 평가 세션이 없어 새로 생성합니다")
                session_id = await service_manager.evaluation_service.start_evaluation_session(
                    user_id=user_id,
                    scenario_id=scenario_id
                )
                
                # 대화 로그를 인터랙션으로 변환하여 추가
                for i in range(0, len(formatted_conversation), 2):
                    if i + 1 < len(formatted_conversation):
                        student_msg = formatted_conversation[i]
                        patient_msg = formatted_conversation[i + 1]
                        
                        await service_manager.evaluation_service.add_interaction(
                            session_id=session_id,
                            student_question=student_msg.get("content", ""),
                            patient_response=patient_msg.get("content", ""),
                            audio_file_path=None
                        )
                
                # 평가 세션 종료 및 종합 평가 수행
                evaluation_result = await service_manager.evaluation_service.end_evaluation_session(session_id)
            
            print(f"✅ [{user_id}] 평가 완료 - 총점: {evaluation_result.get('scores', {}).get('total_score', 0)}")
            
            # 평가 세션 정리
            if user_id in audio_processor.user_evaluation_sessions:
                del audio_processor.user_evaluation_sessions[user_id]
                print(f"🧹 [{user_id}] 평가 세션 정리 완료")
            
            # 평가 결과를 데이터베이스에 저장 (향후 구현)
            # await self._save_evaluation_to_database(user_id, evaluation_result)
            
            return evaluation_result
            
        except Exception as e:
            print(f"❌ [{user_id}] 자동 평가 오류: {e}")
            return {
                "error": f"평가 중 오류가 발생했습니다: {str(e)}",
                "scores": {"total_score": 0}
            }
    
    async def _generate_ai_response(self, user_id: str, user_text: str, audio_file_path: str = None) -> Dict[str, Any]:
        """AI 응답 생성 (시나리오 기반)"""
        try:
            # 세션 정보 가져오기
            session = self.get_user_session(user_id)
            
            # 입력 로깅
            print(f"\n🎤 사용자 입력: '{user_text}'")
            
            # LLM 응답 생성 (고정된 시나리오 사용)
            llm_response = await service_manager.llm_service.generate_response(user_text, user_id)
            response_text = llm_response["text"]
            conversation_ended = llm_response["conversation_ended"]
            
            # 출력 로깅
            print(f"🤖 AI 응답: '{response_text}'")
            
            # 대화 로그에 저장
            # conversation_entry = {
            #     "timestamp": datetime.now().isoformat(),
            #     "student_input": user_text,
            #     "patient_response": response_text,
            #     "type": "interaction"
            # }
            # session["conversation_log"].append(conversation_entry)
            
            # TTS 생성
            audio_path = await service_manager.tts_service.generate_speech(response_text)
            logger.info(f"🔊 TTS 파일 생성됨: {audio_path}")
            
            # AI 발화의 큐 적재는 process_complete_utterance에서 수행 (여기서는 하지 않음)
            
            # 응답 데이터 구성
            audio_url = Path(audio_path).name if audio_path else None
            logger.info(f"🔗 WebSocket 전송할 audio_url: {audio_url}")
            response_data = {
                "type": "voice_response",
                "user_text": user_text,
                "ai_text": response_text,
                "audio_url": audio_url,
                "avatar_action": "talking",
                "processing_time": "실시간",
                "conversation_ended": conversation_ended,
            }
            
            # 대화 종료 시 특별 처리(응답 구성만 변경). 평가는 워커가 종료 이벤트 + pending으로 트리거
            if conversation_ended:
                response_data["type"] = "conversation_ended"
                response_data["avatar_action"] = "goodbye"
                response_data["message"] = "진료가 완료되었습니다. 평가 결과는 곧 저장됩니다."
            
            return response_data
            
        except Exception as e:
            logger.error(f"AI 응답 생성 오류: {e}")
            return {
                "type": "error",
                "message": "응답 생성 중 오류가 발생했습니다.",
                "avatar_action": "error",
                "conversation_ended": False
            }
    
    async def _perform_automatic_evaluation(self, user_id: str, session: Dict) -> Dict[str, Any]:
        """대화 종료 시 자동 평가 수행"""
        try:
            # 평가 세션 ID 가져오기
            if user_id not in self.user_evaluation_sessions:
                raise Exception("평가 세션을 찾을 수 없습니다.")
                
            session_id = self.user_evaluation_sessions[user_id]
            print(f"🔍 [{user_id}] 자동 평가 시작 - 세션: {session_id}")
            
            # 평가 서비스를 통해 세션 종료 및 종합 평가 수행
            evaluation_result = await service_manager.evaluation_service.end_evaluation_session(session_id)
            
            if "error" in evaluation_result:
                raise Exception(f"평가 수행 중 오류: {evaluation_result['error']}")
            
            print(f"✅ [{user_id}] 자동 평가 완료")
            return evaluation_result
            
        except Exception as e:
            logger.error(f"자동 평가 오류: {e}")
            return {
                "error": str(e),
                "scores": {"total_score": 0},
                "message": "평가 중 오류가 발생했습니다."
            }
    
    async def _add_user_conversation_entry_async(self, session_id: str, audio_file_path: str, 
                                               text: str, user_id: str):
        """사용자 대화 엔트리를 백그라운드에서 비동기로 추가"""
        try:
            await service_manager.evaluation_service.add_conversation_entry(
                session_id=session_id,
                audio_file_path=audio_file_path,
                text=text,
                speaker_role="user"  # 환자(사용자) 발화
            )
            logger.info(f"[{user_id}] 사용자 대화 엔트리 백그라운드 추가 완료")
        except Exception as e:
            logger.error(f"[{user_id}] 사용자 대화 엔트리 백그라운드 추가 실패: {e}")
    
    async def _add_ai_conversation_entry_async(self, session_id: str, audio_file_path: str, 
                                             text: str, user_id: str):
        """AI 대화 엔트리를 백그라운드에서 비동기로 추가"""
        try:
            await service_manager.evaluation_service.add_conversation_entry(
                session_id=session_id,
                audio_file_path=audio_file_path,
                text=text,
                speaker_role="assistant"  # 의사(AI) 응답
            )
            logger.info(f"[{user_id}] AI 대화 엔트리 백그라운드 추가 완료")
        except Exception as e:
            logger.error(f"[{user_id}] AI 대화 엔트리 백그라운드 추가 실패: {e}")
    
    async def _record_interaction_async(self, session_id: str, user_text: str, 
                                      response_text: str, audio_file_path: str, user_id: str):
        """인터랙션을 백그라운드에서 비동기로 기록"""
        try:
            await service_manager.evaluation_service.record_interaction(
                session_id, user_text, response_text, audio_file_path
            )
            logger.info(f"[{user_id}] 인터랙션 백그라운드 기록 완료")
        except Exception as e:
            logger.error(f"[{user_id}] 인터랙션 백그라운드 기록 실패: {e}")

    async def _enqueue_with_retry(self, func, *args, **kwargs):
        """비동기 큐 적재에 대한 간단한 백오프 재시도 래퍼
        - func: enqueue_user_utterance/enqueue_ai_utterance/enqueue_conversation_ended
        - args: 해당 함수 인자 그대로 전달
        정책: 3회 재시도, 0.1s, 0.3s, 0.9s 지수 백오프
        실패 시 로깅만 하고 드롭(요구사항상 유실 허용)
        """
        delays = [0.1, 0.3, 0.9]
        last_exc = None
        for i, delay in enumerate([0.0] + delays):
            if delay > 0:
                await asyncio.sleep(delay)
            try:
                await func(*args, **kwargs)
                return True
            except Exception as e:
                last_exc = e
                logger.warning(
                    f"enqueue retry {i}/{len(delays)} failed: func={getattr(func, '__name__', str(func))}, err={e}"
                )
        logger.error(
            f"enqueue failed after retries: func={getattr(func, '__name__', str(func))}, args={args}, kwargs={kwargs}, err={last_exc}"
        )
        return False
    
    async def _background_evaluation_workflow(self, context: Dict):
        """백그라운드에서 평가 워크플로우 실행 (WebSocket 독립적)"""
        user_id = context["user_id"]
        session_id = context["session_id"]
        
        try:
            print(f"🔄 [{user_id}] 백그라운드 평가 워크플로우 시작 - 세션: {session_id}")
            
            # 평가 실행 (모든 대화 데이터가 이미 완료된 상태)
            evaluation_result = await service_manager.evaluation_service.end_evaluation_session(session_id)
            
            if "error" in evaluation_result:
                raise Exception(f"평가 오류: {evaluation_result['error']}")
            
            print(f"✅ [{user_id}] 백그라운드 평가 완료 - 총점: {evaluation_result.get('scores', {}).get('total_score', 0)}")
            
            # DB에 평가 결과 저장 (향후 구현)
            await self._save_evaluation_to_database(user_id, evaluation_result)
            
            # 평가 완료 후 세션 정리
            if user_id in audio_processor.user_evaluation_sessions:
                del audio_processor.user_evaluation_sessions[user_id]
                print(f"🧹 [{user_id}] 평가 세션 정리 완료")
            
            print(f"🎉 [{user_id}] 백그라운드 평가 워크플로우 완료 - DB 저장 성공")
            
        except Exception as e:
            print(f"❌ [{user_id}] 백그라운드 평가 워크플로우 오류: {e}")
            logger.error(f"백그라운드 평가 워크플로우 실패 [{user_id}]: {e}")
    
    async def _save_evaluation_to_database(self, user_id: str, evaluation_result: Dict):
        """평가 결과를 데이터베이스에 저장 (향후 구현)"""
        try:
            # TODO: 실제 DB 저장 로직 구현
            # 예시:
            # async with get_db_session() as db:
            #     await db.save_evaluation_result(user_id, evaluation_result)
            print(f"💾 [{user_id}] DB 저장 시뮬레이션 완료")
            # 실제로는 여기서 데이터베이스에 저장
            
        except Exception as e:
            print(f"❌ [{user_id}] DB 저장 오류: {e}")
            logger.error(f"DB 저장 실패 [{user_id}]: {e}")

# 오디오 프로세서 인스턴스
audio_processor = AudioProcessor()

@router.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket 실시간 음성 스트리밍"""
    await websocket.accept()
    logger.info(f"🔗 실시간 음성 연결: {user_id}")
    
    # 사용자 세션 초기화
    session = audio_processor.get_user_session(user_id)
    
    try:
        # 워커 시작 (최초 1회)
        start_worker_once()

        # 시나리오 선택 메시지 전송
        scenarios = service_manager.llm_service.get_available_scenarios()
        scenario_options = "\n".join([f"{k}. {v}" for k, v in scenarios.items()])
        # 기본 시나리오(치매) 설정 및 평가 세션 시작
        default_scenario_id = "3"  # 치매 시나리오
        session["scenario_id"] = default_scenario_id
        session["session_start_time"] = datetime.now().isoformat()
        
        # LLM 서비스에 시나리오 설정
        service_manager.llm_service.select_scenario(default_scenario_id, user_id)
        
        # 평가 세션 시작
        eval_session_id = await service_manager.evaluation_service.start_evaluation_session(
            user_id, default_scenario_id
        )
        audio_processor.user_evaluation_sessions[user_id] = eval_session_id
        
        print(f"🎭 [{user_id}] 기본 시나리오({default_scenario_id}) 설정 및 평가 세션 시작: {eval_session_id}")
        
        # 시작 메시지 전송
        await websocket.send_text(json.dumps({
            # "type": "scenario_selection",
            # "message": f"🏥 CPX 시스템에 연결되었습니다! ({user_id})\n\n📋 시나리오를 선택해주세요:\n{scenario_options}\n\n번호를 입력하고 음성으로 '시작'이라고 말씀해주세요.",
            # "scenarios": scenarios,
            # "avatar_action": "idle"
            "type": "session_started",
            "message": f"🏥 CPX 시스템에 연결되었습니다! ({user_id})\n\n치매 환자 시나리오가 설정되었습니다.\n지금부터 환자에게 말을 걸어보세요.",
            "scenario_id": default_scenario_id,
            "avatar_action": "ready"
        }, ensure_ascii=False))
        
        while True:
            # 실시간 메시지 수신
            message = await websocket.receive()
            
            if message["type"] == "websocket.receive":
                if "bytes" in message:
                    # 음성 청크 처리
                    await handle_audio_chunk(websocket, user_id, message["bytes"], session)
                    
    except WebSocketDisconnect:
        logger.info(f"🔌 연결 해제: {user_id}")
    except Exception as e:
        logger.error(f"WebSocket 오류: {e}")
    finally:
        # 평가 세션 보호: 평가 진행 중이면 정리 연기
        if user_id in audio_processor.user_evaluation_sessions:
            session_id = audio_processor.user_evaluation_sessions[user_id]
            print(f"⚠️ [{user_id}] 평가 세션 보호 - 백그라운드 평가 완료까지 대기: {session_id}")
            # 평가 세션은 백그라운드에서 정리하도록 함
        
        # WebSocket 세션만 정리 (평가 세션은 보존)
        if user_id in audio_processor.user_sessions:
            del audio_processor.user_sessions[user_id]
        
        # LLM 서비스의 사용자 상태도 정리 (메모리 절약)
        service_manager.llm_service.clear_user_memory(user_id)
        logger.info(f"🧹 [{user_id}] WebSocket 세션 정리 완료 (평가 세션 보존)")

async def handle_audio_chunk(websocket: WebSocket, user_id: str, audio_chunk: bytes, session: Dict):
    """음성 청크 처리"""
    try:
        # 대화 종료 확인 - 종료된 경우 음성 처리 차단
        if session.get("conversation_ended", False):
            logger.info(f"🔒 [{user_id}] 대화 종료됨 - 음성 처리 차단")
            return
        
        # 오디오 버퍼에 추가
        session["audio_buffer"].extend(audio_chunk)
        
        # 음성 활동 감지
        has_voice = await audio_processor.detect_voice_activity(audio_chunk)
        
        if has_voice:
            if not session["is_speaking"]:
                # 새로운 발화 시작 - 기존 처리 중이면 취소
                if session["is_processing"]:
                    logger.info(f"[{user_id}] 🔄 새 발화 감지 - 기존 처리 취소")
                    session["should_cancel"] = True
                    session["is_processing"] = False
                    # 기존 버퍼 초기화
                    session["audio_buffer"].clear()
                
                logger.info(f"[{user_id}] 🎤 발화 시작")
                session["is_speaking"] = True
                session["silence_duration"] = 0
                session["should_cancel"] = False
                
                # 리스닝 상태 전송
                await websocket.send_text(json.dumps({
                    "type": "listening",
                    "message": "음성을 듣고 있습니다...",
                    "avatar_action": "listening"
                }, ensure_ascii=False))
            
            session["silence_duration"] = 0
            
            # 짧은 발화는 호응어일 가능성 - 더 빠르게 처리
            if len(session["audio_buffer"]) > 0:
                buffer_duration = len(session["audio_buffer"]) / (16000 * 2)  # 초 단위
                if buffer_duration > 0.3 and buffer_duration < 1.5:  # 0.3초~1.5초 짧은 발화
                    session["max_silence_duration"] = 0.7  # 호응어는 더 빠르게 처리
                else:
                    session["max_silence_duration"] = 1.0  # 일반 발화
        else:
            # 침묵 구간
            if session["is_speaking"]:
                session["silence_duration"] += 0.1  # 100ms 단위
                
                # 충분한 침묵시 발화 종료 처리
                if session["silence_duration"] >= session["max_silence_duration"]:
                    # STT 처리 시작 표시
                    session["is_processing"] = True
                    
                    await audio_processor.process_complete_utterance(
                        websocket, user_id, session["audio_buffer"], session
                    )
                    
                    # 취소되지 않았으면 세션 초기화
                    if not session["should_cancel"]:
                        session["audio_buffer"].clear()
                        session["is_speaking"] = False
                        session["silence_duration"] = 0
                        session["is_processing"] = False
                    
    except Exception as e:
        logger.error(f"오디오 청크 처리 오류: {e}")

 