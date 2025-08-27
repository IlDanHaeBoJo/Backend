import asyncio
import json
import logging


from typing import Dict, Any

import numpy as np
from datetime import datetime
import base64

from fastapi import APIRouter, WebSocket, WebSocketDisconnect


from core.startup import service_manager
from core.config import settings
from infra.inmemory_queue import (
    enqueue_user_utterance,
    enqueue_ai_utterance,
    enqueue_conversation_ended,
    start_worker_once,
)
from services.cpx_service import CpxService
from core.config import get_db
        
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
            voice_threshold = 1000  # 더 높은 임계값으로 노이즈 제거
            
            return rms > voice_threshold
            
        except Exception as e:
            logger.error(f"VAD 처리 오류: {e}")
            return False
    
    def _convert_buffer_to_numpy(self, audio_buffer: bytearray) -> np.ndarray:
        """오디오 버퍼를 numpy 배열로 변환"""
        try:
            # 16-bit PCM으로 변환
            audio_data = np.frombuffer(audio_buffer, dtype=np.int16)
            
            # float32로 정규화 (-1.0 ~ 1.0)
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            return audio_float
            
        except Exception as e:
            logger.error(f"오디오 버퍼 변환 오류: {e}")
            return np.array([])
    
    async def _perform_stt_from_buffer(self, audio_numpy: np.ndarray) -> str:
        """numpy 배열에서 직접 STT 처리 (Google Cloud Speech)"""
        try:
            if len(audio_numpy) == 0:
                return ""
            
            # STT 서비스로 처리
            result = await service_manager.stt_service.transcribe_from_buffer(audio_numpy)
            return result
            
        except Exception as e:
            logger.error(f"STT 처리 오류: {e}")
            return ""
    
    async def process_complete_utterance(self, websocket: WebSocket, user_id: str, audio_buffer: bytearray, session: dict):
        """완전한 발화 처리 (메모리 버퍼 직접 사용)"""
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
            
            # 오디오 버퍼를 numpy 배열로 변환 (파일 저장 없이)
            audio_numpy = self._convert_buffer_to_numpy(audio_buffer)
            
            # 취소 확인
            if session["should_cancel"]:
                logger.info(f"[{user_id}] ⏹️  처리 취소됨 (새 발화 감지)")
                return
            
            # STT 처리 (메모리 버퍼 직접 사용)
            user_text = await self._perform_stt_from_buffer(audio_numpy)
            
            # 취소 확인
            if session["should_cancel"]:
                logger.info(f"[{user_id}] ⏹️  처리 취소됨 (새 발화 감지)")
                return
            
            if user_text:
                logger.info(f"[{user_id}] STT 결과: {user_text}")
                
                # 사용자 발화 큐 적재 (오디오 버퍼 직접 전달)
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
                            audio_buffer,  # 오디오 버퍼 직접 전달
                            user_text,
                        )
                    )
                
                # AI 응답 생성 (오디오 버퍼 직접 전달)
                response_data = await self._generate_ai_response(user_id, user_text, audio_buffer)
                
            # 취소 확인 (마지막 체크)
                if session["should_cancel"]:
                    logger.info(f"[{user_id}] ⏹️  처리 취소됨 (새 발화 감지)")
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
                                response_data.get("tts_audio_buffer"),  # TTS 버퍼 직접 전달
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
                
        except Exception as e:
            logger.error(f"발화 처리 오류: {e}")
            if not session["should_cancel"]:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "음성 처리 중 오류가 발생했습니다.",
                    "avatar_action": "error"
                }, ensure_ascii=False))
    
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
    
    async def _generate_ai_response(self, user_id: str, user_text: str, audio_buffer: bytearray = None) -> Dict[str, Any]:
        """AI 응답 생성 (메모리 버퍼 기반)"""
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
            
            # TTS 생성 (메모리 버퍼로 반환)
            tts_audio_buffer = await service_manager.tts_service.generate_speech(response_text)
            logger.info(f"🔊 TTS 오디오 생성 완료 (메모리 버퍼)")
            
            # TTS 오디오 버퍼를 Base64로 인코딩
            tts_audio_base64 = None
            if tts_audio_buffer:
                tts_audio_base64 = base64.b64encode(tts_audio_buffer).decode('utf-8')
                logger.info(f"🔊 TTS 오디오 Base64 인코딩 완료 ({len(tts_audio_base64)} 문자)")
            
            # 응답 데이터 구성
            response_data = {
                "type": "voice_response",
                "user_text": user_text,
                "ai_text": response_text,
                "tts_audio_base64": tts_audio_base64,  # Base64 인코딩된 오디오
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


        # 기본 시나리오(치매) 설정 및 평가 세션 시작
        default_scenario_id = "1"  # 치매 시나리오
        session["scenario_id"] = default_scenario_id
        session["session_start_time"] = datetime.now().isoformat()
        
        # CPX 결과 생성 (평가 시작)
        cpx_result_id = None
        async for db in get_db():
            cpx_service = CpxService(db)
            cpx_result = await cpx_service.create_cpx_result(
                student_id=int(user_id),
                patient_name="AI 환자",
                evaluation_status="진행중"
            )
            cpx_result_id = cpx_result.result_id
            break
        
        # 세션에 CPX result_id와 user_id 저장
        session["result_id"] = cpx_result_id
        session["user_id"] = user_id
        
        # LLM 서비스에 시나리오 설정
        service_manager.llm_service.select_scenario(default_scenario_id, user_id)
        
        # 평가 세션 시작 (result_id 전달)
        eval_session_id = await service_manager.evaluation_service.start_evaluation_session(
            user_id, default_scenario_id, cpx_result_id
        )
        audio_processor.user_evaluation_sessions[user_id] = eval_session_id
        
        print(f"🎭 [{user_id}] 기본 시나리오({default_scenario_id}) 설정 및 평가 세션 시작: {eval_session_id}")
        
        # 시작 메시지 전송
        await websocket.send_text(json.dumps({
            "type": "session_started",
            "message": f"🏥 CPX 시스템에 연결되었습니다! ({user_id})\n\n 기억력 저하 시나리오가 설정되었습니다.\n지금부터 환자에게 말을 걸어보세요.",
            "scenario_id": default_scenario_id,
            "result_id": cpx_result_id,
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
