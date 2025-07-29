import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any
import wave
import numpy as np

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from google.cloud import speech

from core.startup import service_manager
from core.config import settings

logger = logging.getLogger(__name__)

# WebSocket 라우터 생성
router = APIRouter()

class AudioProcessor:
    """실시간 오디오 처리 클래스"""
    
    def __init__(self):
        # 사용자별 세션 관리
        self.user_sessions: Dict[str, Dict] = {}
    
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
            }
        return self.user_sessions[user_id]
    
    def clear_user_session(self, user_id: str):
        """사용자 세션 정리"""
        if user_id in self.user_sessions:
            del self.user_sessions[user_id]
    
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
            timestamp = int(asyncio.get_event_loop().time())
            temp_path = settings.TEMP_AUDIO_DIR / f"stream_{user_id}_{timestamp}.wav"
            
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
                
                # AI 응답 생성
                response_data = await self._generate_ai_response(user_id, user_text)
                
                # 취소 확인 (마지막 체크)
                if session["should_cancel"]:
                    logger.info(f"[{user_id}] ⏹️  처리 취소됨 (새 발화 감지)")
                    if temp_path.exists():
                        temp_path.unlink()
                    return
                
                # WebSocket으로 응답 전송
                await websocket.send_text(json.dumps(response_data, ensure_ascii=False))
            else:
                # 음성 인식 실패
                if not session["should_cancel"]:
                    await websocket.send_text(json.dumps({
                        "type": "no_speech",
                        "message": "음성을 인식하지 못했습니다. 다시 말씀해 주세요.",
                        "avatar_action": "listening"
                    }, ensure_ascii=False))
            
            # 임시 파일 정리
            if temp_path.exists():
                temp_path.unlink()
                
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
    
    async def _generate_ai_response(self, user_id: str, user_text: str) -> Dict[str, Any]:
        """AI 응답 생성"""
        try:
            # RAG 검색
            relevant_docs = await service_manager.vector_service.search(user_text)
            
            # LLM 응답 생성
            response_text = await service_manager.llm_service.generate_response(
                user_text, relevant_docs, user_id
            )
            
            # TTS 생성
            audio_path = await service_manager.tts_service.generate_speech(response_text)
            
            # 응답 데이터 구성
            return {
                "type": "voice_response",
                "user_text": user_text,
                "ai_text": response_text,
                "audio_url": f"/static/audio/{Path(audio_path).name}" if audio_path else None,
                "avatar_action": "talking",
                "processing_time": "실시간"
            }
            
        except Exception as e:
            logger.error(f"AI 응답 생성 오류: {e}")
            return {
                "type": "error",
                "message": "응답 생성 중 오류가 발생했습니다.",
                "avatar_action": "error"
            }

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
        # 연결 확인 메시지
        await websocket.send_text(json.dumps({
            "type": "connected",
            "message": f"CPX 시스템에 연결되었습니다. ({user_id})",
            "avatar_action": "idle"
        }, ensure_ascii=False))
        
        while True:
            # 실시간 메시지 수신
            message = await websocket.receive()
            
            if message["type"] == "websocket.receive":
                if "bytes" in message:
                    # 음성 청크 처리
                    await handle_audio_chunk(websocket, user_id, message["bytes"], session)
                    
                elif "text" in message:
                    # 텍스트 명령 처리
                    command = json.loads(message["text"])
                    await handle_command(websocket, user_id, command)
                    
    except WebSocketDisconnect:
        logger.info(f"🔌 연결 해제: {user_id}")
    except Exception as e:
        logger.error(f"WebSocket 오류: {e}")
    finally:
        # 세션 정리
        audio_processor.clear_user_session(user_id)

async def handle_audio_chunk(websocket: WebSocket, user_id: str, audio_chunk: bytes, session: Dict):
    """음성 청크 처리"""
    try:
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

async def handle_command(websocket: WebSocket, user_id: str, command: Dict):
    """클라이언트 명령 처리"""
    cmd_type = command.get("type", "")
    
    if cmd_type == "start_session":
        case_id = command.get("case_id", "IM_001")
        logger.info(f"[{user_id}] 🏥 CPX 세션 시작: {case_id}")
        
        response = {
            "type": "session_started",
            "case_id": case_id,
            "message": "CPX 세션이 시작되었습니다. 환자에게 말을 걸어보세요.",
            "avatar_action": "ready"
        }
        await websocket.send_text(json.dumps(response, ensure_ascii=False))
        
    elif cmd_type == "end_session":
        logger.info(f"[{user_id}] 🏁 CPX 세션 종료")
        
        response = {
            "type": "session_ended",
            "message": "CPX 세션이 종료되었습니다. 수고하셨습니다.",
            "avatar_action": "goodbye"
        }
        await websocket.send_text(json.dumps(response, ensure_ascii=False))
        
    elif cmd_type == "ping":
        # 연결 상태 확인
        await websocket.send_text(json.dumps({
            "type": "pong",
            "message": "연결 상태 양호",
            "timestamp": asyncio.get_event_loop().time()
        }, ensure_ascii=False)) 