import asyncio
from dataclasses import dataclass
from typing import Optional, Dict, Union
from datetime import datetime
import logging

# service_manager를 직접 참조하여 평가 서비스를 호출
from core.startup import service_manager


@dataclass
class QueueEvent:
    session_id: str
    user_id: str
    seq: int
    event_type: str  # "doctor", "patient", "ended"
    audio_path: Optional[str]
    text: Optional[str]
    timestamp: str


_queue: Optional[asyncio.Queue] = None
_worker_task: Optional[asyncio.Task] = None

# 세션별 상태 추적: pending 이벤트 수와 종료 플래그
_session_state: Dict[str, Dict[str, Union[int, bool]]] = {}
logger = logging.getLogger(__name__)

def _get_queue() -> asyncio.Queue:
    global _queue
    if _queue is None:
        _queue = asyncio.Queue()
    return _queue


def _ensure_session_state(session_id: str):
    if session_id not in _session_state:
        _session_state[session_id] = {"pending": 0, "ended": False}


async def enqueue_user_utterance(session_id: str, user_id: str, seq: int, audio_path: str, text: str):
    _ensure_session_state(session_id)
    _session_state[session_id]["pending"] += 1
    await _get_queue().put(
        QueueEvent(
            session_id=session_id,
            user_id=user_id,
            seq=seq,
            event_type="doctor",
            audio_path=audio_path,
            text=text,
            timestamp=datetime.utcnow().isoformat(),
        )
    )
    logger.info(f"유저 발화 세션 큐 등록 = {session_id}, seq={seq}")


async def enqueue_ai_utterance(session_id: str, user_id: str, seq: int, audio_path: Optional[str], text: str):
    _ensure_session_state(session_id)
    _session_state[session_id]["pending"] += 1
    await _get_queue().put(
        QueueEvent(
            session_id=session_id,
            user_id=user_id,
            seq=seq,
            event_type="patient",
            audio_path=audio_path,
            text=text,
            timestamp=datetime.utcnow().isoformat(),
        )
    )
    logger.info(f"AI 발화 세션 큐 등록 = {session_id}, seq={seq}")


async def enqueue_conversation_ended(session_id: str, user_id: str, seq: int):
    _ensure_session_state(session_id)
    # ended 이벤트 자체는 pending 증가 없이 넣고, ended 플래그만 세팅
    await _get_queue().put(
        QueueEvent(
            session_id=session_id,
            user_id=user_id,
            seq=seq,
            event_type="ended",
            audio_path=None,
            text=None,
            timestamp=datetime.utcnow().isoformat(),
        )
    )
    _session_state[session_id]["ended"] = True
    logger.info(f"대화 종료")


async def _process_event(ev: QueueEvent):
    """각 이벤트를 처리. doctor 음성인 경우 SER 분석 후 evaluation 서비스로 전달.
    처리 성공 시 pending 감소.
    """
    try:
        emotion_analysis = None
        
        # doctor 음성인 경우 SER 분석 수행
        if ev.event_type == "doctor" and ev.audio_path:
            try:
                emotion_result = await service_manager.ser_service.analyze_emotion(ev.audio_path)
                if emotion_result.get("success"):
                    emotion_analysis = {
                        "predicted_emotion": emotion_result["predicted_emotion"],
                        "confidence": emotion_result["confidence"],
                        "emotion_scores": emotion_result["emotion_scores"]
                    }
                    logger.info(f"🎭 [{ev.session_id}] SER 분석 완료: {emotion_analysis['predicted_emotion']} ({emotion_analysis['confidence']:.2f})")
                else:
                    logger.warning(f"⚠️ [{ev.session_id}] SER 분석 실패: {emotion_result.get('error', 'Unknown error')}")
            except Exception as ser_error:
                logger.error(f"❌ [{ev.session_id}] SER 분석 오류: {ser_error}")
                # SER 실패해도 평가는 계속 진행
        
        # evaluation 서비스로 전달 (SER 결과 포함)
        if ev.event_type in ("doctor", "patient"):
            await service_manager.evaluation_service.add_conversation_entry(
                session_id=ev.session_id,
                audio_file_path=ev.audio_path or "",
                text=ev.text or "",
                speaker_role=ev.event_type,  # "doctor" or "patient"
                emotion_analysis=emotion_analysis  # SER 결과 전달
            )
            _session_state[ev.session_id]["pending"] -= 1

        elif ev.event_type == "ended":
            # 종료 이벤트는 상태 플래그만 반영(이미 enqueue 시 true), 여기서 즉시 평가는 하지 않고 아래에서 통합 판단
            pass

    except Exception as e:
        # 실패 시 pending 되돌림(patient/doctor만), 재시도 전략은 추후 확장 가능
        try:
            if ev.event_type in ("doctor", "patient"):
                _session_state[ev.session_id]["pending"] = max(
                    0, int(_session_state[ev.session_id]["pending"]) - 1
                )
        except Exception:
            pass
        # 로깅은 평가 서비스 내부 로거에 위임하거나 여기서 print
        logger.error(f"Queue event processing error: session={ev.session_id}, type={ev.event_type}, err={e}")


async def _maybe_finalize_session(session_id: str):
    """세션이 종료 플래그이며 pending이 0이면 평가 마감 수행"""
    st = _session_state.get(session_id)
    if not st:
        return
    if bool(st.get("ended")) and int(st.get("pending", 0)) == 0:
        try:
            result = await service_manager.evaluation_service.end_evaluation_session(session_id)
            # 저장은 evaluation_service가 파일로 처리. 추가 저장 필요 시 여기에 구현 가능
            print(
                f"Evaluation finalized for session={session_id}, total_score={result.get('scores', {}).get('total_score', 0)}"
            )
        except Exception as e:
            print(f"Finalize error for session={session_id}: {e}")
        finally:
            # 세션 상태 정리
            _session_state.pop(session_id, None)


async def _worker_loop():
    q = _get_queue()
    while True:
        ev: QueueEvent = await q.get()
        await _process_event(ev)
        await _maybe_finalize_session(ev.session_id)
        q.task_done()


def start_worker_once():
    global _worker_task
    if _worker_task is None:
        _worker_task = asyncio.create_task(_worker_loop())


