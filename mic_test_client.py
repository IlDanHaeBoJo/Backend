#!/usr/bin/env python3
import asyncio
import json
import numpy as np
import sounddevice as sd
import websockets
import logging
import queue
import pygame
import requests
import tempfile
import os
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MicClient")

class MicrophoneClient:
    def __init__(self, server_url="ws://localhost:8000", user_id="test_user"):
        self.server_url = f"{server_url}/ws/{user_id}"  
        self.base_url = server_url.replace("ws://", "http://").replace("wss://", "https://")
        self.user_id = user_id
        self.websocket = None
        self.is_connected = False
        self.is_recording = False
        self.is_playing_tts = False  # TTS 재생 중 플래그
        
        # 오디오 설정 (서버와 동일하게)
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1600  # 100ms @ 16kHz
        self.dtype = np.int16
        
        # pygame 초기화 (음성 재생용)
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        
        logger.info(f"🎤 마이크 클라이언트 초기화 - 사용자: {user_id}")
        logger.info(f"🔊 음성 재생 시스템 준비 완료")
    
    async def connect(self):
        """서버에 WebSocket 연결"""
        try:
            logger.info(f"🔗 서버 연결 시도: {self.server_url}")
            self.websocket = await websockets.connect(self.server_url)
            self.is_connected = True
            logger.info("✅ 서버에 연결되었습니다!")
            
            # 연결 응답 수신
            response = await self.websocket.recv()
            logger.info(f"📨 서버 응답: {response}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 연결 실패: {e}")
            return False
    
    async def disconnect(self):
        """연결 해제"""
        self.is_recording = False
        if self.websocket:
            await self.websocket.close()
        self.is_connected = False
        logger.info("🔌 연결이 해제되었습니다.")
    
    async def start_cpx_session(self, case_id="IM_001"):
        """CPX 세션 시작"""
        if not self.is_connected:
            return
        
        command = {
            "type": "start_session",
            "case_id": case_id
        }
        
        await self.websocket.send(json.dumps(command))
        logger.info(f"🏥 CPX 세션 시작 요청: {case_id}")
    
    def audio_callback(self, indata, frames, time, status):
        """오디오 콜백 - 마이크에서 데이터 수신"""
        if status:
            logger.warning(f"⚠️  오디오 상태: {status}")
        
        if self.is_recording and self.websocket:
            # int16으로 변환하여 바이트로 전송
            audio_data = (indata * 32767).astype(np.int16)
            audio_bytes = audio_data.tobytes()
            
            # TTS 재생 중에는 호응어만 감지 (볼륨 임계값 높임)
            if self.is_playing_tts:
                # 더 높은 임계값으로 큰 소리(호응어)만 감지
                rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
                if rms < 1000:  # 호응어 임계값 (조정 가능)
                    return
                # 호응어 감지 시 특별 표시
                logger.info("🗣️  호응어 감지 (TTS 재생 중)")
            
            # 오디오 큐에 추가 (동기 방식)
            if hasattr(self, 'audio_queue'):
                try:
                    self.audio_queue.put_nowait(audio_bytes)
                except:
                    pass  # 큐가 가득 찬 경우 무시
    
    async def send_audio_chunk(self, audio_bytes):
        """오디오 청크를 서버로 전송"""
        try:
            if self.websocket and not self.websocket.closed:
                await self.websocket.send(audio_bytes)
        except Exception as e:
            logger.error(f"오디오 전송 오류: {e}")
    
    async def play_tts_audio(self, audio_url):
        """TTS 음성 파일 다운로드 및 재생"""
        try:
            logger.info(f"🔊 TTS 음성 재생 시작: {audio_url}")
            
            # 마이크 입력 차단
            self.is_playing_tts = True
            logger.info("🎤 마이크 입력 일시 차단 (TTS 재생 중)")
            
            # 서버에서 오디오 파일 다운로드
            full_url = f"{self.base_url}/cache/tts/{audio_url}"
            logger.info(f"🌐 실제 요청 URL: {full_url}")
            response = requests.get(full_url, timeout=10)
            
            if response.status_code == 200:
                # 임시 파일로 저장
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                    temp_file.write(response.content)
                    temp_path = temp_file.name
                
                # pygame으로 재생
                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.play()
                
                # 재생 완료까지 대기
                while pygame.mixer.music.get_busy():
                    await asyncio.sleep(0.1)
                
                # 재생 완료 후 추가 대기 (에코 방지)
                await asyncio.sleep(0.5)
                
                # 임시 파일 삭제
                os.unlink(temp_path)
                logger.info("✅ TTS 음성 재생 완료")
                
            else:
                logger.error(f"TTS 파일 다운로드 실패: {response.status_code}")
                
        except Exception as e:
            logger.error(f"TTS 재생 오류: {e}")
        finally:
            # 마이크 입력 재개
            self.is_playing_tts = False
            logger.info("🎤 마이크 입력 재개")
    
    async def start_recording(self):
        """마이크 녹음 시작"""
        if not self.is_connected:
            logger.error("서버에 먼저 연결해주세요!")
            return
        
        logger.info("🎤 마이크 녹음 시작...")
        self.is_recording = True
        
        # 오디오 큐 생성
        self.audio_queue = queue.Queue(maxsize=100)
        
        # sounddevice로 실시간 스트림 시작
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='float32',
            blocksize=self.chunk_size,
            callback=self.audio_callback
        ) as stream:
            logger.info("🎙️  말씀하세요! (Ctrl+C로 종료)")
            
            try:
                while self.is_recording:
                    # 오디오 큐에서 데이터를 읽어 WebSocket으로 전송
                    try:
                        audio_bytes = self.audio_queue.get_nowait()
                        await self.send_audio_chunk(audio_bytes)
                    except queue.Empty:
                        pass  # 큐에 데이터가 없으면 계속
                    
                    # 서버 응답 수신
                    if self.websocket:
                        try:
                            response = await asyncio.wait_for(
                                self.websocket.recv(), timeout=0.001
                            )
                            await self.handle_server_response(response)
                        except asyncio.TimeoutError:
                            pass  # 타임아웃은 정상 (계속 진행)
                        except websockets.exceptions.ConnectionClosed:
                            logger.warning("서버 연결이 끊어졌습니다.")
                            break
                    
                    await asyncio.sleep(0.001)  # CPU 사용량 최적화
                    
            except KeyboardInterrupt:
                logger.info("🛑 사용자가 중단했습니다.")
            finally:
                self.is_recording = False
    
    async def handle_server_response(self, response_text):
        """서버 응답 처리"""
        try:
            response = json.loads(response_text)
            msg_type = response.get("type", "unknown")
            message = response.get("message", "")
            
            if msg_type == "connected":
                logger.info(f"🔗 {message}")
            elif msg_type == "session_started":
                logger.info(f"🏥 {message}")
            elif msg_type == "listening":
                logger.info(f"👂 {message}")
            elif msg_type == "processing":
                logger.info(f"🧠 {message}")
            elif msg_type == "voice_response":
                user_text = response.get("user_text", "")
                ai_text = response.get("ai_text", "")
                audio_url = response.get("audio_url", "")
                
                logger.info(f"👤 학생: {user_text}")
                
                # API 오류 메시지 체크
                if ai_text.startswith("❌"):
                    logger.error(f"🚨 API 오류: {ai_text}")
                else:
                    logger.info(f"🤖 환자: {ai_text}")
                
                # TTS 음성이 있으면 자동 재생 (오류 메시지가 아닌 경우만)
                if audio_url and not ai_text.startswith("❌"):
                    await self.play_tts_audio(audio_url)
                elif ai_text.startswith("❌"):
                    logger.warning("🔇 API 오류로 인해 TTS 생성 안됨")
                else:
                    logger.warning("🔇 TTS 음성 파일이 없습니다")
            elif msg_type == "no_speech":
                logger.warning(f"🔇 {message}")
            elif msg_type == "error":
                logger.error(f"❌ {message}")
            else:
                logger.info(f"📨 응답: {response}")
                
        except json.JSONDecodeError:
            logger.warning(f"잘못된 응답 형식: {response_text}")

async def main():
    """메인 실행 함수"""
    print("🎤 CPX 마이크 테스트 클라이언트")
    print("=" * 50)
    
    # 시나리오 선택
    print("\n📋 시나리오 선택:")
    print("1. 흉통 케이스 (김철수, 45세 남성)")
    print("2. 복통 케이스 (박영희, 32세 여성)")
    
    while True:
        try:
            choice = input("\n시나리오 번호를 선택하세요 (1 or 2): ").strip()
            if choice in ["1", "2"]:
                break
            else:
                print("❌ 1 또는 2를 입력해주세요.")
        except KeyboardInterrupt:
            print("\n👋 프로그램을 종료합니다.")
            return
    
    client = MicrophoneClient()
    
    try:
        # 서버 연결
        if not await client.connect():
            return
        
        # 시나리오 선택 메시지 전송
        scenario_message = {
            "type": "select_scenario",
            "scenario_id": choice
        }
        await client.websocket.send(json.dumps(scenario_message))
        
        print(f"✅ 시나리오 {choice}번 선택됨!")
        print("🎤 마이크 녹음을 시작합니다. 환자에게 말을 걸어보세요!")
        await asyncio.sleep(2)  # 시나리오 설정 대기
        
        # 마이크 녹음 시작
        await client.start_recording()
        
    except KeyboardInterrupt:
        logger.info("프로그램을 종료합니다...")
    finally:
        await client.disconnect()

if __name__ == "__main__":
    print("\n🎤 마이크 테스트 시작")
    print("📋 사용법:")
    print("  1. 프로그램 실행 후 서버 연결 대기")
    print("  2. '마이크 녹음 시작' 메시지 후 말하기")
    print("  3. Ctrl+C로 종료")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 테스트를 종료합니다.") 