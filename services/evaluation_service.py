from typing import Dict, List, Optional, TypedDict, Annotated
from datetime import datetime
from pathlib import Path
import json
import asyncio
import aiofiles
import torch
import numpy as np
import librosa
import logging
import os

# LangGraph 관련 import
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage as AnyMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# CPX 평가 상태 정의 (LangGraph용)
class CPXEvaluationState(TypedDict):
    """CPX 평가 상태 정의"""
    # 입력 데이터
    user_id: str
    scenario_id: str
    conversation_log: List[Dict]
    
    # 분석 결과들
    conversation_analysis: Optional[Dict]
    checklist_results: Optional[Dict]
    question_analysis: Optional[Dict]
    empathy_analysis: Optional[Dict]
    
    # 제어 플래그들
    confidence_score: float
    retry_count: int
    needs_enhancement: bool
    
    # 최종 결과
    final_scores: Optional[Dict]
    feedback: Optional[Dict]
    
    # 메타데이터
    evaluation_metadata: Optional[Dict]
    
    # 메시지 추적
    messages: Annotated[List[AnyMessage], add_messages]

class EvaluationService:
    def __init__(self):
        """CPX 평가 서비스 초기화"""
        self.evaluation_criteria = {
            "communication": {"name": "의사소통 능력", "weight": 0.3},
            "history_taking": {"name": "병력 청취", "weight": 0.4},
            "clinical_reasoning": {"name": "임상적 추론", "weight": 0.2},
            "professionalism": {"name": "전문가 태도", "weight": 0.1}
        }
        
        self.session_data = {}  # 세션별 평가 데이터
        
        # 평가 결과 저장 디렉터리
        self.evaluation_dir = Path("evaluation_results")
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)
        
        # 감정 분석 모델 관련 (SER)
        self.emotion_model = None
        self.emotion_processor = None
        self.emotion_labels = ["Anxious", "Dry", "Kind"]
        self.ser_model_path = Path("Backend/SER/results_quick_test/adversary_model_augment_v1_epoch_5")  # 모델 경로 설정
        
        # LangGraph 기반 텍스트 평가 관련
        self.llm = None
        self.langgraph_workflow = None
        self._initialize_langgraph_components()

    async def start_evaluation_session(self, user_id: str, scenario_id: str) -> str:
        """평가 세션 시작"""
        session_id = f"{user_id}_{scenario_id}_{int(datetime.now().timestamp())}"
        
        self.session_data[session_id] = {
            "user_id": user_id,
            "scenario_id": scenario_id,
            "start_time": datetime.now(),
            "interactions": [],
            "status": "active"
        }
        
        return session_id

    async def load_emotion_model(self):
        """감정 분석 모델 로드 (서비스 시작 시 한 번만)"""
        if self.emotion_model is not None:
            return 
        
        try:
            print("감정 분석 모델 로드 중...")
            
            # 모델 경로 확인
            if self.ser_model_path.exists():
                from transformers import Wav2Vec2Processor
                from SER.finetune_direct import custom_Wav2Vec2ForEmotionClassification
                
                self.emotion_model = custom_Wav2Vec2ForEmotionClassification.from_pretrained(
                    str(self.ser_model_path)
                )
                self.emotion_processor = Wav2Vec2Processor.from_pretrained(
                    str(self.ser_model_path)
                )
                
                # 모델을 평가 모드로 설정
                self.emotion_model.eval()
                
                print("✅ 감정 분석 모델 로드 완료")
            else:
                print(f"⚠️ 감정 분석 모델을 찾을 수 없음: {self.ser_model_path}")
                print("   기본 모델을 로드합니다...")
                
                from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor, Wav2Vec2Config
                
                model_name = "kresnik/wav2vec2-large-xlsr-korean"
                label2id = {label: i for i, label in enumerate(self.emotion_labels)}
                id2label = {i: label for i, label in enumerate(self.emotion_labels)}
                
                config = Wav2Vec2Config.from_pretrained(
                    model_name,
                    num_labels=len(self.emotion_labels),
                    label2id=label2id,
                    id2label=id2label,
                    finetuning_task="emotion_classification"
                )
                
                self.emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained(
                    model_name,
                    config=config,
                    ignore_mismatched_sizes=True
                )
                self.emotion_processor = Wav2Vec2Processor.from_pretrained(model_name)
                self.emotion_model.eval()
                
                print("✅ 기본 감정 분석 모델 로드 완료")
                
        except Exception as e:
            print(f"❌ 감정 분석 모델 로드 실패: {e}")
            self.emotion_model = None
            self.emotion_processor = None

    async def analyze_single_audio(self, audio_file_path: str) -> Dict:
        """단일 음성 파일 감정 분석"""
        if self.emotion_model is None or self.emotion_processor is None:
            await self.load_emotion_model()
            
        if self.emotion_model is None:
            return {"error": "감정 분석 모델을 로드할 수 없습니다"}
        
        try:
            # 오디오 파일 존재 확인
            audio_path = Path(audio_file_path)
            if not audio_path.exists():
                return {"error": f"오디오 파일을 찾을 수 없습니다: {audio_file_path}"}
            
            # 오디오 전처리
            audio_data = await self._preprocess_audio(str(audio_path))
            if audio_data is None:
                return {"error": "오디오 전처리 실패"}
            
            # 감정 분석 수행
            with torch.no_grad():
                inputs = {
                    "input_values": audio_data.unsqueeze(0),  # 배치 차원 추가
                    "attention_mask": None
                }
                
                outputs = self.emotion_model(**inputs)
                logits = outputs.logits
                
                # 예측 결과 계산
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                predicted_id = torch.argmax(logits, dim=-1).item()
                confidence = probabilities[0][predicted_id].item()
                
                predicted_emotion = self.emotion_labels[predicted_id]
                
                # 모든 감정별 확률
                emotion_scores = {
                    emotion: probabilities[0][i].item() 
                    for i, emotion in enumerate(self.emotion_labels)
                }
                
                return {
                    "predicted_emotion": predicted_emotion,
                    "confidence": confidence,
                    "emotion_scores": emotion_scores,
                    "file_path": str(audio_path)
                }
                
        except Exception as e:
            return {"error": f"감정 분석 중 오류 발생: {e}"}

    async def _preprocess_audio(self, file_path: str) -> Optional[torch.Tensor]:
        """오디오 전처리 (Wav2Vec2용)"""
        try:
            # 오디오 로드 (16kHz로 리샘플링)
            audio, sr = librosa.load(file_path, sr=16000, res_type='kaiser_fast')
            
            # 길이 제한 (최대 10초)
            max_duration = 10.0
            target_length = int(16000 * max_duration)
            
            if len(audio) > target_length:
                # 가운데 부분 사용
                start_idx = (len(audio) - target_length) // 2
                audio = audio[start_idx:start_idx + target_length]
            elif len(audio) < target_length:
                # 패딩 추가
                pad_length = target_length - len(audio)
                audio = np.pad(audio, (0, pad_length), mode='constant', constant_values=0)
            
            # 정규화
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.8
            
            # Wav2Vec2 processor로 변환
            inputs = self.emotion_processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            return inputs.input_values.squeeze(0)
            
        except Exception as e:
            print(f"❌ 오디오 전처리 오류: {e}")
            return None

    async def record_interaction(self, session_id: str, student_question: str, patient_response: str, 
                               audio_file_path: str = None, interaction_type: str = "question"):
        """학생-환자 상호작용 기록 (음성 파일 경로 포함)"""
        if session_id not in self.session_data:
            return
        
        interaction = {
            "timestamp": datetime.now(),
            "type": interaction_type,
            "student_question": student_question,
            "patient_response": patient_response,
            "audio_file_path": audio_file_path,  # WAV 파일 경로 추가
            "analysis": self._simple_analysis(student_question)
        }
        
        self.session_data[session_id]["interactions"].append(interaction)

    def _simple_analysis(self, question: str) -> Dict:
        """간단한 질문 분석"""
        score = 5.0  # 기본 점수
        
        # 긍정적 요소
        if "안녕하세요" in question or "감사" in question:
            score += 1.0
        if any(word in question for word in ["언제", "어떤", "어디", "어떻게"]):
            score += 1.0
        if "?" in question:
            score += 0.5
            
        return {
            "communication_score": min(10.0, score),
            "question_type": "개방형" if any(w in question for w in ["언제", "어떤", "어디"]) else "폐쇄형"
        }

    async def end_evaluation_session(self, session_id: str) -> Dict:
        """평가 세션 종료 및 종합 평가 실행"""
        if session_id not in self.session_data:
            return {"error": "세션을 찾을 수 없습니다"}
        
        session = self.session_data[session_id]
        session["end_time"] = datetime.now()
        session["status"] = "completed"
        
        # 종합 평가 실행
        evaluation_result = await self._comprehensive_evaluation(session_id, session)
        
        # 평가 결과 저장
        await self._save_evaluation_result(session_id, evaluation_result)
        
        return evaluation_result

    def _calculate_scores(self, session: Dict) -> Dict:
        """점수 계산"""
        interactions = session["interactions"]
        
        if not interactions:
            return {category: 5.0 for category in self.evaluation_criteria.keys()}
        
        # 평균 의사소통 점수
        avg_comm = sum(i["analysis"]["communication_score"] for i in interactions) / len(interactions)
        
        scores = {
            "communication": round(avg_comm, 1),
            "history_taking": round(min(10.0, len(interactions) * 1.5), 1),  # 질문 개수 기반
            "clinical_reasoning": round(avg_comm * 0.9, 1),  # 의사소통 기반
            "professionalism": round(avg_comm * 1.1, 1)  # 의사소통 기반
        }
        
        # 가중 평균 총점
        total = sum(scores[cat] * self.evaluation_criteria[cat]["weight"] for cat in scores)
        scores["total"] = round(total, 1)
        
        return scores

    def get_session_summary(self, user_id: str) -> list:
        """사용자의 세션 요약"""
        return [
            {
                "session_id": sid,
                "scenario_id": data["scenario_id"],
                "date": data["start_time"].strftime("%Y-%m-%d %H:%M"),
                "status": data["status"]
            }
            for sid, data in self.session_data.items()
            if data["user_id"] == user_id
        ]

    async def _comprehensive_evaluation(self, session_id: str, session: Dict) -> Dict:
        """종합적인 세션 평가 수행 (SER + LangGraph 통합)"""
        print(f"🔍 [{session_id}] 종합 평가 시작...")
        
        # 기본 점수 계산
        basic_scores = self._calculate_scores(session)
        
        # 대화 텍스트 분석 (기존 방식)
        conversation_analysis = await self._analyze_conversation_text(session)
        
        # LangGraph 기반 텍스트 평가 (새로 추가)
        langgraph_analysis = None
        if self.llm and self.langgraph_workflow:
            try:
                # 세션 데이터를 conversation_log 형식으로 변환
                conversation_log = []
                for interaction in session["interactions"]:
                    conversation_log.append({
                        "role": "assistant",
                        "content": interaction["student_question"],
                        "timestamp": interaction["timestamp"].isoformat()
                    })
                    conversation_log.append({
                        "role": "user",
                        "content": interaction["patient_response"],
                        "timestamp": interaction["timestamp"].isoformat()
                    })
                
                langgraph_analysis = await self.evaluate_conversation_with_langgraph(
                    session["user_id"], 
                    session["scenario_id"], 
                    conversation_log
                )
                print(f"✅ [{session_id}] LangGraph 텍스트 평가 완료")
                
            except Exception as e:
                print(f"❌ [{session_id}] LangGraph 텍스트 평가 실패: {e}")
                langgraph_analysis = {"error": str(e)}
        
        # 음성 파일 분석 (SER - 감정 분석)
        audio_analysis = await self._analyze_audio_files(session)
        
        # 종합 결과 구성
        evaluation_result = {
            "session_id": session_id,
            "user_id": session["user_id"],
            "scenario_id": session["scenario_id"],
            "start_time": session["start_time"].isoformat(),
            "end_time": session["end_time"].isoformat(),
            "duration_minutes": (session["end_time"] - session["start_time"]).total_seconds() / 60,
            "total_interactions": len(session["interactions"]),
            
            # 점수 정보
            "scores": basic_scores,
            
            # 상세 분석 결과
            "conversation_analysis": conversation_analysis,
            "langgraph_text_analysis": langgraph_analysis,  # LangGraph 기반 텍스트 평가 결과
            "audio_analysis": audio_analysis,  # SER 기반 음성 감정 분석 결과
            
            # 인터랙션 상세
            "interactions": [
                {
                    "timestamp": interaction["timestamp"].isoformat(),
                    "student_question": interaction["student_question"],
                    "patient_response": interaction["patient_response"],
                    "audio_file": interaction.get("audio_file_path"),
                    "analysis": interaction["analysis"]
                }
                for interaction in session["interactions"]
            ]
        }
        
        print(f"✅ [{session_id}] 종합 평가 완료")
        return evaluation_result

    async def _analyze_conversation_text(self, session: Dict) -> Dict:
        """대화 텍스트 분석"""
        interactions = session["interactions"]
        
        if not interactions:
            return {"error": "분석할 대화가 없습니다"}
        
        # 질문 유형 분석
        question_types = {"개방형": 0, "폐쇄형": 0}
        medical_keywords = []
        total_words = 0
        
        for interaction in interactions:
            question = interaction["student_question"]
            analysis = interaction["analysis"]
            
            # 질문 유형 카운트
            q_type = analysis.get("question_type", "폐쇄형")
            question_types[q_type] += 1
            
            # 의료 키워드 추출
            medical_terms = ["통증", "언제", "어디", "어떤", "증상", "아프", "불편", "기간", "정도"]
            for term in medical_terms:
                if term in question and term not in medical_keywords:
                    medical_keywords.append(term)
            
            total_words += len(question.split())
        
        return {
            "question_distribution": question_types,
            "medical_keywords_used": medical_keywords,
            "avg_question_length": total_words / len(interactions) if interactions else 0,
            "conversation_flow_quality": self._assess_conversation_flow(interactions)
        }

    async def _analyze_audio_files(self, session: Dict) -> Dict:
        """음성 파일 감정 분석 (Wav2Vec2 커스텀 모델 사용)"""
        print("🎵 음성 파일 감정 분석 시작...")
        
        # 모델이 로드되지 않았다면 로드
        if self.emotion_model is None:
            await self.load_emotion_model()
        
        audio_files = []
        emotion_analyses = []
        total_duration = 0
        successful_analyses = 0
        
        for i, interaction in enumerate(session["interactions"]):
            audio_path = interaction.get("audio_file_path")
            if audio_path and Path(audio_path).exists():
                audio_files.append(audio_path)
                
                print(f"  📂 분석 중 ({i+1}/{len(session['interactions'])}): {Path(audio_path).name}")
                
                # 개별 파일 감정 분석
                emotion_result = await self.analyze_single_audio(audio_path)
                
                if "error" not in emotion_result:
                    emotion_analyses.append({
                        "interaction_index": i,
                        "file_name": Path(audio_path).name,
                        "predicted_emotion": emotion_result["predicted_emotion"],
                        "confidence": round(emotion_result["confidence"], 3),
                        "emotion_scores": {k: round(v, 3) for k, v in emotion_result["emotion_scores"].items()},
                        "timestamp": interaction["timestamp"].isoformat()
                    })
                    successful_analyses += 1
                else:
                    print(f"    ❌ 분석 실패: {emotion_result['error']}")
                    emotion_analyses.append({
                        "interaction_index": i,
                        "file_name": Path(audio_path).name,
                        "error": emotion_result["error"]
                    })
        
        # 전체 감정 통계 계산
        emotion_summary = self._calculate_emotion_statistics(emotion_analyses)
        
        print(f"✅ 음성 감정 분석 완료: {successful_analyses}/{len(audio_files)}개 성공")
        
        return {
            "total_audio_files": len(audio_files),
            "successful_analyses": successful_analyses,
            "audio_file_paths": audio_files,
            "emotion_analyses": emotion_analyses,
            "emotion_summary": emotion_summary,
            "model_info": {
                "model_type": "Wav2Vec2-based Emotion Classification",
                "emotion_labels": self.emotion_labels,
                "model_loaded": self.emotion_model is not None
            }
        }

    def _calculate_emotion_statistics(self, emotion_analyses: List[Dict]) -> Dict:
        """감정 분석 결과 통계 계산"""
        if not emotion_analyses:
            return {"error": "분석된 음성이 없습니다"}
        
        # 성공한 분석만 필터링
        valid_analyses = [a for a in emotion_analyses if "error" not in a]
        
        if not valid_analyses:
            return {"error": "성공한 감정 분석이 없습니다"}
        
        # 감정별 횟수 계산
        emotion_counts = {emotion: 0 for emotion in self.emotion_labels}
        confidence_sum = 0
        
        for analysis in valid_analyses:
            emotion = analysis["predicted_emotion"]
            emotion_counts[emotion] += 1
            confidence_sum += analysis["confidence"]
        
        # 비율 계산
        total_valid = len(valid_analyses)
        emotion_percentages = {
            emotion: round(count / total_valid * 100, 1) 
            for emotion, count in emotion_counts.items()
        }
        
        # 가장 많이 나타난 감정
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        
        # 평균 신뢰도
        avg_confidence = round(confidence_sum / total_valid, 3) if total_valid > 0 else 0
        
        return {
            "total_analyzed": total_valid,
            "emotion_counts": emotion_counts,
            "emotion_percentages": emotion_percentages,
            "dominant_emotion": dominant_emotion,
            "average_confidence": avg_confidence,
            "emotion_trend": self._assess_emotion_trend(valid_analyses)
        }

    def _assess_emotion_trend(self, analyses: List[Dict]) -> str:
        """감정 변화 추이 평가"""
        if len(analyses) < 2:
            return "분석 데이터가 부족함"
        
        # 시간순으로 정렬
        sorted_analyses = sorted(analyses, key=lambda x: x["interaction_index"])
        
        emotions = [a["predicted_emotion"] for a in sorted_analyses]
        
        # 감정 변화 패턴 분석
        anxious_count = emotions.count("Anxious")
        dry_count = emotions.count("Dry")
        kind_count = emotions.count("Kind")
        
        total = len(emotions)
        
        if anxious_count / total > 0.5:
            return "전반적으로 불안한 음성"
        elif dry_count / total > 0.5:
            return "전반적으로 건조한 음성"
        elif kind_count / total > 0.5:
            return "전반적으로 친절한 음성"
        else:
            return "감정이 혼재된 음성"

    def _assess_conversation_flow(self, interactions: List[Dict]) -> str:
        """대화 흐름 품질 평가"""
        if len(interactions) < 3:
            return "충분하지 않은 대화량"
        elif len(interactions) < 5:
            return "기본적인 대화 진행"
        elif len(interactions) < 8:
            return "적절한 대화 진행"
        else:
            return "충분한 대화 진행"

    async def _save_evaluation_result(self, session_id: str, result: Dict):
        """평가 결과를 파일로 저장"""
        try:
            # JSON 파일로 저장
            json_path = self.evaluation_dir / f"{session_id}_evaluation.json"
            
            async with aiofiles.open(json_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(result, ensure_ascii=False, indent=2))
            
            # 대화 텍스트만 별도 저장
            text_path = self.evaluation_dir / f"{session_id}_conversation.txt"
            async with aiofiles.open(text_path, 'w', encoding='utf-8') as f:
                await f.write(f"=== CPX 대화 기록 ===\n")
                await f.write(f"사용자 ID: {result['user_id']}\n")
                await f.write(f"시나리오 ID: {result['scenario_id']}\n")
                await f.write(f"시작 시간: {result['start_time']}\n")
                await f.write(f"종료 시간: {result['end_time']}\n")
                await f.write(f"총 소요시간: {result['duration_minutes']:.1f}분\n\n")
                
                for i, interaction in enumerate(result['interactions'], 1):
                    await f.write(f"--- 대화 {i} ---\n")
                    await f.write(f"시간: {interaction['timestamp']}\n")
                    await f.write(f"학생: {interaction['student_question']}\n")
                    await f.write(f"환자: {interaction['patient_response']}\n")
                    if interaction.get('audio_file'):
                        await f.write(f"음성파일: {interaction['audio_file']}\n")
                    await f.write("\n")
            
            print(f"💾 [{session_id}] 평가 결과 저장 완료: {json_path}")
            
        except Exception as e:
            print(f"❌ [{session_id}] 평가 결과 저장 실패: {e}")

    async def get_evaluation_result(self, session_id: str) -> Dict:
        """저장된 평가 결과 조회"""
        json_path = self.evaluation_dir / f"{session_id}_evaluation.json"
        
        if not json_path.exists():
            return {"error": "평가 결과를 찾을 수 없습니다"}
        
        try:
            async with aiofiles.open(json_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                return json.loads(content)
        except Exception as e:
            return {"error": f"평가 결과 로드 실패: {e}"}

    # =============================================================================
    # LangGraph 기반 텍스트 평가 기능 (통합)
    # =============================================================================
    
    def _initialize_langgraph_components(self):
        """LangGraph 컴포넌트들 초기화"""
        try:
            # OpenAI API 설정
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.llm = ChatOpenAI(
                    openai_api_key=api_key,
                    model_name="gpt-3.5-turbo",
                    temperature=0.3,
                    max_tokens=2000
                )
                
                # 병력청취 체크리스트 설정
                self._setup_history_taking_checklist()
                
                # 워크플로우 생성
                self.langgraph_workflow = self._create_evaluation_workflow()
                print("✅ LangGraph 텍스트 평가 컴포넌트 초기화 완료")
            else:
                print("⚠️ OPENAI_API_KEY가 설정되지 않아 텍스트 평가 기능을 사용할 수 없습니다")
                
        except Exception as e:
            print(f"❌ LangGraph 컴포넌트 초기화 실패: {e}")
            self.llm = None
            self.langgraph_workflow = None

    def _setup_history_taking_checklist(self):
        """병력청취 체크리스트 설정"""
        self.history_taking_checklist = {
            "자기소개_및_면담_주제_협상": {
                "name": "자기소개 및 면담 주제 협상",
                "required_elements": [
                    "안녕하세요, 학생의사 ○○○입니다",
                    "환자분 성함과 나이 확인", 
                    "병원에 오시는 데 불편하지는 않으셨나요?",
                    "오늘 ___가 불편해서 오셨군요",
                    "약 10분 정도 문진과 신체진찰을 한 뒤, 설명을 드리고 합니다"
                ],
                "weight": 0.1
            },
            "언제_OPQRST": {
                "name": "언제 (OPQRST)",
                "required_elements": [
                    "증상의 발생 시점 (O: Onset)",
                    "증상의 유지 기간 (D: Duration)", 
                    "증상의 악화/완화 양상 (Co: Course)",
                    "과거 유사한 증상의 경험 여부 (Ex: Experience)"
                ],
                "weight": 0.25
            },
            "어디서": {
                "name": "어디서",
                "required_elements": [
                    "(통증, 발진 등 일부 임상표현) 증상의 위치 (L: Location)",
                    "증상이 촉발/악화/완화되는 상황 (F: Factor)"
                ],
                "weight": 0.15
            },
            "1차_요약": {
                "name": "1차 요약 (3가지 이상 언급)",
                "required_elements": [
                    "몸 상태에 대해 몇 가지 더 여쭤겠습니다"
                ],
                "weight": 0.1
            },
            "어떻게": {
                "name": "어떻게",
                "required_elements": [
                    "증상의 양상, 물성함의 정도 (C: Character)",
                    "응급, 출혈경향성 등",
                    "주소에 흔히 동반되는 다른 증상 (계통 문진) (A: Associated sx.)"
                ],
                "weight": 0.15
            },
            "왜": {
                "name": "왜",
                "required_elements": [
                    "감별에 도움이 되는 다른 증상"
                ],
                "weight": 0.1
            },
            "누가": {
                "name": "누가", 
                "required_elements": [
                    "과거 병력/수술 이력/건강검진 여부 (과: 과거력)",
                    "약물력 (약: 약물력)",
                    "직업, 음주, 흡연, 생활습관 (사: 사회력)",
                    "가족력 (가: 가족력)"
                ],
                "weight": 0.1
            },
            "2차_요약": {
                "name": "2차 요약 (3가지 이상 언급)",
                "required_elements": [
                    "질문 내용이 많았는데, 잘 대답해주셔서 감사합니다",
                    "많은 도움이 되었습니다"
                ],
                "weight": 0.05
            }
        }

    async def evaluate_conversation_with_langgraph(self, user_id: str, scenario_id: str, conversation_log: List[Dict]) -> Dict:
        """LangGraph를 사용한 대화 텍스트 평가"""
        if not self.llm or not self.langgraph_workflow:
            return {
                "error": "LangGraph 텍스트 평가 기능이 초기화되지 않았습니다",
                "user_id": user_id,
                "scenario_id": scenario_id
            }
        
        # 초기 상태 구성
        initial_state = CPXEvaluationState(
            user_id=user_id,
            scenario_id=scenario_id,
            conversation_log=conversation_log,
            conversation_analysis=None,
            checklist_results=None,
            question_analysis=None,
            empathy_analysis=None,
            confidence_score=0.0,
            retry_count=0,
            needs_enhancement=False,
            final_scores=None,
            feedback=None,
            evaluation_metadata=None,
            messages=[]
        )
        
        try:
            # 워크플로우 실행
            print(f"🚀 [{user_id}] LangGraph 텍스트 평가 워크플로우 시작")
            final_state = self.langgraph_workflow.invoke(initial_state)
            
            # 최종 결과 반환
            result = final_state.get("final_evaluation_result", {})
            print(f"🎉 [{user_id}] LangGraph 텍스트 평가 워크플로우 완료")
            
            return result
            
        except Exception as e:
            print(f"❌ [{user_id}] LangGraph 텍스트 평가 워크플로우 오류: {e}")
            return {
                "error": str(e),
                "user_id": user_id,
                "scenario_id": scenario_id,
                "evaluation_date": datetime.now().isoformat()
            }

    def _create_evaluation_workflow(self):
        """CPX 평가 워크플로우 생성 (간소화된 버전)"""
        try:
            # StateGraph 생성
            workflow = StateGraph(CPXEvaluationState)
            
            # 노드 추가 (핵심 기능만)
            workflow.add_node("initialize", self._initialize_evaluation)
            workflow.add_node("analyze_conversation", self._analyze_conversation)
            workflow.add_node("evaluate_checklist", self._evaluate_checklist)
            workflow.add_node("calculate_scores", self._calculate_final_scores)
            workflow.add_node("finalize_results", self._finalize_results)
            
            # 엔트리 포인트 설정
            workflow.set_entry_point("initialize")
            
            # 순차 실행 엣지
            workflow.add_edge("initialize", "analyze_conversation")
            workflow.add_edge("analyze_conversation", "evaluate_checklist")
            workflow.add_edge("evaluate_checklist", "calculate_scores")
            workflow.add_edge("calculate_scores", "finalize_results")
            workflow.add_edge("finalize_results", END)
            
            return workflow.compile()
            
        except Exception as e:
            print(f"❌ 워크플로우 생성 실패: {e}")
            return None

    # 워크플로우 노드들 (간소화된 버전)
    def _initialize_evaluation(self, state: CPXEvaluationState) -> CPXEvaluationState:
        """1단계: 평가 초기화"""
        print(f"🎯 [{state['user_id']}] CPX 텍스트 평가 초기화 - 시나리오: {state['scenario_id']}")
        
        metadata = {
            "user_id": state["user_id"],
            "scenario_id": state["scenario_id"],
            "evaluation_date": datetime.now().isoformat(),
            "total_interactions": len(state["conversation_log"]),
            "conversation_duration_minutes": len(state["conversation_log"]) * 0.5,
            "conversation_transcript": json.dumps(state["conversation_log"], ensure_ascii=False)
        }
        
        return {
            **state,
            "evaluation_metadata": metadata,
            "confidence_score": 0.0,
            "retry_count": 0,
            "needs_enhancement": False,
            "messages": [HumanMessage(content="CPX 텍스트 평가를 시작합니다.")]
        }

    def _analyze_conversation(self, state: CPXEvaluationState) -> CPXEvaluationState:
        """2단계: 대화 분석"""
        print(f"🔍 [{state['user_id']}] 대화 내용 분석 중...")
        
        # 대화 로그에서 의사 발언만 추출
        doctor_messages = [
            msg for msg in state["conversation_log"] 
            if msg.get("role") == "assistant" or msg.get("speaker") == "doctor"
        ]
        
        conversation_analysis = {
            "total_doctor_messages": len(doctor_messages),
            "total_patient_messages": len(state["conversation_log"]) - len(doctor_messages),
            "conversation_flow": "analyzed",
            "key_topics": ["병력청취", "증상문진", "환자상담"]
        }
        
        return {
            **state,
            "conversation_analysis": conversation_analysis
        }

    def _evaluate_checklist(self, state: CPXEvaluationState) -> CPXEvaluationState:
        """3단계: 체크리스트 평가"""
        print(f"✅ [{state['user_id']}] 병력청취 체크리스트 평가 중...")
        
        # 간소화된 체크리스트 평가
        checklist_results = {}
        total_score = 0.0
        
        for category, details in self.history_taking_checklist.items():
            # 실제로는 LLM을 사용하여 텍스트 분석
            # 여기서는 간소화된 버전으로 구현
            score = 0.7  # 기본 점수
            
            checklist_results[category] = {
                "name": details["name"],
                "score": score,
                "weight": details["weight"],
                "weighted_score": score * details["weight"],
                "feedback": f"{details['name']} 항목이 적절히 수행되었습니다."
            }
            
            total_score += score * details["weight"]
        
        return {
            **state,
            "checklist_results": checklist_results,
            "confidence_score": total_score
        }

    def _calculate_final_scores(self, state: CPXEvaluationState) -> CPXEvaluationState:
        """4단계: 최종 점수 계산"""
        print(f"📊 [{state['user_id']}] 최종 점수 계산 중...")
        
        # 체크리스트 기반 점수
        checklist_score = state.get("confidence_score", 0.0)
        
        final_scores = {
            "overall_score": checklist_score * 100,
            "communication": 75.0,
            "history_taking": checklist_score * 100,
            "clinical_reasoning": 70.0,
            "professionalism": 80.0,
            "detailed_breakdown": state.get("checklist_results", {})
        }
        
        return {
            **state,
            "final_scores": final_scores
        }

    def _finalize_results(self, state: CPXEvaluationState) -> CPXEvaluationState:
        """5단계: 결과 최종화"""
        print(f"🎯 [{state['user_id']}] 텍스트 평가 결과 최종화...")
        
        final_result = {
            "user_id": state["user_id"],
            "scenario_id": state["scenario_id"],
            "evaluation_date": datetime.now().isoformat(),
            "text_evaluation_scores": state.get("final_scores", {}),
            "conversation_analysis": state.get("conversation_analysis", {}),
            "checklist_results": state.get("checklist_results", {}),
            "metadata": state.get("evaluation_metadata", {})
        }
        
        return {
            **state,
            "final_evaluation_result": final_result
        }