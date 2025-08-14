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
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# CPX 평가 상태 정의 (Multi-Step Reasoning 전용)
class CPXEvaluationState(TypedDict):
    """CPX 평가 상태 정의 - Multi-Step Reasoning 전용"""
    # 입력 데이터
    user_id: str
    scenario_id: str
    conversation_log: List[Dict]
    
    # Multi-Step Reasoning 결과들 (핵심)
    medical_context_analysis: Optional[Dict]
    question_intent_analysis: Optional[Dict]
    completeness_assessment: Optional[Dict]
    quality_evaluation: Optional[Dict]
    appropriateness_validation: Optional[Dict]
    
    # 종합 평가 결과
    comprehensive_evaluation: Optional[Dict]
    
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
        self.scenario_applicable_elements = {
            "1": {  # 흉통 케이스 (모든 OPQRST 적용)
                "name": "흉통 케이스 (김철수, 45세 남성)",
                "applicable_categories": {
                    # 병력청취 - 모든 OPQRST 요소 적용
                    "O_onset": True,
                    "L_location": True, 
                    "D_duration": True,
                    "Co_course": True,
                    "Ex_experience": True,
                    "C_character": True,
                    "A_associated": True,
                    "F_factor": True,
                    "E_exam": True,
                    # 추가 병력
                    "trauma_history": True,
                    "past_medical_history": True,
                    "medication_history": True,
                    "family_history": True,  # 심혈관 가족력 중요
                    "social_history": True,  # 흡연, 음주 중요
                    "gynecologic_history": False  # 남성이므로 해당없음
                }
            },
            "2": {  # 복통 케이스 (모든 OPQRST 적용)
                "name": "복통 케이스 (박영희, 32세 여성)",
                "applicable_categories": {
                    # 병력청취 - 모든 OPQRST 요소 적용
                    "O_onset": True,
                    "L_location": True,
                    "D_duration": True, 
                    "Co_course": True,
                    "Ex_experience": True,
                    "C_character": True,
                    "A_associated": True,
                    "F_factor": True,
                    "E_exam": True,
                    # 추가 병력
                    "trauma_history": True,
                    "past_medical_history": True,
                    "medication_history": True,
                    "family_history": True,
                    "social_history": True,
                    "gynecologic_history": True  # 여성이므로 해당
                }
            },
            "3": {
                "name": "신경과 치매 케이스 (나몰라, 63세 남성)",
                "applicable_categories": {
                    # 병력청취
                    "O_onset": True,
                    "L_location": False,
                    "D_duration": False,
                    "Co_course": True,
                    "Ex_experience": True,
                    "C_character": True,
                    "A_associated": True,
                    "F_factor": True,
                    "E_exam": True,
                    "trauma_history": True,
                    "past_medical_history": True,
                    "medication_history": True,
                    "family_history": True,
                    "social_history": True,
                    "gynecologic_history": False,
                    # 신체진찰
                    "examination_preparation": True,
                    "vital_signs": True,
                    "physical_examination_technique": True,
                    "examination_attitude": True,
                    # 환자교육
                    "condition_explanation": True,
                    "lifestyle_guidance": True,
                    "treatment_plan": True,
                    # 의사소통
                    "communication_skills": True
                }
            }
        }
                
        self.evaluation_checklist = {
            "history_taking": {
                "name": "병력 청취",
                "categories": {
                    "O_onset": {
                        "name": "O (Onset) - 발병 시기",
                        "required_elements": [
                            "증상의 발생 시점 확인",
                            "급성/만성 여부 파악",
                            "발생 상황 및 계기 탐색",
                            "이전 유사 경험 확인",
                            "초기 대처 방법 문의"
                        ],
                        "weight": 0.10
                    },
                    "L_location": {
                        "name": "L (Location) - 위치",
                        "required_elements": [
                            "증상 위치 정확한 확인",
                            "환자가 직접 가리키도록 요청",
                            "방사통이나 이동성 여부 파악"
                        ],
                        "weight": 0.08
                    },
                    "D_duration": {
                        "name": "D (Duration) - 지속 시간/빈도",
                        "required_elements": [
                            "증상 지속 기간 확인",
                            "하루 중 발생 빈도 파악",
                            "지속적/간헐적 양상 구분",
                            "간헐적인 경우 주기성 확인",
                            "시간 경과에 따른 변화 파악"
                        ],
                        "weight": 0.10
                    },
                    "Co_course": {
                        "name": "Co (Course) - 경과",
                        "required_elements": [
                            "증상의 악화/완화 경향 파악",
                            "증상 변동성(fluctuation) 확인"
                        ],
                        "weight": 0.08
                    },
                    "Ex_experience": {
                        "name": "Ex (Experience) - 유사한 경험",
                        "required_elements": [
                            "과거 유사한 증상 경험 확인"
                        ],
                        "weight": 0.06
                    },
                    "C_character": {
                        "name": "C (Character) - 특징",
                        "required_elements": [
                            "증상의 성질/양상 구체적 확인",
                            "통증의 경우 강도 평가(NRS 0-10점)"
                        ],
                        "weight": 0.10
                    },
                    "A_associated": {
                        "name": "A (Associated symptom) - 동반 증상",
                        "required_elements": [
                            "동반 증상 탐색 시작 안내",
                            "함께 나타나는 증상 확인",
                            "주요 감별질환 관련 증상 확인"
                        ],
                        "weight": 0.10
                    },
                    "F_factor": {
                        "name": "F (Factor) - 악화/완화요인",
                        "required_elements": [
                            "증상 악화 요인 확인",
                            "증상 완화 요인 파악"
                        ],
                        "weight": 0.08
                    },
                    "E_exam": {
                        "name": "E (Exam) - 이전 검진결과",
                        "required_elements": [
                            "기존 진단받은 질환 확인",
                            "최근 건강검진 결과 문의",
                            "과거 수술/입원/외상력 확인"
                        ],
                        "weight": 0.08
                    },
                    "trauma_history": {
                        "name": "외상력",
                        "required_elements": [
                            "교통사고 및 외상 경험 확인"
                        ],
                        "weight": 0.04
                    },
                    "past_medical_history": {
                        "name": "과거력",
                        "required_elements": [
                            "기존 질병 및 지병 확인"
                        ],
                        "weight": 0.06
                    },
                    "medication_history": {
                        "name": "약물력",
                        "required_elements": [
                            "현재 복용 중인 처방약 확인",
                            "일반의약품 복용 여부 파악",
                            "약물 알레르기 반응 확인"
                        ],
                        "weight": 0.06
                    },
                    "family_history": {
                        "name": "가족력",
                        "required_elements": [
                            "가족 내 유사 증상 확인",
                            "가족력상 주요 질환 파악"
                        ],
                        "weight": 0.06
                    },
                    "social_history": {
                        "name": "사회력",
                        "required_elements": [
                            "음주/흡연/카페인 섭취 확인",
                            "식습관 및 운동습관 파악",
                            "직업적 스트레스 요인 확인"
                        ],
                        "weight": 0.06
                    },
                    "gynecologic_history": {
                        "name": "여성력 (해당시)",
                        "required_elements": [
                            "월경력 및 월경 관련 증상 확인",
                            "임신/출산/성생활 관련 병력 파악"
                        ],
                        "weight": 0.04
                    },
                },
                "weight": 0.50
            },
            "physical_examination": {
                "name": "신체 진찰 수기, 신체진찰 태도",
                "categories": {
                    "examination_preparation": {
                        "name": "진찰 준비",
                        "required_elements": [
                            "손 위생 (소독/세정)",
                            "진찰 전 동의 구하기",
                            "환자 자세 및 노출 정도 적절히 조정",
                            "프라이버시 보호"
                        ],
                        "weight": 0.2
                    },
                    "vital_signs": {
                        "name": "활력징후 측정",
                        "required_elements": [
                            "혈압 측정",
                            "맥박 측정",
                            "체온 측정",
                            "호흡수 확인"
                        ],
                        "weight": 0.3
                    },
                    "physical_examination_technique": {
                        "name": "신체진찰 수행",
                        "required_elements": [
                            "시진 (Inspection) - 관찰",
                            "촉진 (Palpation) - 만지기",
                            "청진 (Auscultation) - 듣기",
                            "타진 (Percussion) - 두드리기"
                        ],
                        "weight": 0.3
                    },
                    "examination_attitude": {
                        "name": "진찰 태도",
                        "required_elements": [
                            "부드럽고 신중한 접근",
                            "환자 불편감 최소화",
                            "진찰 과정 설명",
                            "전문적 태도 유지"
                        ],
                        "weight": 0.2
                    }
                },
                "weight": 0.10
            },
            "patient_education": {
                "name": "환자 교육",
                "categories": {
                    "condition_explanation": {
                        "name": "상태 설명",
                        "required_elements": [
                            "현재 상태에 대한 설명",
                            "이해하기 쉬운 용어 사용",
                            "환자 질문에 대한 답변",
                            "추가 검사 필요성 설명"
                        ],
                        "weight": 0.4
                    },
                    "lifestyle_guidance": {
                        "name": "생활 지도",
                        "required_elements": [
                            "일상생활 주의사항",
                            "증상 관리 방법",
                            "언제 병원 재방문할지 안내",
                            "응급상황 대처법"
                        ],
                        "weight": 0.3
                    },
                    "treatment_plan": {
                        "name": "치료 계획",
                        "required_elements": [
                            "치료 방향 설명",
                            "약물 치료 계획 (해당시)",
                            "추가 검사 계획",
                            "예후 설명"
                        ],
                        "weight": 0.3
                    }
                },
                "weight": 0.10
            },
            "patient_doctor_interaction": {
                "name": "환자-의사-상호 작용",
                "categories": {
                    "communication_skills": {
                        "name": "의사소통 기술",
                        "required_elements": [
                            "효율적으로 잘 물어 보았다",
                            "나의 말을 잘 들어 주었다",
                            "나의 입장을 이해하려고 노력하였다",
                            "환자가 이해하기 쉽게 설명하였다",
                            "나와 좋은 유대 관계를 형성하려고 했다"
                        ],
                        "weight": 1.0
                    }
                },
                "weight": 0.30
            }
        }

        self.diagnosis_plan_checklist = {
            "differential_diagnosis": {
                "name": "감별진단",
                "required_elements": [
                    "주요 감별진단 2-3가지 제시",
                    "각 진단의 근거 설명",
                    "가능성 순서대로 배열"
                ],
                "weight": 0.4
            },
            "diagnostic_plan": {
                "name": "진단 계획",
                "required_elements": [
                    "필요한 추가 검사 계획",
                    "검사의 목적과 필요성 설명",
                    "검사 우선순위 고려"
                ],
                "weight": 0.3
            },
            "treatment_plan": {
                "name": "치료 계획",
                "required_elements": [
                    "초기 치료 방향 제시",
                    "약물 치료 계획 (해당시)",
                    "비약물적 치료 방법",
                    "추적 관찰 계획"
                ],
                "weight": 0.3
            }
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
        self.workflow = None
        self._initialize_langgraph_components()

        # FAISS 인덱스 로드
        self.embedding_model = "intfloat/multilingual-e5-large"
        self.faiss_index_path = Path("Backend/RAG/medical_db")
        self.faiss_index = FAISS.load_local(self.faiss_index_path, HuggingFaceEmbeddings(model_name=self.embedding_model), allow_dangerous_deserialization=True)

    async def start_evaluation_session(self, user_id: str, scenario_id: str) -> str:
        """평가 세션 시작"""
        session_id = f"{user_id}_{scenario_id}_{int(datetime.now().timestamp())}"
        
        self.session_data[session_id] = {
            "user_id": user_id,
            "scenario_id": scenario_id,
            "start_time": datetime.now(),
            "conversation_entries": [],  # 실시간 대화 데이터
            # "audio_files": [],  # 임시 저장된 wav 파일 경로들
            "status": "active"
        }
        
        return session_id

    async def add_conversation_entry(self, session_id: str, audio_file_path: str, 
                                   text: str, speaker_role: str) -> Dict:
        """실시간 대화 엔트리 추가 (음성 분석 포함)"""
        if session_id not in self.session_data:
            return {"error": "세션을 찾을 수 없습니다"}
        
        try:
            timestamp = datetime.now()
            emotion_analysis = None
            
            # 의사(doctor) 음성인 경우에만 감정 분석 수행
            if speaker_role == "doctor":
                await self.load_emotion_model()  # 모델이 로드되지 않았다면 로드
                
                if self.emotion_model is not None:
                    emotion_result = await self.analyze_single_audio(audio_file_path)
                    if "error" not in emotion_result:
                        emotion_analysis = {
                            "predicted_emotion": emotion_result["predicted_emotion"],
                            "confidence": emotion_result["confidence"],
                            "emotion_scores": emotion_result["emotion_scores"]
                        }
                        print(f"🎭 [{session_id}] 감정 분석 완료: {emotion_analysis['predicted_emotion']} ({emotion_analysis['confidence']:.2f})")
            
            # 대화 엔트리 생성
            conversation_entry = {
                "timestamp": timestamp.isoformat(),
                "text": text,
                "emotion": emotion_analysis,
                "speaker_role": speaker_role,  # "doctor" (의사) or "patient" (환자)
                "audio_file_path": audio_file_path
            }
            
            # 세션 데이터에 추가
            session = self.session_data[session_id]
            session["conversation_entries"].append(conversation_entry)
            session["audio_files"].append(audio_file_path)
            
            print(f"📝 [{session_id}] 대화 엔트리 추가: {speaker_role} - {text[:50]}...")
            
            # 평가 완료 후 임시 WAV 파일들 삭제
            try:
                await self._cleanup_audio_files(audio_file_path)
            except Exception as e:
                print(f"❌ [{audio_file_path}] 임시 WAV 파일 삭제 실패: {e}")
            
            return {
                "success": True,
                "entry": conversation_entry,
                "total_entries": len(session["conversation_entries"])
            }
            
        except Exception as e:
            print(f"❌ [{session_id}] 대화 엔트리 추가 실패: {e}")
            return {"error": str(e)}

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
                    "input_values": audio_data,  # 이미 배치 차원이 있음 (1, sequence_length)
                    "attention_mask": None
                }
                
                outputs = self.emotion_model(**inputs)
                logits = outputs['emotion_logits']
                
                # 예측 결과 계산
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                predicted_id = torch.argmax(logits, dim=-1).item()
                confidence = probabilities[0][predicted_id].item()
                
                # 예측 감정 결과
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
                start_idx = np.random.randint(0, len(audio) - target_length + 1)
                audio = audio[start_idx:start_idx + target_length]
            elif len(audio) < target_length:
                # 패딩 추가
                pad_length = target_length - len(audio)
                audio = np.pad(audio, (0, pad_length), mode='constant', constant_values=0)
            
            # 정규화
            # if np.max(np.abs(audio)) > 0:
            #     audio = audio / np.max(np.abs(audio)) * 0.8
            
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
        
        # LangGraph 기반 텍스트 평가 (새로운 대화 데이터 사용)
        langgraph_analysis = None
        if self.llm and self.workflow:
            try:
                # 새로운 conversation_entries를 conversation_log 형식으로 변환
                conversation_log = []
                for entry in session.get("conversation_entries", []):
                    conversation_log.append({
                        "role": entry["speaker_role"],
                        "content": entry["text"],
                        "timestamp": entry["timestamp"],
                        "emotion": entry.get("emotion")
                    })
                
                if conversation_log:  # 대화 데이터가 있는 경우에만 평가
                    langgraph_analysis = await self.evaluate_conversation_with_langgraph(
                        session["user_id"], 
                        session["scenario_id"], 
                        conversation_log
                    )
                    print(f"✅ [{session_id}] LangGraph 텍스트 평가 완료")
                else:
                    print(f"⚠️ [{session_id}] 대화 데이터가 없어 LangGraph 평가를 건너뜁니다")
                
            except Exception as e:
                print(f"❌ [{session_id}] LangGraph 텍스트 평가 실패: {e}")
                langgraph_analysis = {"error": str(e)}
        
        # 종합 결과 구성
        evaluation_result = {
            "session_id": session_id,
            "user_id": session["user_id"],
            "scenario_id": session["scenario_id"],
            "start_time": session["start_time"].isoformat(),
            "end_time": session["end_time"].isoformat(),
            "duration_minutes": (session["end_time"] - session["start_time"]).total_seconds() / 60,
            
            # 상세 분석 결과
            "langgraph_text_analysis": langgraph_analysis,  # LangGraph 기반 텍스트 평가 결과
            
            # 실시간 대화 데이터 (감정 분석 포함)
            "conversation_entries": [
                {
                    "timestamp": entry["timestamp"],
                    "text": entry["text"],
                    "speaker_role": entry["speaker_role"],
                    "emotion": entry.get("emotion"),
                    "audio_file": entry["audio_file_path"]
                }
                for entry in session.get("conversation_entries", [])
            ]
        }
        
        print(f"✅ [{session_id}] 종합 평가 완료")
        return evaluation_result

    async def _save_evaluation_result(self, session_id: str, result: Dict):
        """평가 결과를 파일로 저장"""
        try:
            # JSON 파일로 저장
            json_path = self.evaluation_dir / f"{session_id}_evaluation.json"
            
            async with aiofiles.open(json_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(result, ensure_ascii=False, indent=2))
            
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
                
                # 워크플로우 생성
                self.workflow = self._create_evaluation_workflow()
                print("✅ LangGraph 텍스트 평가 컴포넌트 초기화 완료")
            else:
                print("⚠️ OPENAI_API_KEY가 설정되지 않아 텍스트 평가 기능을 사용할 수 없습니다")
                
        except Exception as e:
            print(f"❌ LangGraph 컴포넌트 초기화 실패: {e}")
            self.llm = None
            self.workflow = None

    async def evaluate_conversation(self, user_id: str, scenario_id: str, conversation_log: List[Dict]) -> Dict:
        """LangGraph 워크플로우를 사용한 CPX 평가 실행"""
        # 초기 상태 구성 (Multi-Step 전용)
        initial_state = CPXEvaluationState(
            user_id=user_id,
            scenario_id=scenario_id,
            conversation_log=conversation_log,
            medical_context_analysis=None,
            question_intent_analysis=None,
            completeness_assessment=None,
            quality_evaluation=None,
            appropriateness_validation=None,
            comprehensive_evaluation=None,
            final_scores=None,
            feedback=None,
            evaluation_metadata=None,
            messages=[]
        )
        
        try:
            # 워크플로우 실행
            print(f"🚀 [{user_id}] CPX 평가 워크플로우 시작")
            final_state = self.workflow.invoke(initial_state)
            
            # 간단한 대화 요약 정보 생성
            student_questions = [msg for msg in conversation_log if msg.get("role") == "student"]
            conversation_summary = {
                "total_questions": len(student_questions),
                "duration_minutes": len(conversation_log) * 0.5
            }
            
            # 최종 결과 구성
            result = {
                "evaluation_metadata": final_state.get("evaluation_metadata", {}),
                "scores": final_state.get("final_scores", {}),
                "feedback": final_state.get("feedback", {}),
                "conversation_summary": conversation_summary,
                "detailed_analysis": {
                    "medical_context": final_state.get("medical_context_analysis", {}),
                    "question_intent": final_state.get("question_intent_analysis", {}),
                    "completeness": final_state.get("completeness_assessment", {}),
                    "quality": final_state.get("quality_evaluation", {}),
                    "appropriateness": final_state.get("appropriateness_validation", {}),
                    "comprehensive": final_state.get("comprehensive_evaluation", {})
                },
                "evaluation_method": "6단계 의학적 분석",
                "system_info": {
                    "version": "v2.0",
                    "evaluation_steps": 6
                }
            }
            
            print(f"🎉 [{user_id}] CPX 평가 워크플로우 완료")
            
            return result
            
        except Exception as e:
            print(f"❌ [{user_id}] 평가 워크플로우 오류: {e}")
            return {
                "error": str(e),
                "user_id": user_id,
                "scenario_id": scenario_id,
                "evaluation_date": datetime.now().isoformat()
            }

    # 기존 함수와의 호환성을 위한 별칭
    async def evaluate_conversation_with_langgraph(self, user_id: str, scenario_id: str, conversation_log: List[Dict]) -> Dict:
        """LangGraph를 사용한 대화 텍스트 평가 (호환성 유지)"""
        return await self.evaluate_conversation(user_id, scenario_id, conversation_log)

    def _create_evaluation_workflow(self):
        """CPX 평가 워크플로우 생성"""
        workflow = StateGraph(CPXEvaluationState)

        workflow.add_node("initialize", self._initialize_evaluation)
        workflow.add_node("medical_context", self._analyze_medical_context)
        workflow.add_node("question_intent", self._analyze_question_intent)
        workflow.add_node("completeness", self._assess_medical_completeness)
        workflow.add_node("quality", self._evaluate_question_quality)
        workflow.add_node("appropriateness", self._validate_scenario_appropriateness)
        workflow.add_node("comprehensive_evaluation", self._generate_comprehensive_evaluation)
        workflow.add_node("calculate_scores", self._calculate_final_scores)
        workflow.add_node("generate_feedback", self._generate_feedback)
        workflow.add_node("finalize_results", self._finalize_results)

        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "medical_context")
        workflow.add_edge("medical_context", "question_intent")
        workflow.add_edge("question_intent", "completeness")
        workflow.add_edge("completeness", "quality")
        workflow.add_edge("quality", "appropriateness")
        workflow.add_edge("appropriateness", "comprehensive_evaluation")
        workflow.add_edge("comprehensive_evaluation", "calculate_scores")
        workflow.add_edge("calculate_scores", "generate_feedback")
        workflow.add_edge("generate_feedback", "finalize_results")
        workflow.add_edge("finalize_results", END)

        return workflow.compile()

    def _initialize_evaluation(self, state: CPXEvaluationState) -> CPXEvaluationState:
        print(f"🎯 [{state['user_id']}] CPX 평가 초기화 - 시나리오: {state['scenario_id']}")
        
        metadata = {
            "user_id": state["user_id"],
            "scenario_id": state["scenario_id"],
            "evaluation_date": datetime.now().isoformat(),
            "conversation_duration_minutes": len(state["conversation_log"]) * 0.5,
            "voice_recording_path": "s3로 저장",
            "conversation_transcript": json.dumps(state["conversation_log"], ensure_ascii=False)
        }
        
        return {
            **state,
            "evaluation_metadata": metadata,
            "messages": [HumanMessage(content="CPX 평가를 시작합니다.")]
        }

    def _is_category_applicable(self, category_id: str, scenario_id: str) -> bool:
        """시나리오별 카테고리 해당 여부 확인"""
        if scenario_id not in self.scenario_applicable_elements:
            raise ValueError(f"시나리오 ID '{scenario_id}'를 찾을 수 없습니다. 사용 가능한 ID: {list(self.scenario_applicable_elements.keys())}")
        
        scenario_info = self.scenario_applicable_elements[scenario_id]
        applicable_categories = scenario_info["applicable_categories"]
        
        if category_id not in applicable_categories:
            raise ValueError(f"카테고리 ID '{category_id}'를 시나리오 '{scenario_id}'에서 찾을 수 없습니다")
        
        return applicable_categories[category_id]

    def _analyze_medical_context(self, state: CPXEvaluationState) -> CPXEvaluationState:
        """Step 1: 의학적 맥락 이해"""
        print(f"🧠 [{state['user_id']}] Step 1: 의학적 맥락 분석 시작")
        
        scenario_id = state["scenario_id"]
        scenario_info = self.scenario_applicable_elements.get(scenario_id, {})
        scenario_name = scenario_info.get("name", f"시나리오 {scenario_id}")
        
        medical_context_prompt = f"""
            당신은 의학교육 전문가입니다. 다음 시나리오의 의학적 맥락을 분석하세요.

            【시나리오 정보】: {scenario_name}

            다음 관점에서 분석하세요:
            1. 주요 감별진단들과 각각의 위험도
            2. 놓치면 안 되는 Critical 정보들
            3. 시간 효율성 측면에서 우선순위
            4. 환자 안전을 위해 반드시 확인해야 할 요소들

            다음 JSON 형식으로 응답하세요:
            {{
                "primary_differentials": ["주요 감별진단 리스트"],
                "critical_elements": ["놓치면 위험한 핵심 요소들"],
                "time_priority": ["시간 제약 하에서 우선순위 요소들"],
                "safety_concerns": ["환자 안전 관련 필수 확인사항"],
                "medical_importance_score": 의학적 중요도(1-10)
            }}
        """
        
        messages = [
            SystemMessage(content="당신은 경험 많은 의학교육 전문가입니다."),
            HumanMessage(content=medical_context_prompt)
        ]
        
        response = self.llm(messages)
        result_text = response.content.strip()
        
        import re
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if not json_match:
            raise ValueError(f"Step 1에서 JSON 형식 응답을 찾을 수 없습니다. LLM 응답: {result_text[:100]}")
        
        medical_context = json.loads(json_match.group())
        
        print(f"✅ [{state['user_id']}] Step 1: 의학적 맥락 분석 완료")
        
        return {
            **state,
            "medical_context_analysis": medical_context,
            "messages": state["messages"] + [HumanMessage(content="Step 1: 의학적 맥락 분석 완료")]
        }

    def _analyze_question_intent(self, state: CPXEvaluationState) -> CPXEvaluationState:
        """Step 2: 질문 의도 분석"""
        print(f"🎯 [{state['user_id']}] Step 2: 질문 의도 분석 시작")
        
        conversation_text = self._build_conversation_text(state["conversation_log"])
        medical_context = state.get("medical_context_analysis", {})
        
        question_intent_prompt = f"""
            당신은 의학교육 평가 전문가입니다. 학생의 질문들의 의도를 분석하세요.

            【의학적 맥락】: {medical_context}

            【학생-환자 대화】: {conversation_text}

            다음 관점에서 질문 의도를 분석하세요:
            1. 의학적 목적의 명확성 - 각 질문이 명확한 의학적 목적을 가지고 있는가?
            2. 체계적 접근성 - 논리적이고 체계적인 순서로 질문했는가?
            3. 환자 중심성 - 환자가 이해하기 쉽고 편안하게 답할 수 있도록 질문했는가?
            4. 시간 효율성 - 제한된 시간 내에서 효율적으로 정보를 수집하려 했는가?

            다음 JSON 형식으로 응답하세요:
            {{
                "medical_purpose_clarity": 의학적 목적 명확성 점수(1-10),
                "systematic_approach": 체계적 접근성 점수(1-10),
                "patient_centeredness": 환자 중심성 점수(1-10),
                "time_efficiency": 시간 효율성 점수(1-10),
                "overall_intent_score": 전체 의도 점수(1-10),
                "intent_analysis": "질문 의도에 대한 구체적 분석"
            }}
        """
        
        messages = [
            SystemMessage(content="당신은 의학교육 평가 전문가입니다."),
            HumanMessage(content=question_intent_prompt)
        ]
        
        response = self.llm(messages)
        result_text = response.content.strip()
        
        import re
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if not json_match:
            raise ValueError(f"Step 2에서 JSON 형식 응답을 찾을 수 없습니다. LLM 응답: {result_text[:100]}")
        
        question_intent = json.loads(json_match.group())
        
        print(f"✅ [{state['user_id']}] Step 2: 질문 의도 분석 완료")
        
        return {
            **state,
            "question_intent_analysis": question_intent,
            "messages": state["messages"] + [HumanMessage(content="Step 2: 질문 의도 분석 완료")]
        }

    def _build_conversation_text(self, conversation_log: List[Dict]) -> str:
        """대화 로그를 텍스트로 변환"""
        conversation_parts = []
        for msg in conversation_log:
            speaker = "학생" if msg.get("role") == "student" else "환자"
            content = msg.get("content", "")
            conversation_parts.append(f"{speaker}: {content}")
        return "\n".join(conversation_parts)

    def _assess_medical_completeness(self, state: CPXEvaluationState) -> CPXEvaluationState:
        """Step 3: 의학적 완성도 평가 - 개별 카테고리별 평가"""
        print(f"📋 [{state['user_id']}] Step 3: 의학적 완성도 평가 시작")
        
        conversation_text = self._build_conversation_text(state["conversation_log"])
        scenario_id = state["scenario_id"]
        
        # 해당 시나리오의 applicable categories만 추출
        applicable_categories = []
        for area_id, area_info in self.evaluation_checklist.items():
            for category_id, category_info in area_info["categories"].items():
                if self._is_category_applicable(category_id, scenario_id):
                    applicable_categories.append({
                        "category_id": category_id,
                        "name": category_info["name"],
                        "required_elements": category_info["required_elements"]
                    })
        
        # 각 카테고리별로 개별 평가 수행
        category_results = {}
        critical_gaps = []
        for category in applicable_categories:
            print(f" 📝 [{category['name']}] 개별 평가 중...")
            result = self._evaluate_single_category(
                conversation_text, category, scenario_id
            )
            category_results[category['category_id']] = result
            
            # Critical gap 확인
            if result.get('completion_level') == 'none':
                critical_gaps.append(category['name'])
        
        # 전체 완성도 점수 계산
        if category_results:
            total_score = sum(r.get('completeness_score', 0) for r in category_results.values())
            overall_score = total_score / len(category_results)
        else:
            overall_score = 0
        
        completeness = {
            "category_completeness": category_results,
            "overall_completeness_score": overall_score,
            "critical_gaps": critical_gaps,
            "medical_completeness_analysis": f"개별 카테고리 평가를 통해 {len(category_results)}개 항목 중 {len(critical_gaps)}개 항목이 누락되었습니다."
        }
        
        print(f"✅ [{state['user_id']}] Step 3: 의학적 완성도 평가 완료")
        
        return {
            **state,
            "completeness_assessment": completeness,
            "messages": state["messages"] + [HumanMessage(content="Step 3: 의학적 완성도 평가 완료")]
        }

    def _evaluate_single_category(self, conversation_text: str, category: Dict, scenario_id: str) -> Dict:
        """단일 카테고리에 대한 집중 평가"""
        # 카테고리별 일반적 예시 추가
        category_examples = {
            "외상력": "머리 다침, 교통사고, 낙상, 외상, 부상, 골절 등 외상 경험에 관한 질문",
            "가족력": "가족 중 질병력, 부모님 병력, 형제자매 질환, 유전적 질환 등에 관한 질문",
            "과거력": "기존 질병, 과거 병원 치료, 수술 경험, 입원력 등에 관한 질문",
            "약물력": "현재 복용 약물, 처방약, 일반의약품, 알레르기 등에 관한 질문",
            "사회력": "흡연, 음주, 직업, 생활습관, 운동 등에 관한 질문",
            "O (Onset) - 발병 시기": "증상 시작 시점, 언제부터, 얼마나 오래 등 시간 관련 질문",
            "C (Character) - 특징": "증상의 성질, 양상, 강도, 정도 등에 관한 질문",
            "A (Associated symptom) - 동반 증상": "함께 나타나는 증상, 관련 증상 등에 관한 질문",
            "F (Factor) - 악화/완화요인": "증상을 악화시키는 요인, 완화시키는 요인 등에 관한 질문"
        }
        
        example_text = category_examples.get(category['name'], "")
        
        single_category_prompt = f"""
            당신은 의학교육 평가 전문가입니다. 다음 병력청취 대화에서 "{category['name']}" 항목만 집중적으로 평가해주세요.

            【평가 대상】: {category['name']}
            【필수 요소들】: {category['required_elements']}
            【예시】: {example_text}

            【학생-환자 대화】: {conversation_text}

            이 대화에서 "{category['name']}" 관련 내용이 어느 정도 다뤄졌는지만 평가하세요:
            1. 직접적 완료: 명시적으로 질문하여 정보 수집함
            2. 간접적 완료: 대화 맥락에서 정보가 파악됨
            3. 부분적 완료: 불완전하지만 시도함
            4. 미완료: 전혀 다뤄지지 않음

            다음 JSON 형식으로 응답하세요:
            {{
                "completion_level": "direct/indirect/partial/none",
                "medical_risk_level": "high/medium/low",
                "completeness_score": 점수(1-10),
                "evidence": "판단 근거가 되는 대화 내용"
            }}
        """
        
        messages = [
            SystemMessage(content="당신은 의학교육 평가 전문가입니다."),
            HumanMessage(content=single_category_prompt)
        ]
        
        response = self.llm(messages)
        result_text = response.content.strip()
        
        import re
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if not json_match:
            # 평가 실패 시 기본값 반환
            return {
                "completion_level": "none",
                "medical_risk_level": "medium",
                "completeness_score": 0,
                "evidence": f"JSON 파싱 실패: {result_text[:100]}"
            }
        
        try:
            result = json.loads(json_match.group())
            return result
        except json.JSONDecodeError:
            return {
                "completion_level": "none",
                "medical_risk_level": "medium",
                "completeness_score": 0,
                "evidence": f"JSON 디코딩 실패: {result_text[:100]}"
            }

    def _evaluate_question_quality(self, state: CPXEvaluationState) -> CPXEvaluationState:
        """Step 4: 질적 수준 평가"""
        print(f"⭐ [{state['user_id']}] Step 4: 질적 수준 평가 시작")
        
        conversation_text = self._build_conversation_text(state["conversation_log"])
        scneario_num = self.session_data[state["user_id"]]
        
        quality_prompt = f"""
            당신은 의학교육 평가 전문가입니다. 학생 질문들의 질적 수준을 평가하세요.

            【학생-환자 대화】: {conversation_text}

            다음 4가지 기준으로 질문 품질을 평가하세요:
            1. 의학적 정확성 (1-10점)
            2. 소통 효율성 (1-10점)  
            3. 임상적 실용성 (1-10점)
            4. 환자 배려 (1-10점)

            다음 JSON 형식으로 응답하세요:
            {{
                "medical_accuracy": 의학적 정확성 점수(1-10),
                "communication_efficiency": 소통 효율성 점수(1-10),
                "clinical_practicality": 임상적 실용성 점수(1-10),
                "patient_care": 환자 배려 점수(1-10),
                "overall_quality_score": 전체 품질 점수(1-10),
                "quality_analysis": "질적 수준에 대한 구체적 분석"
            }}
        """
        
        messages = [
            SystemMessage(content="당신은 의학교육 평가 전문가입니다."),
            HumanMessage(content=quality_prompt)
        ]
        
        response = self.llm(messages)
        result_text = response.content.strip()
        
        import re
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if not json_match:
            raise ValueError(f"Step 4에서 JSON 형식 응답을 찾을 수 없습니다. LLM 응답: {result_text[:100]}")
        
        quality = json.loads(json_match.group())
        
        print(f"✅ [{state['user_id']}] Step 4: 질적 수준 평가 완료")
        
        return {
            **state,
            "quality_evaluation": quality,
            "messages": state["messages"] + [HumanMessage(content="Step 4: 질적 수준 평가 완료")]
        }

    def _validate_scenario_appropriateness(self, state: CPXEvaluationState) -> CPXEvaluationState:
        """Step 5: 시나리오 적합성 검증"""
        print(f"🎭 [{state['user_id']}] Step 5: 시나리오 적합성 검증 시작")
        
        conversation_text = self._build_conversation_text(state["conversation_log"])
        scenario_id = state["scenario_id"]
        scenario_info = self.scenario_applicable_elements.get(scenario_id, {})
        scenario_name = scenario_info.get("name", f"시나리오 {scenario_id}")
        
        appropriateness_prompt = f"""
            당신은 의학교육 평가 전문가입니다. 학생의 질문들이 해당 시나리오에 적합했는지 검증하세요.

            【시나리오 정보】: {scenario_name}
            【학생-환자 대화】: {conversation_text}

            다음 관점에서 시나리오 적합성을 검증하세요:
            1. 부적절한 질문 체크
            2. 적절성 평가

            다음 JSON 형식으로 응답하세요:
            {{
                "inappropriate_questions": ["부적절한 질문들과 이유"],
                "scenario_specific_score": 시나리오 특화 점수(1-10),
                "patient_profile_score": 환자 프로필 적합성 점수(1-10),
                "time_allocation_score": 시간 배분 적절성 점수(1-10),
                "overall_appropriateness_score": 전체 적합성 점수(1-10),
                "appropriateness_analysis": "시나리오 적합성에 대한 구체적 분석"
            }}
        """
        
        messages = [
            SystemMessage(content="당신은 의학교육 평가 전문가입니다."),
            HumanMessage(content=appropriateness_prompt)
        ]
        
        response = self.llm(messages)
        result_text = response.content.strip()
        
        import re
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if not json_match:
            raise ValueError(f"Step 5에서 JSON 형식 응답을 찾을 수 없습니다. LLM 응답: {result_text[:100]}")
        
        appropriateness = json.loads(json_match.group())
        
        print(f"✅ [{state['user_id']}] Step 5: 시나리오 적합성 검증 완료")
        
        return {
            **state,
            "appropriateness_validation": appropriateness,
            "messages": state["messages"] + [HumanMessage(content="Step 5: 시나리오 적합성 검증 완료")]
        }

    def _generate_comprehensive_evaluation(self, state: CPXEvaluationState) -> CPXEvaluationState:
        """Step 6: 종합 평가 및 최종 점수 계산"""
        print(f"🎯 [{state['user_id']}] Step 6: 종합 평가 시작")
        
        # Multi-Step 결과들 수집
        medical_context = state.get("medical_context_analysis", {})
        question_intent = state.get("question_intent_analysis", {})
        completeness = state.get("completeness_assessment", {})
        quality = state.get("quality_evaluation", {})
        appropriateness = state.get("appropriateness_validation", {})
        
        comprehensive_prompt = f"""
            당신은 의학교육 평가 전문가입니다. 다음 Multi-Step 분석 결과들을 종합하여 최종 평가를 수행하세요.

            【Step 1 - 의학적 맥락】: {medical_context}
            【Step 2 - 질문 의도】: {question_intent}
            【Step 3 - 의학적 완성도】: {completeness}
            【Step 4 - 질적 수준】: {quality}
            【Step 5 - 시나리오 적합성】: {appropriateness}

            종합 평가 기준:
            1. 기본 완료율: Step 3의 완성도 기반 (40% 가중치)
            2. 품질 가중치: Step 4의 질적 수준 반영 (30% 가중치)
            3. 적합성 보정: Step 5의 시나리오 적합성 (20% 가중치)
            4. 의도 점수: Step 2의 질문 의도 (10% 가중치)

            반드시 아래의 정확한 JSON 형식으로만 응답하세요:
            {{
                "final_completion_rate": 0.8,
                "final_quality_score": 7.5,
                "weighted_scores": {{
                    "completeness_weighted": 32.0,
                    "quality_weighted": 22.5,
                    "appropriateness_weighted": 16.0,
                    "intent_weighted": 8.5
                }},
                "detailed_feedback": {{
                    "strengths": ["구체적인 강점 1", "구체적인 강점 2"],
                    "weaknesses": ["구체적인 약점 1", "구체적인 약점 2"],
                    "medical_insights": ["의학적 통찰 1", "의학적 통찰 2"]
                }},
                "comprehensive_analysis": "종합 분석 내용을 여기에 작성"
            }}
        """
        
        messages = [
            SystemMessage(content="당신은 경험 많은 의학교육 평가 전문가입니다."),
            HumanMessage(content=comprehensive_prompt)
        ]
        
        response = self.llm(messages)
        result_text = response.content.strip()
        
        import re
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if not json_match:
            raise ValueError(f"Step 6에서 JSON 형식 응답을 찾을 수 없습니다. LLM 응답: {result_text[:100]}")
        
        try:
            comprehensive = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"Step 6에서 JSON 파싱 실패: {e}")
        
        print(f"✅ [{state['user_id']}] Step 6: 종합 평가 완료")
        
        return {
            **state,
            "comprehensive_evaluation": comprehensive,
            "messages": state["messages"] + [HumanMessage(content="Step 6: 종합 평가 완료")]
        }

    def _calculate_final_scores(self, state: CPXEvaluationState) -> CPXEvaluationState:
        """최종 점수 계산"""
        print(f"🧮 [{state['user_id']}] 최종 점수 계산 시작")
        
        evaluation_result = state["comprehensive_evaluation"]
        completion_rate = evaluation_result["final_completion_rate"]
        quality_score = evaluation_result["final_quality_score"]
        weighted_scores = evaluation_result["weighted_scores"]
        
        final_total_score = (completion_rate * 70) + (quality_score * 3)
        final_total_score = min(100, max(0, final_total_score))
        
        scores = {
            "total_score": round(final_total_score, 1),
            "completion_rate": round(completion_rate, 2),
            "quality_score": round(quality_score, 1),
            "weighted_breakdown": {
                "completeness_score": round(weighted_scores["completeness_weighted"], 1),
                "quality_score": round(weighted_scores["quality_weighted"], 1),
                "appropriateness_score": round(weighted_scores["appropriateness_weighted"], 1),
                "intent_score": round(weighted_scores["intent_weighted"], 1)
            },
            "grade": self._calculate_grade(final_total_score)
        }
        
        print(f"✅ [{state['user_id']}] 최종 점수 계산 완료 - 총점: {final_total_score:.1f}")
        
        return {
            **state,
            "final_scores": scores,
            "messages": state["messages"] + [HumanMessage(content=f"최종 점수 계산 완료 - 총점: {final_total_score:.1f}점")]
        }

    def _generate_feedback(self, state: CPXEvaluationState) -> CPXEvaluationState:
        """피드백 생성"""
        print(f"📝 [{state['user_id']}] 피드백 생성 시작")
        
        evaluation_result = state["comprehensive_evaluation"]
        final_scores = state["final_scores"]
        evaluation_feedback = evaluation_result["detailed_feedback"]
        
        feedback = {
            "overall_feedback": f"6단계 의학적 분석을 통한 종합 평가입니다. 총점: {final_scores['total_score']}점",
            "strengths": evaluation_feedback["strengths"],
            "weaknesses": evaluation_feedback["weaknesses"],
            "medical_insights": evaluation_feedback["medical_insights"],
            "comprehensive_analysis": evaluation_result["comprehensive_analysis"],
            "evaluation_method": "6단계 의학적 분석"
        }
        
        print(f"✅ [{state['user_id']}] 피드백 생성 완료")
        
        return {
            **state,
            "feedback": feedback,
            "messages": state["messages"] + [HumanMessage(content="피드백 생성 완료")]
        }

    def _finalize_results(self, state: CPXEvaluationState) -> CPXEvaluationState:
        """결과 최종화"""
        print(f"🎯 [{state['user_id']}] 평가 결과 최종화")
        
        total_score = state.get('final_scores', {}).get('total_score', 0)
        print(f"🎉 [{state['user_id']}] CPX 평가 완료 - 총점: {total_score}점")
        
        return {
            **state,
            "messages": state["messages"] + [HumanMessage(content="CPX 평가가 성공적으로 완료되었습니다.")]
        }

    def _calculate_grade(self, score: float) -> str:
        """점수에 따른 등급 계산"""
        if score >= 90:
            return "A+"
        elif score >= 85:
            return "A"
        elif score >= 80:
            return "B+"
        elif score >= 75:
            return "B"
        elif score >= 70:
            return "C+"
        elif score >= 65:
            return "C"
        else:
            return "F"

    def _extract_scenario_keywords(self, scenario_info: str) -> str:
        """시나리오 정보에서 핵심 키워드 추출"""
        try:
            # 정규표현식을 사용하여 증상, 나이, 성별 추출
            import re
            
            # 증상 추출 (케이스 앞부분)
            symptom_match = re.search(r'^(.+?)\s*케이스', scenario_info)
            symptom = symptom_match.group(1).strip() if symptom_match else ""
            
            # 나이 추출
            age_match = re.search(r'(\d+)세', scenario_info)
            age = age_match.group(1) if age_match else ""
            
            # 성별 추출
            gender_match = re.search(r'(남성|여성)', scenario_info)
            gender = gender_match.group(1) if gender_match else ""
            
            # 키워드 조합
            keywords = []
            if symptom:
                keywords.append(symptom)
            if age:
                keywords.append(f"{age}세")
            if gender:
                keywords.append(gender)
            
            result = " ".join(keywords)
            print(f"🔍 시나리오 키워드 추출: '{scenario_info}' -> '{result}'")
            return result
            
        except Exception as e:
            print(f"❌ 시나리오 키워드 추출 실패: {e}")
            return scenario_info

    def _compare_conversation_with_medical_knowledge(self, conversation_log: List[Dict], state: CPXEvaluationState) -> str:
        """대화 로그를 요약"""
        try:
            conversation_text = self._build_conversation_text(conversation_log)
            scenario_num = state["scenario_id"]
            scenario_info = self.scenario_applicable_elements[scenario_num]["name"]
            
            scenario_keywords = self._extract_scenario_keywords(scenario_info)
            medical_knowledge = self._retrieve_medical_knowledge(scenario_keywords)
            print(f"🔍 의학 지식 검색 완료: {scenario_keywords}")

            summary_prompt = f"""
                당신은 의사와 환자 사이의 대화와 검색된 의학 지식 결과를 비교하는 비교 전문가야.

                【학생-환자 대화】: {conversation_text}
                【시나리오 정보】: {scenario_info}
                【의학지식 검색 결과】 : {medical_knowledge}

                환자의 주요 증상과 의사가 수행한 질문들을 바탕으로 의학 지식 검색 결과와 의사의 진찰 결과나 치료 방안이 적절한지 비교 및 평가하세요:
                1. 환자의 주요 증상
                2. 의사가 수행한 주요 질문들
                3. 의사의 진찰 결과
                4. 의사의 치료 방안

                간결하고 명확하게 요약해주세요.
            """

            messages = [
                SystemMessage(content="당신은 의사와 환자 사이의 대화를 요약하는 전문가입니다."),
                HumanMessage(content=summary_prompt)
            ]

            response = self.llm(messages)
            summary = response.content.strip()
            print(f"📝 대화 요약 완료")

            return summary
            
        except Exception as e:
            print(f"❌ 대화 요약 실패: {e}")
            return f"대화 요약 중 오류가 발생했습니다: {str(e)}"

    def _retrieve_medical_knowledge(self, query: str) -> str:
        """의학 지식 검색"""
        try:
            if self.faiss_index is None:
                print("⚠️ FAISS 인덱스가 로드되지 않았습니다")
                return "의학 지식 DB를 사용할 수 없습니다."
            
            # FAISS 검색 수행
            docs_and_scores = self.faiss_index.similarity_search_with_score(
                query, k=3  # 상위 3개 결과만 가져오기
            )
            
            if not docs_and_scores:
                return f"'{query}'에 대한 의학 지식 정보를 찾을 수 없습니다."
            
            # 검색 결과를 텍스트로 구성
            knowledge_text = f"【{query} 관련 의학 지식】\n\n"
            
            for i, (doc, score) in enumerate(docs_and_scores, 1):
                content = doc.page_content
                metadata = doc.metadata
                
                # 메타데이터에서 출처 정보 추출
                source = metadata.get('source', '의학 지식 DB')
                
                knowledge_text += f"{i}. {content}\n"
            
            print(f"🔍 의학 지식 검색 완료: '{query}' -> {len(docs_and_scores)}개 결과")
            return knowledge_text
            
        except Exception as e:
            print(f"❌ 의학 지식 검색 실패: {e}")
            return f"의학 지식 검색 중 오류가 발생했습니다: {str(e)}"

    # =============================================================================
    # 새로운 실시간 대화 데이터 분석 메서드들
    # =============================================================================
    
    async def _analyze_conversation_entries(self, session: Dict) -> Dict:
        """새로운 실시간 대화 데이터 분석"""
        conversation_entries = session.get("conversation_entries", [])
        
        if not conversation_entries:
            return {"error": "분석할 대화 엔트리가 없습니다"}
        
        # 역할별 분리
        doctor_entries = [entry for entry in conversation_entries if entry["speaker_role"] == "doctor"]
        patient_entries = [entry for entry in conversation_entries if entry["speaker_role"] == "patient"]
        
        # 감정 분석 통계 (의사 발언만)
        emotion_stats = {}
        if doctor_entries:
            emotions = [entry.get("emotion", {}).get("predicted_emotion") for entry in doctor_entries if entry.get("emotion")]
            emotions = [e for e in emotions if e]  # None 제거
            
            if emotions:
                from collections import Counter
                emotion_counts = Counter(emotions)
                total_emotional_entries = len(emotions)
                
                emotion_stats = {
                    "dominant_emotion": emotion_counts.most_common(1)[0][0] if emotion_counts else None,
                    "emotion_distribution": {
                        emotion: count / total_emotional_entries 
                        for emotion, count in emotion_counts.items()
                    },
                    "total_emotional_entries": total_emotional_entries,
                    "emotional_consistency": max(emotion_counts.values()) / total_emotional_entries if total_emotional_entries > 0 else 0
                }
        
        # 대화 패턴 분석
        conversation_pattern = {
            "total_entries": len(conversation_entries),
            "doctor_utterances": len(doctor_entries),
            "patient_utterances": len(patient_entries),
            "conversation_balance": len(patient_entries) / len(doctor_entries) if len(doctor_entries) > 0 else 0,
            "avg_doctor_utterance_length": sum(len(entry["text"]) for entry in doctor_entries) / len(doctor_entries) if doctor_entries else 0,
            "avg_patient_utterance_length": sum(len(entry["text"]) for entry in patient_entries) / len(patient_entries) if patient_entries else 0
        }
        
        # 시간 분석
        if len(conversation_entries) >= 2:
            first_time = datetime.fromisoformat(conversation_entries[0]["timestamp"])
            last_time = datetime.fromisoformat(conversation_entries[-1]["timestamp"])
            duration_seconds = (last_time - first_time).total_seconds()
            
            time_analysis = {
                "conversation_duration_seconds": duration_seconds,
                "conversation_duration_minutes": duration_seconds / 60,
                "entries_per_minute": len(conversation_entries) / (duration_seconds / 60) if duration_seconds > 0 else 0
            }
        else:
            time_analysis = {
                "conversation_duration_seconds": 0,
                "conversation_duration_minutes": 0,
                "entries_per_minute": 0
            }
        
        return {
            "emotion_analysis": emotion_stats,
            "conversation_pattern": conversation_pattern,
            "time_analysis": time_analysis,
            "quality_indicators": {
                "has_emotional_data": len([e for e in doctor_entries if e.get("emotion")]) > 0,
                "conversation_completeness": len(conversation_entries) >= 10,  # 최소 10개 발언
                "balanced_interaction": 0.3 <= conversation_pattern["conversation_balance"] <= 3.0
            }
        }

    async def _cleanup_audio_files(self, audio_file_path: str):
        """평가 완료 후 임시 WAV 파일들만 삭제 (TTS 캐시 파일은 보존)"""

        try:
            file_path_obj = Path(audio_file_path)
            # TTS 캐시 파일은 삭제하지 않음
            if "cache/tts" in str(file_path_obj):
                print(f"🔒 TTS 캐시 파일 보존: {audio_file_path}")
                return
                
            if file_path_obj.exists() and file_path_obj.suffix == '.wav':
                file_path_obj.unlink()  # WAV 파일만 삭제
                print(f"🗑️ 임시 WAV 파일 삭제: {audio_file_path}")
                    
        except Exception as e:
            print(f"❌ WAV 파일 삭제 실패 ({audio_file_path}): {e}")