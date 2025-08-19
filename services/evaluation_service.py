from typing import Dict, List, Optional, TypedDict, Annotated
from datetime import datetime
from pathlib import Path
import json
import asyncio
import aiofiles
import logging
import os
import sys
import re
from collections import Counter

# LangGraph 관련 import
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage as AnyMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# CPX 관련 import
from services.cpx_service import CpxService
from core.config import get_db

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
        # 카테고리별 평가 체크리스트 로드
        self.evaluation_checklists = self._load_evaluation_checklists()
        
        # 기존 하드코딩된 시나리오 정보는 제거하고 JSON 기반으로 통합
        
        self.session_data = {}  # 세션별 평가 데이터
        
        # 평가 결과 저장 디렉터리
        self.evaluation_dir = Path("evaluation_results")
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)
        
        # LangGraph 기반 텍스트 평가 관련
        self.llm = None
        self.workflow = None
        self._initialize_langgraph_components()
        
        # RAG 기반 가이드라인 검색기 초기화
        self.guideline_retriever = None
        self._initialize_guideline_retriever()
        
        # 시나리오별 적용 요소들 정의
        self.scenario_applicable_elements = self._initialize_scenario_elements()

    def _initialize_guideline_retriever(self):
        """RAG 기반 가이드라인 검색기 초기화"""
        try:
            # RAG 디렉토리의 guideline_retriever를 import
            rag_path = Path(__file__).parent.parent / "RAG"
            sys.path.append(str(rag_path))
            
            from guideline_retriever import GuidelineRetriever
            
            # 가이드라인 검색기 초기화
            index_path = rag_path / "faiss_guideline_index"
            self.guideline_retriever = GuidelineRetriever(index_path=str(index_path))
            
            if self.guideline_retriever.vectorstore:
                print("✅ RAG 가이드라인 검색기 초기화 완료")
            else:
                print("⚠️ RAG 가이드라인 검색기 초기화 실패")
                self.guideline_retriever = None
                
        except Exception as e:
            print(f"❌ 가이드라인 검색기 초기화 오류: {e}")
            self.guideline_retriever = None

    def _initialize_scenario_elements(self) -> Dict:
        """시나리오별 적용 요소들 초기화 - scenarios/ 디렉토리에서 로드"""
        scenario_elements = {}
        scenario_dir = Path("scenarios")
        
        if not scenario_dir.exists():
            print("⚠️ scenarios 디렉토리가 없습니다.")
            return {}
        
        for json_file in scenario_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                scenario_info = data.get("scenario_info", {})
                category = scenario_info.get("category", "unknown")
                
                # 시나리오 ID를 키로 사용 (예: memory_impairment)
                scenario_id = category.replace(" ", "_").lower()
                
                scenario_elements[scenario_id] = {
                    "name": category,
                    "description": scenario_info.get("case_presentation", ""),
                    "patient_name": scenario_info.get("patient_name", ""),
                    "primary_diagnosis": scenario_info.get("primary_diagnosis", ""),
                    "differential_diagnoses": scenario_info.get("differential_diagnoses", []),
                    "applicable_areas": [
                        "history_taking",
                        "physical_examination", 
                        "patient_education"
                    ]
                }
                
                print(f"✅ 시나리오 로드: {category} ({json_file.name})")
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON 파싱 오류 ({json_file.name}): {e}")
            except Exception as e:
                print(f"❌ 시나리오 로드 오류 ({json_file.name}): {e}")
        
        return scenario_elements

    def _load_evaluation_checklists(self) -> Dict:
        """카테고리별 평가 체크리스트 로드"""
        checklists = {}
        checklist_dir = Path("evaluation_checklists")
        
        if not checklist_dir.exists():
            print("⚠️ evaluation_checklists 디렉터리가 없습니다.")
            return checklists
        
        for json_file in checklist_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    category = data.get("category")
                    if category:
                        checklists[category] = data
                        print(f"✅ 평가 체크리스트 로드: {category}")
                    else:
                        print(f"⚠️ {json_file.name}에서 category 필드를 찾을 수 없습니다.")
            except json.JSONDecodeError as e:
                print(f"❌ JSON 파싱 오류 ({json_file.name}): {e}")
            except Exception as e:
                print(f"❌ 체크리스트 로드 오류 ({json_file.name}): {e}")
        
        return checklists

    def get_evaluation_checklist(self, category: str) -> Optional[Dict]:
        """카테고리별 평가 체크리스트 반환"""
        return self.evaluation_checklists.get(category)

    def _get_scenario_category(self, scenario_id: str) -> Optional[str]:
        """시나리오 파일에서 카테고리 정보 로드"""
        try:
            scenario_path = Path(f"scenarios/neurology_dementia_case.json")  # 현재는 하나의 시나리오만
            if not scenario_path.exists():
                return None
            
            with open(scenario_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("scenario_info", {}).get("category")
        except Exception as e:
            print(f"❌ 시나리오 카테고리 로드 실패: {e}")
            return None

    def _extract_applicable_categories(self, checklist: Dict) -> List[Dict]:
        """체크리스트에서 적용 가능한 카테고리들 추출"""
        categories = []
        
        for area_name, area_data in checklist.get("evaluation_areas", {}).items():
            for subcat_id, subcat_data in area_data.get("subcategories", {}).items():
                # applicable이 False인 경우 제외
                if subcat_data.get("applicable", True):
                    required_elements = subcat_data.get("required_questions", subcat_data.get("required_actions", []))
                    categories.append({
                        "category_id": subcat_id,
                        "name": subcat_data["name"],
                        "required_questions": required_elements,
                        "required_elements": required_elements,  # 추가: LangGraph에서 사용
                        "weight": subcat_data.get("weight", 0.1),
                        "area": area_name
                    })
        
        return categories

    def _create_default_completeness_result(self, state: CPXEvaluationState) -> CPXEvaluationState:
        """기본 완성도 결과 생성 (오류 시)"""
        completeness = {
            "category_completeness": {},
            "overall_completeness_score": 0.0,
            "missing_items": [],
            "medical_completeness_analysis": "카테고리 정보를 찾을 수 없어 기본 평가를 수행합니다."
        }
        
        return {
            **state,
            "completeness_assessment": completeness,
            "messages": state["messages"] + [HumanMessage(content="Step 3: 기본 완성도 평가 완료")]
        }

    async def start_evaluation_session(self, user_id: str, scenario_id: str, result_id: Optional[int] = None) -> str:
        """평가 세션 시작"""
        session_id = f"{user_id}_{scenario_id}_{int(datetime.now().timestamp())}"
        
        self.session_data[session_id] = {
            "user_id": user_id,
            "scenario_id": scenario_id,
            "result_id": result_id,  # CPX result_id 저장
            "start_time": datetime.now(),
            "conversation_entries": [],  # 실시간 대화 데이터
            # "audio_files": [],  # 임시 저장된 wav 파일 경로들
            "status": "active"
        }
        
        return session_id

    async def add_conversation_entry(self, session_id: str, audio_file_path: str, 
                                   text: str, speaker_role: str, emotion_analysis: Optional[Dict] = None) -> Dict:
        """실시간 대화 엔트리 추가 (SER 결과는 queue에서 전달받음)"""
        if session_id not in self.session_data:
            return {"error": "세션을 찾을 수 없습니다"}
        
        try:
            timestamp = datetime.now()
            
            # SER 결과 로깅 (queue에서 전달받은 경우)
            if emotion_analysis:
                print(f"🎭 [{session_id}] 감정 분석 결과 수신: {emotion_analysis['predicted_emotion']} ({emotion_analysis['confidence']:.2f})")
            
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
            if "audio_files" not in session:
                session["audio_files"] = []
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

    async def end_evaluation_session(self, session_id: str) -> Dict:
        """평가 세션 종료 및 종합 평가 실행"""
        if session_id not in self.session_data:
            return {"error": "세션을 찾을 수 없습니다"}
        
        session = self.session_data[session_id]
        session["end_time"] = datetime.now()
        session["status"] = "completed"
        
        # 종합 평가 실행
        evaluation_result = await self._comprehensive_evaluation(session_id, session)
        
        # CPX 데이터베이스 업데이트
        await self._update_cpx_database_after_evaluation(session_id, evaluation_result)
        
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

    async def evaluate_with_rag_guidelines(self, conversation_log: List[Dict], category: str) -> Dict:
        """
        RAG 가이드라인을 사용한 대화 평가
        
        Args:
            conversation_log: 대화 로그
            category: 평가할 카테고리 (예: "기억력 저하")
            
        Returns:
            RAG 기반 평가 결과
        """
        if not self.guideline_retriever:
            return {
                "error": "RAG 가이드라인 검색기가 초기화되지 않았습니다.",
                "category": category,
                "completeness": 0.0
            }
        
        try:
            print(f"🔍 [{category}] RAG 가이드라인 기반 평가 시작...")
            
            # 가이드라인 기반 완성도 평가
            evaluation_result = self.guideline_retriever.evaluate_conversation_completeness(
                conversation_log, category
            )
            
            # 추가 분석을 위한 세부 정보
            detailed_analysis = {
                "category": category,
                "overall_completeness": evaluation_result.get("overall_completeness", 0.0),
                "area_breakdown": evaluation_result.get("area_results", {}),
                "completed_count": len(evaluation_result.get("completed_items", [])),
                "missing_count": len(evaluation_result.get("missing_items", [])),
                "total_expected_items": len(evaluation_result.get("completed_items", [])) + len(evaluation_result.get("missing_items", [])),
                "completed_items": evaluation_result.get("completed_items", []),
                "missing_items": evaluation_result.get("missing_items", []),
                "evaluation_method": "rag_guideline",
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"✅ [{category}] RAG 평가 완료 - 완성도: {detailed_analysis['overall_completeness']:.2%}")
            
            return detailed_analysis
            
        except Exception as e:
            print(f"❌ [{category}] RAG 가이드라인 평가 실패: {e}")
            return {
                "error": str(e),
                "category": category,
                "completeness": 0.0,
                "evaluation_method": "rag_guideline_error"
            }

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
        """CPX 평가 워크플로우 생성 (3단계 명확화)"""
        workflow = StateGraph(CPXEvaluationState)

        workflow.add_node("initialize", self._initialize_evaluation)
        workflow.add_node("step1_rag_completeness", self._evaluate_rag_completeness)
        workflow.add_node("step2_quality_assessment", self._evaluate_quality_assessment)
        workflow.add_node("step3_comprehensive_results", self._generate_comprehensive_results)

        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "step1_rag_completeness")
        workflow.add_edge("step1_rag_completeness", "step2_quality_assessment")
        workflow.add_edge("step2_quality_assessment", "step3_comprehensive_results")
        workflow.add_edge("step3_comprehensive_results", END)

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



    def _evaluate_rag_completeness(self, state: CPXEvaluationState) -> CPXEvaluationState:
        """1단계: RAG 기반 완성도 평가 (병력청취, 신체진찰, 환자교육)"""
        print(f"📋 [{state['user_id']}] 1단계: RAG 기반 완성도 평가 시작")
        
        conversation_text = self._build_conversation_text(state["conversation_log"])
        scenario_id = state["scenario_id"]
        
        # 시나리오에서 카테고리 정보 로드
        scenario_category = self._get_scenario_category(scenario_id)
        rag_data = {"category": scenario_category or scenario_id}
        
        # 3개 영역별 RAG 기반 평가
        areas_evaluation = {
            "history_taking": self._evaluate_single_area(conversation_text, "병력 청취", rag_data),
            "physical_examination": self._evaluate_single_area(conversation_text, "신체 진찰", rag_data),
            "patient_education": self._evaluate_single_area(conversation_text, "환자 교육", rag_data)
        }
        
        # 전체 완성도 점수 계산
        area_scores = [area.get("area_score", 0) for area in areas_evaluation.values()]
        overall_completeness = sum(area_scores) / (len(area_scores) * 10) if area_scores else 0  # 0-1 스케일
        
        # 전체 완료/누락 항목 수집
        all_completed_items = []
        all_missing_items = []
        for area_data in areas_evaluation.values():
            all_completed_items.extend(area_data.get("completed_items", []))
            all_missing_items.extend(area_data.get("missing_items", []))
        
        rag_completeness_result = {
            "category": scenario_category or scenario_id,
            "overall_completeness": round(overall_completeness, 2),
            "areas_evaluation": areas_evaluation,
            "total_completed_items": len(all_completed_items),
            "total_missing_items": len(all_missing_items),
            "completed_items": all_completed_items,
            "missing_items": all_missing_items,
            "evaluation_method": "rag_three_areas"
        }
        
        print(f"✅ [{state['user_id']}] 1단계: RAG 기반 완성도 평가 완료 - 완성도: {overall_completeness:.2%}")
        
        return {
            **state,
            "completeness_assessment": rag_completeness_result,
            "messages": state["messages"] + [HumanMessage(content="1단계: RAG 기반 완성도 평가 완료")]
        }

    
    def _evaluate_single_area(self, conversation_text: str, area_name: str, rag_data: Dict) -> Dict:
        """RAG 가이드라인 기반 단일 영역 평가"""
        
        # 영역명 매핑
        area_mapping = {
            "병력 청취": "history_taking",
            "신체 진찰": "physical_examination", 
            "환자 교육": "patient_education"
        }
        
        area_key = area_mapping.get(area_name, area_name)
        
        # RAG에서 해당 영역의 구체적 평가 기준 가져오기
        if self.guideline_retriever and hasattr(self.guideline_retriever, 'vectorstore'):
            try:
                # 시나리오 카테고리 정보 가져오기
                scenario_category = rag_data.get("category", "기억력 저하")
                
                # 해당 영역의 세부 기준 검색
                area_query = f"{scenario_category} {area_name} 평가 기준"
                docs = self.guideline_retriever.vectorstore.similarity_search(area_query, k=3)
                
                if docs:
                    # 검색된 문서에서 해당 영역 정보 추출
                    area_guidelines = ""
                    for doc in docs:
                        if area_key in doc.page_content:
                            area_guidelines += doc.page_content + "\n"
                    
                    if area_guidelines:
                        area_prompt = f"""
당신은 의학교육 평가 전문가입니다. 다음 RAG 가이드라인을 기반으로 "{area_name}" 영역을 평가하세요.

【학생-환자 대화】: {conversation_text}

【RAG 가이드라인 - {area_name} 평가 기준】:
{area_guidelines}

위 가이드라인의 구체적 항목들을 기준으로 다음을 평가하세요:
1. 필수 질문/행동들을 얼마나 수행했는가
2. 가이드라인에 명시된 세부 사항들을 다뤘는가
3. 의학적 정확성과 체계성을 보였는가

다음 JSON 형식으로 응답하세요:
{{
    "area_score": 점수(1-10),
    "completeness_level": "excellent/good/fair/poor",
    "completed_items": ["가이드라인 기준으로 완료된 항목들"],
    "missing_items": ["가이드라인 기준으로 누락된 항목들"],
    "strengths": ["RAG 기준으로 잘한 점들"],
    "improvements": ["RAG 기준으로 개선 필요한 점들"],
    "guideline_compliance": "가이드라인 준수도에 대한 구체적 분석"
}}
"""
        
                        try:
                            messages = [
                                SystemMessage(content="당신은 의학교육 평가 전문가입니다."),
                                HumanMessage(content=area_prompt)
                                                ]
        
                            response = self.llm(messages)
                            result_text = response.content.strip()
                            
                            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                            if json_match:
                                result = json.loads(json_match.group())
                                result["evaluation_method"] = "rag_guideline_based"
                                return result
                            else:
                                raise ValueError(f"RAG 기반 {area_name} 평가에서 JSON 패턴을 찾을 수 없습니다. LLM 응답: {result_text}")
                        except Exception as e:
                            print(f"❌ RAG 기반 {area_name} 평가 실패: {e}")
                            raise e
            
            except Exception as e:
                print(f"❌ RAG 검색 실패 ({area_name}): {e}")
                raise e
        
        # RAG 실패 시 기본 평가
        basic_prompt = f"""
당신은 의학교육 평가 전문가입니다. 다음 대화에서 "{area_name}" 영역의 수행도를 평가하세요.

【대화 내용】: {conversation_text}

【{area_name} 일반 평가 기준】:
- 완성도: 필요한 항목들을 얼마나 다뤘는가
- 정확성: 의학적으로 정확한 접근인가  
- 체계성: 논리적 순서로 진행했는가

다음 JSON 형식으로 응답하세요:
{{
    "area_score": 점수(1-10),
    "completeness_level": "excellent/good/fair/poor",
    "completed_items": ["완료된 항목들"],
    "missing_items": ["누락된 항목들"],
    "strengths": ["강점들"],
    "improvements": ["개선 필요점들"],
    "guideline_compliance": "일반적 기준 기반 분석"
}}
"""
        
        try:
            messages = [
                SystemMessage(content="당신은 의학교육 평가 전문가입니다."),
                HumanMessage(content=basic_prompt)
            ]
            
            response = self.llm(messages)
            result_text = response.content.strip()
            
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                result["evaluation_method"] = "basic_fallback"
                return result
            else:
                raise ValueError(f"기본 {area_name} 평가에서 JSON 패턴을 찾을 수 없습니다. LLM 응답: {result_text}")
        except Exception as e:
            print(f"❌ 기본 {area_name} 영역 평가 실패: {e}")
            raise e

    def _build_conversation_text(self, conversation_log: List[Dict]) -> str:
        """대화 로그를 텍스트로 변환"""
        conversation_parts = []
        for msg in conversation_log:
            speaker = "학생" if msg.get("role") == "student" else "환자"
            content = msg.get("content", "")
            conversation_parts.append(f"{speaker}: {content}")
        return "\n".join(conversation_parts)

    def _evaluate_quality_assessment(self, state: CPXEvaluationState) -> CPXEvaluationState:
        print(f"⭐ [{state['user_id']}] 2단계: 품질 평가 시작")
        
        conversation_text = self._build_conversation_text(state["conversation_log"])
        scenario_id = state["scenario_id"]
        scenario_info = self.scenario_applicable_elements.get(scenario_id, {})
        
        quality_assessment_prompt = f"""
당신은 의학교육 평가 전문가입니다. 다음 CPX 대화의 품질을 4가지 기준으로 평가하세요.

【학생-환자 대화】: {conversation_text}
【시나리오 정보】: {scenario_info.get('name', scenario_id)}

다음 4가지 품질 기준으로 평가하세요:

【1. 의학적 정확성 (Medical Accuracy)】:
- 질문의 의학적 타당성과 정확성
- 진단적 접근의 논리성
- 의학 용어 사용의 적절성
- 임상적 판단의 합리성

【2. 의사소통 효율성 (Communication Efficiency)】:
- 환자가 이해하기 쉬운 언어 사용
- 질문의 명확성과 구체성
- 환자 반응에 대한 적절한 후속 질문
- 대화 흐름의 자연스러움

【3. 전문성 (Professionalism)】:
- 의료진다운 태도와 예의
- 환자에 대한 공감과 배려
- 체계적이고 논리적인 접근
- 자신감 있는 진료 태도

【4. 시나리오 적합성 (Scenario Appropriateness)】:
- 주어진 시나리오에 맞는 접근
- 환자 연령/성별/상황 고려
- 시간 제약 내 효율적 진행
- 우선순위에 따른 체계적 접근

각 항목을 1-10점으로 평가하고, 전체 품질 점수를 산출하세요.

다음 JSON 형식으로 응답하세요:
{{
    "medical_accuracy": 의학적정확성점수(1-10),
    "communication_efficiency": 의사소통효율성점수(1-10),
    "professionalism": 전문성점수(1-10),
    "scenario_appropriateness": 시나리오적합성점수(1-10),
    "overall_quality_score": 전체품질점수(1-10),
    "quality_strengths": ["품질 면에서 우수한 점들"],
    "quality_improvements": ["품질 면에서 개선이 필요한 점들"],
    "detailed_analysis": {{
        "medical_accuracy_detail": "의학적 정확성에 대한 구체적 분석",
        "communication_detail": "의사소통에 대한 구체적 분석",
        "professionalism_detail": "전문성에 대한 구체적 분석",
        "scenario_fit_detail": "시나리오 적합성에 대한 구체적 분석"
    }}
}}
"""
        
        try:
            messages = [
                SystemMessage(content="당신은 의학교육 평가 전문가입니다."),
                HumanMessage(content=quality_assessment_prompt)
            ]
            
            response = self.llm(messages)
            result_text = response.content.strip()
            
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                quality_assessment = json.loads(json_match.group())
                
                print(f"✅ [{state['user_id']}] 2단계: 품질 평가 완료 - 종합 점수: {quality_assessment.get('overall_quality_score', 0)}")
                
                return {
                    **state,
                    "quality_evaluation": quality_assessment,
                    "messages": state["messages"] + [HumanMessage(content="2단계: 품질 평가 완료")]
                }
            else:
                raise ValueError(f"품질 평가에서 JSON 패턴을 찾을 수 없습니다. LLM 응답: {result_text}")
        except Exception as e:
            print(f"❌ [{state['user_id']}] 품질 평가 실패: {e}")
            raise e

    def _generate_comprehensive_results(self, state: CPXEvaluationState) -> CPXEvaluationState:
        """3단계: 종합 평가 및 최종 결과 생성"""
        print(f"🎯 [{state['user_id']}] 3단계: 종합 평가 시작")
        
        # 1단계와 2단계 결과 수집
        rag_completeness = state.get("completeness_assessment", {})
        quality_assessment = state.get("quality_evaluation", {})
        
        # 최종 점수 계산 (가중치: 완성도 60%, 품질 40%)
        completeness_score = rag_completeness.get("overall_completeness", 0.5) * 10  # 0-10 스케일로 변환
        quality_score = quality_assessment.get("overall_quality_score", 6)
        
        # 가중치 적용: 완성도 60%, 품질 40%
        final_score = (completeness_score * 0.6) + (quality_score * 0.4)
        final_score = min(10, max(0, final_score))  # 0-10 범위로 제한
        
        # 종합 피드백 생성
        strengths = []
        improvements = []
        
        # 1단계 RAG 평가에서 강점/개선점 수집
        for area_name, area_data in rag_completeness.get("areas_evaluation", {}).items():
            if isinstance(area_data, dict):
                strengths.extend(area_data.get("strengths", []))
                improvements.extend(area_data.get("improvements", []))
        
        # 2단계 품질 평가에서 강점/개선점 추가
        strengths.extend(quality_assessment.get("quality_strengths", []))
        improvements.extend(quality_assessment.get("quality_improvements", []))
        
        # 상세 분석 생성
        detailed_analysis_parts = []
        detailed_analysis_parts.append(f"【완성도 평가】 RAG 기반 평가 결과 {rag_completeness.get('overall_completeness', 0):.1%} 완성")
        detailed_analysis_parts.append(f"【품질 평가】 4가지 품질 기준 평균 {quality_score:.1f}점")
        
        if rag_completeness.get("total_completed_items", 0) > 0:
            detailed_analysis_parts.append(f"총 {rag_completeness.get('total_completed_items', 0)}개 항목 완료")
        
        if rag_completeness.get("total_missing_items", 0) > 0:
            detailed_analysis_parts.append(f"{rag_completeness.get('total_missing_items', 0)}개 항목 누락")
        
        comprehensive_result = {
            "final_score": round(final_score, 1),
            "grade": self._calculate_grade(final_score * 10),  # 100점 스케일로 변환
            "score_breakdown": {
                "completeness_score": round(completeness_score, 1),
            "quality_score": round(quality_score, 1),
                "weighted_completeness": round(completeness_score * 0.6, 1),
                "weighted_quality": round(quality_score * 0.4, 1)
            },
            "detailed_feedback": {
                "strengths": list(set(strengths))[:5] if strengths else ["평가를 성실히 완료했습니다"],
                "improvements": list(set(improvements))[:5] if improvements else ["지속적인 학습과 연습이 필요합니다"],
                "overall_analysis": " | ".join(detailed_analysis_parts)
            },
            "evaluation_summary": {
                "method": "3단계 RAG+품질 평가",
                "steps_completed": 3,
                "completeness_rate": rag_completeness.get("overall_completeness", 0),
                "quality_details": quality_assessment.get("detailed_analysis", {}),
                "total_items_evaluated": rag_completeness.get("total_completed_items", 0) + rag_completeness.get("total_missing_items", 0)
            }
        }
        
        print(f"✅ [{state['user_id']}] 3단계: 종합 평가 완료 - 최종 점수: {final_score:.1f}/10 ({comprehensive_result['grade']})")
        
        return {
            **state,
            "comprehensive_evaluation": comprehensive_result,
            "final_scores": {
                "total_score": round(final_score * 10, 1),  # 100점 스케일
                "completion_rate": rag_completeness.get("overall_completeness", 0.5),
                "quality_score": quality_score,
                "grade": comprehensive_result["grade"]
            },
            "feedback": comprehensive_result["detailed_feedback"],
            "messages": state["messages"] + [HumanMessage(content=f"3단계: 종합 평가 완료 - {final_score:.1f}점 ({comprehensive_result['grade']})")]
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

    async def _update_cpx_database_after_evaluation(self, session_id: str, evaluation_result: dict):
        """평가 완료 후 CPX Details만 업데이트"""
        try:
            session = self.session_data[session_id]
            result_id = session["result_id"]
            user_id = session["user_id"]
            
            # CPX Details만 업데이트 (시스템 평가 데이터)
            async for db in get_db():
                cpx_service = CpxService(db)
                
                await cpx_service.update_cpx_details(
                    result_id=result_id,
                    user_id=int(user_id),
                    system_evaluation_data=evaluation_result
                )
                
                print(f"✅ CPX Details 업데이트 완료: result_id={result_id}, session_id={session_id}")
                break
                
        except Exception as e:
            print(f"❌ CPX Details 업데이트 실패: {e}")

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