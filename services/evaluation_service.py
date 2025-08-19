from typing import Dict, List, Optional, TypedDict, Annotated
from datetime import datetime
from pathlib import Path
import json
import asyncio
import aiofiles
# SER 관련 import는 제거됨 (ser_service로 분리)
import logging
import os

# LangGraph 관련 import
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage as AnyMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

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
        
        # SER 관련 코드는 ser_service와 queue로 분리됨
        
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
            import sys
            from pathlib import Path
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
                    ],
                    "critical_points": self._extract_critical_points_from_scenario(data)
                }
                
                print(f"✅ 시나리오 로드: {category} ({json_file.name})")
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON 파싱 오류 ({json_file.name}): {e}")
            except Exception as e:
                print(f"❌ 시나리오 로드 오류 ({json_file.name}): {e}")
        
        return scenario_elements
    
    def _extract_critical_points_from_scenario(self, scenario_data: Dict) -> List[str]:
        """시나리오 데이터에서 중요 포인트 추출"""
        critical_points = []
        
        # history_taking에서 중요 포인트 추출
        history_taking = scenario_data.get("history_taking", {})
        if "O_onset" in history_taking:
            critical_points.append("발병 시기와 양상")
        if "A_associated" in history_taking:
            critical_points.append("동반 증상 확인")
        if any(key in history_taking for key in ["past_medical_history", "family_history"]):
            critical_points.append("과거력 및 가족력")
        
        # physical_examination에서 중요 포인트 추출
        physical_exam = scenario_data.get("physical_examination", {})
        if "MMSE" in physical_exam:
            critical_points.append("인지기능 평가")
        if any(key in physical_exam for key in ["CN_exam", "신경학적검사"]):
            critical_points.append("신경학적 검사")
        
        # 기본값 설정
        if not critical_points:
            critical_points = ["병력 청취", "신체 진찰", "환자 교육"]
        
        return critical_points

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
            "critical_gaps": [],
            "medical_completeness_analysis": "카테고리 정보를 찾을 수 없어 기본 평가를 수행합니다."
        }
        
        return {
            **state,
            "completeness_assessment": completeness,
            "messages": state["messages"] + [HumanMessage(content="Step 3: 기본 완성도 평가 완료")]
        }

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

    # SER 관련 메서드들은 ser_service로 분리됨





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
        """Step 3: 의학적 완성도 평가 - 시나리오 카테고리 기반 평가"""
        print(f"📋 [{state['user_id']}] Step 3: 의학적 완성도 평가 시작")
        
        conversation_text = self._build_conversation_text(state["conversation_log"])
        scenario_id = state["scenario_id"]
        
        # 시나리오에서 카테고리 정보 로드
        scenario_category = self._get_scenario_category(scenario_id)
        if not scenario_category:
            print(f"⚠️ 시나리오 {scenario_id}의 카테고리를 찾을 수 없습니다.")
            return self._create_default_completeness_result(state)
        
        # 해당 카테고리의 평가 체크리스트 로드
        checklist = self.get_evaluation_checklist(scenario_category)
        if not checklist:
            print(f"⚠️ '{scenario_category}' 카테고리의 평가 체크리스트를 찾을 수 없습니다.")
            return self._create_default_completeness_result(state)
        
        print(f"✅ 카테고리 '{scenario_category}' 체크리스트 사용")
        
        # 평가 영역별로 평가 수행
        applicable_categories = self._extract_applicable_categories(checklist)
        
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
        # 카테고리별 일반적 예시 및 유사 표현 추가
        category_examples = {
            "외상력": "머리 다침, 교통사고, 낙상, 외상, 부상, 골절 등 외상 경험에 관한 질문",
            "가족력": "가족 중 질병력, 부모님 병력, 형제자매 질환, 유전적 질환 등에 관한 질문",
            "과거력": "기존 질병, 과거 병원 치료, 수술 경험, 입원력 등에 관한 질문",
            "약물력": "현재 복용 약물, 처방약, 일반의약품, 알레르기 등에 관한 질문",
            "사회력": "흡연, 음주, 직업, 생활습관, 운동 등에 관한 질문",
            "O (Onset) - 발병 시기": "증상 시작 시점, 언제부터, 얼마나 오래 등 시간 관련 질문. 유사표현: '언제부터', '얼마나 오래', '시작된 시기', '처음 느낀 때'",
            "C (Character) - 특징": "증상의 성질, 양상, 강도, 정도 등에 관한 질문. 유사표현: '어떤 증상', '어떻게 느껴지는지', '증상의 특징', '어떤 기억력 문제'",
            "A (Associated symptom) - 동반 증상": "함께 나타나는 증상, 관련 증상 등에 관한 질문. 유사표현: '다른 증상', '함께 나타나는', '동반되는', '추가 증상'",
            "F (Factor) - 악화/완화요인": "증상을 악화시키는 요인, 완화시키는 요인 등에 관한 질문. 유사표현: '악화되는 때', '나아지는 때', '스트레스', '휴식'",
            "인지 검사 (간이 MMSE)": "MMSE, 인지검사, 기억력검사, 간이정신상태검사 등. 유사표현: 'MMSE', 'mnse', '엠엠에스이', '인지검사', '기억력 테스트', '간이 검사'",
            "신경학적 검사": "뇌신경검사, 신경학적 검사, 반사검사 등. 유사표현: '뇌신경', '신경검사', '반사', '신경학적', '뇌 검사'",
            "운동 검사": "보행검사, 걸음걸이, 운동기능 등. 유사표현: '보행', '걸어보세요', '걸음', '운동', '움직임'"
        }
        
        example_text = category_examples.get(category['name'], "")
        
        single_category_prompt = f"""
당신은 의학교육 평가 전문가입니다. 다음 병력청취 대화에서 "{category['name']}" 항목만 집중적으로 평가해주세요.

【평가 대상】: {category['name']}
【필수 요소들】: {category['required_elements']}
【예시 및 유사 표현】: {example_text}

【학생-환자 대화】: {conversation_text}

⚠️ **평가 원칙**:
- **관대한 평가**: 비슷한 의미의 질문이면 점수 부여
- **STT 오류 고려**: 발음 차이나 인식 오류 감안 (예: MMSE → mnse, 엠엠에스이)
- **의도 중심**: 정확한 용어보다는 질문/검사 의도가 있는지 판단
- **구술 검사**: 실제 검사 수행이 아닌 구술로 검사 언급만 해도 인정

이 대화에서 "{category['name']}" 관련 내용이 어느 정도 다뤄졌는지만 평가하세요:
1. 직접적 완료: 명시적으로 질문하거나 검사 언급함
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
당신은 의과대학 CPX(Clinical Performance Examination) 평가 전문가입니다. 
아래 5단계 분석 결과를 바탕으로 학생의 실제 수행도를 객관적으로 평가하세요.

=== 5단계 분석 결과 ===
【Step 1 - 의학적 맥락 분석】: {medical_context}
【Step 2 - 질문 의도 분석】: {question_intent}
【Step 3 - 의학적 완성도 평가】: {completeness}
【Step 4 - 질적 수준 평가】: {quality}
【Step 5 - 시나리오 적합성 검증】: {appropriateness}

=== 점수 산출 공식 ===
• 완성도 (40%): Step 3의 필수 항목 달성률
• 품질 (30%): Step 4의 질문/대화 수준  
• 적합성 (20%): Step 5의 시나리오 부합도
• 의도 (10%): Step 2의 질문 의도 적절성

점수 계산 방법:
1. 완성도: 0.0~1.0 범위로 평가 → ×40 (최대 40점)
2. 품질: 0~10 범위로 평가 → ×3 (최대 30점)  
3. 적합성: 0.0~1.0 범위로 평가 → ×20 (최대 20점)
4. 의도: 0.0~1.0 범위로 평가 → ×10 (최대 10점)

최종 점수 = 완성도×40 + 품질×3 + 적합성×20 + 의도×10 (총 100점 만점)

중요: 각 Step의 실제 분석 결과를 바탕으로 객관적 점수를 산출하세요.

반드시 아래 JSON 형식으로만 응답하세요:
{{
    "final_completion_rate": 0.0~1.0_사이의_실제_완성도,
    "final_quality_score": 0~10_사이의_실제_품질점수,
    "weighted_scores": {{
        "completeness_weighted": 완성도에_40을_곱한_값,
        "quality_weighted": 품질에_3을_곱한_값,
        "appropriateness_weighted": 적합성에_20을_곱한_값,
        "intent_weighted": 의도에_10을_곱한_값
    }},
    "detailed_feedback": {{
        "strengths": ["대화에서_실제_관찰된_강점1", "구체적_강점2", "구체적_강점3"],
        "weaknesses": ["대화에서_실제_발견된_약점1", "구체적_약점2", "구체적_약점3"],
        "medical_insights": ["의학적_우수점_또는_부족점1", "의학적_통찰2"]
    }},
    "comprehensive_analysis": "위_5단계_분석을_종합한_상세한_평가_내용(최소_100자)"
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
        
        # 프롬프트에서 명시한 공식 사용: (완성도×40) + (품질×3) + (적합성×20) + (의도×10)
        final_total_score = (
            weighted_scores.get("completeness_weighted", 0) +
            weighted_scores.get("quality_weighted", 0) +
            weighted_scores.get("appropriateness_weighted", 0) +
            weighted_scores.get("intent_weighted", 0)
        )
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