"""
CPX 평가 서비스 - 정리된 버전
의료 시뮬레이션 대화 평가를 위한 서비스
"""

from typing import Dict, List, Optional, TypedDict, Annotated
from datetime import datetime
from pathlib import Path
import json
import logging
import os
import re
import time

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage as AnyMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from services.cpx_service import CpxService
from core.config import get_db

from RAG.guideline_retriever import GuidelineRetriever

logger = logging.getLogger(__name__)


class CPXEvaluationState(TypedDict):
    """CPX 평가 상태 정의"""
    # 입력 데이터
    user_id: str
    scenario_id: str
    conversation_log: List[Dict]
    
    # 평가 결과들
    completeness_assessment: Optional[Dict]
    quality_evaluation: Optional[Dict]
    
    # 종합 평가 결과
    comprehensive_evaluation: Optional[Dict]
    
    # 최종 결과
    final_scores: Optional[Dict]
    feedback: Optional[Dict]
    markdown_feedback: Optional[str]  # 마크다운 피드백 추가
    
    # 메타데이터
    evaluation_metadata: Optional[Dict]
    
    # 메시지 추적
    messages: Annotated[List[AnyMessage], add_messages]


class EvaluationService:
    """CPX 평가 서비스"""
    
    def __init__(self):
        """CPX 평가 서비스 초기화"""
        self.session_data = {}
        
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

    # ================================
    # 1. 초기화 관련 메서드들
    # ================================
    
    def _initialize_langgraph_components(self):
        """LangGraph 컴포넌트들 초기화"""
        try:
            # OpenAI API 설정
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.llm = ChatOpenAI(
                    openai_api_key=api_key,
                    model_name="gpt-5-nano",
                    max_tokens=4000
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

    def _initialize_guideline_retriever(self):
        """RAG 기반 가이드라인 검색기 초기화"""
        try:
            # 가이드라인 검색기 초기화
            rag_path = Path(__file__).parent.parent / "RAG"
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



    # ================================
    # 2. 세션 관리 메서드들 (외부 API)
    # ================================
    
    async def start_evaluation_session(self, user_id: str, scenario_id: str, result_id: Optional[int] = None) -> str:
        """평가 세션 시작"""
        session_id = f"{user_id}_{scenario_id}_{int(datetime.now().timestamp())}"
        
        self.session_data[session_id] = {
            "user_id": user_id,
            "scenario_id": scenario_id,
            "result_id": result_id,  # CPX result_id 저장
            "start_time": datetime.now(),
            "conversation_entries": [],  # 실시간 대화 데이터
            "status": "active"
        }
        
        return session_id

    async def add_conversation_entry(self, session_id: str, text: str, role: str, emotion_analysis: Optional[Dict] = None) -> Dict:
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
                "role": role,  # "doctor" (의사) or "patient" (환자)
            }
            
            # 세션 데이터에 추가
            session = self.session_data[session_id]
            session["conversation_entries"].append(conversation_entry)
            
            print(f"📝 [{session_id}] 대화 엔트리 추가: {role} - {text[:50]}...")
            
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



    # ================================
    # 3. LangGraph 워크플로우 관련
    # ================================
    
    def _create_evaluation_workflow(self):
        """CPX 평가 워크플로우 생성 (3단계)"""
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
        """평가 초기화"""
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

    async def _evaluate_rag_completeness(self, state: CPXEvaluationState) -> CPXEvaluationState:
        """1단계: RAG 기반 완성도 평가 (병력청취, 신체진찰, 환자교육)"""
        print(f"📋 [{state['user_id']}] 1단계: RAG 기반 완성도 평가 시작")
        
        conversation_text = self._build_conversation_text(state["conversation_log"])
        scenario_id = state["scenario_id"]
        
        # 시나리오에서 카테고리 정보 로드
        scenario_category = self._get_scenario_category(scenario_id)
        if not scenario_category:
            raise ValueError(f"시나리오 '{scenario_id}'의 카테고리를 찾을 수 없습니다.")
                    
        # 3개 영역별 청크 기반 평가
        areas_evaluation = {}
        
        for area_key, area_name in [("history_taking", "병력 청취"), ("physical_examination", "신체 진찰"), ("patient_education", "환자 교육")]:
            # RAG에서 가이드라인 가져오기
            criteria_data = self.guideline_retriever.get_evaluation_criteria(scenario_category, area_name)
            documents = criteria_data.get("documents", [])
            
            if not documents or not documents[0]:
                raise ValueError(f"❌ {area_name} 가이드라인을 찾을 수 없습니다.")
            
            # 구조화된 섹션 파싱
            structured_sections = self._parse_structured_sections(documents[0])
            
            # 간단한 RAG 가이드라인 비교 평가 실행
            areas_evaluation[area_key] = await self._evaluate_area_simple(
                conversation_text, area_name, structured_sections
            )
        
        # 전체 완성도 점수 계산
        total_guidelines = sum(area.get("total_guidelines", 0) for area in areas_evaluation.values())
        completed_guidelines = sum(area.get("completed_guidelines", 0) for area in areas_evaluation.values())
        overall_completeness = completed_guidelines / total_guidelines if total_guidelines > 0 else 0
        
        # 전체 완료/누락 항목 수집 (간단화)
        all_completed_items = []
        all_missing_items = []
        for area_data in areas_evaluation.values():
            completed_count = area_data.get("completed_guidelines", 0)
            total_count = area_data.get("total_guidelines", 0)
            missing_count = total_count - completed_count
            
            if completed_count > 0:
                all_completed_items.append(f"{area_data.get('area_name', 'Unknown')}: {completed_count}개 항목 완료")
            if missing_count > 0:
                all_missing_items.append(f"{area_data.get('area_name', 'Unknown')}: {missing_count}개 항목 누락")
        
        rag_completeness_result = {
            "category": scenario_category or scenario_id,
            "overall_completeness": round(overall_completeness, 2),
            "areas_evaluation": areas_evaluation,
            "total_completed_items": len(all_completed_items),
            "total_missing_items": len(all_missing_items),
            "evaluation_method": "rag_three_areas"
        }
        
        print(f"✅ [{state['user_id']}] 1단계: RAG 기반 완성도 평가 완료 - 완성도: {overall_completeness:.2%}")
        
        return {
            **state,
            "completeness_assessment": rag_completeness_result,
            "messages": state["messages"] + [HumanMessage(content=f"1단계: RAG 기반 완성도 평가 완료 - {overall_completeness:.1%}")]
        }

    async def _evaluate_quality_assessment(self, state: CPXEvaluationState) -> CPXEvaluationState:
        """2단계: 대화 품질 평가 (친절함, 공감, 전문성 등)"""
        print(f"⭐ [{state['user_id']}] 2단계: 품질 평가 시작")
        
        conversation_text = self._build_conversation_text(state["conversation_log"])
        scenario_id = state["scenario_id"]
        
        quality_prompt = f"""
당신은 의학교육 평가 전문가입니다. 다음 CPX 대화의 품질을 4가지 기준으로 평가하세요.

【의사-환자 대화】:
{conversation_text}

【시나리오】: {scenario_id}

다음 4가지 품질 기준으로 평가하세요:

【1. 의학적 정확성】:
- 질문의 의학적 타당성과 정확성
- 진단적 접근의 논리성
- 의학 용어 사용의 적절성

【2. 의사소통 효율성】:
- 환자가 이해하기 쉬운 언어 사용
- 질문의 명확성과 구체성
- 대화 흐름의 자연스러움

【3. 전문성】:
- 의료진다운 태도와 예의
- 환자에 대한 공감과 배려
- 체계적이고 논리적인 접근

【4. 시나리오 적합성】:
- 주어진 시나리오에 맞는 접근
- 환자 상황 고려
- 효율적 진행

각 항목을 1-10점으로 평가하세요.

JSON 응답:
{{
    "medical_accuracy": 점수(1-10),
    "communication_efficiency": 점수(1-10),
    "professionalism": 점수(1-10),
    "scenario_appropriateness": 점수(1-10),
    "overall_quality_score": 전체품질점수(1-10),
    "quality_strengths": ["우수한 점들"],
    "quality_improvements": ["개선이 필요한 점들"]
}}"""

        try:
            messages = [SystemMessage(content=quality_prompt)]
            response = await self.llm.ainvoke(messages)
            result_text = response.content
            
            print(f"[품질] LLM 응답 원문:\n{result_text[:300]}...")
            
            # JSON 파싱
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                quality_result = json.loads(json_str)
                print(f"[품질] JSON 파싱 성공")
            else:
                print(f"[품질] JSON 형식을 찾을 수 없습니다.")
                quality_result = {
                    "medical_accuracy": 6,
                    "communication_efficiency": 6,
                    "professionalism": 6,
                    "scenario_appropriateness": 6,
                    "overall_quality_score": 6,
                    "quality_strengths": ["RAG 기반 평가 완료"],
                    "quality_improvements": ["품질 평가 개선 필요"]
                }
            
            print(f"✅ [{state['user_id']}] 2단계: 품질 평가 완료 - 평균: {quality_result.get('overall_quality_score', 6):.1f}점")
            
            return {
                **state,
                "quality_evaluation": quality_result,
                "messages": state["messages"] + [HumanMessage(content=f"2단계: 품질 평가 완료 - {quality_result.get('overall_quality_score', 6):.1f}점")]
            }
            
        except Exception as e:
            print(f"❌ [{state['user_id']}] 품질 평가 실패: {e}")
            return {
                **state,
                "quality_evaluation": {
                    "medical_accuracy": 6,
                    "communication_efficiency": 6,
                    "professionalism": 6,
                    "scenario_appropriateness": 6,
                    "overall_quality_score": 6,
                    "quality_strengths": ["기본 평가 완료"],
                    "quality_improvements": ["품질 평가 오류로 기본값 사용"]
                }
            }

    async def _generate_comprehensive_results(self, state: CPXEvaluationState) -> CPXEvaluationState:
        """3단계: SER 감정 분석 + 종합 평가 및 최종 결과 생성"""
        print(f"🎯 [{state['user_id']}] 3단계: SER 감정 분석 + 종합 평가 시작")
        
        # 1단계와 2단계 결과 수집
        rag_completeness = state.get("completeness_assessment", {})
        quality_assessment = state.get("quality_evaluation", {})
        
        # SER 감정 분석 수행
        ser_evaluation = await self._evaluate_ser_emotions(state["conversation_log"])
        
        # 최종 점수 계산 (가중치: 완성도 30%, 품질 30%, SER 감정 40%)
        completeness_score = rag_completeness.get("overall_completeness", 0.5) * 10  # 0-10 스케일로 변환
        quality_score = quality_assessment.get("overall_quality_score", 6)
        ser_score = ser_evaluation.get("ser_score", 6)  # SER 기반 감정 점수
        
        # 가중치 적용: 완성도 30%, 품질 30%, SER 감정 40%
        final_score = (completeness_score * 0.3) + (quality_score * 0.3) + (ser_score * 0.4)
        final_score = min(10, max(0, final_score))  # 0-10 범위로 제한
        
        # 등급 계산
        grade = self._calculate_grade(final_score * 10)  # 100점 스케일로 변환
        
        # 종합 피드백 생성
        strengths = []
        improvements = []
        
        # 1단계에서 강점/개선점 수집
        for area_key, area_data in rag_completeness.get("areas_evaluation", {}).items():
            area_name = area_data.get('area_name', area_key)
            completion_rate = area_data.get("completion_rate", 0)
            if completion_rate > 0.7:
                strengths.append(f"{area_name} 영역 우수 ({completion_rate:.1%})")
            elif completion_rate < 0.5:
                improvements.append(f"{area_name} 영역 보완 필요 ({completion_rate:.1%})")
        
        # 2단계에서 강점/개선점 추가
        strengths.extend(quality_assessment.get("quality_strengths", []))
        improvements.extend(quality_assessment.get("quality_improvements", []))
        
        # 3단계 SER에서 강점/개선점 추가
        strengths.extend(ser_evaluation.get("ser_strengths", []))
        improvements.extend(ser_evaluation.get("ser_improvements", []))
        
        comprehensive_result = {
            "final_score": round(final_score, 1),
            "grade": grade,
            "detailed_feedback": {
                "strengths": strengths[:5],  # 최대 5개
                "improvements": improvements[:5],  # 최대 5개
                "overall_analysis": f"3단계 통합 평가 결과 {final_score * 10:.1f}점 (완성도 30% + 품질 30% + 감정 40%)"
            },
            "ser_evaluation": ser_evaluation  # SER 평가 결과 포함
        }
        
        print(f"✅ [{state['user_id']}] 3단계: 종합 평가 완료 - 최종 점수: {final_score:.1f}/10 ({grade})")
        
        # 최종 결과 구성
        final_result = {
            **state,
            "comprehensive_evaluation": comprehensive_result,
            "final_scores": {
                "total_score": round(final_score * 10, 1),  # 100점 스케일
                "completion_rate": rag_completeness.get("overall_completeness", 0.5),
                "quality_score": quality_score,
                "grade": grade
            },
            "feedback": comprehensive_result["detailed_feedback"],
            "messages": state["messages"] + [HumanMessage(content=f"3단계: 종합 평가 완료 - {final_score:.1f}점 ({grade})")]
        }
        
        # 마크다운 피드백 생성
        print(f"🔍 DEBUG [{state['user_id']}] 마크다운 피드백 생성 시작")
        try:
            # generate_evaluation_markdown 함수가 기대하는 전체 구조 생성
            evaluation_data = {
                "user_id": state['user_id'],
                "scenario_id": state['scenario_id'],
                "duration_minutes": 0,  # 실제 duration이 있다면 사용
                "langgraph_text_analysis": {
                    "scores": final_result["final_scores"],
                    "feedback": final_result["feedback"], 
                    "detailed_analysis": {
                        "completeness": rag_completeness,
                        "quality": quality_assessment,
                        "comprehensive": comprehensive_result
                    },
                    "conversation_summary": {},
                    # 기존 구조도 유지 (호환성)
                    "completeness_assessment": rag_completeness,
                    "quality_evaluation": quality_assessment,
                    "comprehensive_evaluation": comprehensive_result,
                    "final_scores": final_result["final_scores"]
                }
            }
            print(f"🔍 DEBUG [{state['user_id']}] evaluation_data 구성 완료")
            
            markdown_feedback = self.generate_evaluation_markdown(evaluation_data)
            print(f"🔍 DEBUG [{state['user_id']}] generate_evaluation_markdown 호출 결과: type={type(markdown_feedback)}, len={len(markdown_feedback) if markdown_feedback else 0}")
            
            if markdown_feedback:
                print(f"🔍 DEBUG [{state['user_id']}] 마크다운 내용 미리보기: {markdown_feedback[:100]}...")
            else:
                print(f"🔍 DEBUG [{state['user_id']}] ⚠️ 마크다운이 비어있음!")
            
            print(f"✅ [{state['user_id']}] 마크다운 피드백 생성 완료 ({len(markdown_feedback) if markdown_feedback else 0}자)")
            
            # final_result에 markdown_feedback 추가
            final_result["markdown_feedback"] = markdown_feedback
            print(f"🔍 DEBUG [{state['user_id']}] final_result에 markdown_feedback 추가 완료")
            
            return final_result
            
        except Exception as e:
            print(f"❌ [{state['user_id']}] 마크다운 피드백 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            
            # 오류 발생 시
            final_result["markdown_feedback"] = None
            return final_result

    # ================================
    # 4. 핵심 평가 로직
    # ================================
    
    async def _comprehensive_evaluation(self, session_id: str, session: Dict) -> Dict:
        """종합적인 세션 평가 수행 (SER + LangGraph 통합)"""
        print(f"🔍 [{session_id}] 종합 평가 시작...")
        
        # LangGraph 기반 텍스트 평가 (직접 워크플로우 실행)
        langgraph_analysis = None
        if self.llm and self.workflow:
            try:
                # 새로운 conversation_entries를 conversation_log 형식으로 변환
                conversation_log = []
                for entry in session.get("conversation_entries", []):
                    conversation_log.append({
                        "role": entry["role"],
                        "content": entry["text"],
                        "timestamp": entry["timestamp"],
                        "emotion": entry.get("emotion")
                    })
                
                if conversation_log:  # 대화 데이터가 있는 경우에만 평가
                    # LangGraph 워크플로우 직접 실행
                    initial_state = CPXEvaluationState(
                        user_id=session["user_id"],
                        scenario_id=session["scenario_id"],
                        conversation_log=conversation_log,
                        completeness_assessment=None,
                        quality_evaluation=None,
                        comprehensive_evaluation=None,
                        final_scores=None,
                        feedback=None,
                        markdown_feedback=None,  # 마크다운 피드백 초기화
                        evaluation_metadata=None,
                        messages=[]
                    )
                    
                    print(f"🚀 [{session_id}] LangGraph 워크플로우 시작")
                    final_state = await self.workflow.ainvoke(initial_state)
                    
                    print(f"🔍 DEBUG [{session_id}] final_state keys: {list(final_state.keys()) if isinstance(final_state, dict) else 'Not dict'}")
                    print(f"🔍 DEBUG [{session_id}] final_state.markdown_feedback: {final_state.get('markdown_feedback')}")
                    
                    # LangGraph 분석 결과 구성
                    student_questions = [msg for msg in conversation_log if msg.get("role") == "doctor"]
                    conversation_summary = {
                        "total_questions": len(student_questions),
                        "duration_minutes": (session["end_time"] - session["start_time"]).total_seconds() / 60
                    }
                    
                    markdown_from_state = final_state.get("markdown_feedback")
                    print(f"🔍 DEBUG [{session_id}] markdown_from_state: type={type(markdown_from_state)}, value={markdown_from_state}")
                    
                    langgraph_analysis = {
                        "evaluation_metadata": final_state.get("evaluation_metadata", {}),
                        "scores": final_state.get("final_scores", {}),
                        "feedback": final_state.get("feedback", {}),
                        "conversation_summary": conversation_summary,
                        "detailed_analysis": {
                            "completeness": final_state.get("completeness_assessment", {}),
                            "quality": final_state.get("quality_evaluation", {}),
                            "comprehensive": final_state.get("comprehensive_evaluation", {})
                        },
                        "markdown_feedback": markdown_from_state,
                        "evaluation_method": "3단계 의학적 분석",
                        "system_info": {
                            "version": "v2.0",
                            "evaluation_steps": 3
                        }
                    }
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
                    "role": entry["role"],
                    "emotion": entry.get("emotion")
                }
                for entry in session.get("conversation_entries", [])
            ]
        }
        
        # 평가 결과를 JSON 파일로 저장
        try:
            timestamp = int(time.time())
            filename = f"evaluation_{session_id}_{timestamp}.json"
            file_path = self.evaluation_dir / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_result, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"💾 [{session_id}] 평가 결과 저장 완료: {filename}")
        except Exception as e:
            print(f"❌ [{session_id}] 평가 결과 파일 저장 실패: {e}")
        
        print(f"✅ [{session_id}] 종합 평가 완료")
        return evaluation_result

    async def _evaluate_area_simple(self, conversation_text: str, area_name: str, structured_sections: dict) -> dict:
        """단일 단계 RAG 가이드라인 비교 평가 - GPT-4o 통합 평가"""
        
        # 가이드라인 텍스트 구성
        detailed_guideline_text = ""
        for section_name, section_data in structured_sections.items():
            required_items = section_data.get('required_questions', []) + section_data.get('required_actions', [])
            if required_items:
                detailed_guideline_text += f"\n【{section_name}】\n"
                detailed_guideline_text += "이 항목에서 확인해야 할 구체적 내용들:\n"
                for item in required_items:
                    detailed_guideline_text += f"  • {item}\n"
                detailed_guideline_text += "\n"
        
        print(f"[통합 평가] {area_name} 영역 평가 시작...")
        
        prompt = f"""{area_name} 영역 평가

전체 대화:
{conversation_text}

평가할 항목들:
{detailed_guideline_text}

**평가 방법**:
1. 먼저 전체 대화에서 {area_name} 영역에 해당하는 부분을 파악하세요
2. 해당 영역의 대화 내용만을 기준으로 각 항목을 평가하세요
3. 다른 영역의 발언은 사용하지 마세요

**영역 파악 가이드**:
**일반적 순서**: 병력청취(초반) → 신체진찰(중반) → 환자교육(후반)

**영역별 특징**:
- **병력청취**: 정보 수집 목적
  * 특징: 질문형 발언, "언제부터", "어떻게", "있으세요" 등
  * 위치: 대화 초반~중반 (진찰 시작 전)
  * 목적: 증상, 병력, 가족력 등 정보 탐색

- **신체진찰**: 검사 수행 목적
  * 특징: "진찰하겠습니다", "검사하겠습니다" 등 행위형 발언
  * 위치: 병력청취 후 명시적 진찰 구간
  * 목적: 물리적 검사 실시

- **환자교육**: 정보 전달 목적
  * 특징: "가능성", "때문에", "입니다" 등 설명형 발언
  * 위치: 신체진찰 후~대화 종료
  * 목적: 진단 설명, 치료 계획 안내

**평가 규칙**:
- {area_name} 영역 발언만 evidence로 사용
- 대화 원문을 정확히 복사 (변경 금지)
- 관련 내용이 다뤄졌으면 completed: true

JSON 응답:
{{
{', '.join([f'    "{section_name}": {{"completed": true/false, "evidence": []}}' for section_name in structured_sections.keys()])}
}}"""
        
        result = await self._process_evaluation_response(prompt, area_name, structured_sections, stage="통합")
        
        print(f"[검증] evidence 실제 존재 여부 확인...")
        print(f"[검증] 대화 텍스트 샘플: {conversation_text[:200]}...")
        # evidence 검증 단계 추가
        verified_result = self._verify_evidence_exists(conversation_text, result)
        
        return verified_result
        


    # ================================
    # 5. 유틸리티 및 헬퍼 메서드들
    # ================================
    
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

    def _parse_structured_sections(self, document) -> dict:
        """문서에서 구조화된 섹션 파싱 - RAG 가이드라인 JSON 형식 처리"""
        structured_sections = {}
        
        # 문서를 문자열로 변환
        if hasattr(document, 'page_content'):
            document_text = document.page_content
        elif isinstance(document, dict):
            document_text = document.get('content', '') or document.get('page_content', '') or str(document)
        else:
            document_text = str(document)
        
        # 섹션 패턴: 【섹션명】
        section_pattern = re.compile(r'【([^】]+)】')
        # 항목 패턴: • 또는 - 로 시작하는 줄
        bullet_pattern = re.compile(r'^\s*[•\-\*]\s+(.+)$', re.MULTILINE)
        
        sections = section_pattern.split(document_text)
        
        for i in range(1, len(sections), 2):
            if i + 1 < len(sections):
                section_name = sections[i].strip()
                section_content = sections[i + 1]
                
                # 섹션 내용에서 필수 항목 추출
                required_items = bullet_pattern.findall(section_content)
                if required_items:
                    # 각 섹션을 딕셔너리 형태로 저장 (required_questions 키 사용)
                    structured_sections[section_name] = {
                        'required_questions': required_items,
                        'required_actions': []  # 기본값
                    }
        
        return structured_sections

    def _build_conversation_text(self, conversation_log: List[Dict]) -> str:
        """대화 로그를 텍스트로 변환"""
        conversation_parts = []
        for msg in conversation_log:
            speaker = "의사" if msg.get("role") == "doctor" else "환자"
            content = msg.get("content", "")
            conversation_parts.append(f"{speaker}: {content}")
        return "\n".join(conversation_parts)

    async def _process_evaluation_response(self, prompt: str, area_name: str, structured_sections: dict, stage: str = "") -> dict:
        """평가 응답 처리 공통 함수"""
        try:
            messages = [{"role": "user", "content": prompt}]
            response = await self.llm.ainvoke(messages)
            result_text = response.content
            
            print(f"[{stage}] LLM 응답 원문:\n{result_text[:300]}...")
            
            # 개선된 JSON 추출 및 파싱
            # 1. 코드 블록 내 JSON 찾기 시도
            code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', result_text, re.DOTALL)
            if code_block_match:
                json_str = code_block_match.group(1)
            else:
                # 2. 일반 JSON 패턴 찾기
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', result_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                else:
                    error_msg = f"[{stage}] JSON 형식을 찾을 수 없습니다."
                    print(error_msg)
                    raise ValueError(error_msg)
            
            # JSON 파싱 시도
            try:
                result = json.loads(json_str)
                print(f"[{stage}] JSON 파싱 성공: {len(result)}개 항목")
            except json.JSONDecodeError as json_error:
                error_msg = f"[{stage}] JSON 파싱 실패: {json_error}"
                print(error_msg)
                raise ValueError(error_msg)
            
            # 결과 변환 및 검증
            guideline_evaluations = []
            for section_name in structured_sections.keys():
                section_result = result.get(section_name, {})
                
                # 데이터 타입 검증 및 보정
                found_value = section_result.get("completed", False) or section_result.get("found", False)
                if isinstance(found_value, str):
                    found_value = found_value.lower() in ['true', 'yes', '1', 'found', 'completed']
                elif not isinstance(found_value, bool):
                    found_value = False
                
                evidence_value = section_result.get("evidence", [])
                if not isinstance(evidence_value, list):
                    # 문자열인 경우 배열로 변환
                    if isinstance(evidence_value, str) and evidence_value:
                        evidence_value = [evidence_value]
                    else:
                        evidence_value = []
                
                # completed가 false인 경우 required_action 추가
                required_action = []
                if not found_value:
                    # RAG 결과에서 해당 항목의 required_questions나 required_actions 가져오기
                    for area_name_key, area_data in structured_sections.items():
                        if area_name_key == section_name:
                            # required_questions가 있으면 추가
                            if "required_questions" in area_data:
                                required_action.extend(area_data["required_questions"])
                            # required_actions가 있으면 추가  
                            if "required_actions" in area_data:
                                required_action.extend(area_data["required_actions"])
                            break

                guideline_evaluations.append({
                    "guideline_item": section_name,
                    "completed": found_value,
                    "evidence": evidence_value,
                    "required_action": required_action if not found_value else []
                })
            
            # 통계 계산
            total_guidelines = len(guideline_evaluations)
            completed_guidelines = sum(1 for item in guideline_evaluations if item["completed"])
            completion_rate = completed_guidelines / total_guidelines if total_guidelines > 0 else 0
            
            print(f"[{stage}] 평가 완료: {completed_guidelines}/{total_guidelines} ({completion_rate:.1%})")
            
            return {
                "area_name": area_name,
                "total_guidelines": total_guidelines,
                "completed_guidelines": completed_guidelines,
                "completion_rate": completion_rate,
                "guideline_evaluations": guideline_evaluations
            }
                
        except Exception as e:
            error_msg = f"[{stage}] 평가 실패: {e}"
            print(error_msg)
            logger.error(f"Traceback: {e}", exc_info=True)
            raise RuntimeError(error_msg)

    def _verify_evidence_exists(self, conversation_text: str, evaluation_result: dict) -> dict:
        """evidence array의 각 항목이 실제 대화에 존재하는지 검증"""
        
        verified_evaluations = []
        
        for item in evaluation_result['guideline_evaluations']:
            evidence_array = item['evidence']
            
            if not evidence_array:
                # evidence가 비어있으면 그대로 유지
                verified_evaluations.append(item)
                continue
            
            # evidence array의 각 항목을 개별적으로 검증
            verified_evidence = []
            
            for evidence_item in evidence_array:
                evidence_item = evidence_item.strip()
                if not evidence_item:
                    continue
                
                # "의사:" 또는 "환자:" 접두사 제거하고 실제 content만 추출
                content_to_check = evidence_item
                if evidence_item.startswith('의사: '):
                    content_to_check = evidence_item[4:]  # "의사: " 제거
                elif evidence_item.startswith('환자: '):
                    content_to_check = evidence_item[4:]  # "환자: " 제거
                
                # 실제 대화에서 해당 content가 정확히 존재하는지 확인
                if content_to_check and content_to_check in conversation_text:
                    verified_evidence.append(evidence_item)
                elif content_to_check:
                    print(f"⚠️ 존재하지 않는 발언: {content_to_check[:50]}...")
                else:
                    verified_evidence.append(evidence_item)
            
            # 검증된 evidence가 있으면 completed 유지, 없으면 false로 변경
            has_valid_evidence = len(verified_evidence) > 0
            final_completed = item["completed"] and has_valid_evidence
            
            verified_evaluations.append({
                "guideline_item": item["guideline_item"],
                "completed": final_completed,
                "evidence": verified_evidence,
                "required_action": item.get("required_action", []) if not final_completed else []
            })
            
            if not has_valid_evidence and item["completed"]:
                print(f"❌ {item['guideline_item']}: 잘못된 evidence 제거")
        
        # 통계 재계산
        verified_result = evaluation_result.copy()
        verified_result['guideline_evaluations'] = verified_evaluations
        
        total_guidelines = len(verified_evaluations)
        completed_guidelines = sum(1 for item in verified_evaluations if item["completed"])
        completion_rate = completed_guidelines / total_guidelines if total_guidelines > 0 else 0
        
        verified_result.update({
            "completed_guidelines": completed_guidelines,
            "completion_rate": completion_rate
        })
        
        print(f"[검증 완료] {completed_guidelines}/{total_guidelines} ({completion_rate:.1%})")
        
        return verified_result

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

    async def _evaluate_ser_emotions(self, conversation_log: List[Dict]) -> Dict:
        """3단계: SER 감정 분석 평가 - Kind 높아야 하고 문맥에 맞는 감정인지 LLM이 판단"""
        print(f"🎭 SER 감정 분석 평가 시작")
        
        # SER 감정 분석 결과 수집
        emotion_analysis = self._analyze_conversation_emotions(conversation_log)
        
        if emotion_analysis["total_analyzed_utterances"] == 0:
            return {
                "ser_score": 5,  # 중간 점수
                "emotion_analysis": "감정 분석 데이터가 없습니다.",
                "ser_strengths": [],
                "ser_improvements": ["음성 감정 분석 데이터 확보 필요"],
                "detailed_feedback": "SER 감정 분석을 수행할 수 있는 데이터가 부족합니다."
            }
        
        # 대화 텍스트와 감정 데이터 구성
        conversation_text = self._build_conversation_text(conversation_log)
        emotion_summary = self._generate_emotion_summary(emotion_analysis)
        
        # 각 의사 발화별 감정 분석 결과 상세 정보
        emotion_details = ""
        for emotion_data in emotion_analysis["doctor_emotions"]:
            emotion_details += f"발화: \"{emotion_data['text']}\"\n"
            emotion_details += f"  → 감정: {emotion_data['predicted_emotion']} (신뢰도: {emotion_data['confidence']:.2f})\n"
            emotion_details += f"  → 감정 점수: {emotion_data['emotion_scores']}\n\n"
        
        ser_prompt = f"""
당신은 의료 커뮤니케이션 전문가입니다. 의사의 음성 감정 분석 결과를 평가하세요.

【전체 대화】:
{conversation_text}

【감정 분석 통계】:
{emotion_summary}

【발화별 상세 감정 분석】:
{emotion_details}

다음 기준으로 의사의 감정적 적절성을 평가하세요:

【평가 기준】:
1. **Kind(친절함) 비율**: 의료진은 기본적으로 친절해야 함 (높을수록 좋음)
2. **상황별 감정 적절성**: 
   - 환자가 걱정을 표현할 때 → Anxious(공감)가 적절할 수 있음
   - 정보 전달/설명 시 → Kind(친절함)가 적절
   - 진찰/검사 시 → 약간의 Dry(건조함)는 전문성으로 볼 수 있음
3. **감정 일관성**: 급격한 감정 변화가 있는지
4. **환자 상황 고려**: 환자의 상태/말에 맞는 감정인지

【점수 기준】:
- Kind 비율이 70% 이상: 8-10점
- Kind 비율이 50-70%: 6-8점  
- Kind 비율이 30-50%: 4-6점
- Kind 비율이 30% 미만: 1-4점
- 단, 상황에 맞는 Anxious나 적절한 Dry는 가점 요소

1-10점으로 평가하고 구체적인 피드백을 제공하세요.

JSON 응답:
{{
    "ser_score": 점수(1-10),
    "kind_ratio_assessment": "Kind 비율에 대한 평가",
    "contextual_appropriateness": "상황별 감정 적절성 평가", 
    "emotional_consistency": "감정 일관성 평가",
    "patient_consideration": "환자 상황 고려도 평가",
    "ser_strengths": ["감정적으로 우수한 점들"],
    "ser_improvements": ["감정적으로 개선이 필요한 점들"],
    "detailed_feedback": "종합적인 감정 평가 피드백"
}}"""

        try:
            messages = [SystemMessage(content=ser_prompt)]
            response = await self.llm.ainvoke(messages)
            result_text = response.content
            
            print(f"[SER] LLM 응답 원문:\n{result_text[:300]}...")
            
            # JSON 파싱
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                ser_result = json.loads(json_str)
                print(f"[SER] JSON 파싱 성공")
            else:
                print(f"[SER] JSON 형식을 찾을 수 없습니다.")
                ser_result = {
                    "ser_score": 6,
                    "kind_ratio_assessment": "Kind 비율 분석 실패",
                    "contextual_appropriateness": "상황별 적절성 분석 실패",
                    "emotional_consistency": "감정 일관성 분석 실패", 
                    "patient_consideration": "환자 고려도 분석 실패",
                    "ser_strengths": ["기본 감정 분석 완료"],
                    "ser_improvements": ["SER 평가 개선 필요"],
                    "detailed_feedback": "SER 분석에서 오류가 발생했습니다."
                }
            
            # 감정 통계 데이터 추가
            ser_result["emotion_statistics"] = emotion_analysis["emotion_statistics"]
            ser_result["total_analyzed_utterances"] = emotion_analysis["total_analyzed_utterances"]
            
            print(f"✅ SER 감정 분석 평가 완료 - 점수: {ser_result.get('ser_score', 6):.1f}점")
            
            return ser_result
            
        except Exception as e:
            print(f"❌ SER 감정 분석 평가 실패: {e}")
            return {
                "ser_score": 5,
                "kind_ratio_assessment": "평가 오류",
                "contextual_appropriateness": "평가 오류",
                "emotional_consistency": "평가 오류",
                "patient_consideration": "환자 상황 고려도 평가", 
                "ser_strengths": ["기본 평가 완료"],
                "ser_improvements": ["SER 평가 오류로 기본값 사용"],
                "detailed_feedback": f"SER 평가 중 오류 발생: {str(e)}",
                "emotion_statistics": emotion_analysis.get("emotion_statistics", {}),
                "total_analyzed_utterances": emotion_analysis.get("total_analyzed_utterances", 0)
            }

    def _analyze_conversation_emotions(self, conversation_log: List[Dict]) -> Dict:
        """대화에서 의사의 감정 분석 결과 수집 및 분석"""
        doctor_emotions = []
        emotion_stats = {"Kind": 0, "Anxious": 0, "Dry": 0, "total_utterances": 0}
        
        for msg in conversation_log:
            if msg.get("role") == "doctor" and msg.get("emotion"):
                emotion_data = msg.get("emotion")
                predicted_emotion = emotion_data.get("predicted_emotion")
                confidence = emotion_data.get("confidence", 0)
                emotion_scores = emotion_data.get("emotion_scores", {})
                
                doctor_emotions.append({
                    "text": msg.get("content", ""),
                    "predicted_emotion": predicted_emotion,
                    "confidence": confidence,
                    "emotion_scores": emotion_scores
                })
                
                # 통계 수집
                if predicted_emotion in emotion_stats:
                    emotion_stats[predicted_emotion] += 1
                emotion_stats["total_utterances"] += 1
        
        # 전체 감정 비율 계산
        if emotion_stats["total_utterances"] > 0:
            for emotion in ["Kind", "Anxious", "Dry"]:
                emotion_stats[f"{emotion}_ratio"] = emotion_stats[emotion] / emotion_stats["total_utterances"]
        
        return {
            "doctor_emotions": doctor_emotions,
            "emotion_statistics": emotion_stats,
            "total_analyzed_utterances": len(doctor_emotions)
        }

    def _generate_emotion_summary(self, emotion_analysis: Dict) -> str:
        """감정 분석 결과를 요약 텍스트로 생성"""
        stats = emotion_analysis.get("emotion_statistics", {})
        doctor_emotions = emotion_analysis.get("doctor_emotions", [])
        
        if not doctor_emotions:
            return "감정 분석 데이터가 없습니다."
        
        total = stats.get("total_utterances", 0)
        if total == 0:
            return "분석 가능한 의사 발화가 없습니다."
        
        # 감정별 비율
        kind_ratio = stats.get("Kind_ratio", 0)
        anxious_ratio = stats.get("Anxious_ratio", 0) 
        dry_ratio = stats.get("Dry_ratio", 0)
        
        # 주요 감정 파악
        dominant_emotion = "Kind"
        if anxious_ratio > kind_ratio and anxious_ratio > dry_ratio:
            dominant_emotion = "Anxious"
        elif dry_ratio > kind_ratio and dry_ratio > anxious_ratio:
            dominant_emotion = "Dry"
        
        # 평균 신뢰도 계산
        avg_confidence = sum(e.get("confidence", 0) for e in doctor_emotions) / len(doctor_emotions)
        
        summary = f"""총 {total}개 발화 분석:
- Kind (친절함): {kind_ratio:.1%}
- Anxious (불안함): {anxious_ratio:.1%} 
- Dry (건조함): {dry_ratio:.1%}
- 주요 감정: {dominant_emotion}
- 평균 신뢰도: {avg_confidence:.2f}

감정별 발화 예시:"""
        
        # 각 감정별 대표 발화 1개씩 추가
        for emotion in ["Kind", "Anxious", "Dry"]:
            emotion_examples = [e for e in doctor_emotions if e.get("predicted_emotion") == emotion]
            if emotion_examples:
                best_example = max(emotion_examples, key=lambda x: x.get("confidence", 0))
                summary += f"\n- {emotion}: \"{best_example.get('text', '')[:30]}...\" (신뢰도: {best_example.get('confidence', 0):.2f})"
        
        return summary

    def generate_evaluation_markdown(self, evaluation_result: Dict) -> str:
        """평가 결과를 마크다운 형식으로 생성"""
        try:
            langgraph_analysis = evaluation_result.get("langgraph_text_analysis", {})
            scores = langgraph_analysis.get("scores", {})
            feedback = langgraph_analysis.get("feedback", {})
            detailed_analysis = langgraph_analysis.get("detailed_analysis", {})
            conversation_summary = langgraph_analysis.get("conversation_summary", {})
            
            # 기본 정보
            user_id = evaluation_result.get("user_id", "Unknown")
            scenario_id = evaluation_result.get("scenario_id", "Unknown")
            duration = evaluation_result.get("duration_minutes", 0)
            
            markdown_content = f"""# CPX 실습 평가 결과

## 1. 종합 점수
- **총점**: {scores.get('total_score', 0):.1f}점 / 100점 ({scores.get('grade', 'F')}등급)
- **필수항목 달성률**: {scores.get('completion_rate', 0):.1%}
- **진료 수행도**: {scores.get('quality_score', 0):.1f}점 / 10점

## 2. 각 단계별 결과

### 필수항목 달성 평가
"""
            
            # 완성도 평가 상세 내용
            completeness = detailed_analysis.get("completeness", {})
            areas_evaluation = completeness.get("areas_evaluation", {})
            
            for area_key, area_data in areas_evaluation.items():
                area_name = area_data.get("area_name", area_key)
                completion_rate = area_data.get("completion_rate", 0)
                completed_guidelines = area_data.get("completed_guidelines", 0)
                total_guidelines = area_data.get("total_guidelines", 0)
                
                # 완성도 상태
                if completion_rate >= 0.8:
                    status = "우수"
                elif completion_rate >= 0.5:
                    status = "보통"
                else:
                    status = "미흡"
                
                markdown_content += f"- **{area_name}**: {completion_rate:.1%} ({status})\n"

            # 의사소통 평가 상세 내용
            quality = detailed_analysis.get("quality", {})
            markdown_content += f"""
### 진료 수행 평가
- **의학적 정확성**: {quality.get('medical_accuracy', 0):.1f}/10점
- **의사소통 능력**: {quality.get('communication_efficiency', 0):.1f}/10점
- **전문성**: {quality.get('professionalism', 0):.1f}/10점
- **시나리오 적합성**: {quality.get('scenario_appropriateness', 0):.1f}/10점
"""

            # SER 감정 분석 결과 추가
            ser_evaluation = detailed_analysis.get("comprehensive", {}).get("ser_evaluation", {})
            if ser_evaluation:
                markdown_content += f"""
### 감정 평가
- ** 감정 점수**: {ser_evaluation.get('ser_score', 0):.1f}/10점

**상세 피드백**: {ser_evaluation.get('detailed_feedback', '음성 톤 분석 결과가 없습니다.')}
"""

            # 종합 결과 및 권장사항
            strengths = feedback.get("strengths", [])
            improvements = feedback.get("improvements", [])
            
            markdown_content += "\n## 3. 종합 결과 및 권장사항\n"
            
            if strengths:
                markdown_content += "\n### 우수한 점\n"
                for strength in strengths[:3]:  # 최대 3개만
                    markdown_content += f"- {strength}\n"
            
            if improvements:
                markdown_content += "\n### 개선 필요사항\n"
                for improvement in improvements[:3]:  # 최대 3개만
                    markdown_content += f"- {improvement}\n"
            
            # 학습 권장사항 (간단하게)
            markdown_content += f"""
### 학습 권장사항
1. 완성도가 낮은 영역({', '.join([area for area, data in areas_evaluation.items() if data.get('completion_rate', 0) < 0.5])}) 집중 학습
2. 질문 기법 및 의사소통 스킬 향상
3. 시나리오별 반복 연습
"""
            
            return markdown_content
            
        except Exception as e:
            print(f"❌ 마크다운 생성 실패: {e}")
            return f"# CPX 평가 결과\n\n평가 결과를 불러오는 중 오류가 발생했습니다.\n\n오류 내용: {str(e)}"

    # ================================
    # 6. 데이터베이스 및 파일 관리
    # ================================
    
    async def _update_cpx_database_after_evaluation(self, session_id: str, evaluation_result: dict):
        """평가 완료 후 CPX Details만 업데이트"""
        try:
            session = self.session_data[session_id]
            result_id = session.get("result_id")
            user_id = session["user_id"]
            
            if result_id is None:
                print(f"❌ [{session_id}] result_id가 None입니다. CPX 결과 생성에 실패했을 가능성이 있습니다.")
                print(f"❌ [{session_id}] 데이터베이스 업데이트를 건너뜁니다. JSON 파일만 저장됩니다.")
                return
            
            # CPX Details 및 Evaluations 업데이트
            db_gen = get_db()
            db = await db_gen.__anext__()
            try:
                cpx_service = CpxService(db)
                
                # CPX Details 업데이트 (시스템 평가 데이터)
                await cpx_service.update_cpx_details(
                    result_id=result_id,
                    user_id=int(user_id),
                    system_evaluation_data=evaluation_result
                )
                
                # CPX Results 상태 업데이트 (자동 평가 완료)
                await cpx_service.update_cpx_result_status(
                    result_id=result_id,
                    new_status="완료"
                )
                
                print(f"✅ CPX Details 및 Evaluations 업데이트 완료: result_id={result_id}, session_id={session_id}")
                
            finally:
                await db_gen.aclose()
                
        except Exception as e:
            print(f"❌ CPX Details 업데이트 실패: {e}")
            import traceback
            traceback.print_exc()

