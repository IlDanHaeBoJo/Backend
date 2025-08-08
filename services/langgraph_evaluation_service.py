import os
import json
from typing import Dict, List, Optional, TypedDict, Annotated
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage as AnyMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

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

class LangGraphEvaluationService:
    def __init__(self):
        """LangGraph 기반 CPX 평가 서비스 초기화"""
        # OpenAI API 설정
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 필요합니다")
            
        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name="gpt-3.5-turbo",
            temperature=0.3,
            max_tokens=2000
        )
        
        # 병력청취 체크리스트 (기존과 동일)
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
        
        # 워크플로우 생성
        self.workflow = self._create_evaluation_workflow()

    def _create_evaluation_workflow(self):
        """CPX 평가 워크플로우 생성"""
        
        # StateGraph 생성 (LangGraph 0.6.3 방식)
        workflow = StateGraph(CPXEvaluationState)
        
        # 노드 추가
        workflow.add_node("initialize", self._initialize_evaluation)
        workflow.add_node("analyze_conversation", self._analyze_conversation)
        workflow.add_node("evaluate_checklist", self._evaluate_checklist)
        workflow.add_node("evaluate_questions", self._evaluate_questions)
        workflow.add_node("evaluate_empathy", self._evaluate_empathy)
        workflow.add_node("quality_check", self._quality_check)
        workflow.add_node("enhanced_evaluation", self._enhanced_evaluation)
        workflow.add_node("calculate_scores", self._calculate_final_scores)
        workflow.add_node("generate_feedback", self._generate_feedback)
        workflow.add_node("finalize_results", self._finalize_results)
        
        # 엔트리 포인트 설정 (LangGraph 0.6.3 방식)
        workflow.set_entry_point("initialize")
        
        # 순차 실행 엣지
        workflow.add_edge("initialize", "analyze_conversation")
        workflow.add_edge("analyze_conversation", "evaluate_checklist")
        workflow.add_edge("evaluate_checklist", "evaluate_questions")
        workflow.add_edge("evaluate_questions", "evaluate_empathy")
        workflow.add_edge("evaluate_empathy", "quality_check")
        
        # 조건부 라우팅 (LangGraph 0.6.3 방식)
        workflow.add_conditional_edges(
            "quality_check",
            self._should_enhance_evaluation,
            {
                "enhance": "enhanced_evaluation",
                "calculate": "calculate_scores"
            }
        )
        
        workflow.add_edge("enhanced_evaluation", "calculate_scores")
        workflow.add_edge("calculate_scores", "generate_feedback")
        workflow.add_edge("generate_feedback", "finalize_results")
        workflow.add_edge("finalize_results", END)
        
        return workflow.compile()

    # =============================================================================
    # 워크플로우 노드들 (각각 단일 책임)
    # =============================================================================
    
    def _initialize_evaluation(self, state: CPXEvaluationState) -> CPXEvaluationState:
        """1단계: 평가 초기화"""
        print(f"🎯 [{state['user_id']}] CPX 평가 초기화 - 시나리오: {state['scenario_id']}")
        
        # 메타데이터 생성
        metadata = {
            "user_id": state["user_id"],
            "scenario_id": state["scenario_id"],
            "evaluation_date": datetime.now().isoformat(),
            "total_interactions": len(state["conversation_log"]),
            "conversation_duration_minutes": len(state["conversation_log"]) * 0.5,
            "voice_recording_path": "s3로 저장",
            "conversation_transcript": json.dumps(state["conversation_log"], ensure_ascii=False)
        }
        
        return {
            **state,
            "evaluation_metadata": metadata,
            "confidence_score": 0.0,
            "retry_count": 0,
            "needs_enhancement": False,
            "messages": [HumanMessage(content="CPX 평가를 시작합니다.")]
        }

    def _analyze_conversation(self, state: CPXEvaluationState) -> CPXEvaluationState:
        """2단계: 대화 기본 분석"""
        print(f"📊 [{state['user_id']}] 대화 분석 시작")
        
        conversation_log = state["conversation_log"]
        
        if not conversation_log:
            return {
                **state,
                "conversation_analysis": {"total_questions": 0, "duration_minutes": 0, "question_types": {}},
                "messages": state["messages"] + [HumanMessage(content="대화 로그가 비어있습니다.")]
            }
        
        total_questions = len([msg for msg in conversation_log if msg.get("type") == "student"])
        
        # 질문 유형 분석
        open_questions = 0
        closed_questions = 0
        
        for msg in conversation_log:
            if msg.get("type") == "student":
                question = msg.get("content", "")
                if any(word in question for word in ["어떤", "어떻게", "왜", "언제", "어디"]):
                    open_questions += 1
                elif any(word in question for word in ["있나요", "없나요", "맞나요"]):
                    closed_questions += 1
        
        analysis = {
            "total_questions": total_questions,
            "duration_minutes": len(conversation_log) * 0.5,
            "question_types": {
                "open_questions": open_questions,
                "closed_questions": closed_questions,
                "open_ratio": open_questions / max(total_questions, 1)
            }
        }
        
        print(f"✅ [{state['user_id']}] 대화 분석 완료 - 총 질문: {total_questions}")
        
        return {
            **state,
            "conversation_analysis": analysis,
            "messages": state["messages"] + [HumanMessage(content=f"대화 분석 완료: 총 {total_questions}개 질문")]
        }

    def _evaluate_checklist(self, state: CPXEvaluationState) -> CPXEvaluationState:
        """3단계: 병력청취 체크리스트 평가"""
        print(f"📋 [{state['user_id']}] 체크리스트 평가 시작")
        
        conversation_text = self._build_conversation_text(state["conversation_log"])
        checklist_results = {}
        total_confidence = 0.0
        
        # 각 카테고리별 평가 (동기적으로 실행)
        for category_id, category_info in self.history_taking_checklist.items():
            try:
                result = self._evaluate_checklist_category_sync(
                    conversation_text, 
                    category_info, 
                    state["scenario_id"]
                )
                checklist_results[category_id] = result
                total_confidence += result.get("completion_rate", 0) * result.get("quality_score", 5) / 10
                
            except Exception as e:
                print(f"❌ [{state['user_id']}] 체크리스트 카테고리 {category_id} 평가 오류: {e}")
                checklist_results[category_id] = {
                    "completed_elements": [],
                    "missing_elements": category_info['required_elements'],
                    "completion_rate": 0.0,
                    "quality_score": 5,
                    "specific_feedback": f"평가 오류: {str(e)}"
                }
        
        # 전체 신뢰도 계산
        confidence_score = total_confidence / len(self.history_taking_checklist)
        
        print(f"✅ [{state['user_id']}] 체크리스트 평가 완료 - 신뢰도: {confidence_score:.2f}")
        
        return {
            **state,
            "checklist_results": checklist_results,
            "confidence_score": confidence_score,
            "messages": state["messages"] + [HumanMessage(content=f"체크리스트 평가 완료 - 신뢰도: {confidence_score:.2f}")]
        }

    def _evaluate_questions(self, state: CPXEvaluationState) -> CPXEvaluationState:
        """4단계: 질문 기법 평가"""
        print(f"❓ [{state['user_id']}] 질문 기법 평가 시작")
        
        conversation_log = state["conversation_log"]
        student_questions = [msg for msg in conversation_log if msg.get("type") == "student"]
        
        if not student_questions:
            return {
                **state,
                "question_analysis": {
                    "total_questions": 0,
                    "positive_techniques": 0,
                    "negative_techniques": 0,
                    "technique_analysis": {}
                },
                "messages": state["messages"] + [HumanMessage(content="학생 질문이 없습니다.")]
            }
        
        positive_count = 0
        negative_count = 0
        technique_details = {}
        
        for i, question_msg in enumerate(student_questions):
            question = question_msg.get("content", "")
            
            # 긍정적 기법 체크
            if self._is_open_question(question):
                positive_count += 1
                technique_details[f"개방형_질문_{i}"] = {
                    "type": "positive",
                    "technique": "개방형 질문 사용",
                    "example": question[:50] + "..."
                }
            
            if self._is_empathetic_question(question):
                positive_count += 1
                technique_details[f"공감적_질문_{i}"] = {
                    "type": "positive", 
                    "technique": "공감적 표현 사용",
                    "example": question[:50] + "..."
                }
            
            # 부정적 기법 체크
            if self._is_leading_question(question):
                negative_count += 1
                technique_details[f"유도_질문_{i}"] = {
                    "type": "negative",
                    "technique": "유도 질문 사용 (개선 필요)",
                    "example": question[:50] + "..."
                }
        
        technique_score = max(1, min(10, (positive_count * 2 - negative_count) + 5))
        
        analysis = {
            "total_questions": len(student_questions),
            "positive_techniques": positive_count,
            "negative_techniques": negative_count,
            "technique_score": technique_score,
            "technique_analysis": technique_details
        }
        
        print(f"✅ [{state['user_id']}] 질문 기법 평가 완료 - 점수: {technique_score}")
        
        return {
            **state,
            "question_analysis": analysis,
            "messages": state["messages"] + [HumanMessage(content=f"질문 기법 평가 완료 - 점수: {technique_score}")]
        }

    def _evaluate_empathy(self, state: CPXEvaluationState) -> CPXEvaluationState:
        """5단계: 공감 능력 평가 (새로운 평가 기준)"""
        print(f"💝 [{state['user_id']}] 공감 능력 평가 시작")
        
        conversation_log = state["conversation_log"]
        student_questions = [msg for msg in conversation_log if msg.get("type") == "student"]
        
        empathy_score = 5.0  # 기본 점수
        empathy_indicators = []
        
        for question_msg in student_questions:
            question = question_msg.get("content", "")
            
            # 공감적 표현 체크
            empathy_keywords = ["힘드시겠어요", "걱정", "이해", "불편", "괜찮으시", "편하게", "천천히"]
            if any(keyword in question for keyword in empathy_keywords):
                empathy_score += 1.0
                empathy_indicators.append(f"공감적 표현: '{question[:30]}...'")
            
            # 환자 중심적 표현 체크
            patient_centered = ["환자분", "어떠세요", "말씀해주세요", "설명해주시겠어요"]
            if any(phrase in question for phrase in patient_centered):
                empathy_score += 0.5
                empathy_indicators.append(f"환자 중심적: '{question[:30]}...'")
        
        empathy_score = min(10.0, empathy_score)
        
        analysis = {
            "empathy_score": empathy_score,
            "empathy_indicators": empathy_indicators,
            "total_empathetic_expressions": len(empathy_indicators)
        }
        
        print(f"✅ [{state['user_id']}] 공감 능력 평가 완료 - 점수: {empathy_score}")
        
        return {
            **state,
            "empathy_analysis": analysis,
            "messages": state["messages"] + [HumanMessage(content=f"공감 능력 평가 완료 - 점수: {empathy_score}")]
        }

    def _quality_check(self, state: CPXEvaluationState) -> CPXEvaluationState:
        """6단계: 평가 품질 체크"""
        print(f"🔍 [{state['user_id']}] 평가 품질 체크")
        
        confidence_score = state.get("confidence_score", 0.0)
        checklist_results = state.get("checklist_results", {})
        
        # 품질 체크 기준
        needs_enhancement = False
        quality_issues = []
        
        # 신뢰도 체크
        if confidence_score < 0.6:
            needs_enhancement = True
            quality_issues.append(f"낮은 신뢰도: {confidence_score:.2f}")
        
        # 완료율 체크
        low_completion_categories = []
        for category_id, result in checklist_results.items():
            if result.get("completion_rate", 0) < 0.3:
                low_completion_categories.append(category_id)
        
        if len(low_completion_categories) > 3:
            needs_enhancement = True
            quality_issues.append(f"낮은 완료율 카테고리 {len(low_completion_categories)}개")
        
        # 재시도 횟수 체크
        if state.get("retry_count", 0) >= 2:
            needs_enhancement = False  # 더 이상 재시도하지 않음
            quality_issues.append("최대 재시도 횟수 도달")
        
        print(f"🔍 [{state['user_id']}] 품질 체크 완료 - 향상 필요: {needs_enhancement}")
        
        return {
            **state,
            "needs_enhancement": needs_enhancement,
            "quality_issues": quality_issues,
            "messages": state["messages"] + [HumanMessage(content=f"품질 체크 완료 - 이슈: {len(quality_issues)}개")]
        }

    def _enhanced_evaluation(self, state: CPXEvaluationState) -> CPXEvaluationState:
        """7단계: 향상된 평가 (필요시)"""
        print(f"🚀 [{state['user_id']}] 향상된 평가 시작 (재시도: {state.get('retry_count', 0) + 1})")
        
        # 더 상세한 분석을 위한 향상된 프롬프트 사용
        conversation_text = self._build_conversation_text(state["conversation_log"])
        enhanced_results = {}
        
        # 기존 결과에서 낮은 점수를 받은 카테고리만 재평가
        checklist_results = state.get("checklist_results", {})
        
        for category_id, category_info in self.history_taking_checklist.items():
            existing_result = checklist_results.get(category_id, {})
            
            # 완료율이 낮은 카테고리만 재평가
            if existing_result.get("completion_rate", 0) < 0.5:
                try:
                    enhanced_result = self._evaluate_checklist_category_enhanced(
                        conversation_text, 
                        category_info, 
                        state["scenario_id"]
                    )
                    enhanced_results[category_id] = enhanced_result
                    print(f"🔄 [{state['user_id']}] {category_id} 재평가 완료")
                    
                except Exception as e:
                    print(f"❌ [{state['user_id']}] {category_id} 재평가 오류: {e}")
                    enhanced_results[category_id] = existing_result
            else:
                enhanced_results[category_id] = existing_result
        
        # 신뢰도 재계산
        total_confidence = 0.0
        for result in enhanced_results.values():
            total_confidence += result.get("completion_rate", 0) * result.get("quality_score", 5) / 10
        
        new_confidence = total_confidence / len(enhanced_results)
        
        print(f"✅ [{state['user_id']}] 향상된 평가 완료 - 신뢰도: {new_confidence:.2f}")
        
        return {
            **state,
            "checklist_results": enhanced_results,
            "confidence_score": new_confidence,
            "retry_count": state.get("retry_count", 0) + 1,
            "messages": state["messages"] + [HumanMessage(content=f"향상된 평가 완료 - 신뢰도: {new_confidence:.2f}")]
        }

    def _calculate_final_scores(self, state: CPXEvaluationState) -> CPXEvaluationState:
        """8단계: 최종 점수 계산"""
        print(f"🧮 [{state['user_id']}] 최종 점수 계산 시작")
        
        checklist_results = state.get("checklist_results", {})
        question_analysis = state.get("question_analysis", {})
        empathy_analysis = state.get("empathy_analysis", {})
        conversation_analysis = state.get("conversation_analysis", {})
        
        # 1. 체크리스트 기반 점수 (60%)
        checklist_score = 0
        total_weight = 0
        
        for category_id, result in checklist_results.items():
            category_weight = self.history_taking_checklist[category_id]["weight"]
            category_score = result.get("completion_rate", 0) * result.get("quality_score", 5)
            checklist_score += category_score * category_weight
            total_weight += category_weight
        
        if total_weight > 0:
            checklist_score = checklist_score / total_weight
        
        # 2. 질문 기법 점수 (25%)
        technique_score = question_analysis.get("technique_score", 5)
        
        # 3. 공감 능력 점수 (10%)
        empathy_score = empathy_analysis.get("empathy_score", 5)
        
        # 4. 전반적 의사소통 점수 (5%)
        communication_score = min(10, max(1, conversation_analysis.get("question_types", {}).get("open_ratio", 0.3) * 10))
        
        # 최종 점수 (100점 만점)
        final_score = (
            checklist_score * 0.6 + 
            technique_score * 0.25 + 
            empathy_score * 0.1 +
            communication_score * 0.05
        ) * 10
        
        scores = {
            "total_score": round(final_score, 1),
            "checklist_score": round(checklist_score * 10, 1),
            "technique_score": round(technique_score, 1),
            "empathy_score": round(empathy_score, 1),
            "communication_score": round(communication_score, 1),
            "grade": self._calculate_grade(final_score)
        }
        
        print(f"✅ [{state['user_id']}] 최종 점수 계산 완료 - 총점: {final_score:.1f}")
        
        return {
            **state,
            "final_scores": scores,
            "messages": state["messages"] + [HumanMessage(content=f"최종 점수 계산 완료 - 총점: {final_score:.1f}점")]
        }

    def _generate_feedback(self, state: CPXEvaluationState) -> CPXEvaluationState:
        """9단계: 상세 피드백 생성"""
        print(f"📝 [{state['user_id']}] 피드백 생성 시작")
        
        checklist_results = state.get("checklist_results", {})
        question_analysis = state.get("question_analysis", {})
        empathy_analysis = state.get("empathy_analysis", {})
        
        # 강점 분석
        strengths = []
        for category_id, result in checklist_results.items():
            if result.get("completion_rate", 0) > 0.7:
                category_name = self.history_taking_checklist[category_id]["name"]
                strengths.append(f"{category_name} 영역을 잘 수행했습니다")
        
        if question_analysis.get("positive_techniques", 0) > question_analysis.get("negative_techniques", 0):
            strengths.append("적절한 질문 기법을 사용했습니다")
            
        if empathy_analysis.get("empathy_score", 0) > 7:
            strengths.append("환자에 대한 공감적 태도를 보였습니다")
        
        # 개선점 분석  
        improvements = []
        for category_id, result in checklist_results.items():
            if result.get("completion_rate", 0) < 0.5:
                category_name = self.history_taking_checklist[category_id]["name"]
                improvements.append(f"{category_name} 영역의 필수 요소들을 더 확인하세요")
        
        if question_analysis.get("negative_techniques", 0) > 0:
            improvements.append("유도 질문보다는 개방형 질문을 사용하세요")
            
        if empathy_analysis.get("empathy_score", 0) < 6:
            improvements.append("환자의 감정에 더 공감하는 표현을 사용하세요")
        
        feedback = {
            "overall_feedback": f"전반적인 병력청취 수행도에 대한 평가입니다. 총점: {state.get('final_scores', {}).get('total_score', 0)}점",
            "strengths": strengths[:3],  # 상위 3개
            "improvements": improvements[:3],  # 상위 3개  
            "specific_recommendations": [
                "OPQRST 구조를 활용한 체계적 병력청취를 연습하세요",
                "환자의 감정에 공감하는 표현을 더 사용하세요",
                "의학용어보다는 환자가 이해하기 쉬운 표현을 사용하세요"
            ]
        }
        
        print(f"✅ [{state['user_id']}] 피드백 생성 완료")
        
        return {
            **state,
            "feedback": feedback,
            "messages": state["messages"] + [HumanMessage(content="상세 피드백 생성 완료")]
        }

    def _finalize_results(self, state: CPXEvaluationState) -> CPXEvaluationState:
        """10단계: 결과 최종화"""
        print(f"🎯 [{state['user_id']}] 평가 결과 최종화")
        
        # 최종 결과 구조화 (기존과 동일한 형식)
        final_result = {
            "evaluation_metadata": state.get("evaluation_metadata", {}),
            "scores": state.get("final_scores", {}),
            "checklist_results": state.get("checklist_results", {}),
            "question_analysis": state.get("question_analysis", {}),
            "empathy_analysis": state.get("empathy_analysis", {}),  # 새 항목
            "feedback": state.get("feedback", {}),
            "conversation_summary": state.get("conversation_analysis", {}),
            "quality_info": {
                "confidence_score": state.get("confidence_score", 0),
                "retry_count": state.get("retry_count", 0),
                "quality_issues": state.get("quality_issues", [])
            }
        }
        
        print(f"🎉 [{state['user_id']}] CPX 평가 완료 - 총점: {final_result['scores'].get('total_score', 0)}점")
        
        return {
            **state,
            "final_evaluation_result": final_result,
            "messages": state["messages"] + [HumanMessage(content="CPX 평가가 성공적으로 완료되었습니다.")]
        }

    # =============================================================================
    # 조건부 라우팅 함수들
    # =============================================================================
    
    def _should_enhance_evaluation(self, state: CPXEvaluationState) -> str:
        """평가 향상이 필요한지 결정"""
        needs_enhancement = state.get("needs_enhancement", False)
        retry_count = state.get("retry_count", 0)
        
        if needs_enhancement and retry_count < 2:
            return "enhance"
        else:
            return "calculate"

    # =============================================================================
    # 헬퍼 메서드들 (기존과 동일)
    # =============================================================================
    
    def _build_conversation_text(self, conversation_log: List[Dict]) -> str:
        """대화 로그를 텍스트로 변환"""
        conversation_parts = []
        for msg in conversation_log:
            speaker = "학생" if msg.get("type") == "student" else "환자"
            content = msg.get("content", "")
            conversation_parts.append(f"{speaker}: {content}")
        
        return "\n".join(conversation_parts)

    def _evaluate_checklist_category_sync(self, conversation_text: str, category_info: Dict, scenario_id: str) -> Dict:
        """체크리스트 카테고리별 평가 (동기 버전)"""
        
        evaluation_prompt = f"""
당신은 의과대학 CPX 평가 전문가입니다. 
다음 병력청취 대화에서 "{category_info['name']}" 카테고리의 필수 요소들이 얼마나 잘 수행되었는지 평가하세요.

【평가 카테고리】: {category_info['name']}

【필수 요소들】:
{chr(10).join([f"- {element}" for element in category_info['required_elements']])}

【대화 내용】:
{conversation_text}

【평가 기준】:
- 각 필수 요소가 대화에 포함되었는지 확인
- 자연스럽고 적절한 방식으로 수행되었는지 평가
- 환자와의 소통이 효과적이었는지 판단

다음 JSON 형식으로 응답하세요:
{{
    "completed_elements": ["완료된 필수 요소들"],
    "missing_elements": ["누락된 필수 요소들"],
    "completion_rate": 완료율(0.0-1.0),
    "quality_score": 수행 품질 점수(1-10),
    "specific_feedback": "구체적인 피드백"
}}
"""

        try:
            messages = [
                SystemMessage(content="당신은 의과대학 CPX 평가 전문가입니다."),
                HumanMessage(content=evaluation_prompt)
            ]
            
            response = self.llm(messages)
            result_text = response.content.strip()
            
            # JSON 파싱 시도
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # 파싱 실패 시 기본값
                return {
                    "completed_elements": [],
                    "missing_elements": category_info['required_elements'],
                    "completion_rate": 0.0,
                    "quality_score": 5,
                    "specific_feedback": "평가 시스템 오류로 상세 분석이 불가합니다."
                }
                
        except Exception as e:
            print(f"❌ 체크리스트 평가 오류: {e}")
            return {
                "completed_elements": [],
                "missing_elements": category_info['required_elements'],
                "completion_rate": 0.0,
                "quality_score": 5,
                "specific_feedback": f"평가 오류: {str(e)}"
            }

    def _evaluate_checklist_category_enhanced(self, conversation_text: str, category_info: Dict, scenario_id: str) -> Dict:
        """향상된 체크리스트 평가 (더 상세한 분석)"""
        
        enhanced_prompt = f"""
당신은 의과대학 CPX 평가 전문가입니다. 
다음 병력청취 대화에서 "{category_info['name']}" 카테고리를 매우 상세히 분석해주세요.

【평가 카테고리】: {category_info['name']}

【필수 요소들】:
{chr(10).join([f"- {element}" for element in category_info['required_elements']])}

【대화 내용】:
{conversation_text}

【향상된 평가 기준】:
- 각 필수 요소의 직접적/간접적 포함 여부 모두 고려
- 의도는 있었으나 표현이 부족한 경우도 부분 점수 부여
- 환자의 반응을 통해 유추할 수 있는 내용도 고려
- 전체적인 맥락에서 해당 카테고리의 목적 달성도 평가

더욱 정확하고 관대한 평가를 해주세요:
{{
    "completed_elements": ["완료된 필수 요소들"],
    "partially_completed": ["부분적으로 완료된 요소들"],
    "missing_elements": ["완전히 누락된 필수 요소들"],
    "completion_rate": 완료율(0.0-1.0),
    "quality_score": 수행 품질 점수(1-10),
    "specific_feedback": "상세한 피드백과 근거"
}}
"""

        try:
            messages = [
                SystemMessage(content="당신은 경험 많은 의과대학 CPX 평가 전문가입니다. 학생의 노력을 인정하되 정확한 평가를 해주세요."),
                HumanMessage(content=enhanced_prompt)
            ]
            
            response = self.llm(messages)
            result_text = response.content.strip()
            
            # JSON 파싱 시도
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                # 부분 완료된 요소들을 완료된 요소에 0.5 가중치로 추가
                partial_count = len(result.get("partially_completed", []))
                complete_count = len(result.get("completed_elements", []))
                total_elements = len(category_info['required_elements'])
                
                enhanced_completion_rate = min(1.0, (complete_count + partial_count * 0.5) / max(total_elements, 1))
                result["completion_rate"] = enhanced_completion_rate
                
                return result
            else:
                return self._evaluate_checklist_category_sync(conversation_text, category_info, scenario_id)
                
        except Exception as e:
            print(f"❌ 향상된 체크리스트 평가 오류: {e}")
            return self._evaluate_checklist_category_sync(conversation_text, category_info, scenario_id)

    def _is_open_question(self, question: str) -> bool:
        """개방형 질문 여부 판단"""
        open_keywords = ["어떤", "어떻게", "왜", "언제", "어디서", "무엇", "어느", "설명", "얘기"]
        return any(keyword in question for keyword in open_keywords)
    
    def _is_empathetic_question(self, question: str) -> bool:
        """공감적 질문 여부 판단"""
        empathy_keywords = ["힘드시겠어요", "걱정", "이해", "불편", "괜찮으시", "편하게"]
        return any(keyword in question for keyword in empathy_keywords)
    
    def _is_leading_question(self, question: str) -> bool:
        """유도 질문 여부 판단"""
        leading_patterns = ["~아니에요?", "~죠?", "~것 같은데", "~일까요?"]
        return any(pattern in question for pattern in leading_patterns)

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
    # 공개 메서드들
    # =============================================================================
    
    async def evaluate_conversation(self, user_id: str, scenario_id: str, conversation_log: List[Dict]) -> Dict:
        """LangGraph 워크플로우를 사용한 CPX 평가 실행"""
        
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
            print(f"🚀 [{user_id}] LangGraph CPX 평가 워크플로우 시작")
            final_state = self.workflow.invoke(initial_state)
            
            # 최종 결과 반환
            result = final_state.get("final_evaluation_result", {})
            print(f"🎉 [{user_id}] LangGraph CPX 평가 워크플로우 완료")
            
            return result
            
        except Exception as e:
            print(f"❌ [{user_id}] LangGraph 평가 워크플로우 오류: {e}")
            return {
                "error": str(e),
                "user_id": user_id,
                "scenario_id": scenario_id,
                "evaluation_date": datetime.now().isoformat()
            }

    def get_evaluation_summary(self, user_id: str) -> Dict:
        """사용자의 평가 요약 정보"""
        return {
            "user_id": user_id,
            "total_evaluations": 0,
            "average_score": 0.0,
            "recent_evaluations": []
        }