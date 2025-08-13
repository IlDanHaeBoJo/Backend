import os
import json
from typing import Dict, List, Optional, TypedDict, Annotated
from datetime import datetime
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage as AnyMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

class CPXEvaluationState(TypedDict):
    """CPX 평가 상태 정의 - Multi-Step Reasoning 전용"""
    # 입력 데이터
    user_id: str
    scenario_id: str
    conversation_log: List[Dict]
    
    # 기본 분석 결과
    conversation_analysis: Optional[Dict]
    
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


        self.workflow = self._create_evaluation_workflow()

    def _create_evaluation_workflow(self):
        """CPX 평가 워크플로우 생성"""
        

        workflow = StateGraph(CPXEvaluationState)
            

        workflow.add_node("initialize", self._initialize_evaluation)
        workflow.add_node("analyze_conversation", self._analyze_conversation)
        

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
            

        workflow.add_edge("initialize", "analyze_conversation")
        workflow.add_edge("analyze_conversation", "medical_context")
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
        print(f"📊 [{state['user_id']}] 대화 분석 시작")
        
        conversation_log = state["conversation_log"]
        
        if not conversation_log:
            return {
                **state,
                "conversation_analysis": {"total_questions": 0, "duration_minutes": 0, "question_types": {}},
                "messages": state["messages"] + [HumanMessage(content="대화 로그가 비어있습니다.")]
            }
        
        total_questions = len([msg for msg in conversation_log if msg.get("role") == "student"])
        

        open_questions = 0
        closed_questions = 0
        
        for msg in conversation_log:
            if msg.get("role") == "student":
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

    def _is_category_applicable(self, category_id: str, scenario_id: str) -> bool:
        """시나리오별 카테고리 해당 여부 확인"""
        if scenario_id not in self.scenario_applicable_elements:
            raise ValueError(f"시나리오 ID '{scenario_id}'를 찾을 수 없습니다. 사용 가능한 ID: {list(self.scenario_applicable_elements.keys())}")
        
        scenario_info = self.scenario_applicable_elements[scenario_id]
        applicable_categories = scenario_info["applicable_categories"]
        
        if category_id not in applicable_categories:
            raise ValueError(f"카테고리 ID '{category_id}'를 시나리오 '{scenario_id}'에서 찾을 수 없습니다")
        
        return applicable_categories[category_id]



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
        """10단계: 결과 최종화"""
        print(f"🎯 [{state['user_id']}] 평가 결과 최종화")
        
        # 총점 확인 및 로그
        total_score = state.get('final_scores', {}).get('total_score', 0)
        print(f"🎉 [{state['user_id']}] CPX 평가 완료 - 총점: {total_score}점")
        
        return {
            **state,
            "messages": state["messages"] + [HumanMessage(content="CPX 평가가 성공적으로 완료되었습니다.")]
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

【학생-환자 대화】:
{conversation_text}

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

    def _assess_medical_completeness(self, state: CPXEvaluationState) -> CPXEvaluationState:
        """Step 3: 의학적 완성도 평가 - 개별 카테고리별 평가"""
        print(f"📋 [{state['user_id']}] Step 3: 의학적 완성도 평가 시작")
        
        conversation_text = self._build_conversation_text(state["conversation_log"])
        medical_context = state.get("medical_context_analysis", {})
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
            print(f"  📝 [{category['name']}] 개별 평가 중...")
            result = self._evaluate_single_category(
                conversation_text, 
                category, 
                scenario_id
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
당신은 의학교육 평가 전문가입니다. 
다음 병력청취 대화에서 "{category['name']}" 항목만 집중적으로 평가해주세요.

【평가 대상】: {category['name']}
【필수 요소들】: {category['required_elements']}
【예시】: {example_text}

【학생-환자 대화】:
{conversation_text}

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
        question_intent = state.get("question_intent_analysis", {})
        
        quality_prompt = f"""
당신은 의학교육 평가 전문가입니다. 학생 질문들의 질적 수준을 평가하세요.

【질문 의도 분석 결과】: {question_intent}

【학생-환자 대화】:
{conversation_text}

다음 4가지 기준으로 질문 품질을 평가하세요:

1. 의학적 정확성 (1-10점)
   - 용어 사용이 적절한가?
   - 의학적 논리가 맞는가?
   - 전문적 지식이 반영되었는가?

2. 소통 효율성 (1-10점)
   - 환자가 이해하기 쉬운가?
   - 불필요한 반복은 없는가?
   - 명확하고 간결한가?

3. 임상적 실용성 (1-10점)
   - 실제 진료에서 유용한 정보를 얻을 수 있는가?
   - 시간 대비 효율적인가?
   - 진단에 도움이 되는가?

4. 환자 배려 (1-10점)
   - 환자의 상황을 고려했는가?
   - 공감적 태도를 보였는가?
   - 환자 중심적 접근인가?

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
【해당 카테고리들】: {list(scenario_info.get("applicable_categories", {}).keys())}

【학생-환자 대화】:
{conversation_text}

다음 관점에서 시나리오 적합성을 검증하세요:

1. 부적절한 질문 체크:
   - 해당 시나리오에 맞지 않는 질문들
   - 환자 특성(나이, 성별)에 부적합한 질문들
   - 시급성을 고려하지 않은 질문들

2. 적절성 평가:
   - 시나리오에 특화된 질문을 했는가?
   - 환자 프로필에 맞는 접근을 했는가?
   - 시간 배분이 적절했는가?

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
        """Step 6: 종합 평가 및 최종 점수 계산 (Multi-Step 결과 통합)"""
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

반드시 아래의 정확한 JSON 형식으로만 응답하세요. 다른 텍스트는 포함하지 마세요:

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
            print(f"❌ [{state['user_id']}] Step 6: JSON 형식을 찾을 수 없음")
            print(f"LLM 응답: {result_text[:200]}...")
            raise ValueError(f"Step 6에서 JSON 형식 응답을 찾을 수 없습니다. LLM 응답: {result_text[:100]}")
        
        try:
            comprehensive = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            print(f"❌ [{state['user_id']}] Step 6: JSON 파싱 실패")
            print(f"JSON 텍스트: {json_match.group()[:200]}...")
            raise ValueError(f"Step 6에서 JSON 파싱 실패: {e}")
        
        print(f"✅ [{state['user_id']}] Step 6: 종합 평가 완료")
        
        return {
            **state,
            "comprehensive_evaluation": comprehensive,
            "messages": state["messages"] + [HumanMessage(content="Step 6: 종합 평가 완료")]
        }




    
    def _build_conversation_text(self, conversation_log: List[Dict]) -> str:
        """대화 로그를 텍스트로 변환"""
        conversation_parts = []
        for msg in conversation_log:
            speaker = "학생" if msg.get("role") == "student" else "환자"
            content = msg.get("content", "")
            conversation_parts.append(f"{speaker}: {content}")
        
        return "\n".join(conversation_parts)

    def _evaluate_checklist_category_sync(self, conversation_text: str, category_info: Dict, scenario_id: str) -> Dict:
        """체크리스트 카테고리별 평가 (동기 버전)"""
        
        # 시나리오별 컨텍스트 정보 추가
        scenario_info = self.scenario_applicable_elements.get(scenario_id, {})
        scenario_name = scenario_info.get("name", f"시나리오 {scenario_id}")
        
        evaluation_prompt = f"""
당신은 의과대학 CPX 평가 전문가입니다. 
다음 병력청취 대화에서 "{category_info['name']}" 카테고리의 필수 요소들이 얼마나 잘 수행되었는지 평가하세요.

【시나리오 정보】: {scenario_name}
【평가 카테고리】: {category_info['name']}

【필수 요소들】:
{chr(10).join([f"- {element}" for element in category_info['required_elements']])}

【대화 내용】:
{conversation_text}

【평가 기준】:
- 각 필수 요소가 대화에 포함되었는지 확인
- 시나리오에 적합한 방식으로 수행되었는지 평가
- 자연스럽고 적절한 방식으로 수행되었는지 평가
- 환자와의 소통이 효과적이었는지 판단
- 시간 효율성을 고려하여 핵심 요소 위주로 평가

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

    def _evaluate_diagnosis_category_sync(self, conversation_text: str, category_info: Dict, scenario_id: str) -> Dict:
        """진단 계획 카테고리별 평가 (동기 버전)"""
        
        evaluation_prompt = f"""
당신은 의과대학 CPX 진단 계획 평가 전문가입니다. 
다음 병력청취 대화에서 "{category_info['name']}" 카테고리의 진단 계획 요소들이 얼마나 잘 수행되었는지 평가하세요.

【평가 카테고리】: {category_info['name']}

【필수 요소들】:
{chr(10).join([f"- {element}" for element in category_info['required_elements']])}

【대화 내용】:
{conversation_text}

【시나리오】: {scenario_id}

【평가 기준】:
- 시나리오에 적합한 진단 계획이 제시되었는지 확인
- 각 필수 요소가 대화에 포함되었는지 평가
- 의학적 근거와 논리적 사고 과정 평가
- 환자에게 적절히 설명되었는지 판단

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
                SystemMessage(content="당신은 의과대학 CPX 진단 계획 평가 전문가입니다."),
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
                    "specific_feedback": "진단 계획 평가 시스템 오류로 상세 분석이 불가합니다."
                }
                    
        except Exception as e:
            print(f"❌ 진단 계획 평가 오류: {e}")
            return {
                "completed_elements": [],
                "missing_elements": category_info['required_elements'],
                "completion_rate": 0.0,
                "quality_score": 5,
                "specific_feedback": f"평가 오류: {str(e)}"
            }

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


    
    async def evaluate_conversation(self, user_id: str, scenario_id: str, conversation_log: List[Dict]) -> Dict:
        """LangGraph 워크플로우를 사용한 CPX 평가 실행"""
        
        # 초기 상태 구성 (Multi-Step 전용)
        initial_state = CPXEvaluationState(
            user_id=user_id,
            scenario_id=scenario_id,
            conversation_log=conversation_log,
            conversation_analysis=None,
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
            
            # 최종 결과 구성
            result = {
                "evaluation_metadata": final_state.get("evaluation_metadata", {}),
                "scores": final_state.get("final_scores", {}),
                "feedback": final_state.get("feedback", {}),
                "conversation_summary": final_state.get("conversation_analysis", {}),
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

    def get_evaluation_summary(self, user_id: str) -> Dict:
        """사용자의 평가 요약 정보"""
        return {
            "user_id": user_id,
            "total_evaluations": 0,
            "average_score": 0.0,
            "recent_evaluations": []
        }

