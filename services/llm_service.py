import os
import json
from typing import Dict, List
from pathlib import Path

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


class LLMService:
    def __init__(self):
        """LLM 서비스 초기화"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 필요합니다")

        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1000
        )

        # 사용자별 상태 관리 (가장 일반적인 패턴)
        self.user_states = {}  # user_id -> {scenario, system_prompt, memories}

        # 공통 기본 프롬프트
        self.base_prompt = self._get_base_cpx_prompt()

        # 시나리오 JSON 파일 로드
        self.scenario_data = self._load_scenario_json()

    def _load_scenario_json(self) -> Dict:
        """시나리오 JSON 파일 로드"""
        try:
            scenario_path = Path("scenarios/neurology_dementia_case.json")
            with open(scenario_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ 시나리오 로드 완료: {data.get('scenario_info', {}).get('patient_name', 'Unknown')}")
            return data
        except Exception as e:
            print(f"❌ 시나리오 로드 실패: {e}")
            return {}

    def _convert_scenario_to_prompt(self, scenario_data: Dict) -> str:
        """시나리오 데이터를 LLM 프롬프트로 변환 - 모든 정보 포함"""
        if not scenario_data:
            return "시나리오 정보를 불러올 수 없습니다."
        
        scenario_info = scenario_data.get("scenario_info", {})
        history_taking = scenario_data.get("history_taking", {})
        physical_examination = scenario_data.get("physical_examination", {})
        patient_education = scenario_data.get("patient_education", {})
        
        prompt_parts = []
        
        # 환자 기본 정보
        prompt_parts.append("【환자 기본 정보】")
        prompt_parts.append(f"당신은 표준화 환자 \"{scenario_info.get('patient_name', 'Unknown')}\"입니다.")
        prompt_parts.append(f"- {scenario_info.get('case_presentation', '')}")
        prompt_parts.append(f"- Vital signs: {scenario_info.get('vital_signs', '')}")
        prompt_parts.append(f"- 주요 진단: {scenario_info.get('primary_diagnosis', '')}")
        
        # 감별 진단
        diff_diagnoses = scenario_info.get("differential_diagnoses", [])
        if diff_diagnoses:
            prompt_parts.append(f"- 감별진단: {', '.join(diff_diagnoses)}")
        
        # 병력 청취 정보 (모든 카테고리)
        prompt_parts.append("\n【병력 청취 정보】")
        history_labels = {
            "O_onset": "발병 시기/경과",
            "L_location": "위치",
            "D_duration": "지속 시간/패턴",
            "Co_course": "경과/변화",
            "Ex_experience": "과거 경험/가족력",
            "C_character": "증상 특징",
            "A_associated": "동반 증상",
            "F_factor": "악화/완화 요인",
            "E_exam": "기존 검사/진단",
            "trauma_history": "외상력",
            "past_medical_history": "과거 병력",
            "medication_history": "복용 약물",
            "family_history": "가족력",
            "social_history": "사회력",
            "gynecologic_history": "산부인과력"
        }
        
        for key, label in history_labels.items():
            if key in history_taking and history_taking[key]:
                prompt_parts.append(f"- {label}: {history_taking[key]}")
        
        # 신체 검사 정보
        physical_examination = scenario_data.get("physical_examination", {})
        if physical_examination:
            prompt_parts.append("\n【신체 검사 정보】")
            for key, value in physical_examination.items():
                prompt_parts.append(f"- {key}: {value}")
        
        # 환자 교육 정보
        patient_education = scenario_data.get("patient_education", {})
        if patient_education:
            prompt_parts.append("\n【환자 교육 관련 정보】")
            if isinstance(patient_education, dict):
                for key, value in patient_education.items():
                    prompt_parts.append(f"- {key}: {value}")
            else:
                prompt_parts.append(f"- 교육 내용: {patient_education}")
        
        # 카테고리 정보
        category = scenario_info.get("category", "")
        if category:
            prompt_parts.append(f"\n【진료 카테고리】: {category}")
        
        # 환자 역할 지침
        prompt_parts.append("\n【환자 역할 지침】")
        prompt_parts.append("⚠️ **중요: 위 정보를 바탕으로 환자 역할을 하되, 90%는 질문에만 간단히 답하고 10%만 추가 설명을 제공하세요**")
        prompt_parts.append("")
        prompt_parts.append("✅ **간결한 답변 스타일 (90%)**:")
        prompt_parts.append('- "자꾸 깜빡깜빡하는 것 같아요"')
        prompt_parts.append('- "한 6개월 전부터 그런 것 같습니다"')
        prompt_parts.append('- "그런 건 없는 것 같습니다"')
        prompt_parts.append('- "마트에 물건을 사러 갔는데 뭘 사러 갔는지 잘 생각이 안 나고요"')
        prompt_parts.append("")
        prompt_parts.append("📝 **가끔 추가 설명 (10%)**:")
        prompt_parts.append('- 의사가 "편하게 얘기해보세요"라고 할 때만 자세히 설명')
        prompt_parts.append("- 같은 질문을 반복할 때 조금 더 구체적으로 답변")
        prompt_parts.append("- 중요한 증상에 대해서는 2-3개의 예시 제공")
        prompt_parts.append("")
        prompt_parts.append("🎭 **말하는 성격**:")
        prompt_parts.append("- 치매 걱정이 있는 63세 남성")
        prompt_parts.append("- 침착하고 성실하지만 **말수가 적음**")
        prompt_parts.append("- 묻는 것에만 답하는 스타일")
        prompt_parts.append("- 불필요한 추가 정보는 제공하지 마세요")
        
        return "\n".join(prompt_parts)

    def _get_or_create_user_state(self, user_id: str) -> Dict:
        """사용자 상태 가져오기 또는 생성 (일반적인 패턴)"""
        if user_id not in self.user_states:
            self.user_states[user_id] = {
                'scenario': None,
                'system_prompt': '',
                'memories': []
            }
        return self.user_states[user_id]

    def select_scenario(self, scenario_id: str, user_id: str) -> bool:
        """사용자별 시나리오 선택하고 LLM 프롬프트 고정"""
        if not self.scenario_data:
            print(f"❌ [{user_id}] 시나리오 데이터가 로드되지 않았습니다.")
            return False
            
        # 현재는 하나의 시나리오만 지원 (scenario_id "1")
        expected_id = self.scenario_data.get("scenario_info", {}).get("scenario_id", "1")
        if scenario_id != expected_id:
            print(f"❌ [{user_id}] 지원하지 않는 시나리오: {scenario_id} (사용 가능: {expected_id})")
            return False
            
        user_state = self._get_or_create_user_state(user_id)
        user_state['scenario'] = scenario_id
        
        # 공통 프롬프트 + 시나리오 정보 조합
        case_info = self._convert_scenario_to_prompt(self.scenario_data)
        user_state['system_prompt'] = self.base_prompt + "\n\n" + case_info
        
        patient_name = self.scenario_data.get("scenario_info", {}).get("patient_name", "Unknown")
        print(f"✅ [{user_id}] 시나리오 선택: {patient_name} 케이스")
        return True

    def get_available_scenarios(self) -> Dict[str, str]:
        """사용 가능한 시나리오 목록 반환"""
        if not self.scenario_data:
            return {}
        
        scenario_info = self.scenario_data.get("scenario_info", {})
        scenario_id = scenario_info.get("scenario_id", "1")
        patient_name = scenario_info.get("patient_name", "Unknown")
        case_presentation = scenario_info.get("case_presentation", "")
        
        return {scenario_id: f"{patient_name} - {case_presentation}"}

    def _get_base_cpx_prompt(self) -> str:
        """CPX 공통 기본 프롬프트"""
        return """
당신은 의과대학 CPX(Clinical Performance Examination) 실기시험을 위한 한국어 가상 표준화 환자입니다.

【중요: 절대 의사가 되지 마세요!】
- 당신은 병원에 온 환자입니다
- 진료하지 마세요
- "무슨 일로 오셨나요?", "어디가 아프세요?" 같은 의사 말은 절대 하지 마세요
- 오직 환자로서 본인의 증상과 걱정만 이야기하세요

【환자 역할】
- 의사의 질문에 환자로서 대답하세요
- 본인의 아픈 곳과 증상을 표현하세요  
- 의사의 진료를 받는 입장입니다
- 걱정되는 마음을 솔직하게 표현하세요

【말하는 방식】
 **핵심 룰: 의사가 묻는 것에만 답하고 부가적인 설명 금지**
- 질문 한 개 → 답변 한 개 (끝)
- 묻지 않은 내용은 절대 말하지 마세요
- "편하게 얘기해보세요"라고 할 때만 2-3개 예시 제공
- 한국어로 일반인처럼 자연스럽게 말하기
- "아파요", "쓰려요", "답답해요", "불안해요" 같은 자연스러운 표현
- 감정을 솔직하게 표현 (걱정, 불안, 아픔, 두려움 등)
- 질문에 대해 모르면 "잘 모르겠어요" 솔직하게 말하기
- 의학 지식을 과시하지 마세요

의사가 아닌 환자 역할만 하세요!
"""



    async def generate_response(self, user_input: str, user_id: str = "default") -> dict:
        """사용자 입력에 대한 AI 응답 생성 (사용자별 상태 관리)"""
        user_state = self._get_or_create_user_state(user_id)
        
        # 사용자별 시나리오 확인
        if not user_state['scenario']:
            return {"text": "먼저 시나리오를 선택해주세요.", "conversation_ended": False}

        # 사용자별 대화 기록 사용
        memory = user_state['memories']

        # 메시지 구성 (사용자별 시나리오 프롬프트 사용)
        messages = [SystemMessage(content=user_state['system_prompt'])]

        # 최근 대화 추가 (최근 5개만)
        for msg in memory[-5:]:
            messages.extend(msg)

        # 현재 입력
        messages.append(HumanMessage(content=user_input))

        # LLM 호출
        response = self.llm(messages)
        response_text = response.content.strip()

        # 사용자별 대화 기록 저장
        memory.append([
            HumanMessage(content=user_input),
            response
        ])

        # 대화 종료 의도 감지
        conversation_ended = False
        if self._detect_conversation_ending(user_input, response_text):
            response_text = await self._generate_natural_farewell(
                user_input, response_text, user_state, user_id
            )
            conversation_ended = True
            print(f"🏁 [{user_id}] 대화 종료 감지됨 - 음성 처리 중단됩니다")

        return {"text": response_text, "conversation_ended": conversation_ended}

    def _detect_conversation_ending(self, user_input: str, ai_response: str) -> bool:
        """대화 종료 의도 감지 (의사의 마무리 멘트 감지)"""
        # 의사(사용자)가 진료 마무리할 때 하는 말들
        doctor_ending_keywords = [
            "처방해드릴게요", "처방해드리겠습니다", "약을 드리겠습니다",
            "괜찮으실 거예요", "괜찮을 거예요", "걱정하지 마세요",
            "조심하세요", "몸조심하세요", "건강하세요", 
            "더 아프시면 오세요", "악화되면 오세요", "변화있으면 오세요",
            "안녕히 가세요", "들어가세요", "수고하셨습니다",
            "진료 마치겠습니다", "이상으로", "오늘은 여기까지"
        ]
        
        # 의사의 마무리 멘트 감지
        doctor_ending = any(keyword in user_input for keyword in doctor_ending_keywords)
        
        return doctor_ending

    async def _generate_natural_farewell(self, doctor_input: str, ai_response: str, user_state: dict, user_id: str) -> str:
        """LLM을 사용해 대화 맥락에 맞는 자연스러운 마무리 인사 생성"""
        
        # 마무리 인사 생성을 위한 특별 프롬프트
        farewell_prompt = f"""
{user_state['system_prompt']}

【중요: 지금은 진료가 끝나는 상황입니다】
의사가 마무리 멘트를 했으므로, 환자로서 자연스럽고 감사한 마음을 담아 인사하세요.

의사의 마지막 말: "{doctor_input}"
당신의 일반적인 응답: "{ai_response}"

이제 의사에게 감사 인사와 함께 자연스럽게 작별 인사를 하세요.
- 의사에 대한 감사 표현
- 처방이나 조언에 대한 수용적 태도  
- 환자 캐릭터에 맞는 말투 유지
- 너무 길지 않게, 자연스럽게

응답은 위의 일반적인 응답에 자연스럽게 이어지도록 작성하세요.
"""
        
        # LLM에게 자연스러운 마무리 인사 요청
        farewell_messages = [SystemMessage(content=farewell_prompt)]
        farewell_messages.append(HumanMessage(content="자연스러운 마무리 인사를 해 주세요."))
        
        try:
            farewell_response = self.llm(farewell_messages)
            natural_farewell = farewell_response.content.strip()
            
            # 기존 응답과 자연스럽게 결합
            return f"{ai_response}\n\n{natural_farewell}"
            
        except Exception as e:
            print(f"❌ 마무리 인사 생성 오류: {e}")
            # 오류 시 기본 마무리 인사 사용
            default_farewell = "네, 감사합니다 선생님. 안녕히 계세요."
            return f"{ai_response}\n\n{default_farewell}"

    def clear_user_memory(self, user_id: str):
        """사용자 상태 전체 초기화 (대화 기록 + 시나리오)"""
        if user_id in self.user_states:
            del self.user_states[user_id]
            print(f"✅ [{user_id}] 사용자 상태 초기화 완료")

    def get_conversation_summary(self, user_id: str) -> str:
        """사용자별 대화 요약"""
        user_state = self._get_or_create_user_state(user_id)
        
        if not user_state['memories']:
            return "대화 내역이 없습니다."

        count = len(user_state['memories'])
        if user_state['scenario'] and self.scenario_data:
            patient_name = self.scenario_data.get("scenario_info", {}).get("patient_name", "Unknown")
            scenario_name = f"{patient_name} 케이스"
        else:
            scenario_name = "시나리오 미선택"
        return f"현재 시나리오: {scenario_name}\n총 {count}번의 대화가 있었습니다."

