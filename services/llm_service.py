import os
import json
from typing import Dict, List
from pathlib import Path

from langchain_openai import ChatOpenAI
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
        """시나리오 데이터를 답변 참고 정보로 변환"""
        if not scenario_data:
            return "시나리오 정보를 불러올 수 없습니다."
        
        scenario_info = scenario_data.get("scenario_info", {})
        history_taking = scenario_data.get("history_taking", {})
        physical_examination = scenario_data.get("physical_examination", {})
        
        prompt_parts = []
        
        # 환자 정보
        prompt_parts.append("【환자 정보】")
        prompt_parts.append(f"이름: {scenario_info.get('patient_name', 'Unknown')}")
        prompt_parts.append(f"상황: {scenario_info.get('case_presentation', '')}")
        prompt_parts.append(f"진단: {scenario_info.get('primary_diagnosis', '')}")
        prompt_parts.append("")
        prompt_parts.append("【환자 심리 상태】")
        prompt_parts.append("- 증상 때문에 병원에 온 상황 → 걱정되고 불안함")
        prompt_parts.append("- 본인의 건강 상태에 대한 우려와 궁금증")
        prompt_parts.append("- 진단 결과나 치료에 대한 관심과 걱정")
        prompt_parts.append("- 위 상황에 맞는 환자 성격으로 행동하세요")
        
        # 의사 질문별 답변 가이드
        prompt_parts.append("\n【의사 질문에 따른 답변 참고 정보】")
        
        # 주요 질문 카테고리별 답변 정보 매핑
        question_answer_map = {
            "발병시기/언제부터": history_taking.get("O_onset", ""),
            "지속시간/얼마나": history_taking.get("D_duration", ""),
            "경과/변화": history_taking.get("Co_course", ""),
            "과거경험/가족력": history_taking.get("Ex_experience", ""),
            "증상특징/어떤증상": history_taking.get("C_character", ""),
            "동반증상": history_taking.get("A_associated", ""),
            "악화완화요인": history_taking.get("F_factor", ""),
            "기존검사": history_taking.get("E_exam", ""),
            "외상력": history_taking.get("trauma_history", ""),
            "과거병력": history_taking.get("past_medical_history", ""),
            "복용약물": history_taking.get("medication_history", ""),
            "가족력": history_taking.get("family_history", ""),
            "사회력": history_taking.get("social_history", "")
        }
        
        for question_type, answer_info in question_answer_map.items():
            if answer_info and answer_info != "해당없음":
                prompt_parts.append(f"• {question_type} 관련 질문 시 → {answer_info}")
        
        # 신체 검사 관련 정보
        if physical_examination:
            prompt_parts.append("\n【신체 검사 관련 답변 정보】")
            for key, value in physical_examination.items():
                if value:
                    prompt_parts.append(f"• {key} 관련 → {value}")
        
        # 환자 교육 관련 정보
        patient_education = scenario_data.get("patient_education", "")
        if patient_education:
            prompt_parts.append(f"\n【환자 교육 시 반응】")
            prompt_parts.append("의사가 설명할 때:")
            prompt_parts.append('- "네", "아...", "그렇구나" 같은 수용적 반응만')
            prompt_parts.append("- 의사 말을 절대 반복하지 마세요")
            prompt_parts.append("")
            prompt_parts.append('의사가 "궁금한 점 있으세요?" 물으면:')
            prompt_parts.append("- 아래 교육 내용에 언급 안된 것이 있으면 → 그것에 대해 1-2개 질문")
            prompt_parts.append("- 없으면 → '없습니다' 또는 '괜찮습니다'")
            prompt_parts.append(f"\n교육 내용: {patient_education}")
        
        # 답변 스타일 가이드
        prompt_parts.append("\n【답변 스타일】")
        prompt_parts.append("- 일반인이 병원에서 말하는 방식으로 답변")
        prompt_parts.append("- 의학 용어 사용 금지 → 쉬운 말로 표현")
        prompt_parts.append("- 간단하고 자연스럽게 (1-2문장)")
        prompt_parts.append("- 예시: '최근에 깜빡깜빡해요', '머리가 아파요', '잘 기억나요'")
        prompt_parts.append("- 위 정보를 바탕으로 환자 입장에서 답변")
        prompt_parts.append("- 의사가 구체적으로 물어보면 세부사항 추가 제공")
        
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
        """사용자별 시나리오 선택하고 LLM 프롬프트 고정 - 기억력 저하 시나리오 고정"""
        if not self.scenario_data:
            print(f"❌ [{user_id}] 시나리오 데이터가 로드되지 않았습니다.")
            return False
            
        # 기억력 저하 시나리오(1번) 고정 사용
        fixed_scenario_id = "1"
        user_state = self._get_or_create_user_state(user_id)
        user_state['scenario'] = fixed_scenario_id
        
        # 공통 프롬프트 + 시나리오 정보 조합
        case_info = self._convert_scenario_to_prompt(self.scenario_data)
        user_state['system_prompt'] = self.base_prompt + "\n\n" + case_info
        
        patient_name = self.scenario_data.get("scenario_info", {}).get("patient_name", "Unknown")
        print(f"✅ [{user_id}] 기억력 저하 시나리오 자동 선택: {patient_name} 케이스")
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
        """CPX 기본 프롬프트"""
        return """
당신은 의과대학 CPX(Clinical Performance Examination) 실기시험을 위한 가상 표준화 환자입니다.

【상황 설정】
- 의대생이 의사 역할을 하며 당신에게 문진을 합니다
- 당신은 특정 질환을 가진 환자 역할을 연기합니다
- 실제 병원 진료실과 같은 상황입니다

【금지】
- 의사 역할 금지 (질문, 진단, 처방 등)
- 의사 말 반복 금지
- 불필요한 추가 설명 금지

【답변 방식】
- 아래 시나리오 정보를 바탕으로 환자 역할 연기
- 질문받은 것만 간단히 답변
- 모르면 "잘 모르겠어요"
- 자연스럽고 솔직하게
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
        print(f"🔍 [{user_id}] 시스템 프롬프트 길이: {len(user_state['system_prompt'])} 문자")
        print(f"🔍 [{user_id}] 시스템 프롬프트 앞부분: {user_state['system_prompt'][:200]}...")
        messages = [SystemMessage(content=user_state['system_prompt'])]

        # 최근 대화 추가 (최근 5개만)
        for msg in memory[-5:]:
            messages.extend(msg)

        # 대화 종료 의도 사전 감지
        conversation_ended = False
        if self._detect_conversation_ending(user_input):
            # 마무리 인사 직접 생성
            response_text = await self._generate_natural_farewell(user_input, user_state, user_id)
            conversation_ended = True
            print(f"🏁 [{user_id}] 대화 종료 감지됨 - 음성 처리 중단됩니다")
        else:
            # 일반 대화 - LLM 호출
            messages.append(HumanMessage(content=user_input))
            response = self.llm(messages)
            response_text = response.content.strip()
            
            # 사용자별 대화 기록 저장
            memory.append([
                HumanMessage(content=user_input),
                response
            ])

        return {"text": response_text, "conversation_ended": conversation_ended}

    def _detect_conversation_ending(self, user_input: str) -> bool:
        """대화 종료 의도 감지 (의사의 마무리 멘트 감지)"""
        # 의사(사용자)가 진료 마무리할 때 하는 말들
        doctor_ending_keywords = [
            # 처방 관련
            "처방해드릴게요", "처방해드리겠습니다", "약을 드리겠습니다", "약 받으세요",
            
            # 안심시키는 말
            "괜찮으실 거예요", "괜찮을 거예요", "걱정하지 마세요", "크게 걱정 안하셔도",
            
            # 건강 관련 당부
            "조심하세요", "몸조심하세요", "건강하세요", "조심히 가세요", "조심히 들어가세요",
            "몸 관리 잘하세요", "무리하지 마세요", "푹 쉬세요",
            
            # 재방문 안내
            "더 아프시면 오세요", "악화되면 오세요", "변화있으면 오세요", "이상하면 다시 오세요",
            "문제되면 언제든 오세요", "필요하면 다시 오세요",
            
            # 인사말
            "안녕히 가세요", "들어가세요", "수고하셨습니다", "고생하셨습니다",
            
            # 진료 마무리
            "진료 마치겠습니다", "이상으로", "오늘은 여기까지", "진료 끝내겠습니다",
            "이만 마치겠습니다", "진료 완료하겠습니다",
            
            # 환자 마무리 응답도 감지
            "감사합니다", "고맙습니다", "안녕히 계세요", "좋은 하루", "검사 후에 뵙겠습니다"
        ]
        
        # 의사의 마무리 멘트 감지
        doctor_ending = any(keyword in user_input for keyword in doctor_ending_keywords)
        
        return doctor_ending

    async def _generate_natural_farewell(self, doctor_input: str, user_state: dict, user_id: str) -> str:
        """간단한 마무리 인사 생성"""
        
        # 매우 간단한 마무리 인사 프롬프트
        farewell_prompt = f"""
의사: "{doctor_input}"

환자로서 1문장으로 간단히 감사 인사하세요.
예시: "네, 감사합니다 선생님."
"""
        
        try:
            farewell_messages = [SystemMessage(content=farewell_prompt)]
            farewell_response = self.llm(farewell_messages)
            farewell = farewell_response.content.strip()
            
            return farewell
            
        except Exception as e:
            print(f"❌ 마무리 인사 생성 오류: {e}")
            return "네, 감사합니다 선생님."

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

