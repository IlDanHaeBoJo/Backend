import os
from typing import Dict, List

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

        self.user_memories = {}  # 사용자별 대화 기록
        self.current_scenario = None  # 현재 선택된 시나리오
        self.system_prompt = ""  # 시나리오 선택 후 설정됨

        # 공통 기본 프롬프트
        self.base_prompt = self._get_base_cpx_prompt()

        # 사용 가능한 시나리오들 (케이스별 고유 정보만)
        self.scenarios = {
            "1": {
                "name": "흉통 케이스 (김철수, 45세 남성)",
                "case_info": self._get_chest_pain_case_info()
            },
            "2": {
                "name": "복통 케이스 (박영희, 32세 여성)",
                "case_info": self._get_abdominal_pain_case_info()
            }
        }

    def select_scenario(self, scenario_id: str) -> bool:
        """시나리오 선택하고 LLM 프롬프트 고정"""
        if scenario_id in self.scenarios:
            self.current_scenario = scenario_id
            # 공통 프롬프트 + 케이스별 정보 조합
            case_info = self.scenarios[scenario_id]["case_info"]
            self.system_prompt = self.base_prompt + "\n\n" + case_info
            print(f"✅ 시나리오 선택: {self.scenarios[scenario_id]['name']}")
            return True
        else:
            print(f"❌ 존재하지 않는 시나리오: {scenario_id}")
            return False

    def get_available_scenarios(self) -> Dict[str, str]:
        """사용 가능한 시나리오 목록 반환"""
        return {k: v["name"] for k, v in self.scenarios.items()}

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
- 한국어로 일반인처럼 자연스럽게 말하기
- "아파요", "쓰려요", "답답해요", "불안해요" 같은 자연스러운 표현
- 감정을 솔직하게 표현 (걱정, 불안, 아픔, 두려움 등)
- 질문에 대해 모르면 "잘 모르겠어요" 솔직하게 말하기
- 의학 지식을 과시하지 마세요

【대화 예시】
의사: "안녕하세요"
환자: "안녕하세요... 몸이 좀 안 좋아서 왔어요" ✅

의사가 아닌 환자 역할만 하세요!
"""

    def _get_chest_pain_case_info(self) -> str:
        """흉통 케이스 고유 정보 (김철수)"""
        return """
【환자 정보】
당신은 표준화 환자 "김철수"입니다.
- 이름: 김철수 (45세, 남성)
- 주증상: 가슴 왼쪽 압박감과 통증 (3일 전부터)
- 성격: 스트레스 많은 회사 중간관리직, 건강에 무관심했던 전형적 중년남성

【증상 특징】
- 가슴 왼쪽이 조이는 듯한 압박감
- 운동이나 계단 오르면 악화, 쉬면 좋아짐
- 목과 왼쪽 어깨로 퍼지는 통증
- 가끔 식은땀, 숨차는 느낌
- 통증 강도: 10점 중 6-7점
- 지속시간: 각 에피소드마다 5-10분 정도

【배경 정보】
- 고혈압 진단 5년 전 (현재 약물 복용 중)
- 흡연: 1갑/일 × 20년 (금연 시도 중)
- 아버지가 심근경색으로 60세에 돌아가심
- 회사 스트레스 많음, 불규칙한 생활
- 음주: 주 2-3회, 소주 1병 정도

【이 환자의 말하는 특징】
- 전형적인 중년 남성의 말투와 태도
- 스트레스와 업무 부담을 많이 받는 직장인의 특징
- 건강 관리에 소홀했던 것에 대한 후회와 불안감
"""

    def _get_abdominal_pain_case_info(self) -> str:
        """복통 케이스 고유 정보 (박영희)"""
        return """
【환자 정보】
당신은 표준화 환자 "박영희"입니다.
- 이름: 박영희 (32세, 여성)
- 주증상: 오른쪽 윗배 통증 (2일 전부터)
- 성격: 젊은 직장 여성, 평소 건강했어서 현재 상황에 놀라고 있음

【증상 특징】
- 오른쪽 윗배 (우상복부) 심한 통증
- 치킨 먹은 후 시작됨 (2일 전 저녁)
- 등쪽으로 퍼지는 통증
- 기름진 음식 먹으면 악화
- 통증 강도: 10점 중 7-8점
- 구토 2-3차례 (어제 밤)
- 열감은 없음, 식욕 완전 없어짐

【배경 정보】
- 특별한 병력 없음 (평소 건강했음)
- 평소 기름진 음식 좋아함
- 다이어트로 불규칙한 식사 패턴
- 가족력: 어머니 담석증
- 직업: 회사원 (사무직)
- 음주: 주 1-2회 적당히, 흡연: 안 함

【이 환자의 말하는 특징】
- 젊은 여성답게 감정 표현이 솔직함
- 평소 건강했던 사람의 당황스러움과 걱정
- 통증으로 인한 불편함을 솔직하게 표현
"""

    async def generate_response(self, user_input: str, user_id: str = "default") -> str:
        """사용자 입력에 대한 AI 응답 생성 (RAG 의존성 제거)"""
        if not self.current_scenario:
            return "먼저 시나리오를 선택해주세요."

        # 대화 기록 관리 (최근 5개만)
        if user_id not in self.user_memories:
            self.user_memories[user_id] = []

        memory = self.user_memories[user_id]

        # 메시지 구성 (고정된 시나리오 프롬프트 사용)
        messages = [SystemMessage(content=self.system_prompt)]

        # 최근 대화 추가
        for msg in memory[-5:]:
            messages.extend(msg)

        # 현재 입력
        messages.append(HumanMessage(content=user_input))

        # LLM 호출
        response = self.llm(messages)
        response_text = response.content.strip()

        # 대화 기록 저장
        memory.append([
            HumanMessage(content=user_input),
            response
        ])

        return response_text

    def clear_user_memory(self, user_id: str):
        """사용자 대화 기록 초기화"""
        if user_id in self.user_memories:
            del self.user_memories[user_id]

    def get_conversation_summary(self, user_id: str) -> str:
        """대화 요약"""
        if user_id not in self.user_memories:
            return "대화 내역이 없습니다."

        count = len(self.user_memories[user_id])
        scenario_name = self.scenarios[self.current_scenario]["name"] if self.current_scenario else "시나리오 미선택"
        return f"현재 시나리오: {scenario_name}\n총 {count}번의 대화가 있었습니다."
