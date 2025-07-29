import os
import logging
from typing import List, Optional
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        """LLM 서비스 초기화"""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        
        # GPT-4o 모델 초기화
        self.llm = ChatOpenAI(
            openai_api_key=self.api_key,
            model_name="gpt-4o",
            temperature=0.7,
            max_tokens=1000
        )
        
        # 대화 메모리 (사용자별로 관리해야 함)
        self.user_memories = {}
        
        # CPX 표준화 환자 시스템 프롬프트 (한국어 의료 특화)
        self.system_prompt = """
당신은 의과대학 CPX(Clinical Performance Examination) 실기시험을 위한 한국어 가상 표준화 환자입니다.

⚠️ **역할 구분 - 절대 지킬 것**: 
- 당신: 아픈 환자 (의사에게 치료받으러 온 사람)
- 상대방: 의과대학생 (의사 역할)

❌ **절대 금지 - 의사가 하는 질문들**:
- "어디가 불편하세요?" / "어디가 아파요?"
- "언제부터 아프셨어요?" / "언제부터 그랬어요?"
- "어떤 증상이 있으세요?"
- "무엇을 도와드릴까요?"
- "검사를 받아보시겠어요?"
→ 이런 질문들은 의사가 환자에게 묻는 것입니다!

✅ **환자가 할 수 있는 질문들**:
- "선생님, 이게 심각한 건가요?"
- "수술해야 하나요?"
- "언제쯤 나을까요?"
- "약은 어떻게 먹어야 하나요?"
→ 이런 질문들은 환자가 의사에게 묻는 것입니다.

**올바른 상호작용:**
- 학생: "안녕하세요" → 환자: "안녕하세요, 선생님" (그 후 조용히 기다림)
- 학생: "어디가 아프세요?" → 환자: "배가 아파요"
- 학생: "언제부터 아프셨어요?" → 환자: "어제 저녁부터요"

다음 지침을 엄격히 따라주세요:

1. **역할 구분**:
   - 당신: 아픈 환자 (질문을 받고 답변함)
   - 상대방: 의과대학생 (질문을 하는 사람)
   - 수동적이고 도움을 구하는 자세 유지

2. **한국어 증상 표현**: 
   - 일반적인 표현: "속이 아파요", "토할 것 같아요", "어지러워요"
   - 의학용어 금지: "복통"보다는 "배가 아파요"
   - 자연스러운 표현: "시큰시큰해요", "욱신욱신해요", "뻐근해요"

3. **학생 질문 대응**: 
   - 개방형 질문: 자세하고 자연스럽게 답변 (2-3문장)
   - 폐쇄형 질문: 간단명료하게 "네/아니요" + 간단한 부연설명
4. **감정과 어투**: 
   - 연령대에 맞는 존댓말 사용
   - 불안, 걱정, 고통 등 적절한 감정 표현
   - "음...", "아...", "그런데..." 같은 자연스러운 망설임 포함
5. **일관성**: 같은 질문에는 항상 동일한 답변, 모순되지 않게
6. **현실적 반응**: 
   - 의학지식 없는 일반인 관점으로 답변
   - 증상을 일상 언어로 생생하게 묘사
   - 시간, 정도, 부위를 구체적으로 표현

**한국어 표현 예시:**
- 통증: "아야", "쿡쿡 쑤셔요", "찌릿찌릿해요"
- 시간: "어제부터", "3일 전쯤", "새벽에"
- 정도: "조금", "많이", "견딜 만해요", "너무 아파요"

참고 문서의 케이스 정보를 바탕으로 해당 환자 역할을 연기하세요.
당신은 아픈 환자일 뿐, 의사가 아닙니다.
"""
        
        logger.info("LLM 서비스 초기화 완료")

    def get_user_memory(self, user_id: str) -> ConversationBufferWindowMemory:
        """사용자별 대화 메모리 반환"""
        if user_id not in self.user_memories:
            self.user_memories[user_id] = ConversationBufferWindowMemory(
                k=10,  # 최근 10개 대화만 기억
                return_messages=True
            )
        return self.user_memories[user_id]

    async def generate_response(
        self, 
        user_input: str, 
        relevant_docs: List[str] = None,
        user_id: str = "default"
    ) -> str:
        """사용자 입력에 대한 AI 응답 생성"""
        try:
            # 사용자별 메모리 가져오기
            memory = self.get_user_memory(user_id)
            
            # 컨텍스트 구성
            context = ""
            if relevant_docs:
                context = "\n참고 문서:\n" + "\n".join(relevant_docs[:3])  # 상위 3개만 사용
            
            # 프롬프트 템플릿 구성
            messages = [
                SystemMessage(content=self.system_prompt),
            ]
            
            # 대화 히스토리 추가
            chat_history = memory.chat_memory.messages
            messages.extend(chat_history)
            
            # 현재 사용자 입력 추가
            user_message_content = user_input
            if context:
                user_message_content += f"\n\n{context}"
            
            messages.append(HumanMessage(content=user_message_content))
            
            # LLM 호출
            response = self.llm(messages)
            response_text = response.content.strip()
            
            # 메모리에 대화 저장
            memory.chat_memory.add_user_message(user_input)
            memory.chat_memory.add_ai_message(response_text)
            
            logger.info(f"[{user_id}] LLM 응답 생성 완료: {response_text[:50]}...")
            return response_text
            
        except Exception as e:
            logger.error(f"LLM 응답 생성 실패: {e}")
            # API 실패를 명확하게 알림
            if "429" in str(e):
                return f"❌ OpenAI API 할당량 초과: {e}"
            elif "invalid_api_key" in str(e):
                return f"❌ OpenAI API 키 오류: {e}"
            elif "insufficient_quota" in str(e):
                return f"❌ OpenAI 크레딧 부족: {e}"
            else:
                return f"❌ LLM API 오류: {e}"

    

    def clear_user_memory(self, user_id: str):
        """사용자 대화 메모리 초기화"""
        if user_id in self.user_memories:
            self.user_memories[user_id].clear()
            logger.info(f"[{user_id}] 대화 메모리 초기화 완료")

    def get_conversation_summary(self, user_id: str) -> str:
        """사용자의 대화 요약 반환"""
        memory = self.get_user_memory(user_id)
        messages = memory.chat_memory.messages
        
        if not messages:
            return "대화 내역이 없습니다."
        
        # 간단한 대화 요약
        summary = f"총 {len(messages)//2}번의 대화가 있었습니다."
        return summary 