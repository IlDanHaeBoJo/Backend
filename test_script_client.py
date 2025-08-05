#!/usr/bin/env python3
"""
CPX 스크립트 테스트 클라이언트
STT 없이 텍스트로 바로 테스트할 수 있는 클라이언트
"""

import asyncio
import websockets
import json

class CPXScriptTester:
    def __init__(self, user_id="test_user"):
        self.user_id = user_id
        self.websocket = None
        
    async def connect(self):
        """웹소켓 연결"""
        uri = f"ws://localhost:8000/ws/{self.user_id}"
        print(f"🔗 연결 중: {uri}")
        self.websocket = await websockets.connect(uri)
        print("✅ 연결 완료!")
        
        # 초기 메시지 받기
        response = await self.websocket.recv()
        data = json.loads(response)
        print(f"📋 {data.get('message', '')}")
        
    async def select_scenario(self, scenario_id):
        """시나리오 선택"""
        command = {
            "type": "select_scenario",
            "scenario_id": scenario_id
        }
        await self.websocket.send(json.dumps(command))
        
        response = await self.websocket.recv()
        data = json.loads(response)
        print(f"🎭 {data.get('message', '')}")
        
    async def send_text(self, text):
        """텍스트 직접 전송 (의사 발언)"""
        command = {
            "type": "text_input",
            "text": text
        }
        await self.websocket.send(json.dumps(command))
        
        response = await self.websocket.recv()
        data = json.loads(response)
        
        print(f"\n🩺 의사: {text}")
        print(f"🤒 환자: {data.get('ai_text', '응답 없음')}")
        
        # 대화 종료 확인
        if data.get('conversation_ended', False):
            print("🏁 대화가 종료되었습니다.")
            return True
        
        return False
        
    async def run_script_test(self, scenario_id, doctor_lines):
        """스크립트 기반 테스트 실행"""
        await self.connect()
        await self.select_scenario(scenario_id)
        
        print("\n" + "="*50)
        print("🏥 CPX 테스트 시작")
        print("="*50)
        
        for line in doctor_lines:
            ended = await self.send_text(line)
            if ended:
                break
            await asyncio.sleep(0.5)  # 약간의 대기
        
        print("\n" + "="*50)
        print("✅ 테스트 완료")
        print("="*50)
        
        await self.websocket.close()

# 신경과 치매 케이스 스크립트 (의사 발언만)
NEUROLOGY_DOCTOR_SCRIPT = [
    "안녕하세요. 저는 신경과라고 합니다.",
    "환자분 성함과 나이, 등록번호가 어떻게 되세요?",
    "네, 어떻게 오셨어요?",
    "기억력이 떨어지시는 것 같다. 그러면 그게 언제부터 그러시는 거죠?",
    "그러면 어떻게 깜빡하시고 기억력이 떨어지시는지 편하게 얘기 한번 해보시겠어요?",
    "그러면 그게 이제 비교적 최근의 일들을 잘 까먹으시나요? 아니면 옛날 일도 많이 잊어버리시나요?",
    "그러면 혹시 최근에 성격 변화 같은 건 없으세요? 급해지거나 짜증이 나거나 많이 다투시거나 그런 건?",
    "물건 이름이나 단어 같은 거 생각이 잘 안나거나 말하는 게 좀 어둔하거나 그런건 없으세요?",
    "최근에 길을 잃어버렸다거나 다니던 길인데 잘 모르겠다거나 그런 적은 없으세요?",
    "계산은 잘 하세요? 예를 들면 물건 살 때 돈 내는 거 그런 거?",
    "알겠습니다. 일단은 기억력도 떨어지시는 것 같은데 일상생활을 할 때 그 기억력 때문에 문제가 되거나 그런 일상생활에 장애가 있으세요?",
    "예를 들면 직장생활에서 기억을 못 해가지고 실제로 상사와 문제가 되었거나 아니면 은행을 보는데 그게 문제가 되었거나 그런 게 있으세요?",
    "친구와 약속 같은 거는 깜빡하시거나 그런 적은 있으세요?",
    "조금은 있으신데 약간 애매하시다. 알겠습니다.",
    "가족 중에서 혹시 치매 환자가 있으세요?",
    "몇 살 때 정도 그러셨어요?",
    "환자분은 당뇨, 고혈압, 협심증 같은 혈관성 질환이 있으세요?",
    "그거 말고는 뭐 우울증이나 진통제나 다른 약 같은 거 드시는 건 없으세요?",
    "몸이 많이 피곤하거나 최근에 최근에 몸무게가 많이 찌거나 아니면 갑상선 질환 같은 것도 없으시고요?",
    "최근에 우울함이 많이 심하거나 의욕이 없거나 그런 게 있으세요?",
    "혹시 환각 같은 헛개비가 보이거나 엉뚱한 행동을 하거나 이상한 소리 한 번씩 하시는 건 없으세요?",
    "손이 떨리거나 몸이 뻣뻣하거나 느려지는 건 없으세요?",
    "걸음걸이가 종종 걸었거나 불편하시거나 아니면 소변 조절이 잘 안 되시는 불편함은 없으세요?",
    "평소에 술 많이 드시는 편이세요?",
    "혹시 머리를 다치시거나 뇌염 같은 거 뇌질환 앓은 적은 없으세요?",
    "알겠습니다. 그러면 제가 이제 신체 진찰을 하도록 하겠습니다.",
    "이제 진찰은 끝났고요 혹시 걱정되는 거 있으세요?",
    "음 일단은 가족력이 있으시고 또 기억력이 떨어지시는 것 때문에 치매의 가장 흔한 유형인 알츠하이머 치매 가능성을 고려하긴 해야 될 것 같습니다",
    "하지만 그 극심한 스트레스 때문에 최근에 우울증 증상이 좀 있어 보이셔서 우울증에 의한 가성 침해의 가능성도 고려해야 합니다.",
    "또 조금 가능성이 높지는 않지만 고혈압이 있고 고혈압 조절이 안되시는 것으로 봐서는 혈관성 치매 가능성도 함께 고려해야 할 것 같습니다.",
    "일단 피검사를 할 거고요. 그리고 인지기능 검사 자세하게 하고 뇌 MRI를 찍어서 정확한 이유와 원인을 확인하도록 하겠습니다.",
    "경도인지장애와 치매는 둘 다 인지기능, 기억력이나 언어능력이나 판단력 같은 인지기능이 떨어지는 것은 맞는데",
    "치매는 그로 인한 인질기능 저하 때문에 일상생활에 장애가 있는 것을 치매라고 말하고요. 일상생활에 장애가 없는 단계를 경도인지저하라고 치매 전 단계 정도로 생각하고 있습니다.",
    "혹시 또 다른 궁금한 거 있으세요?",
    "네. 그러면 검사 후에 뵙도록 하겠습니다. 조심해서 가세요."
]

async def main():
    """메인 실행 함수"""
    tester = CPXScriptTester("script_test_001")
    
    print("🏥 CPX 스크립트 테스터")
    print("=" * 50)
    
    # 시나리오 선택
    print("1. 흉통 케이스")
    print("2. 복통 케이스") 
    print("3. 신경과 치매 케이스 (스크립트 준비됨)")
    
    choice = input("\n시나리오 번호를 선택하세요 (1-3): ").strip()
    
    if choice == "3":
        # 신경과 케이스 자동 실행
        await tester.run_script_test("3", NEUROLOGY_DOCTOR_SCRIPT)
    else:
        # 수동 테스트 모드
        await tester.connect()
        await tester.select_scenario(choice)
        
        print("\n📝 텍스트 입력 모드")
        print("종료하려면 'quit' 입력")
        print("-" * 30)
        
        while True:
            text = input("\n의사 발언: ").strip()
            if text.lower() == 'quit':
                break
                
            ended = await tester.send_text(text)
            if ended:
                break
        
        await tester.websocket.close()

if __name__ == "__main__":
    asyncio.run(main())