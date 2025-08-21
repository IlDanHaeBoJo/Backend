import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any

from services.evaluation_service_clean import EvaluationService


class DummyDataGenerator:
    """실제 CPX 대화 기반 더미 데이터 생성기"""
    
    def __init__(self):
        self.evaluation_service = EvaluationService()
        
    def generate_memory_loss_conversation(self) -> list:
        """실제 의사-환자 대화 쌍 (test_script_client 기반, SER 분석 완료)"""
        
        # 실제 CPX 대화 쌍 데이터
        conversation_pairs = [
            ("안녕하세요. 저는 신경과라고 합니다.", "네, 안녕하세요."),
            ("환자분 성함과 나이, 등록번호가 어떻게 되세요?", "김영희, 75세입니다. 등록번호는 1234567입니다."),
            ("네, 어떻게 오셨어요?", "요즘 기억력이 많이 떨어지는 것 같아서 왔습니다."),
            ("기억력이 떨어지시는 것 같다. 그러면 그게 언제부터 그러시는 거죠?", "한 6개월 전부터 심해진 것 같아요."),
            ("그러면 어떻게 깜빡하시고 기억력이 떨어지시는지 편하게 얘기 한번 해보시겠어요?", "방금 뭘 하려고 했는지 까먹고, 물건을 어디 뒀는지 자꾸 잊어버려요."),
            ("그러면 그게 이제 비교적 최근의 일들을 잘 까먹으시나요? 아니면 옛날 일도 많이 잊어버리시나요?", "최근 일들을 더 많이 까먹는 것 같아요. 옛날 일은 기억나는데."),
            ("그러면 혹시 최근에 성격 변화 같은 건 없으세요? 급해지거나 짜증이 나거나 많이 다투시거나 그런 건?", "예전보다 짜증이 많이 나는 것 같아요."),
            ("물건 이름이나 단어 같은 거 생각이 잘 안나거나 말하는 게 좀 어둔하거나 그런건 없으세요?", "네, 가끔 단어가 생각이 안 날 때가 있어요."),
            ("최근에 길을 잃어버렸다거나 다니던 길인데 잘 모르겠다거나 그런 적은 없으세요?", "아직까지는 길을 잃어버린 적은 없어요."),
            ("계산은 잘 하세요? 예를 들면 물건 살 때 돈 내는 거 그런 거?", "계산은 아직 괜찮은 것 같아요."),
            ("알겠습니다. 일단은 기억력도 떨어지시는 것 같은데 일상생활을 할 때 그 기억력 때문에 문제가 되거나 그런 일상생활에 장애가 있으세요?", "가끔 요리할 때 가스불 끄는 걸 깜빡해서 걱정이에요."),
            ("예를 들면 직장생활에서 기억을 못 해가지고 실제로 상사와 문제가 되었거나 아니면 은행을 보는데 그게 문제가 되었거나 그런 게 있으세요?", "은퇴해서 직장은 안 다니지만, 은행 업무는 아직 괜찮아요."),
            ("친구와 약속 같은 거는 깜빡하시거나 그런 적은 있으세요?", "네, 가끔 약속을 깜빡할 때가 있어요."),
            ("조금은 있으신데 약간 애매하시다. 알겠습니다.", "네."),
            ("가족 중에서 혹시 치매 환자가 있으세요?", "어머니께서 치매를 앓으셨어요."),
            ("몇 살 때 정도 그러셨어요?", "80세쯤부터 시작되셨던 것 같아요."),
            ("환자분은 당뇨, 고혈압, 협심증 같은 혈관성 질환이 있으세요?", "고혈압이 있어서 약을 먹고 있어요."),
            ("그거 말고는 뭐 우울증이나 진통제나 다른 약 같은 거 드시는 건 없으세요?", "우울증약은 안 먹고 있어요."),
            ("몸이 많이 피곤하거나 최근에 최근에 몸무게가 많이 찌거나 아니면 갑상선 질환 같은 것도 없으시고요?", "몸무게는 비슷하고 갑상선은 괜찮아요."),
            ("최근에 우울함이 많이 심하거나 의욕이 없거나 그런 게 있으세요?", "가끔 우울할 때가 있어요."),
            ("혹시 환각 같은 헛개비가 보이거나 엉뚱한 행동을 하거나 이상한 소리 한 번씩 하시는 건 없으세요?", "그런 건 없어요."),
            ("손이 떨리거나 몸이 뻣뻣하거나 느려지는 건 없으세요?", "그런 건 없어요."),
            ("걸음걸이가 종종 걸었거나 불편하시거나 아니면 소변 조절이 잘 안 되시는 불편함은 없으세요?", "아직까지는 괜찮아요."),
            ("평소에 술 많이 드시는 편이세요?", "거의 안 마셔요."),
            ("혹시 머리를 다치시거나 뇌염 같은 거 뇌질환 앓은 적은 없으세요?", "없어요."),
            ("알겠습니다. 그러면 제가 이제 신체 진찰을 하도록 하겠습니다.", "네."),
            ("이제 진찰은 끝났고요 혹시 걱정되는 거 있으세요?", "치매가 맞는 건가요?"),
            ("음 일단은 가족력이 있으시고 또 기억력이 떨어지시는 것 때문에 치매의 가장 흔한 유형인 알츠하이머 치매 가능성을 고려하긴 해야 될 것 같습니다", "네."),
            ("하지만 그 극심한 스트레스 때문에 최근에 우울증 증상이 좀 있어 보이셔서 우울증에 의한 가성 침해의 가능성도 고려해야 합니다.", "네."),
            ("또 조금 가능성이 높지는 않지만 고혈압이 있고 고혈압 조절이 안되시는 것으로 봐서는 혈관성 치매 가능성도 함께 고려해야 할 것 같습니다.", "네."),
            ("일단 피검사를 할 거고요. 그리고 인지기능 검사 자세하게 하고 뇌 MRI를 찍어서 정확한 이유와 원인을 확인하도록 하겠습니다.", "네, 알겠습니다."),
            ("경도인지장애와 치매는 둘 다 인지기능, 기억력이나 언어능력이나 판단력 같은 인지기능이 떨어지는 것은 맞는데", "네."),
            ("치매는 그로 인한 인질기능 저하 때문에 일상생활에 장애가 있는 것을 치매라고 말하고요. 일상생활에 장애가 없는 단계를 경도인지저하라고 치매 전 단계 정도로 생각하고 있습니다.", "네, 알겠습니다."),
            ("혹시 또 다른 궁금한 거 있으세요?", "없어요."),
            ("네. 그러면 검사 후에 뵙도록 하겠습니다. 조심해서 가세요.", "네, 감사합니다.")
        ]
        
        conversations = []
        
        # 실제 대화 쌍을 SER 분석 완료된 형태로 변환
        for i, (doctor_text, patient_text) in enumerate(conversation_pairs):
            # 의사 발언 (SER 감정 분석 포함)
            doctor_entry = {
                "content": doctor_text,
                "role": "doctor",
                "emotion_analysis": self._generate_doctor_emotion(i),
                "timestamp": datetime.now().isoformat(),
                "audio_file_path": f"dummy_doctor_{i:02d}.wav"
            }
            conversations.append(doctor_entry)
            
            # 환자 응답 (SER 분석 없음)
            patient_entry = {
                "content": patient_text,
                "role": "patient",
                "emotion_analysis": None,
                "timestamp": datetime.now().isoformat(),
                "audio_file_path": f"dummy_patient_{i:02d}.wav"
            }
            conversations.append(patient_entry)
            
        return conversations

    def _generate_doctor_emotion(self, index: int) -> dict:
        """의사 발언에 대한 SER 감정 분석 결과 생성"""
        
        # 실제 SER 분석과 유사한 패턴으로 생성
        emotion_patterns = [
            {"predicted_emotion": "Kind", "confidence": 0.87, "emotion_scores": {"Anxious": 0.05, "Dry": 0.08, "Kind": 0.87}},
            {"predicted_emotion": "Dry", "confidence": 0.72, "emotion_scores": {"Anxious": 0.15, "Dry": 0.72, "Kind": 0.13}},
            {"predicted_emotion": "Kind", "confidence": 0.84, "emotion_scores": {"Anxious": 0.08, "Dry": 0.08, "Kind": 0.84}},
            {"predicted_emotion": "Anxious", "confidence": 0.68, "emotion_scores": {"Anxious": 0.68, "Dry": 0.22, "Kind": 0.10}},
            {"predicted_emotion": "Kind", "confidence": 0.81, "emotion_scores": {"Anxious": 0.09, "Dry": 0.10, "Kind": 0.81}},
            {"predicted_emotion": "Dry", "confidence": 0.75, "emotion_scores": {"Anxious": 0.12, "Dry": 0.75, "Kind": 0.13}}
        ]
        
        return emotion_patterns[index % len(emotion_patterns)]

    async def run_evaluation_test(self):
        """더미 데이터로 평가 시스템 테스트"""
        
        # 1. 더미 대화 데이터 생성
        print("🎭 실제 CPX 대화 기반 더미 데이터 생성 중...")
        conversations = self.generate_memory_loss_conversation()
        print(f"✅ {len(conversations)}개 대화 항목 생성 완료")
        
        # 2. 평가 실행
        print("\n🔍 LangGraph 기반 평가 시스템 실행 중...")
        
        try:
            # 테스트 세션 생성
            session_id = await self.evaluation_service.start_evaluation_session(
                user_id="1",
                scenario_id="1"
            )
            
            # 대화 데이터를 세션에 추가
            for conversation in conversations:
                await self.evaluation_service.add_conversation_entry(
                    session_id=session_id,
                    audio_file_path="test_audio.wav",  # 테스트용 더미 파일
                    text=conversation.get("content", ""),
                    role=conversation.get("role", "doctor"),
                    emotion_analysis=None
                )
            
            # 종합 평가 실행
            evaluation_result = await self.evaluation_service.end_evaluation_session(session_id)
            
            print("✅ 평가 완료!")
            
            return evaluation_result
            
        except Exception as e:
            print(f"❌ 평가 실행 중 오류: {e}")
            return None

    def generate_conversation_summary(self, conversations: List[Dict]) -> Dict:
        """대화 요약 통계 생성"""
        
        doctor_messages = [c for c in conversations if c["role"] == "doctor"]
        patient_messages = [c for c in conversations if c["role"] == "patient"]
        
        # 감정 분석 통계
        emotion_stats = {"Kind": 0, "Dry": 0, "Anxious": 0}
        for msg in doctor_messages:
            if msg.get("emotion_analysis"):
                emotion = msg["emotion_analysis"]["predicted_emotion"]
                emotion_stats[emotion] += 1
        
        return {
            "total_exchanges": len(conversations) // 2,
            "doctor_messages": len(doctor_messages),
            "patient_messages": len(patient_messages),
            "emotion_distribution": emotion_stats,
            "conversation_duration": "약 15분 (추정)",
            "cpx_stages": {
                "history_taking": "1-24번 대화 (병력청취)",
                "physical_examination": "25번 대화 (신체진찰)",
                "patient_education": "26-32번 대화 (환자교육)"
            }
        }


async def main():
    """메인 테스트 실행"""
    
    print("🏥 CPX 평가 시스템 더미 데이터 테스트")
    print("=" * 50)
    
    generator = DummyDataGenerator()
    
    # 1. 더미 대화 생성
    conversations = generator.generate_memory_loss_conversation()
    
    # 2. 평가 시스템 테스트
    print("\n" + "=" * 50)
    evaluation_result = await generator.run_evaluation_test()
    
    if evaluation_result:
        print("\n🎯 테스트 성공! 실제 CPX 대화 기반 평가 시스템이 정상 작동합니다.")
        
        # 결과 파일 저장
        result_filename = f"cpx_demo_evaluation_{int(datetime.now().timestamp())}.json"
        with open(f"evaluation_results/{result_filename}", "w", encoding="utf-8") as f:
            json.dump({
                "conversations": conversations,
                "evaluation_result": evaluation_result
            }, f, indent=2, ensure_ascii=False)
        
        print(f"📁 결과 저장: evaluation_results/{result_filename}")
    else:
        print("\n❌ 테스트 실패")


if __name__ == "__main__":
    asyncio.run(main())