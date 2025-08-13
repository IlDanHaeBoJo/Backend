#!/usr/bin/env python3
"""
CPX 평가 시스템 테스트 클라이언트
EvaluationService를 직접 테스트하는 클라이언트
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict

# 환경변수 로드
from dotenv import load_dotenv
load_dotenv()

# 테스트용 데이터베이스 설정
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test.db")

from services.evaluation_service import EvaluationService

class EvaluationTester:
    def __init__(self):
        """평가 테스터 초기화"""
        self.evaluation_service = EvaluationService()
        print("✅ EvaluationService 초기화 완료")
    
    def create_dummy_conversation(self) -> List[Dict]:
        """더미 대화 데이터 생성 (신경과 치매 케이스)"""
        return self._create_neurology_conversation()
    
    def _create_neurology_conversation(self) -> List[Dict]:
        """신경과 치매 케이스 더미 데이터"""
        base_time = datetime.now()
        conversation = []
        
        # 의사-환자 대화 쌍들 (test_script_client.py의 NEUROLOGY_DOCTOR_SCRIPT 기반)
        dialogues = [
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
            ("네. 그러면 검사 후에 뵙도록 하겠습니다. 조심해서 가세요.", "네, 감사합니다."),
        ]
        
        # 감정 더미 데이터 (의사 발언에만 추가)
        emotions = [
            {"predicted_emotion": "Kind", "confidence": 0.85, "emotion_scores": {"Kind": 0.85, "Anxious": 0.10, "Dry": 0.05}},
            {"predicted_emotion": "Kind", "confidence": 0.78, "emotion_scores": {"Kind": 0.78, "Anxious": 0.15, "Dry": 0.07}},
            {"predicted_emotion": "Kind", "confidence": 0.82, "emotion_scores": {"Kind": 0.82, "Anxious": 0.12, "Dry": 0.06}},
            {"predicted_emotion": "Kind", "confidence": 0.89, "emotion_scores": {"Kind": 0.89, "Anxious": 0.08, "Dry": 0.03}},
            {"predicted_emotion": "Kind", "confidence": 0.76, "emotion_scores": {"Kind": 0.76, "Anxious": 0.18, "Dry": 0.06}},
        ]
        
        emotion_idx = 0
        for i, (doctor_msg, patient_msg) in enumerate(dialogues):
            # 의사 발언 (student)
            conversation.append({
                "role": "student",
                "content": doctor_msg,
                "timestamp": (base_time + timedelta(minutes=i*2)).isoformat(),
                "emotion": emotions[emotion_idx % len(emotions)]
            })
            
            # 환자 발언 (patient)  
            conversation.append({
                "role": "patient", 
                "content": patient_msg,
                "timestamp": (base_time + timedelta(minutes=i*2 + 1)).isoformat(),
                "emotion": None  # 환자 발언에는 감정 분석 없음
            })
            
            emotion_idx += 1
        
        return conversation
    

    
    async def test_evaluation(self, user_id="test_user_001"):
        """평가 테스트 실행"""
        print(f"\n🏥 CPX 평가 테스트 시작 - 신경과 치매 케이스")
        print("=" * 60)
        
        # 더미 대화 데이터 생성
        conversation_log = self.create_dummy_conversation()
        print(f"📋 생성된 대화 수: {len(conversation_log)}개")
        print(f"📋 학생 질문 수: {len([msg for msg in conversation_log if msg['role'] == 'student'])}개")
        
        # 평가 실행
        print("\n🚀 평가 시작...")
        start_time = datetime.now()
        
        try:
            result = await self.evaluation_service.evaluate_conversation(
                user_id=user_id,
                scenario_id="3",
                conversation_log=conversation_log
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print(f"✅ 평가 완료! (소요시간: {duration:.2f}초)")
            
            # 결과 출력
            self._print_evaluation_results(result)
            
            return result
            
        except Exception as e:
            print(f"❌ 평가 실패: {e}")
            return None
    
    def _print_evaluation_results(self, result: Dict):
        """평가 결과 출력 (Multi-Step 평가 시스템 대응)"""
        print("\n" + "=" * 60)
        print("📊 평가 결과")
        print("=" * 60)
        
        if "error" in result:
            print(f"❌ 오류: {result['error']}")
            return
        
        # Multi-Step 점수 정보
        scores = result.get("scores", {})
        print(f"\n🎯 최종 점수: {scores.get('total_score', 0)}점 ({scores.get('grade', 'F')})")
        
        # Multi-Step 가중치 세부 점수
        weighted_breakdown = scores.get("weighted_breakdown", {})
        if weighted_breakdown:
            print(f"   📊 세부 점수 (가중치 적용):")
            print(f"      - 완성도: {weighted_breakdown.get('completeness_score', 0)}점 (40%)")
            print(f"      - 품질: {weighted_breakdown.get('quality_score', 0)}점 (30%)")  
            print(f"      - 적합성: {weighted_breakdown.get('appropriateness_score', 0)}점 (20%)")
            print(f"      - 의도: {weighted_breakdown.get('intent_score', 0)}점 (10%)")
        
        # 대화 요약 정보
        conversation_summary = result.get("conversation_summary", {})
        print(f"\n❓ 대화 분석:")
        print(f"   - 총 질문 수: {conversation_summary.get('total_questions', 0)}개")
        print(f"   - 대화 시간: {conversation_summary.get('duration_minutes', 0):.1f}분")
        

        
        # 상세 분석 결과
        detailed_analysis = result.get("detailed_analysis", {})
        if detailed_analysis:
            print(f"\n🧠 상세 분석:")
            
            # 완성도 분석
            completeness = detailed_analysis.get("completeness", {})
            if completeness:
                overall_score = completeness.get("overall_completeness_score", 0)
                print(f"   - 의학적 완성도: {overall_score}/10점")
            
            # 품질 분석
            quality = detailed_analysis.get("quality", {})
            if quality:
                overall_quality = quality.get("overall_quality_score", 0)
                print(f"   - 질문 품질: {overall_quality}/10점")
                
            # 적합성 분석
            appropriateness = detailed_analysis.get("appropriateness", {})
            if appropriateness:
                overall_appropriate = appropriateness.get("overall_appropriateness_score", 0)
                print(f"   - 시나리오 적합성: {overall_appropriate}/10점")
        
        # 체크리스트 결과 (상위 3개만)
        checklist_results = result.get("checklist_results", {})
        if checklist_results:
            print(f"\n📋 체크리스트 결과 (상위 3개):")
            sorted_checklist = sorted(checklist_results.items(), 
                                    key=lambda x: x[1].get('completion_rate', 0), 
                                    reverse=True)[:3]
            for category, data in sorted_checklist:
                rate = data.get('completion_rate', 0)
                print(f"   - {category}: {rate:.1%} 완료")
        
        # Multi-Step 피드백
        feedback = result.get("feedback", {})
        if feedback:
            print(f"\n📝 피드백:")
            print(f"   전체: {feedback.get('overall_feedback', '')}")
            
            strengths = feedback.get("strengths", [])
            if strengths:
                print(f"   강점:")
                for strength in strengths[:3]:
                    print(f"     • {strength}")
            
            weaknesses = feedback.get("weaknesses", [])
            if weaknesses:
                print(f"   개선점:")
                for weakness in weaknesses[:3]:
                    print(f"     • {weakness}")
                    
            medical_insights = feedback.get("medical_insights", [])
            if medical_insights:
                print(f"   의학적 통찰:")
                for insight in medical_insights[:2]:
                    print(f"     • {insight}")
        
        # 평가 방법 정보
        evaluation_method = result.get("evaluation_method", "")
        if evaluation_method:
            print(f"\n🔬 평가 방법: {evaluation_method}")
        
        # 시스템 정보
        system_info = result.get("system_info", {})
        if system_info:
            print(f"\n⚙️ 시스템 정보:")
            print(f"   - 버전: {system_info.get('version', 'Unknown')}")
            print(f"   - 평가 단계: {system_info.get('evaluation_steps', 0)}단계")
    
    def save_test_results(self, result: Dict, filename: str = None):
        """테스트 결과를 JSON 파일로 저장"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_evaluation_result_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n💾 결과 저장 완료: {filename}")
        except Exception as e:
            print(f"❌ 결과 저장 실패: {e}")

async def main():
    """메인 실행 함수"""
    tester = EvaluationTester()
    
    print("🏥 CPX 평가 시스템 테스터")
    print("=" * 50)
    print("신경과 치매 케이스 평가 테스트")
    
    # 평가 실행
    result = await tester.test_evaluation()
    
    if result:
        save_choice = input("\n결과를 파일로 저장하시겠습니까? (y/n): ").strip().lower()
        if save_choice == 'y':
            tester.save_test_results(result)

if __name__ == "__main__":
    asyncio.run(main())

