#!/usr/bin/env python3
"""
SER이 완료된 더미 대화 데이터를 이용한 Evaluation Service 테스트 클라이언트

이 테스트는:
1. SER 분석이 완료된 더미 대화 데이터를 생성
2. Evaluation Service에 직접 전달하여 평가 수행
3. RAG 가이드라인 기반 평가 결과 확인
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# 환경변수 로드
from dotenv import load_dotenv
load_dotenv()

# Backend 디렉토리를 Python path에 추가
sys.path.append(str(Path(__file__).parent))

from services.evaluation_service import EvaluationService

class DummyDataGenerator:
    """SER이 완료된 더미 대화 데이터 생성기"""
    
    def __init__(self):
        self.emotion_labels = ["Anxious", "Dry", "Kind"]
        
    def generate_memory_loss_conversation(self) -> list:
        """기억력 저하 관련 CPX 대화 시나리오 생성"""
        conversations = [
            # 1. 인사 및 시작
            {
                "text": "안녕하세요, 어떤 증상으로 오셨나요?",
                "speaker_role": "doctor",
                "emotion_analysis": {
                    "predicted_emotion": "Kind",
                    "confidence": 0.85,
                    "emotion_scores": {"Anxious": 0.1, "Dry": 0.05, "Kind": 0.85}
                }
            },
            {
                "text": "요즘 기억력이 많이 떨어져서 걱정이 돼요.",
                "speaker_role": "patient",
                "emotion_analysis": None  # 환자는 SER 분석 안함
            },
            
            # 2. O (Onset) - 발병 시기
            {
                "text": "언제부터 기억력 저하를 느끼셨나요?",
                "speaker_role": "doctor",
                "emotion_analysis": {
                    "predicted_emotion": "Kind",
                    "confidence": 0.78,
                    "emotion_scores": {"Anxious": 0.12, "Dry": 0.1, "Kind": 0.78}
                }
            },
            {
                "text": "한 6개월 전부터 서서히 시작된 것 같아요.",
                "speaker_role": "patient",
                "emotion_analysis": None
            },
            
            # 3. L (Location) - 해당없음 확인
            {
                "text": "특정 부위의 통증은 없으시죠?",
                "speaker_role": "doctor",
                "emotion_analysis": {
                    "predicted_emotion": "Dry",
                    "confidence": 0.65,
                    "emotion_scores": {"Anxious": 0.15, "Dry": 0.65, "Kind": 0.2}
                }
            },
            {
                "text": "네, 특별한 통증은 없어요.",
                "speaker_role": "patient",
                "emotion_analysis": None
            },
            
            # 4. D (Duration) - 지속 시간
            {
                "text": "증상이 하루 종일 지속되나요, 아니면 특정 시간에만 나타나나요?",
                "speaker_role": "doctor",
                "emotion_analysis": {
                    "predicted_emotion": "Kind",
                    "confidence": 0.72,
                    "emotion_scores": {"Anxious": 0.08, "Dry": 0.2, "Kind": 0.72}
                }
            },
            {
                "text": "하루 종일 그런 것 같아요. 특히 새로운 것을 기억하기 어려워요.",
                "speaker_role": "patient",
                "emotion_analysis": None
            },
            
            # 5. Co (Course) - 경과
            {
                "text": "증상이 점점 심해지고 있나요?",
                "speaker_role": "doctor",
                "emotion_analysis": {
                    "predicted_emotion": "Kind",
                    "confidence": 0.80,
                    "emotion_scores": {"Anxious": 0.05, "Dry": 0.15, "Kind": 0.80}
                }
            },
            {
                "text": "네, 처음보다는 더 심해진 것 같아요.",
                "speaker_role": "patient",
                "emotion_analysis": None
            },
            
            # 6. Ex (Exacerbating factors) - 악화 요인
            {
                "text": "스트레스를 받거나 피곤할 때 더 심해지나요?",
                "speaker_role": "doctor",
                "emotion_analysis": {
                    "predicted_emotion": "Kind",
                    "confidence": 0.75,
                    "emotion_scores": {"Anxious": 0.1, "Dry": 0.15, "Kind": 0.75}
                }
            },
            {
                "text": "맞아요, 스트레스 받을 때 더 심해지는 것 같아요.",
                "speaker_role": "patient",
                "emotion_analysis": None
            },
            
            # 7. C (Character) - 특성
            {
                "text": "구체적으로 어떤 종류의 기억이 어려우신가요? 최근 일들인가요, 오래된 일들인가요?",
                "speaker_role": "doctor",
                "emotion_analysis": {
                    "predicted_emotion": "Kind",
                    "confidence": 0.82,
                    "emotion_scores": {"Anxious": 0.03, "Dry": 0.15, "Kind": 0.82}
                }
            },
            {
                "text": "주로 최근 일들이에요. 어제 뭘 했는지도 잘 기억이 안 나요.",
                "speaker_role": "patient",
                "emotion_analysis": None
            },
            
            # 8. A (Associated symptoms) - 동반 증상
            {
                "text": "다른 증상도 있나요? 두통이나 어지럼증, 수면 문제는 어떠세요?",
                "speaker_role": "doctor",
                "emotion_analysis": {
                    "predicted_emotion": "Kind",
                    "confidence": 0.77,
                    "emotion_scores": {"Anxious": 0.08, "Dry": 0.15, "Kind": 0.77}
                }
            },
            {
                "text": "잠을 잘 못 자고, 가끔 두통도 있어요.",
                "speaker_role": "patient",
                "emotion_analysis": None
            },
            
            # 9. 과거력 확인
            {
                "text": "과거에 머리 다친 적이나 뇌 관련 질환을 앓은 적이 있나요?",
                "speaker_role": "doctor",
                "emotion_analysis": {
                    "predicted_emotion": "Dry",
                    "confidence": 0.68,
                    "emotion_scores": {"Anxious": 0.12, "Dry": 0.68, "Kind": 0.2}
                }
            },
            {
                "text": "특별한 건 없었던 것 같아요.",
                "speaker_role": "patient",
                "emotion_analysis": None
            },
            
            # 10. 신체 진찰 설명
            {
                "text": "이제 간단한 신체 검사를 해보겠습니다. 괜찮으시죠?",
                "speaker_role": "doctor",
                "emotion_analysis": {
                    "predicted_emotion": "Kind",
                    "confidence": 0.88,
                    "emotion_scores": {"Anxious": 0.02, "Dry": 0.1, "Kind": 0.88}
                }
            },
            {
                "text": "네, 괜찮습니다.",
                "speaker_role": "patient",
                "emotion_analysis": None
            },
            
            # 11. 환자 교육 및 공감
            {
                "text": "기억력 저하로 많이 걱정이 되셨을 텐데, 정확한 진단을 위해 추가 검사가 필요할 것 같습니다.",
                "speaker_role": "doctor",
                "emotion_analysis": {
                    "predicted_emotion": "Kind",
                    "confidence": 0.92,
                    "emotion_scores": {"Anxious": 0.01, "Dry": 0.07, "Kind": 0.92}
                }
            },
            {
                "text": "네, 알겠습니다. 검사는 언제 받을 수 있나요?",
                "speaker_role": "patient",
                "emotion_analysis": None
            }
        ]
        
        # 타임스탬프와 오디오 경로 추가
        for i, conv in enumerate(conversations):
            conv["timestamp"] = datetime.now().isoformat()
            conv["audio_file_path"] = f"dummy_audio_{i:02d}.wav"
            
        return conversations

class EvaluationTestClient:
    """Evaluation Service 테스트 클라이언트"""
    
    def __init__(self):
        self.evaluation_service = None
        
    async def initialize(self):
        """서비스 초기화"""
        print("🔧 Evaluation Service 초기화 중...")
        self.evaluation_service = EvaluationService()
        print("✅ 초기화 완료!")
        
    async def test_conversation_evaluation(self, conversation_data: list):
        """대화 데이터를 이용한 평가 테스트"""
        session_id = f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        user_id = "1"
        
        print(f"\n🎯 테스트 세션 시작: {session_id}")
        print(f"📊 총 대화 수: {len(conversation_data)}개")
        
        # 세션 시작 (실제 메서드 시그니처에 맞게 수정)
        actual_session_id = await self.evaluation_service.start_evaluation_session(
            user_id=user_id,
            scenario_id="memory_impairment",
            result_id=1  # 테스트용 result_id 설정
        )
        print(f"📋 실제 세션 ID: {actual_session_id}")
        
        # 대화 데이터 순차 입력
        print("\n📝 대화 데이터 입력 중...")
        for i, conv in enumerate(conversation_data, 1):
            result = await self.evaluation_service.add_conversation_entry(
                session_id=actual_session_id,  # 실제 생성된 세션 ID 사용
                audio_file_path=conv["audio_file_path"],
                text=conv["text"],
                speaker_role=conv["speaker_role"],
                emotion_analysis=conv["emotion_analysis"]
            )
            
            emotion_info = ""
            if conv["emotion_analysis"]:
                emotion = conv["emotion_analysis"]["predicted_emotion"]
                confidence = conv["emotion_analysis"]["confidence"]
                emotion_info = f" [감정: {emotion} ({confidence:.2f})]"
            
            print(f"  {i:2d}. [{conv['speaker_role']:7s}] {conv['text'][:50]}...{emotion_info}")
            
            # 짧은 대기 (실제 대화 시뮬레이션)
            await asyncio.sleep(0.1)
        
        # 세션 종료 및 최종 평가
        print(f"\n🏁 세션 종료 및 최종 평가 수행...")
        final_result = await self.evaluation_service.end_evaluation_session(actual_session_id)
        
        return final_result
    
    def print_evaluation_results(self, results: dict):
        """평가 결과 출력"""
        print(f"\n" + "="*60)
        print(f"📋 최종 평가 결과")
        print(f"="*60)
        
        # 기본 정보
        print(f"🆔 세션 ID: {results.get('session_id', 'N/A')}")
        print(f"👤 사용자 ID: {results.get('user_id', 'N/A')}")
        print(f"📅 평가 시간: {results.get('evaluation_time', 'N/A')}")
        
        # LangGraph 분석 결과
        if 'langgraph_analysis' in results:
            lg_analysis = results['langgraph_analysis']
            print(f"\n🧠 LangGraph 분석:")
            print(f"  • 총점: {lg_analysis.get('total_score', 'N/A')}")
            print(f"  • 완성도: {lg_analysis.get('completeness_percentage', 'N/A')}%")
            
            if 'detailed_scores' in lg_analysis:
                print(f"  • 세부 점수:")
                for category, score in lg_analysis['detailed_scores'].items():
                    print(f"    - {category}: {score}")
        
        # RAG 가이드라인 분석 결과  
        if 'rag_analysis' in results:
            rag_analysis = results['rag_analysis']
            print(f"\n🔍 RAG 가이드라인 분석:")
            print(f"  • 커버된 항목: {rag_analysis.get('covered_items', 0)}개")
            print(f"  • 누락된 항목: {rag_analysis.get('missing_items', 0)}개")
            print(f"  • 커버리지: {rag_analysis.get('coverage_percentage', 'N/A')}%")
            
            if 'missing_categories' in rag_analysis and rag_analysis['missing_categories']:
                print(f"  • 누락된 카테고리:")
                for category in rag_analysis['missing_categories'][:5]:  # 최대 5개만
                    print(f"    - {category}")
        
        # 감정 분석 통계
        if 'emotion_statistics' in results:
            emotion_stats = results['emotion_statistics']
            print(f"\n🎭 감정 분석 통계:")
            for emotion, count in emotion_stats.items():
                print(f"  • {emotion}: {count}회")
        
        # 추천사항
        if 'recommendations' in results and results['recommendations']:
            print(f"\n💡 개선 추천사항:")
            for i, rec in enumerate(results['recommendations'][:3], 1):
                print(f"  {i}. {rec}")
        
        print(f"\n" + "="*60)

async def main():
    """메인 테스트 함수"""
    print("🏥 CPX Evaluation Service 테스트 클라이언트")
    print("=" * 60)
    
    try:
        # 1. 더미 데이터 생성
        print("📋 더미 대화 데이터 생성 중...")
        dummy_generator = DummyDataGenerator()
        conversation_data = dummy_generator.generate_memory_loss_conversation()
        print(f"✅ {len(conversation_data)}개 대화 데이터 생성 완료")
        
        # 2. 테스트 클라이언트 초기화
        test_client = EvaluationTestClient()
        await test_client.initialize()
        
        # 3. 평가 테스트 실행
        results = await test_client.test_conversation_evaluation(conversation_data)
        
        # 4. 결과 출력
        test_client.print_evaluation_results(results)
        
        print(f"💾 평가 결과는 evaluation_results/ 디렉토리에 자동 저장되었습니다.")
        
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
