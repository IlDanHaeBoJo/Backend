import os
import logging
from typing import List, Dict, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class EvaluationService:
    def __init__(self):
        """CPX 평가 서비스 초기화"""
        self.evaluation_criteria = {
            "communication": {
                "name": "의사소통 능력",
                "weight": 0.3,
                "items": [
                    "적절한 인사 및 자기소개",
                    "환자와 눈 맞춤 및 경청 자세",
                    "공감적 반응 및 이해 표현",
                    "명확하고 이해하기 쉬운 질문"
                ]
            },
            "history_taking": {
                "name": "병력 청취",
                "weight": 0.4,
                "items": [
                    "주증상 상세 파악",
                    "현병력 체계적 질문",
                    "과거병력 및 가족력 확인",
                    "사회력 및 생활습관 조사"
                ]
            },
            "clinical_reasoning": {
                "name": "임상적 추론",
                "weight": 0.2,
                "items": [
                    "감별진단 고려",
                    "추가 검사 계획",
                    "논리적 사고 과정",
                    "우선순위 설정"
                ]
            },
            "professionalism": {
                "name": "전문가 태도",
                "weight": 0.1,
                "items": [
                    "환자 존중 및 배려",
                    "개인정보 보호 의식",
                    "시간 관리",
                    "정확한 의학 용어 사용"
                ]
            }
        }
        
        # 세션별 평가 데이터 저장
        self.session_data = {}
        
        logger.info("CPX 평가 서비스 초기화 완료")

    async def start_evaluation_session(self, student_id: str, case_id: str) -> str:
        """평가 세션 시작"""
        session_id = f"{student_id}_{case_id}_{int(datetime.now().timestamp())}"
        
        self.session_data[session_id] = {
            "student_id": student_id,
            "case_id": case_id,
            "start_time": datetime.now(),
            "interactions": [],
            "scores": {},
            "feedback": {},
            "status": "active"
        }
        
        logger.info(f"평가 세션 시작: {session_id}")
        return session_id

    async def record_interaction(
        self, 
        session_id: str, 
        student_question: str, 
        patient_response: str,
        interaction_type: str = "question"
    ):
        """학생-환자 상호작용 기록"""
        if session_id not in self.session_data:
            logger.warning(f"존재하지 않는 세션: {session_id}")
            return
        
        interaction = {
            "timestamp": datetime.now(),
            "type": interaction_type,
            "student_question": student_question,
            "patient_response": patient_response,
            "analysis": await self._analyze_interaction(student_question, interaction_type)
        }
        
        self.session_data[session_id]["interactions"].append(interaction)
        logger.info(f"상호작용 기록됨: {session_id}")

    async def _analyze_interaction(self, student_question: str, interaction_type: str) -> Dict:
        """개별 상호작용 분석"""
        analysis = {
            "question_type": self._classify_question_type(student_question),
            "communication_score": self._evaluate_communication(student_question),
            "clinical_relevance": self._evaluate_clinical_relevance(student_question),
            "suggestions": []
        }
        
        # 간단한 규칙 기반 분석 (추후 LLM으로 확장 가능)
        if analysis["question_type"] == "open_ended":
            analysis["suggestions"].append("개방형 질문을 잘 활용했습니다.")
        elif analysis["question_type"] == "closed_ended":
            analysis["suggestions"].append("적절한 폐쇄형 질문입니다.")
        
        return analysis

    def _classify_question_type(self, question: str) -> str:
        """질문 유형 분류"""
        # 간단한 규칙 기반 분류
        question_lower = question.lower()
        
        # 개방형 질문 패턴
        open_patterns = ["어떤", "어떻게", "왜", "언제부터", "어디서", "무엇이", "어떤 느낌"]
        if any(pattern in question_lower for pattern in open_patterns):
            return "open_ended"
        
        # 폐쇄형 질문 패턴  
        closed_patterns = ["있나요", "없나요", "아픈가요", "있습니까", "없습니까"]
        if any(pattern in question_lower for pattern in closed_patterns):
            return "closed_ended"
        
        return "unknown"

    def _evaluate_communication(self, question: str) -> float:
        """의사소통 점수 평가 (0-10)"""
        score = 5.0  # 기본 점수
        
        # 긍정적 요소
        if "안녕하세요" in question:
            score += 0.5
        if "괜찮으시다면" in question or "혹시" in question:
            score += 0.5
        if len(question) > 10:  # 충분히 구체적인 질문
            score += 0.5
            
        # 부정적 요소
        if question.count("?") > 2:  # 너무 많은 질문을 한번에
            score -= 0.5
        
        return min(10.0, max(0.0, score))

    def _evaluate_clinical_relevance(self, question: str) -> float:
        """임상적 관련성 점수 평가 (0-10)"""
        score = 5.0
        
        # 의학적 키워드가 있으면 점수 향상
        medical_keywords = ["통증", "증상", "언제부터", "어떤 상황", "약물", "병력", "가족력", "알레르기"]
        keyword_count = sum(1 for keyword in medical_keywords if keyword in question)
        score += keyword_count * 0.5
        
        return min(10.0, max(0.0, score))

    async def end_evaluation_session(self, session_id: str) -> Dict:
        """평가 세션 종료 및 최종 점수 계산"""
        if session_id not in self.session_data:
            return {"error": "존재하지 않는 세션입니다"}
        
        session = self.session_data[session_id]
        session["end_time"] = datetime.now()
        session["duration"] = (session["end_time"] - session["start_time"]).total_seconds()
        session["status"] = "completed"
        
        # 최종 점수 계산
        final_scores = await self._calculate_final_scores(session)
        session["scores"] = final_scores
        
        # 피드백 생성
        feedback = await self._generate_feedback(session)
        session["feedback"] = feedback
        
        logger.info(f"평가 세션 완료: {session_id}")
        return {
            "session_id": session_id,
            "scores": final_scores,
            "feedback": feedback,
            "duration": session["duration"],
            "total_interactions": len(session["interactions"])
        }

    async def _calculate_final_scores(self, session: Dict) -> Dict:
        """최종 점수 계산"""
        interactions = session["interactions"]
        
        if not interactions:
            return {category: 0 for category in self.evaluation_criteria.keys()}
        
        scores = {}
        
        # 각 평가 영역별 점수 계산
        for category, criteria in self.evaluation_criteria.items():
            category_score = 0
            
            if category == "communication":
                category_score = sum(
                    interaction["analysis"]["communication_score"] 
                    for interaction in interactions
                ) / len(interactions)
                
            elif category == "history_taking":
                category_score = sum(
                    interaction["analysis"]["clinical_relevance"] 
                    for interaction in interactions
                ) / len(interactions)
                
            elif category == "clinical_reasoning":
                # 질문의 다양성과 논리적 순서 평가
                question_types = [
                    interaction["analysis"]["question_type"] 
                    for interaction in interactions
                ]
                diversity_score = len(set(question_types)) / len(question_types) * 10
                category_score = diversity_score
                
            elif category == "professionalism":
                # 인사, 정중함, 전문용어 사용 등 평가
                politeness_count = sum(
                    1 for interaction in interactions 
                    if any(word in interaction["student_question"].lower() 
                          for word in ["안녕", "감사", "죄송", "괜찮"])
                )
                category_score = min(10, politeness_count / len(interactions) * 20)
            
            scores[category] = round(category_score, 2)
        
        # 가중 평균 계산
        weighted_score = sum(
            scores[category] * self.evaluation_criteria[category]["weight"]
            for category in scores.keys()
        )
        
        scores["total"] = round(weighted_score, 2)
        return scores

    async def _generate_feedback(self, session: Dict) -> Dict:
        """개인화된 피드백 생성"""
        scores = session["scores"]
        interactions = session["interactions"]
        
        feedback = {
            "summary": f"총 {len(interactions)}번의 상호작용으로 평가를 완료했습니다.",
            "strengths": [],
            "improvements": [],
            "detailed_feedback": {}
        }
        
        # 강점 분석
        for category, score in scores.items():
            if category == "total":
                continue
                
            if score >= 8:
                feedback["strengths"].append(
                    f"{self.evaluation_criteria[category]['name']} 영역에서 우수한 성과를 보였습니다."
                )
            elif score < 6:
                feedback["improvements"].append(
                    f"{self.evaluation_criteria[category]['name']} 영역에서 더 많은 연습이 필요합니다."
                )
        
        # 상세 피드백
        for category, criteria in self.evaluation_criteria.items():
            feedback["detailed_feedback"][category] = {
                "score": scores.get(category, 0),
                "max_score": 10,
                "suggestions": criteria["items"]
            }
        
        return feedback

    def get_session_summary(self, student_id: str) -> List[Dict]:
        """학생의 모든 세션 요약"""
        student_sessions = [
            {
                "session_id": sid,
                "case_id": data["case_id"],
                "date": data["start_time"].strftime("%Y-%m-%d %H:%M"),
                "total_score": data.get("scores", {}).get("total", 0),
                "status": data["status"]
            }
            for sid, data in self.session_data.items()
            if data["student_id"] == student_id
        ]
        
        return sorted(student_sessions, key=lambda x: x["date"], reverse=True)

    def export_session_data(self, session_id: str) -> Optional[Dict]:
        """세션 데이터 내보내기"""
        if session_id not in self.session_data:
            return None
        
        session = self.session_data[session_id].copy()
        
        # datetime 객체를 문자열로 변환
        session["start_time"] = session["start_time"].isoformat()
        if "end_time" in session:
            session["end_time"] = session["end_time"].isoformat()
        
        for interaction in session["interactions"]:
            interaction["timestamp"] = interaction["timestamp"].isoformat()
        
        return session 