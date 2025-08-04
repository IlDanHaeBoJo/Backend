from typing import Dict
from datetime import datetime

class EvaluationService:
    def __init__(self):
        """CPX 평가 서비스 초기화"""
        self.evaluation_criteria = {
            "communication": {"name": "의사소통 능력", "weight": 0.3},
            "history_taking": {"name": "병력 청취", "weight": 0.4},
            "clinical_reasoning": {"name": "임상적 추론", "weight": 0.2},
            "professionalism": {"name": "전문가 태도", "weight": 0.1}
        }
        
        self.session_data = {}  # 세션별 평가 데이터

    async def start_evaluation_session(self, student_id: str, case_id: str) -> str:
        """평가 세션 시작"""
        session_id = f"{student_id}_{case_id}_{int(datetime.now().timestamp())}"
        
        self.session_data[session_id] = {
            "student_id": student_id,
            "case_id": case_id,
            "start_time": datetime.now(),
            "interactions": [],
            "status": "active"
        }
        
        return session_id

    async def record_interaction(self, session_id: str, student_question: str, patient_response: str, interaction_type: str = "question"):
        """학생-환자 상호작용 기록"""
        if session_id not in self.session_data:
            return
        
        interaction = {
            "timestamp": datetime.now(),
            "type": interaction_type,
            "student_question": student_question,
            "patient_response": patient_response,
            "analysis": self._simple_analysis(student_question)
        }
        
        self.session_data[session_id]["interactions"].append(interaction)

    def _simple_analysis(self, question: str) -> Dict:
        """간단한 질문 분석"""
        score = 5.0  # 기본 점수
        
        # 긍정적 요소
        if "안녕하세요" in question or "감사" in question:
            score += 1.0
        if any(word in question for word in ["언제", "어떤", "어디", "어떻게"]):
            score += 1.0
        if "?" in question:
            score += 0.5
            
        return {
            "communication_score": min(10.0, score),
            "question_type": "개방형" if any(w in question for w in ["언제", "어떤", "어디"]) else "폐쇄형"
        }

    async def end_evaluation_session(self, session_id: str) -> Dict:
        """평가 세션 종료"""
        if session_id not in self.session_data:
            return {"error": "세션을 찾을 수 없습니다"}
        
        session = self.session_data[session_id]
        session["end_time"] = datetime.now()
        session["status"] = "completed"
        
        # 간단한 점수 계산
        scores = self._calculate_scores(session)
        
        return {
            "session_id": session_id,
            "scores": scores,
            "total_interactions": len(session["interactions"]),
            "duration": (session["end_time"] - session["start_time"]).total_seconds() / 60
        }

    def _calculate_scores(self, session: Dict) -> Dict:
        """점수 계산"""
        interactions = session["interactions"]
        
        if not interactions:
            return {category: 5.0 for category in self.evaluation_criteria.keys()}
        
        # 평균 의사소통 점수
        avg_comm = sum(i["analysis"]["communication_score"] for i in interactions) / len(interactions)
        
        scores = {
            "communication": round(avg_comm, 1),
            "history_taking": round(min(10.0, len(interactions) * 1.5), 1),  # 질문 개수 기반
            "clinical_reasoning": round(avg_comm * 0.9, 1),  # 의사소통 기반
            "professionalism": round(avg_comm * 1.1, 1)  # 의사소통 기반
        }
        
        # 가중 평균 총점
        total = sum(scores[cat] * self.evaluation_criteria[cat]["weight"] for cat in scores)
        scores["total"] = round(total, 1)
        
        return scores

    def get_session_summary(self, student_id: str) -> list:
        """학생의 세션 요약"""
        return [
            {
                "session_id": sid,
                "case_id": data["case_id"],
                "date": data["start_time"].strftime("%Y-%m-%d %H:%M"),
                "status": data["status"]
            }
            for sid, data in self.session_data.items()
            if data["student_id"] == student_id
        ]