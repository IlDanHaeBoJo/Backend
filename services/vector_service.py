from typing import List

class VectorService:
    def __init__(self):
        """벡터 서비스 초기화 - 의학 지식 및 평가 기준용"""
        # 의학 지식 데이터 (환자 케이스 정보 X)
        self.medical_knowledge = self._get_medical_knowledge()
        self.evaluation_criteria = self._get_evaluation_criteria()

    def _get_medical_knowledge(self) -> List[str]:
        """의학 지식 데이터베이스 (더미)"""
        return [
            """
흉통의 감별진단:
1. 심근경색: ST elevation, 심근효소 상승
2. 협심증: 운동 시 악화, 안정 시 호전
3. 심근염: 바이러스 감염 후, CRP 상승
4. 대동맥박리: 찢어지는 듯한 통증, CT 필요
5. 폐색전증: 호흡곤란, D-dimer 상승
            """,
            """
복통의 감별진단:
1. 담석증: 우상복부 통증, 지방 음식 후 악화
2. 충수염: 우하복부 통증, McBurney point
3. 췌장염: 상복부 통증, 등으로 방사
4. 위염: 상복부 통증, 속쓰림
5. 장폐색: 복부팽만, 구토
            """,
            """
흉통 환자 응급처치:
1. 활력징후 측정
2. 심전도 즉시 시행
3. 산소포화도 모니터링
4. 니트로글리세린 설하정
5. 아스피린 300mg 투여
            """
        ]

    def _get_evaluation_criteria(self) -> List[str]:
        """CPX 평가 기준 (더미)"""
        return [
            """
CPX 흉통 케이스 평가 체크리스트:

【의사소통 능력 (30점)】
- 적절한 인사와 자기소개 (5점)
- 환자의 불안감 공감 (10점)
- 설명의 명확성 (15점)

【병력 청취 (40점)】
- 주증상 상세 문진 (15점)
- 과거력, 가족력 확인 (10점)
- 위험인자 파악 (15점)

【신체검사 (20점)】
- 활력징후 측정 (10점)
- 심폐 청진 (10점)

【임상 추론 (10점)】
- 감별진단 제시 (5점)
- 추가 검사 계획 (5점)
            """,
            """
CPX 복통 케이스 평가 체크리스트:

【의사소통 능력 (30점)】
- 적절한 인사와 자기소개 (5점)
- 환자의 통증 공감 (10점)
- 설명의 명확성 (15점)

【병력 청취 (40점)】
- 통증 특성 문진 (15점)
- 음식 관련성 확인 (10점)
- 동반 증상 파악 (15점)

【신체검사 (20점)】
- 복부 진찰 (15점)
- 압통점 확인 (5점)

【임상 추론 (10점)】
- 감별진단 제시 (5점)
- 추가 검사 계획 (5점)
            """
        ]

    async def search_medical_knowledge(self, query: str) -> List[str]:
        """의학 지식 검색"""
        # 간단한 키워드 매칭 (나중에 실제 벡터 검색으로 교체)
        query_lower = query.lower()
        results = []
        
        for knowledge in self.medical_knowledge:
            if any(keyword in knowledge.lower() for keyword in query_lower.split()):
                results.append(knowledge)
        
        print(f"🔍 의학 지식 검색: '{query}' -> {len(results)}개 결과")
        return results[:3]  # 최대 3개

    async def search_evaluation_criteria(self, case_type: str) -> List[str]:
        """평가 기준 검색"""
        case_lower = case_type.lower()
        results = []
        
        for criteria in self.evaluation_criteria:
            if case_lower in criteria.lower():
                results.append(criteria)
        
        print(f"📊 평가 기준 검색: '{case_type}' -> {len(results)}개 결과")
        return results

    # 기존 search 메서드는 의학 지식 검색으로 리다이렉트
    async def search(self, query: str, k: int = 3) -> List[str]:
        """기본 검색 (의학 지식용)"""
        return await self.search_medical_knowledge(query)

    def get_collection_stats(self) -> dict:
        """통계 정보"""
        return {
            "medical_knowledge_count": len(self.medical_knowledge),
            "evaluation_criteria_count": len(self.evaluation_criteria),
            "status": "medical_knowledge_mode",
            "description": "의학 지식 및 평가 기준 저장용"
        }