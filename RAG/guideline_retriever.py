"""
가이드라인 검색 및 평가를 위한 RAG 유틸리티
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import torch
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 상위 디렉토리의 모듈을 import하기 위한 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

class GuidelineRetriever:
    """CPX 가이드라인 검색 및 평가 도구"""
    
    def __init__(self, index_path: str = "faiss_guideline_index", model_name: str = "intfloat/multilingual-e5-large"):
        """
        가이드라인 검색기 초기화
        
        Args:
            index_path: FAISS 인덱스 경로
            model_name: 임베딩 모델명
        """
        self.index_path = index_path
        self.model_name = model_name
        self.vectorstore = None
        self.embeddings = None
        
        self._initialize_vectorstore()
    
    def _initialize_vectorstore(self):
        """벡터스토어 초기화"""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🔧 가이드라인 검색기 초기화 (장치: {device})")
            
            # 임베딩 모델 초기화
            self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
            
            # FAISS 인덱스 로드
            if os.path.exists(self.index_path):
                self.vectorstore = FAISS.load_local(
                    self.index_path, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                print(f"✅ 가이드라인 인덱스 로드 완료: {len(self.vectorstore.index_to_docstore_id.values())}개 문서")
            else:
                print(f"❌ 가이드라인 인덱스를 찾을 수 없습니다: {self.index_path}")
                print("먼저 RAG/index_guideline.py를 실행하여 인덱스를 생성하세요.")
                
        except Exception as e:
            print(f"❌ 가이드라인 검색기 초기화 실패: {e}")
            self.vectorstore = None
    
    def search_guidelines(self, query: str, k: int = 5, category: Optional[str] = None) -> List[Document]:
        """
        가이드라인에서 관련 내용 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
            category: 특정 카테고리로 필터링 (예: "기억력 저하")
            
        Returns:
            관련 문서들의 리스트
        """
        if not self.vectorstore:
            print("❌ 벡터스토어가 초기화되지 않았습니다.")
            return []
        
        try:
            # 기본 검색
            results = self.vectorstore.similarity_search(query, k=k*2)  # 여유있게 검색
            
            # 카테고리 필터링
            if category:
                filtered_results = []
                for doc in results:
                    if doc.metadata.get("category") == category:
                        filtered_results.append(doc)
                results = filtered_results[:k]
            else:
                results = results[:k]
            
            return results
            
        except Exception as e:
            print(f"❌ 가이드라인 검색 실패: {e}")
            return []
    
    def get_evaluation_criteria(self, category: str, area: str = None) -> Dict:
        """
        특정 카테고리의 평가 기준 가져오기
        
        Args:
            category: 질병/증상 카테고리 (예: "기억력 저하")
            area: 평가 영역 (예: "병력 청취", "신체 진찰", "환자 교육")
            
        Returns:
            평가 기준 딕셔너리
        """
        if not self.vectorstore:
            return {}
        
        # 쿼리 구성
        if area:
            query = f"{category} {area} 평가 기준"
        else:
            query = f"{category} CPX 평가 기준"
        
        # 검색 수행
        results = self.search_guidelines(query, k=10, category=category)
        
        # 결과 구조화
        evaluation_criteria = {
            "category": category,
            "area": area,
            "criteria": [],
            "questions": [],
            "actions": []
        }
        
        for doc in results:
            doc_type = doc.metadata.get("type", "unknown")
            
            if doc_type == "subcategory":
                # 세부 카테고리의 질문/행동 추출
                content_lines = doc.page_content.split('\n')
                current_section = None
                
                for line in content_lines:
                    line = line.strip()
                    if line.startswith("필수 질문들:"):
                        current_section = "questions"
                    elif line.startswith("필수 행동들:"):
                        current_section = "actions"
                    elif line.startswith("- ") and current_section:
                        item = line[2:].strip()  # "- " 제거
                        if current_section == "questions":
                            evaluation_criteria["questions"].append({
                                "question": item,
                                "subcategory": doc.metadata.get("subcategory", ""),
                                "area": doc.metadata.get("area", "")
                            })
                        elif current_section == "actions":
                            evaluation_criteria["actions"].append({
                                "action": item,
                                "subcategory": doc.metadata.get("subcategory", ""),
                                "area": doc.metadata.get("area", "")
                            })
        
        return evaluation_criteria
    
    def evaluate_conversation_completeness(self, conversation_log: List[Dict], category: str) -> Dict:
        """
        대화 로그를 가이드라인과 비교하여 완성도 평가
        
        Args:
            conversation_log: 대화 로그 [{"role": "student/patient", "content": "..."}]
            category: 평가할 카테고리 (예: "기억력 저하")
            
        Returns:
            완성도 평가 결과
        """
        if not self.vectorstore:
            return {"error": "가이드라인 검색기가 초기화되지 않았습니다."}
        
        # 전체 대화 텍스트 구성
        conversation_text = "\n".join([
            f"{entry['role']}: {entry['content']}" 
            for entry in conversation_log
        ])
        
        # 각 평가 영역별로 기준 가져오기
        areas = ["병력 청취", "신체 진찰", "환자 교육"]
        evaluation_results = {
            "category": category,
            "overall_completeness": 0.0,
            "area_results": {},
            "missing_items": [],
            "completed_items": []
        }
        
        total_score = 0
        area_count = 0
        
        for area in areas:
            criteria = self.get_evaluation_criteria(category, area)
            
            if not criteria.get("questions") and not criteria.get("actions"):
                continue
            
            area_result = self._evaluate_area_completeness(
                conversation_text, 
                criteria, 
                area
            )
            
            evaluation_results["area_results"][area] = area_result
            total_score += area_result["completeness_score"]
            area_count += 1
            
            # 누락된 항목과 완료된 항목 수집
            evaluation_results["missing_items"].extend(area_result.get("missing_items", []))
            evaluation_results["completed_items"].extend(area_result.get("completed_items", []))
        
        # 전체 완성도 계산
        if area_count > 0:
            evaluation_results["overall_completeness"] = total_score / area_count
        
        return evaluation_results
    
    def _evaluate_area_completeness(self, conversation_text: str, criteria: Dict, area: str) -> Dict:
        """특정 영역의 완성도 평가"""
        
        result = {
            "area": area,
            "completeness_score": 0.0,
            "total_items": 0,
            "completed_items": [],
            "missing_items": []
        }
        
        # 질문 항목 평가
        questions = criteria.get("questions", [])
        actions = criteria.get("actions", [])
        all_items = questions + actions
        
        if not all_items:
            return result
        
        result["total_items"] = len(all_items)
        completed_count = 0
        
        conversation_lower = conversation_text.lower()
        
        for item in all_items:
            item_text = item.get("question", item.get("action", ""))
            item_key_words = self._extract_keywords(item_text)
            
            # 키워드 기반 매칭 (간단한 방식)
            is_completed = any(
                keyword.lower() in conversation_lower 
                for keyword in item_key_words
                if len(keyword) > 2  # 2글자 이상 키워드만
            )
            
            if is_completed:
                completed_count += 1
                result["completed_items"].append({
                    "item": item_text,
                    "subcategory": item.get("subcategory", ""),
                    "type": "question" if "question" in item else "action"
                })
            else:
                result["missing_items"].append({
                    "item": item_text,
                    "subcategory": item.get("subcategory", ""),
                    "type": "question" if "question" in item else "action"
                })
        
        # 완성도 점수 계산
        result["completeness_score"] = completed_count / len(all_items) if all_items else 0.0
        
        return result
    
    def _extract_keywords(self, text: str) -> List[str]:
        """텍스트에서 핵심 키워드 추출 (간단한 방식)"""
        
        # 불용어 제거를 위한 간단한 리스트
        stop_words = {
            "이", "가", "을", "를", "의", "에", "에서", "으로", "로", "과", "와", "도", "만", "까지", "부터",
            "은", "는", "이다", "있다", "없다", "하다", "되다", "이야", "야", "아", "어", "지", "고", "서",
            "어떻게", "무엇", "누구", "언제", "어디서", "왜", "어떤", "몇", "얼마나"
        }
        
        # 단어 분리 및 정제
        words = []
        for word in text.replace("?", "").replace(".", "").replace(",", "").split():
            word = word.strip()
            if len(word) > 1 and word not in stop_words:
                words.append(word)
        
        return words

def main():
    """테스트 함수"""
    
    print("🧪 가이드라인 검색기 테스트")
    
    retriever = GuidelineRetriever()
    
    if not retriever.vectorstore:
        print("❌ 검색기 초기화 실패")
        return
    
    # 테스트 1: 기본 검색
    print("\n🔍 테스트 1: 기본 검색")
    results = retriever.search_guidelines("기억력 저하 병력청취", k=3)
    for i, doc in enumerate(results):
        print(f"  {i+1}. [{doc.metadata.get('type')}] {doc.page_content[:100]}...")
    
    # 테스트 2: 평가 기준 가져오기
    print("\n📋 테스트 2: 평가 기준")
    criteria = retriever.get_evaluation_criteria("기억력 저하", "병력 청취")
    print(f"  질문 수: {len(criteria.get('questions', []))}")
    print(f"  행동 수: {len(criteria.get('actions', []))}")
    
    # 테스트 3: 대화 완성도 평가
    print("\n⭐ 테스트 3: 대화 완성도 평가")
    test_conversation = [
        {"role": "student", "content": "언제부터 기억력이 떨어지셨나요?"},
        {"role": "patient", "content": "3개월 전부터요."},
        {"role": "student", "content": "가족 중에 치매 환자가 있나요?"},
        {"role": "patient", "content": "아버지가 치매였습니다."}
    ]
    
    evaluation = retriever.evaluate_conversation_completeness(test_conversation, "기억력 저하")
    print(f"  전체 완성도: {evaluation['overall_completeness']:.2%}")
    print(f"  완료된 항목: {len(evaluation['completed_items'])}개")
    print(f"  누락된 항목: {len(evaluation['missing_items'])}개")

if __name__ == "__main__":
    main()
