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
    
    def get_evaluation_criteria(self, category: str, area: str) -> Dict:
        """
        특정 카테고리와 영역의 평가 기준 가져오기
        
        Args:
            category: 질병/증상 카테고리 (예: "기억력 저하")
            area: 평가 영역 (예: "병력 청취", "신체 진찰", "환자 교육")
            
        Returns:
            해당 카테고리+영역의 가이드라인 문서
        """
        if not self.vectorstore:
            return {"documents": [], "category": category, "area": area}
        
        # 정확한 쿼리 구성: "카테고리 영역" 형태
        query = f"{category} {area}"
        
        # 검색 수행 - 정확한 매칭을 위해 카테고리 필터링 적용
        results = self.search_guidelines(query, k=1, category=category)
        
        # 해당 영역 문서가 있는지 확인
        target_doc = None
        for doc in results:
            if (doc.metadata.get("category") == category and 
                doc.metadata.get("area") == area and
                doc.metadata.get("type") == "guideline"):
                target_doc = doc
                break
        
        return {
            "documents": [target_doc] if target_doc else [],
            "category": category,
            "area": area
        }
if __name__ == "__main__":
    print("=== GuidelineRetriever 테스트 ===")
    retriever = GuidelineRetriever()
    
    print("\n=== 기억력 저하 + 병력 청취 검색 ===")
    result = retriever.get_evaluation_criteria('기억력 저하', '병력 청취')
    print(f"검색된 문서 수: {len(result['documents'])}")
    
    if result['documents']:
        doc = result['documents'][0]
        print("\n=== 문서 내용 ===")
        print(doc.page_content)
        print("\n=== 메타데이터 ===")
        print(doc.metadata)
    else:
        print("검색된 문서가 없습니다.")


