"""
체크리스트를 RAG 시스템에 인덱싱하는 모듈
"""

import json
import os
from pathlib import Path
from typing import Dict, List
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# OpenAI API 사용으로 device 함수 불필요

def create_documents_from_guideline(guideline_path: str) -> List[Document]:
    """가이드라인 JSON 파일을 Document 객체들로 변환"""
    
    with open(guideline_path, 'r', encoding='utf-8') as f:
        guideline = json.load(f)
    
    documents = []
    category = guideline['category']
    
    # 각 평가 영역별로 카테고리 포함한 문서 생성
    for area_key, area_data in guideline['evaluation_areas'].items():
        area_name = area_data['name']
        
        # 문서 제목에 카테고리 + 영역명 포함 (한글 name 사용)
        area_content = f"""
제목: {category} {area_name} 평가 가이드라인
카테고리: {category}
평가 영역: {area_name}

=== {category} {area_name} 필수 항목들 ===

"""
        
        # 모든 하위 카테고리의 질문/행동을 한 문서에 통합
        all_questions = []
        all_actions = []
        
        for subcat_key, subcat_data in area_data.get('subcategories', {}).items():
            if not subcat_data.get('applicable', True):
                continue  # applicable이 false인 항목은 건너뛰기
                
            subcat_name = subcat_data['name']
            area_content += f"\n【{subcat_name}】\n"
            
            # 질문들 추가
            questions = subcat_data.get('required_questions', [])
            if questions:
                area_content += "필수 질문:\n"
                for question in questions:
                    area_content += f"  • {question}\n"
                    all_questions.append(question)
            
            # 행동들 추가
            actions = subcat_data.get('required_actions', [])
            if actions:
                area_content += "필수 행동:\n"
                for action in actions:
                    area_content += f"  • {action}\n"
                    all_actions.append(action)
            
            area_content += "\n"
        
        # 요약 정보 추가
        area_content += f"""
=== 요약 ===
총 필수 질문: {len(all_questions)}개
총 필수 행동: {len(all_actions)}개
"""
    
        documents.append(Document(
            page_content=area_content,
            metadata={
                "source": "cpx_textbook",
                "type": "guideline",
                "category": category,
                "area": area_name,
                "total_questions": len(all_questions),
                "total_actions": len(all_actions)
            }
        ))
    
    return documents

def index_guideline_to_faiss(guideline_path: str, index_path: str, model_name: str = "text-embedding-3-small"):
    """체크리스트를 FAISS 인덱스에 추가"""
    
    # OpenAI API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
    
    # 임베딩 모델 초기화 - OpenAI API 사용
    embeddings = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=api_key
    )
    
    # 가이드라인을 문서로 변환
    documents = create_documents_from_guideline(guideline_path)
    print(f"생성된 문서 수: {len(documents)}")
    
    # 기존 인덱스가 있으면 로드, 없으면 새로 생성
    if os.path.exists(index_path):
        print(f"기존 FAISS 인덱스 로드: {index_path}")
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        
        # 새로운 문서들 추가
        vectorstore.add_documents(documents)
        print(f"가이드라인 문서 {len(documents)}개 추가 완료")
    else:
        print(f"새로운 FAISS 인덱스 생성: {index_path}")
        vectorstore = FAISS.from_documents(documents, embeddings)
        print(f"가이드라인 문서 {len(documents)}개로 인덱스 생성 완료")
    
    # 인덱스 저장
    vectorstore.save_local(index_path)
    print(f"FAISS 인덱스 저장 완료: {index_path}")
    
    return vectorstore

def search_guideline(index_path: str, query: str, k: int = 5, model_name: str = "text-embedding-3-small"):
    """체크리스트에서 관련 내용 검색"""
    
    # OpenAI API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
    
    embeddings = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=api_key
    )
    
    # 인덱스 로드
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    
    # 검색 수행
    results = vectorstore.similarity_search(query, k=k)
    
    return results

def main():
    """메인 실행 함수"""
    
    print("🏥 가이드라인 RAG 인덱싱 시작")
    
    # 경로 설정
    guideline_path = "memory_loss_guideline_rag.json"
    index_path = "faiss_guideline_index"
    
    if not os.path.exists(guideline_path):
        print(f"❌ 가이드라인 파일을 찾을 수 없습니다: {guideline_path}")
        print("먼저 extract_memory_loss.py를 실행하여 가이드라인을 생성하세요.")
        return
    
    # 가이드라인 인덱싱
    try:
        vectorstore = index_guideline_to_faiss(guideline_path, index_path)
        print(f"✅ 인덱싱 완료!")
        print(f"총 문서 수: {len(vectorstore.index_to_docstore_id.values())}")
        
        # 테스트 검색 (카테고리 포함)
        print("\n🔍 테스트 검색:")
        test_queries = [
            "기억력 저하 병력 청취",
            "기억력 저하 신체 진찰", 
            "기억력 저하 환자 교육"
        ]
        
        for query in test_queries:
            print(f"\n쿼리: {query}")
            results = search_guideline(index_path, query, k=2)
            for i, doc in enumerate(results):
                print(f"  {i+1}. [{doc.metadata.get('type', 'unknown')}] {doc.page_content[:100]}...")
        
    except Exception as e:
        print(f"❌ 인덱싱 실패: {e}")

if __name__ == "__main__":
    main()
