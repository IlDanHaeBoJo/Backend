"""
체크리스트를 RAG 시스템에 인덱싱하는 모듈
"""

import json
import os
from pathlib import Path
from typing import Dict, List
import torch
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

def load_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def create_documents_from_guideline(guideline_path: str) -> List[Document]:
    """가이드라인 JSON 파일을 Document 객체들로 변환"""
    
    with open(guideline_path, 'r', encoding='utf-8') as f:
        guideline = json.load(f)
    
    documents = []
    
    # 1. 전체 체크리스트를 하나의 문서로
    full_content = f"""
카테고리: {guideline['category']}
설명: {guideline['description']}

이것은 {guideline['category']} CPX 평가 가이드라인입니다.
총 {guideline['metadata']['total_questions']}개의 질문과 행동으로 구성되어 있습니다.
"""
    
    documents.append(Document(
        page_content=full_content,
        metadata={
            "source": "guideline",
            "type": "overview",
            "category": guideline['category'],
            "total_questions": guideline['metadata']['total_questions']
        }
    ))
    
    # 2. 각 평가 영역별로 문서 생성
    for area_key, area_data in guideline['evaluation_areas'].items():
        area_content = f"""
평가 영역: {area_data['name']}
카테고리: {guideline['category']}

"""
        
        # 각 하위 카테고리의 질문들을 포함
        for subcat_key, subcat_data in area_data.get('subcategories', {}).items():
            if not subcat_data.get('applicable', True):
                continue  # applicable이 false인 항목은 건너뛰기
                
            area_content += f"\n{subcat_data['name']}:\n"
            
            # 질문들 추가
            questions = subcat_data.get('required_questions', [])
            for question in questions:
                area_content += f"- {question}\n"
            
            # 행동들 추가
            actions = subcat_data.get('required_actions', [])
            for action in actions:
                area_content += f"- {action}\n"
        
        documents.append(Document(
            page_content=area_content,
            metadata={
                "source": "guideline",
                "type": "evaluation_area",
                "category": guideline['category'],
                "area": area_data['name'],
                "area_key": area_key
            }
        ))
    
    # 3. 각 하위 카테고리별로 세부 문서 생성
    for area_key, area_data in guideline['evaluation_areas'].items():
        for subcat_key, subcat_data in area_data.get('subcategories', {}).items():
            if not subcat_data.get('applicable', True):
                continue
                
            subcat_content = f"""
카테고리: {guideline['category']}
평가 영역: {area_data['name']}
하위 카테고리: {subcat_data['name']}

"""
            
            questions = subcat_data.get('required_questions', [])
            actions = subcat_data.get('required_actions', [])
            
            if questions:
                subcat_content += "필수 질문들:\n"
                for question in questions:
                    subcat_content += f"- {question}\n"
            
            if actions:
                subcat_content += "\n필수 행동들:\n"
                for action in actions:
                    subcat_content += f"- {action}\n"
            
            documents.append(Document(
                page_content=subcat_content,
                metadata={
                    "source": "guideline",
                    "type": "subcategory",
                    "category": guideline['category'],
                    "area": area_data['name'],
                    "area_key": area_key,
                    "subcategory": subcat_data['name'],
                    "subcategory_key": subcat_key,
                    "question_count": len(questions),
                    "action_count": len(actions)
                }
            ))
    
    return documents

def index_guideline_to_faiss(guideline_path: str, index_path: str, model_name: str = "intfloat/multilingual-e5-large"):
    """체크리스트를 FAISS 인덱스에 추가"""
    
    device = load_device()
    print(f"사용 중인 장치: {device}")
    
    # 임베딩 모델 초기화
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
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

def search_guideline(index_path: str, query: str, k: int = 5, model_name: str = "intfloat/multilingual-e5-large"):
    """체크리스트에서 관련 내용 검색"""
    
    device = load_device()
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
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
        
        # 테스트 검색
        print("\n🔍 테스트 검색:")
        test_queries = [
            "기억력 저하 병력청취 질문",
            "MMSE 검사 방법",
            "환자 교육 내용"
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
