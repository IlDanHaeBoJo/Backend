import argparse
import os
from pathlib import Path
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def load_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_faiss_index(index_path: str, model_name: str="intfloat/multilingual-e5-large") -> FAISS:
    """
    FAISS 인덱스를 로드하는 함수
    :param index_path: FAISS 인덱스가 저장된 경로
    :return: FAISS 인덱스 객체
    """
    device = load_device()
    print(f"사용 중인 장치: {device}")
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})

    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    
    # 인덱스 정보 출력
    total_docs = len(vectorstore.index_to_docstore_id.values())
    print(f"총 문서 개수: {total_docs}")
    print(f"FAISS 인덱스 크기: {vectorstore.index.ntotal}")
    
    return vectorstore

def retrieve_faiss_index(index_path: str, query: str, k: int=5, model_name: str="intfloat/multilingual-e5-large") -> list:
    """
    FAISS 인덱스에서 쿼리에 대한 결과를 검색하는 함수
    :param index_path: FAISS 인덱스가 저장된 경로
    :param query: 검색할 쿼리
    :param k: 반환할 결과의 수
    :param model_name: 사용할 모델 이름
    :return: 검색 결과 리스트
    """
    print(f"검색 시작: 쿼리='{query}', k={k}")
    
    faiss_index = load_faiss_index(index_path, model_name)
    
    # 방법 1: 유사도와 함께 검색 (중복 제거 확인)
    try:
        results_with_scores = faiss_index.similarity_search_with_score(query, k=k*2)
        print(f"유사도 검색 완료: {len(results_with_scores)}개 결과 (k*2={k*2})")
        
        # 중복 제거 확인
        unique_contents = set()
        filtered_results = []
        
        for doc, score in results_with_scores:
            if doc.page_content not in unique_contents:
                unique_contents.add(doc.page_content)
                filtered_results.append(doc)
                if len(filtered_results) >= k:
                    break
        
        print(f"중복 제거 후: {len(filtered_results)}개 결과")
        return filtered_results[:k]
        
    except Exception as e:
        print(f"similarity_search_with_score 오류: {e}")
        
        # 방법 2: 더 큰 k로 검색
        try:
            results = faiss_index.similarity_search(query, k=k*3)
            print(f"확장 검색 완료: {len(results)}개 결과 (k*3={k*3})")
            return results[:k]
        except Exception as e2:
            print(f"similarity_search 오류: {e2}")
            
            # 방법 3: retriever 방식으로 시도
            try:
                retriever = faiss_index.as_retriever(search_kwargs={"k": k*2, "fetch_k": k*4})
                results = retriever.get_relevant_documents(query)
                print(f"retriever 검색 완료: {len(results)}개 결과")
                return results[:k]
            except Exception as e3:
                print(f"retriever 오류: {e3}")
                return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FAISS 인덱스 검색 스크립트")
    parser.add_argument("--index_path", type=str, default="/home/ghdrnjs/Backend/RAG/medical_rag_db", help="FAISS 인덱스 경로")
    parser.add_argument("--query", type=str, required=True, help="검색할 쿼리")
    parser.add_argument("--k", type=int, default=6, help="검색 결과의 수")
    parser.add_argument("--model_name", type=str, default="intfloat/multilingual-e5-large", help="사용할 모델 이름")
    args = parser.parse_args()

    index_path = args.index_path
    query = args.query
    k = args.k
    model_name = args.model_name
    results = retrieve_faiss_index(index_path, query, k, model_name)

    if not results:
        print("검색 결과가 없습니다.")
    else:
        print(f"\n[검색 결과 Top {k}]")
        for i, result in enumerate(results, 1):
            print(f"\n--- 결과 {i} ---")
            # page_content(텍스트) 출력
            content = result.page_content if hasattr(result, 'page_content') else result.get('page_content', '')
            print(f"내용: {content}")
            # 메타데이터가 있으면 출력
            meta = result.metadata if hasattr(result, 'metadata') else result.get('metadata', {})
            if meta:
                print("메타데이터:")
                for k, v in meta.items():
                    print(f"  {k}: {v}")