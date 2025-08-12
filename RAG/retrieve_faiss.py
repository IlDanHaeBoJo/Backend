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
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

def retrieve_faiss_index(index_path: str, query: str, k: int=5, model_name: str="intfloat/multilingual-e5-large") -> list:
    """
    FAISS 인덱스에서 쿼리에 대한 결과를 검색하는 함수
    :param index_path: FAISS 인덱스가 저장된 경로
    :param query: 검색할 쿼리
    :param k: 반환할 결과의 수
    :param model_name: 사용할 모델 이름
    :return: 검색 결과 리스트
    """
    faiss_index = load_faiss_index(index_path, model_name)
    retriever = faiss_index.as_retriever(k=k)
    # langchain retriever는 get_relevant_documents(query) 사용
    results = retriever.get_relevant_documents(query)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FAISS 인덱스 검색 스크립트")
    parser.add_argument("--index_path", type=str, required=True, help="FAISS 인덱스 경로")
    parser.add_argument("--query", type=str, required=True, help="검색할 쿼리")
    parser.add_argument("--k", type=int, default=5, help="검색 결과의 수")
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