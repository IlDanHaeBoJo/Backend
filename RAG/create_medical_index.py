"""
의료 HTML JSON 파일들을 FAISS 인덱스로 변환하는 스크립트
"""

import sys
import os
import shutil
sys.path.append('/home/ghdrnjs/Backend/RAG')

try:
    from html_faiss_index import build_medical_faiss_index
    from pathlib import Path
    
    json_path = "/home/ghdrnjs/Backend/RAG/cpx_html_json"
    output_dir = "/home/ghdrnjs/Backend/RAG/medical_db"
    
    print("의료 FAISS 인덱스 생성 중...")

    # FAISS 인덱스 생성
    vectorstore = build_medical_faiss_index(
        html_files_dir=str(json_path),
        index_path=output_dir,
        model_name="intfloat/multilingual-e5-large"
    )
    
    print(f"✅ FAISS 인덱스 생성 완료: {output_dir}")
    
    # 간단한 검색 테스트
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    vectorstore = FAISS.load_local(output_dir, embeddings, allow_dangerous_deserialization=True)
    
    # 검색 테스트
    test_queries = ["발열 증상이 있을 때 적절한 치료 방법", "피로 증상이 있을 때 적절한 병력 청취", "체중 감소의 환자 교육 방법"]
    
    for query in test_queries:
        print(f"\n=== '{query}' 검색 결과 ===")
        results = vectorstore.similarity_search(query, k=5)
        for i, doc in enumerate(results, 1):
            print(f"결과 {i}:\n")
            print(f"  내용: {doc.page_content[:200] if hasattr(doc, 'page_content') else doc.get('page_content', '')}...\n")

except Exception as e:
    print(f"오류 발생: {e}")
    import traceback
    traceback.print_exc()
