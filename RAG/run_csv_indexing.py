#!/usr/bin/env python3
"""
CSV 파일들을 FAISS 인덱스로 변환하는 실행 스크립트
"""

import subprocess
import os
from pathlib import Path

def main():
    # CSV 파일 경로들
    csv_files = {
        "disease5": "/data/ghdrnjs/RAG/disease5.csv",
        "disease4": "/data/ghdrnjs/RAG/disease4.csv", 
        "disease_listen": "/data/ghdrnjs/RAG/disease_listen.csv"
    }
    
    # 출력 디렉터리
    output_dir = "/home/ghdrnjs/Backend/RAG/faiss_index"
    
    # 각 파일 존재 여부 확인
    existing_files = {}
    for name, path in csv_files.items():
        if os.path.exists(path):
            existing_files[name] = path
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name}: {path} (파일이 존재하지 않음)")
    
    if not existing_files:
        print("처리할 CSV 파일이 없습니다.")
        return
    
    # build_faiss_index.py 실행
    cmd = [
        "python", "/home/ghdrnjs/Backend/RAG/build_faiss_index.py",
        "--out_dir", output_dir,
        "--model_name", "intfloat/multilingual-e5-large"
    ]
    
    # 존재하는 CSV 파일들 추가
    if "disease5" in existing_files:
        cmd.extend(["--disease5_csv", existing_files["disease5"]])
    if "disease4" in existing_files:
        cmd.extend(["--disease4_csv", existing_files["disease4"]])
    if "disease_listen" in existing_files:
        cmd.extend(["--disease_listen_csv", existing_files["disease_listen"]])
    
    print(f"\n실행할 명령어:")
    print(" ".join(cmd))
    print("\n인덱싱 시작...")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ 인덱싱 완료!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("❌ 인덱싱 실패:")
        print(e.stderr)
        return
    
    # 결과 확인
    index_path = Path(output_dir)
    if index_path.exists():
        files = list(index_path.glob("*"))
        print(f"\n생성된 파일들:")
        for f in files:
            print(f"  - {f.name}")
    
    print(f"\n🎉 FAISS 인덱스가 생성되었습니다: {output_dir}")

if __name__ == "__main__":
    main()
