import os
import csv
import argparse, json, shutil
from pathlib import Path
from typing import List, Dict
import torch
import boto3
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def load_qa_items(input_path: str, csv_question_col: str = "question", csv_answer_col: str = "answer") -> List[Dict]:
    p = Path(input_path)
    items: List[Dict] = []
    
    if p.suffix.lower() == ".jsonl":
        with p.open("r", encoding="utf-8-sig") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                q = (obj.get("question") or "").strip()
                a = (obj.get("answer") or "").strip()
                if not (q or a):
                    continue
                items.append(obj)
    
    elif p.suffix.lower() == ".json":
        with p.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
            
        data_list = data if isinstance(data, list) else [data]
        for obj in data_list:
            q = (obj.get("question") or "").strip()
            a = (obj.get("answer") or "").strip()
            if not (q or a):
                continue
            items.append(obj)
    
    elif p.suffix.lower() == ".csv":
        items.extend(load_csv_items(input_path, csv_question_col, csv_answer_col))
    
    else:
        raise ValueError("지원하지 않는 포맷입니다. .jsonl/.json/.csv 사용")
    return items

def load_csv_items(input_path: str, csv_question_col: str = "question", csv_answer_col: str = "answer") -> List[Dict]:
    """CSV 파일을 로드하여 다양한 형식을 처리"""
    p = Path(input_path)
    items: List[Dict] = []
    
    with p.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        
        # 첫 번째 행을 읽어서 컬럼 구조 파악
        first_row = next(reader, None)
        if not first_row:
            return items
            
        columns = list(first_row.keys())
        
        # 파일명에 따른 처리 방식 결정
        if "disease5.csv" in input_path:
            items.extend(process_disease5_csv(reader, first_row))
        elif "disease4.csv" in input_path:
            items.extend(process_disease4_csv(reader, first_row))
        elif "disease_listen.csv" in input_path:
            items.extend(process_disease_listen_csv(reader, first_row))
        else:
            # 기존 방식: question/answer 컬럼 기반 처리
            items.extend(process_standard_qa_csv(reader, first_row, csv_question_col, csv_answer_col))
    
    return items

def process_disease5_csv(reader, first_row) -> List[Dict]:
    """disease5.csv 처리: 계통, 원인, 질병, 진단 키워드, 검사 치료"""
    items = []
    
    # 첫 번째 행 처리
    all_rows = [first_row] + list(reader)
    
    for row in all_rows:
        system = (row.get("계통") or "").strip()
        cause = (row.get("원인") or "").strip()
        disease = (row.get("질병") or "").strip()
        diagnosis_keywords = (row.get("진단 키워드") or "").strip()
        examination_treatment = (row.get("검사 치료") or "").strip()
        
        if not (disease or diagnosis_keywords):  # 핵심 정보가 있어야 함
            continue
            
        # 자연스러운 텍스트 형태로 정보 구성
        content = f"{disease}"
        
        if system:
            content += f"은(는) {system}에 속하는 질병입니다."
        if cause:
            content += f" 주요 원인은 {cause}입니다."
        if diagnosis_keywords:
            content += f" 진단 시 주요 증상 및 키워드: {diagnosis_keywords}."
        if examination_treatment:
            content += f" 검사 및 치료 방법: {examination_treatment}."
        
        items.append({
            "content": content,
            "계통": system,
            "원인": cause,
            "질병": disease,
            "진단 키워드": diagnosis_keywords,
            "검사 치료": examination_treatment,
            "data_type": "disease_info",
            "csv_source": "disease5"
        })
    
    return items

def process_disease4_csv(reader, first_row) -> List[Dict]:
    """disease4.csv 처리: 계통, 질병, 진단 키워드, 검사 치료"""
    items = []
    
    # 첫 번째 행 처리
    all_rows = [first_row] + list(reader)
    
    for row in all_rows:
        system = (row.get("계통") or "").strip()
        disease = (row.get("질병") or "").strip()
        diagnosis_keywords = (row.get("진단 키워드") or "").strip()
        examination_treatment = (row.get("검사 치료") or "").strip()
        
        if not (disease or diagnosis_keywords):  # 핵심 정보가 있어야 함
            continue
            
        # 자연스러운 텍스트 형태로 정보 구성
        content = f"{disease}"
        
        if system:
            content += f"은(는) {system}에 속하는 질병입니다."
        if diagnosis_keywords:
            content += f" 진단 시 주요 증상 및 키워드: {diagnosis_keywords}."
        if examination_treatment:
            content += f" 검사 및 치료 방법: {examination_treatment}."
        
        items.append({
            "content": content,
            "계통": system,
            "질병": disease,
            "진단 키워드": diagnosis_keywords,
            "검사 치료": examination_treatment,
            "data_type": "disease_info",
            "csv_source": "disease4"
        })
    
    return items

def process_disease_listen_csv(reader, first_row) -> List[Dict]:
    """disease_listen.csv 처리: 넘버링, 약자 풀이, 의미, 목적"""
    items = []
    
    # 첫 번째 행 처리
    all_rows = [first_row] + list(reader)
    
    for row in all_rows:
        numbering = (row.get("넘버링") or "").strip()
        abbreviation = (row.get("약자 풀이") or "").strip()
        meaning = (row.get("의미") or "").strip()
        purpose = (row.get("목적") or "").strip()
        
        if not (abbreviation or meaning):  # 핵심 정보가 있어야 함
            continue
            
        # 자연스러운 텍스트 형태로 정보 구성
        content = f"{abbreviation}"
        
        if meaning:
            content += f"은(는) {meaning}을 의미합니다."
        if purpose:
            content += f" 이는 {purpose}를 위해 사용됩니다."
        if numbering:
            content += f" (평가 절차 번호: {numbering})"
        
        items.append({
            "content": content,
            "넘버링": numbering,
            "약자 풀이": abbreviation,
            "의미": meaning,
            "목적": purpose,
            "data_type": "evaluation_procedure",
            "csv_source": "disease_listen"
        })
    
    return items

def process_standard_qa_csv(reader, first_row, csv_question_col: str, csv_answer_col: str) -> List[Dict]:
    """기존 방식: question/answer 컬럼 기반 처리"""
    items = []
    
    # 첫 번째 행 처리
    all_rows = [first_row] + list(reader)
    
    for row in all_rows:
        q = (row.get(csv_question_col) or "").strip()
        a = (row.get(csv_answer_col) or "").strip()
        
        if not (q or a):
            continue
            
        # CSV의 추가 컬럼은 메타데이터로 보존
        items.append({
            "question": q,
            "answer": a,
            **{k: v for k, v in row.items() if k not in (csv_question_col, csv_answer_col)}
        })
    
    return items

def build_text(item: Dict) -> str:
    """데이터 타입에 따라 적절한 텍스트 포맷으로 변환"""
    
    # CSV 기반 데이터인 경우 content 필드 사용
    if "content" in item:
        content = item["content"]
        data_type = item.get("data_type", "")
        
        if data_type == "disease_info":
            # 병 정보의 경우
            text = f"의료 정보: {content}"
            
        elif data_type == "evaluation_procedure":
            # 평가 절차의 경우
            text = f"평가 절차: {content}"
            
        else:
            text = content
            
    else:
        # 기존 question-answer 형태 데이터
        q = (item.get("question") or "").strip()
        a = (item.get("answer") or "").strip()
        text = f"질문: {q}\n답변: {a}"
    
    return text

def upload_dir_to_s3(local_dir: str, bucket: str, prefix: str):
    s3 = boto3.client("s3")
    for root, _, files in os.walk(local_dir):
        for name in files:
            full_path = Path(root) / name
            rel_path = str(Path(full_path).relative_to(local_dir))
            key = f"{prefix.rstrip('/')}/{rel_path}"
            s3.upload_file(str(full_path), bucket, key)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", help="QA(JSON/JSONL/CSV) 디렉터리")
    ap.add_argument("--out_dir", required=True, help="로컬 인덱스 출력 디렉터리")
    ap.add_argument("--s3_bucket", help="업로드할 S3 버킷명")
    ap.add_argument("--s3_prefix", help="업로드할 S3 prefix (예: faiss/indexes/v1)")
    ap.add_argument("--model_name", default="intfloat/multilingual-e5-large")
    # CSV QA 컬럼명 지정
    ap.add_argument("--csv_question_col", default="question")
    ap.add_argument("--csv_answer_col", default="answer")
    # 특정 CSV 파일들 직접 지정
    ap.add_argument("--disease5_csv", help="disease5.csv 파일 경로")
    ap.add_argument("--disease4_csv", help="disease4.csv 파일 경로")
    ap.add_argument("--disease_listen_csv", help="disease_listen.csv 파일 경로")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(model_name=args.model_name, model_kwargs={"device": device})

    vs = None
    total = 0
    file_names = []
    
    # 특정 CSV 파일들이 지정된 경우 우선 처리
    specific_csvs = [
        (args.disease5_csv, "disease5.csv"),
        (args.disease4_csv, "disease4.csv"), 
        (args.disease_listen_csv, "disease_listen.csv")
    ]
    
    for csv_path, csv_name in specific_csvs:
        if csv_path and os.path.exists(csv_path):
            file_names.append(csv_path)
            print(f"✅ 특정 CSV 파일 추가: {csv_path}")
    
    if args.input_dir:
        for dir_path, _, filenames in os.walk(args.input_dir):
            for fn in filenames:
                if fn.lower().endswith((".json", ".jsonl", ".csv")):
                    full_path = os.path.join(dir_path, fn)
                    if full_path not in file_names:
                        file_names.append(full_path)

    if not file_names:
        raise ValueError("처리할 파일이 없습니다. --input_dir 또는 특정 CSV 파일 경로를 지정해주세요.")

    out_dir = Path(args.out_dir)
    if out_dir.exists() and len(os.listdir(out_dir)) > 0:
        print(f"📂 기존 인덱스 로드 중: {out_dir}")
        vs = FAISS.load_local(str(out_dir), embeddings, allow_dangerous_deserialization=True)
    else:
        vs = None

    for data_path in file_names:
        print(f"📄 처리 중인 파일: {data_path}")
        items = load_qa_items(
            data_path,
            csv_question_col=args.csv_question_col,
            csv_answer_col=args.csv_answer_col,
        )
        if not items:
            print(f"⚠️  파일에서 처리할 항목이 없습니다: {data_path}")
            continue
            
        print(f"✅ {len(items)}개 항목 로드됨")
        texts = [build_text(it) for it in items]
        metas = [{k: v for k, v in it.items() if k not in ("question", "answer", "content")} for it in items]
        
        if vs is None:
            vs = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metas)
        else:
            vs.add_texts(texts=texts, metadatas=metas)
        total += len(items)

    if vs is None:
        raise RuntimeError("인덱싱할 항목이 없습니다.")

    out_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(out_dir))
    print(f"🎉 총 {total}개 항목으로 FAISS 인덱스 생성 완료: {out_dir}")

    if args.s3_bucket and args.s3_prefix:
        upload_dir_to_s3(str(out_dir), args.s3_bucket, args.s3_prefix)
        print(f"☁️  S3 업로드 완료: s3://{args.s3_bucket}/{args.s3_prefix}")

if __name__ == "__main__":
    main()