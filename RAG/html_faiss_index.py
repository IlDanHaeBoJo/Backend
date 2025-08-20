import re
import json
from bs4 import BeautifulSoup
from pathlib import Path
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def is_symptom_title(text):
    """텍스트가 증상명인지 판단"""
    text = text.strip()
    
    # 48가지 증상 리스트 (정확한 매칭을 위해)
    symptom_list = [
        "급성 복통", "소화불량/만성 복통", "토혈", "혈변", "구토", "배변 이상(변비/설사)", 
        "황달", "가슴 통증", "실신", "두근거림", "고혈압", "이상지질혈증", "기침", 
        "콧물/고막힘", "객혈", "호흡곤란", "소변량 변화(다뇨증/핍뇨)", "붉은색 소변", 
        "배뇨 이상", "발열", "쉽게 멍이 듦", "피로", "체중 감소", "체중 증가/비만", 
        "관절 통증/부기", "목 통증/허리 통증", "피부 발진", "기분 변화", "불안", 
        "수면장애", "기억력 저하", "어지럼", "두통", "경령", "근력/감각 이상", 
        "의식장애", "떨림/운동 이상", "유방통/유방덩이", "질분비물/질출혈", 
        "월경 이상/월경통", "산전 진찰", "성장/발달 지연", "예방접종", "음주/금연 상담", 
        "물질 오남용", "나쁜 소식 전하기", "가정 폭력/성 폭력", "자살"
    ]
    
    # 영문 괄호가 있는 경우 제거해서 비교
    text_clean = re.sub(r'\s*\([^)]*\)', '', text).strip()
    
    # 정확한 증상명 매칭 (대소문자 무시, 공백 정규화)
    normalized_text = re.sub(r'\s+', ' ', text_clean)
    
    for symptom in symptom_list:
        # 정확한 매칭
        if symptom == normalized_text:
            return True
        
        # 유사 매칭 (일부 변형 허용)
        symptom_variants = []
        
        # 슬래시로 구분된 증상들 처리
        if '/' in symptom:
            # "소화불량/만성 복통" -> ["소화불량", "만성 복통"]
            parts = symptom.split('/')
            symptom_variants.extend(parts)
            symptom_variants.append(symptom)
        
        # 괄호 포함 증상들 처리
        if '(' in symptom:
            # "배변 이상(변비/설사)" -> ["배변 이상", "변비", "설사"]
            base = symptom.split('(')[0].strip()
            symptom_variants.append(base)
            inner = re.search(r'\(([^)]+)\)', symptom)
            if inner:
                inner_parts = inner.group(1).split('/')
                symptom_variants.extend([p.strip() for p in inner_parts])
        
        # 기본 증상도 추가
        symptom_variants.append(symptom)
        
        # 변형들과 비교
        for variant in symptom_variants:
            variant_clean = variant.strip()
            if variant_clean and variant_clean == normalized_text:
                return True
    
    # 제외할 패턴들 (확실히 증상이 아닌 것들)
    exclude_patterns = [
        r'CPX', r'전\s*신', r'^\d+$', r'코멘트', r'정리', r'내용', r'참고자료',
        r'요점정리', r'네이버', r'서울대병원', r'인기다경', r'만성.*증후군',
        r'STEP', r'기본.*진찰', r'진단.*계획', r'치료.*계획', r'환자.*교육',
        r'생활.*습관', r'안내', r'한.*권.*으로', r'총론', r'주제별.*내용'
    ]
    
    # 제외 패턴 확인
    for pattern in exclude_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False
    
    return False

def split_by_symptoms(soup):
    """HTML을 증상별로 분리"""
    all_elements = soup.find_all(['h1', 'p', 'table', 'br'])
    
    symptom_sections = []
    current_symptom = None
    current_elements = []
    
    for element in all_elements:
        if element.name == 'h1':
            text = element.get_text(strip=True)
            
            # 새로운 증상명인지 확인
            if is_symptom_title(text):
                # 이전 증상 저장
                if current_symptom and current_elements:
                    symptom_sections.append({
                        'symptom': current_symptom,
                        'elements': current_elements.copy()
                    })
                
                # 새로운 증상 시작
                current_symptom = re.sub(r'\s*\([^)]*\)', '', text).strip()
                current_elements = [element]
            else:
                # 증상명이 아닌 h1은 현재 섹션에 포함
                if current_symptom:
                    current_elements.append(element)
        else:
            # h1이 아닌 요소들은 현재 섹션에 포함
            if current_symptom:
                current_elements.append(element)
    
    # 마지막 증상 저장
    if current_symptom and current_elements:
        symptom_sections.append({
            'symptom': current_symptom,
            'elements': current_elements.copy()
        })
    
    return symptom_sections

def chunk_symptom_section(symptom_name, elements, source_filename, chunk_size=800):
    """특정 증상의 요소들을 청크로 분할"""
    chunks_with_meta = []
    
    # 주요 섹션 구분을 위한 키워드
    section_keywords = {
        "코멘트": "comment",
        "질병별 진단": "diagnosis_treatment", 
        "치료 계획": "diagnosis_treatment",
        "기본 진찰": "examination",
        "병력 청취": "examination", 
        "환자 교육": "patient_education",
        "진단 계획": "diagnosis_plan",
        "치료 계획": "treatment_plan",
        "생활 습관": "lifestyle_education"
    }
    
    current_section = "general"
    current_heading = symptom_name
    accumulated_content = []
    
    for element in elements:
        if element.name == 'h1':
            # 새로운 섹션 시작
            heading_text = element.get_text(strip=True)
            
            # 이전 섹션 내용이 있으면 청크로 저장
            if accumulated_content:
                content_text = ' '.join(accumulated_content)
                if content_text.strip():
                    chunk = create_chunk(
                        content_text, symptom_name, current_section, 
                        current_heading, source_filename
                    )
                    if chunk:  # general 섹션이 아닌 경우만 추가
                        chunks_with_meta.append(chunk)
                accumulated_content = []
            
            # 새로운 섹션 설정
            current_heading = heading_text
            
            # 섹션 타입 결정
            current_section = "general"
            for keyword, section_type in section_keywords.items():
                if keyword in heading_text:
                    current_section = section_type
                    break
            
            # 제목도 내용에 포함
            accumulated_content.append(f"[{heading_text}]")
            
        elif element.name == 'p':
            text = element.get_text(strip=True)
            if text:
                accumulated_content.append(text)
                
        elif element.name == 'table':
            table_text = extract_table_content(element)
            if table_text:
                accumulated_content.append(table_text)
        
        # 청크 크기 확인 후 분할
        current_content = ' '.join(accumulated_content)
        if len(current_content) > chunk_size:
            chunk = create_chunk(
                current_content, symptom_name, current_section,
                current_heading, source_filename
            )
            if chunk:  # general 섹션이 아닌 경우만 추가
                chunks_with_meta.append(chunk)
            accumulated_content = []
    
    # 마지막 남은 내용 처리
    if accumulated_content:
        content_text = ' '.join(accumulated_content)
        if content_text.strip():
            chunk = create_chunk(
                content_text, symptom_name, current_section,
                current_heading, source_filename
            )
            if chunk:  # general 섹션이 아닌 경우만 추가
                chunks_with_meta.append(chunk)
    
    return chunks_with_meta

def chunk_medical_html(html_content, source_filename, chunk_size=800):
    """
    의료 HTML을 증상별로 분리한 후 각각을 청크로 분할
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # footer 태그 제거
    for tag in soup(["footer"]):
        tag.decompose()

    all_chunks = []
    
    # 증상별로 분리
    symptom_sections = split_by_symptoms(soup)
    
    print(f"발견된 증상: {[s['symptom'] for s in symptom_sections]}")
    
    # 각 증상별로 청크 생성
    for section in symptom_sections:
        symptom_chunks = chunk_symptom_section(
            section['symptom'], 
            section['elements'], 
            source_filename, 
            chunk_size
        )
        all_chunks.extend(symptom_chunks)
    
    return all_chunks

def extract_table_content(table_element):
    """테이블 내용을 구조화된 텍스트로 변환"""
    rows = []
    
    # 테이블 헤더 처리
    headers = []
    thead = table_element.find('thead')
    if thead:
        header_row = thead.find('tr')
        if header_row:
            headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
    
    # 테이블 바디 처리
    tbody = table_element.find('tbody') or table_element
    for row in tbody.find_all('tr'):
        cells = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
        if cells:
            if headers and len(headers) == len(cells):
                # 헤더가 있는 경우 구조화
                row_text = ' | '.join([f"{h}: {c}" for h, c in zip(headers, cells) if c])
            else:
                # 헤더가 없는 경우 단순 연결
                cells = [c.strip() for c in cells if c]
                row_text = ':'.join(cells)
            rows.append(row_text)
    
    return '\n'.join(rows) if rows else ""

def create_chunk(content, symptom_name, section_type, heading, source_filename):
    """청크와 메타데이터 생성"""
    # general 섹션은 제외
    if section_type == "general":
        return None
        
    if heading.startswith("STEP 1"):
        heading = heading.replace("STEP 1", "").lstrip()
    elif heading.startswith("STEP 2"):
        heading = heading.replace("STEP 2", "").lstrip()
    elif heading.lower().startswith("1)"):
        heading = heading.replace("1)", "").lstrip()
    elif heading.lower().startswith("2)"):
        heading = heading.replace("2)", "").lstrip()    
    elif heading.lower().startswith("3)"):
        heading = heading.replace("3)", "").lstrip()

    # metadata를 content에 포함
    enhanced_content = f"증상: {symptom_name} | 섹션: {section_type} | 제목: {heading} | 내용: {content}"

    return {
        "content": enhanced_content,
        "metadata": {
            "symptom": symptom_name,
            "section_type": section_type,
            "section_title": heading,
            "source": source_filename,
            "data_type": "medical_symptom"
        }
    }

def build_medical_faiss_index(html_files_dir, index_path, model_name="intfloat/multilingual-e5-large"):
    """
    의료 HTML 파일들을 처리하여 FAISS 인덱스 생성 또는 기존 인덱스에 추가
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})
    
    all_chunks = []
    html_dir = Path(html_files_dir)
    
    # HTML/JSON 파일들 처리
    for file_path in html_dir.glob("*.json"):
        print(f"처리 중: {file_path.name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # JSON 파일인 경우 HTML 추출
            if file_path.suffix == '.json':
                try:
                    json_data = json.loads(content)
                    html_content = json_data.get('html', content)
                    if 'content' in json_data and 'html' in json_data['content']:
                        html_content = json_data['content']['html']
                except:
                    html_content = content
            else:
                html_content = content
            
            # 청크 생성
            chunks = chunk_medical_html(html_content, file_path.stem)
            all_chunks.extend(chunks)
            print(f"  -> {len(chunks)}개 청크 생성")
            
        except Exception as e:
            print(f"파일 처리 오류 {file_path}: {e}")
    
    if not all_chunks:
        raise ValueError("처리할 청크가 없습니다.")
    
    # 텍스트와 메타데이터 분리
    texts = [chunk["content"] for chunk in all_chunks]
    metadatas = [chunk["metadata"] for chunk in all_chunks]
    
    # FAISS 인덱스 생성 또는 로드
    index_path_obj = Path(index_path)
    if index_path_obj.exists() and any(index_path_obj.iterdir()):
        print(f"기존 FAISS 인덱스 로드 중: {index_path}")
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        print(f"새로운 {len(texts)}개 청크를 기존 인덱스에 추가 중...")
        vectorstore.add_texts(texts=texts, metadatas=metadatas)
    else:
        print(f"새로운 FAISS 인덱스 생성 중: 총 {len(texts)}개 청크")
        vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    
    # 인덱스 저장
    index_path_obj.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_path))
    
    print(f"FAISS 인덱스 저장 완료: {index_path}")
    return vectorstore

# 테스트 코드
if __name__ == "__main__":
    # 실제 JSON 파일로 테스트
    try:
        with open("/home/ghdrnjs/Backend/RAG/chunk_181-210.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        html_content = data['content']['html']
        
        # 청크 생성 테스트
        chunks = chunk_medical_html(html_content, "chunk_181-210")
        
        print("=== 청크 생성 결과 ===")
        
        # 증상별 통계
        symptoms = {}
        for chunk in chunks:
            symptom = chunk['metadata']['symptom']
            if symptom not in symptoms:
                symptoms[symptom] = 0
            symptoms[symptom] += 1
        
        print(f"총 {len(chunks)}개 청크 생성됨")
        print("증상별 청크 개수:")
        for symptom, count in symptoms.items():
            print(f"  {symptom}: {count}개")
            
        print(f"\n처음 3개 청크 예시:")
        if chunks:
            for i, chunk in enumerate(chunks[:3]):  # 처음 3개만 출력
                print(f"\n--- 청크 {i+1} ---")
                print(f"  증상: {chunk['metadata']['symptom']}")
                print(f"  섹션: {chunk['metadata']['section_type']}")
                print(f"  제목: {chunk['metadata']['section_title']}")
                print(f"  내용: {chunk['content'][:200]}...")
            
    except Exception as e:
        print(f"테스트 오류: {e}")
        import traceback
        traceback.print_exc()