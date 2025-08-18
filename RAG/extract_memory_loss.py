"""
기억력 저하 전용 추출기
- 범용 추출기를 사용해서 기억력 저하만 추출
- 특화된 키워드와 후처리
"""

import json
from medical_extractor import MedicalExtractor
from typing import Dict, Optional

def extract_memory_loss_from_chunk(chunk_file_path: str) -> Optional[Dict]:
    """
    청크 파일에서 기억력 저하 완전 가이드 추출
    
    Args:
        chunk_file_path: 청크 JSON 파일 경로
        
    Returns:
        기억력 저하 완전 가이드 또는 None
    """
    
    print("🧠 기억력 저하 청크 추출 시작")
    print("=" * 50)
    
    # 1. 범용 추출기 초기화
    extractor = MedicalExtractor()
    
    # 2. 추출 실행 (키워드 없이 LLM이 직접 판단)
    result = extractor.extract_from_json_file(
        json_file_path=chunk_file_path,
        target_condition="기억력 저하"
    )
    
    if not result:
        return None
    
    # 4. 결과 출력
    print_extraction_summary(result)
    
    return result



def print_extraction_summary(result: Dict):
    """체크리스트 추출 결과 요약 출력"""
    
    print("=" * 50)
    print("🎯 기억력 저하 체크리스트 추출 결과:")
    print(f"   📚 카테고리: {result['category']}")
    print(f"   📊 신뢰도: {result['metadata']['confidence']:.2f}")
    print(f"   📋 총 질문/행동: {result['metadata']['total_questions']}개")
    print(f"   🏷️ 키워드: {len(result['metadata']['keywords'])}개")
    
    # 영역별 내용 개수
    print(f"\n📋 영역별 항목:")
    evaluation_areas = result.get('evaluation_areas', {})
    for area_key, area_data in evaluation_areas.items():
        area_name = area_data.get('name', area_key)
        subcategories = area_data.get('subcategories', {})
        item_count = 0
        
        for subcat in subcategories.values():
            if isinstance(subcat, dict):
                questions = subcat.get('required_questions', [])
                actions = subcat.get('required_actions', [])
                item_count += len(questions) + len(actions)
        
        status = "✅" if item_count > 5 else "⚠️" if item_count > 0 else "❌"
        print(f"   {status} {area_name}: {item_count}개 항목")


def create_rag_chunk(checklist: Dict) -> Dict:
    """RAG 시스템용 체크리스트 청크 형태로 변환"""
    
    return {
        "id": checklist['id'],
        "category": checklist['category'],
        "description": checklist['description'],
        "evaluation_areas": checklist['evaluation_areas'],
        "metadata": {
            "source": "cpx_textbook",
            "condition": "기억력저하",
            "type": "guideline",
            "keywords": checklist['metadata']['keywords'],
            "total_questions": checklist['metadata']['total_questions'],
            "confidence": checklist['metadata']['confidence'],
            "extraction_method": checklist['metadata']['extraction_method']
        }
    }

def main():
    """메인 실행 함수"""
    
    print("🏥 기억력 저하 체크리스트 추출기")
    print("📁 대상 파일: chunk_241-270.json")
    print()
    
    # 기억력 저하 추출
    chunk_file = "chunk_241-270.json"
    result = extract_memory_loss_from_chunk(chunk_file)
    
    if result:
        # RAG 청크 생성 및 저장 (RAG 파일만 생성)
        rag_chunk = create_rag_chunk(result)
        
        rag_output = "memory_loss_guideline_rag.json"
        with open(rag_output, 'w', encoding='utf-8') as f:
            json.dump(rag_chunk, f, ensure_ascii=False, indent=2)
        
        print(f"💾 RAG 가이드라인 저장 완료: {rag_output}")
        
        # 체크리스트 미리보기
        print(f"\n📖 체크리스트 구조 미리보기:")
        print("=" * 50)
        
        evaluation_areas = result.get('evaluation_areas', {})
        for area_name, area_data in evaluation_areas.items():
            print(f"\n🔹 {area_data.get('name', area_name)}")
            subcategories = area_data.get('subcategories', {})
            for subcat_key, subcat_data in list(subcategories.items())[:2]:  # 처음 2개만
                print(f"  └ {subcat_data.get('name', subcat_key)}")
                questions = subcat_data.get('required_questions', [])
                actions = subcat_data.get('required_actions', [])
                for q in questions[:2]:  # 처음 2개 질문만
                    print(f"    • {q}")
                for a in actions[:2]:  # 처음 2개 행동만
                    print(f"    • {a}")
        
        print("=" * 50)
        
        print(f"\n✅ 기억력 저하 체크리스트 추출 완료!")
        print(f"🎯 이제 CPX 평가에서 바로 사용할 수 있습니다.")
        
    else:
        print("❌ 기억력 저하 관련 내용을 찾을 수 없습니다.")

if __name__ == "__main__":
    main()
