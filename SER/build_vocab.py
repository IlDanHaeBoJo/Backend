# Backend/SER/build_vocab.py
import os
import json
from tqdm import tqdm
from collections import Counter

DATA_DIR = "/data/ghdrnjs/SER/small/"
OUTPUT_FILE = "char_to_id.json"

def build_vocab():
    """데이터셋의 모든 텍스트를 읽어 Character Vocabulary를 생성합니다."""
    print(f"🔍 데이터 디렉토리에서 모든 전사 파일(.txt)을 찾고 있습니다: {DATA_DIR}")
    transcript_paths = os.path.join(DATA_DIR, "script.txt")
    print(transcript_paths)

    if not transcript_paths:
        print(f"❌ '{DATA_DIR}'에서 전사 파일을 찾을 수 없습니다.")
        return
    
    all_text = ""
    print("📖 전사 파일을 읽는 중...")
    try:
        with open(transcript_paths, 'r', encoding='utf-8') as f:
            all_text = f.read().strip()
    except Exception as e:
        print(f"⚠️ 파일 읽기 오류: {transcript_paths} - {e}")
        return

    print("📊 문자 빈도수 계산 중...")
    # 모든 글자의 빈도 계산
    char_counts = Counter(all_text)
    
    # Vocabulary 생성 (빈도수 높은 순으로 정렬)
    # <pad>: 패딩용, <unk>: 모르는 글자용
    vocab = ['<pad>', '<unk>'] + [char for char, count in char_counts.most_common()]
    
    char_to_id = {char: i for i, char in enumerate(vocab)}
    
    print(f"🎉 총 {len(vocab)}개의 고유한 문자로 Vocabulary 생성 완료!")
    
    # 파일로 저장
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(char_to_id, f, ensure_ascii=False, indent=4)
        
    print(f"💾 Vocabulary가 '{OUTPUT_FILE}' 파일로 저장되었습니다.")
    print("\n-- 예시 --")
    print(list(char_to_id.items())[:20]) # 처음 20개 예시 출력

if __name__ == "__main__":
    build_vocab()