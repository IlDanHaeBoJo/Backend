import numpy as np
from typing import Tuple, List, Optional, Literal
from collections import Counter
import re
import os
import random
from tqdm import tqdm

EMOTION_LABELS = ["Anxious", "Dry", "Kind"]

def simple_augmentation(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """간단한 오디오 증강 (NumPy 2.x 호환)"""
    if np.random.random() < 0.3:  # 30% 확률로 노이즈 추가
        noise = np.random.normal(0, 0.005, audio.shape)
        audio = audio + noise
    
    if np.random.random() < 0.3:  # 30% 확률로 볼륨 조정
        volume_factor = np.random.uniform(0.8, 1.2)
        audio = audio * volume_factor
    
    return audio





def extract_number_from_filename(filename: str, type: Literal['content', 'emotion'] = 'emotion') -> Optional[int]:
    try:
        if type == "content":
            # 파일명에서 마지막 숫자 그룹 전체를 추출 (예: F2001_000123.wav -> 123)
            match = re.search(r'_(\d+)\.wav$', os.path.basename(filename))
            if match:
                return int(match.group(1))
            return None
        else:
            # F..._...xxxD.wav 에서 마지막 숫자 D를 추출
            match = re.search(r'_(\d+)\.wav$', os.path.basename(filename))
            if match:
                return int(match.group(1)) % 10
            return None
    except (ValueError, AttributeError):
        return None



def get_emotion_from_filename(filename: str) -> Optional[str]:
    """파일명에서 번호를 추출하여 감정 라벨 반환"""
    file_num = extract_number_from_filename(filename, type="content")
    if file_num is None:
        return None
        
    if 21 <= file_num <= 30:
        return "Anxious"
    elif 31 <= file_num <= 40:
        return "Kind"
    elif 141 <= file_num <= 150:
        return "Dry"
    else:
        return None


# 데이터 전체를 스캔해서 (경로, 감정, 화자, 스크립트ID) 인덱스 생성
def build_corpus_index(data_dir: str,
                       accept_exts={'.wav', '.flac'},
                       require_emotion=True) -> List[Dict[str, Any]]:
    """
    return: [{"path": p, "emotion": e, "speaker": s, "content_id": c}, ...]
    """
    index = []
    speakers = sorted([d for d in os.listdir(data_dir)
                       if os.path.isdir(os.path.join(data_dir, d))])
    print(f"📁 화자 폴더 수: {len(speakers)}")

    for spk in tqdm(speakers, desc="인덱스 구축"):
        spk_dir = os.path.join(data_dir, spk)
        # 하위 디렉토리를 재귀적으로 탐색 (감정별 폴더/단일 폴더 둘 다 대응)
        for root, _, files in os.walk(spk_dir):
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                if ext not in accept_exts:
                    continue
                path = os.path.join(root, fn)

                # 감정 라벨
                emo = infer_emotion_from_path(path)
                if require_emotion and emo not in EMOTION_LABELS:
                    # 감정 미매칭 샘플은 제외
                    continue

                # 스크립트(대화) ID: 파일명에서 추출 (기존 규칙 그대로)
                cid = extract_number_from_filename(fn, type="content")
                if cid is None:
                    # 스크립트 ID 없으면 제외(불교차 조건을 보장하기 위해)
                    continue

                index.append({
                    "path": path,
                    "emotion": emo,
                    "speaker": spk,
                    "content_id": cid
                })
    print(f"✅ 인덱스 샘플 수: {len(index)}")
    return index


def split_speaker_and_content(
    index: List[Dict[str, Any]],
    val_content_ratio: float = 0.2,
    test_content_ratio: float = 0.2,
    val_speaker_ratio: float = 0.2,
    test_speaker_ratio: float = 0.2,
    seed: int = 42,
    fixed_val_content_ids: Optional[List[int]] = None,
    fixed_test_content_ids: Optional[List[int]] = None,
) -> Tuple[Tuple[List[str], List[str]],
           Tuple[List[str], List[str]],
           Tuple[List[str], List[str]]]:
    """
    index: build_corpus_index() 반환 리스트
    반환: ((train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels))
    """
    rng = random.Random(seed)

    # 전체 스크립트 ID, 화자 목록
    all_contents = sorted(set([it["content_id"] for it in index]))
    all_speakers = sorted(set([it["speaker"] for it in index]))

    # --- 2-1) 스크립트(대화) 불교차 세트 만들기
    if fixed_val_content_ids is not None and fixed_test_content_ids is not None:
        val_contents = set(fixed_val_content_ids)
        test_contents = set(fixed_test_content_ids)
        train_contents = set(all_contents) - val_contents - test_contents
    else:
        contents = all_contents[:]
        rng.shuffle(contents)
        n_val = max(1, int(len(contents) * val_content_ratio))
        n_test = max(1, int(len(contents) * test_content_ratio))
        val_contents = set(contents[:n_val])
        test_contents = set(contents[n_val:n_val+n_test])
        train_contents = set(contents[n_val+n_test:])

    # --- 2-2) 화자 불교차 세트 만들기
    speakers = all_speakers[:]
    rng.shuffle(speakers)
    n_val_spk = max(1, int(len(speakers) * val_speaker_ratio))
    n_test_spk = max(1, int(len(speakers) * test_speaker_ratio))
    val_speakers = set(speakers[:n_val_spk])
    test_speakers = set(speakers[n_val_spk:n_val_spk+n_test_spk])
    train_speakers = set(speakers[n_val_spk+n_test_spk:])

    # --- 2-3) 교집합 제거: 두 조건(화자 세트, 스크립트 세트)을 동시에 만족하는 샘플만 채택
    train_items = [it for it in index
                   if it["speaker"] in train_speakers and it["content_id"] in train_contents]
    val_items   = [it for it in index
                   if it["speaker"] in val_speakers and it["content_id"] in val_contents]
    test_items  = [it for it in index
                   if it["speaker"] in test_speakers and it["content_id"] in test_contents]

    # --- 2-4) 점검 출력
    def summarize(name, items):
        spks = sorted(set([it["speaker"] for it in items]))
        cids = sorted(set([it["content_id"] for it in items]))
        emo_cnt = Counter([it["emotion"] for it in items])
        print(f"\n[{name}] 샘플: {len(items)}, 화자: {len(spks)}, 스크립트ID: {len(cids)}")
        print(f"  감정분포: {dict(emo_cnt)}")
        print(f"  예시 화자(최대 10): {spks[:10]}")
        print(f"  예시 스크립트ID(최대 20): {cids[:20]}")

    summarize("TRAIN", train_items)
    summarize("VAL",   val_items)
    summarize("TEST",  test_items)

    # --- 2-5) 교차 검증: 화자/스크립트 불교차 여부 확인
    assert set([it["speaker"] for it in train_items]).isdisjoint(set([it["speaker"] for it in val_items + test_items])), \
        "Train 화자가 Val/Test와 겹칩니다."
    assert set([it["speaker"] for it in val_items]).isdisjoint(set([it["speaker"] for it in test_items])), \
        "Val 화자가 Test와 겹칩니다."
    assert set([it["content_id"] for it in train_items]).isdisjoint(set([it["content_id"] for it in val_items + test_items])), \
        "Train 스크립트ID가 Val/Test와 겹칩니다."
    assert set([it["content_id"] for it in val_items]).isdisjoint(set([it["content_id"] for it in test_items])), \
        "Val 스크립트ID가 Test와 겹칩니다."

    # --- 2-6) 최종 리스트 변환
    def to_xy(items):
        return [it["path"] for it in items], [it["emotion"] for it in items]

    return to_xy(train_items), to_xy(val_items), to_xy(test_items)


def load_dataset_subset(data_dir: str, max_per_class: int) -> Tuple[List[str], List[str]]:
    audio_paths = []
    labels = []
    emotion_counts = {label: 0 for label in EMOTION_LABELS}
    
    person_folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
    print(f"📁 발견된 person 폴더: {len(person_folders)}개")
    
    for person_folder in tqdm(person_folders, desc="데이터셋 로딩 중"):
        wav_path = os.path.join(data_dir, person_folder, "wav_48000")
        if not os.path.exists(wav_path):
            continue
        
        for audio_file in os.listdir(wav_path):
            if not audio_file.lower().endswith('.wav'):
                continue
            
            emotion_label = get_emotion_from_filename(audio_file)
            if emotion_label and emotion_counts[emotion_label] < max_per_class:
                audio_paths.append(os.path.join(wav_path, audio_file))
                labels.append(emotion_label)
                emotion_counts[emotion_label] += 1
        
        if all(count >= max_per_class for count in emotion_counts.values()):
            break

    print(f"\n📊 로드된 데이터 분포 (빠른 테스트용):")
    for emotion, count in emotion_counts.items():
        percentage = (count / len(audio_paths) * 100) if len(audio_paths) > 0 else 0
        print(f"  {emotion}: {count}개 ({percentage:.1f}%)")
            
    return audio_paths, labels


# (필수) 화자 ID 추출: data_dir 바로 아래 1단계 폴더명이 화자
def extract_speaker_id(audio_path: str, data_dir: str) -> str:
    rel = os.path.relpath(audio_path, data_dir)
    spk = rel.split(os.sep)[0]
    return spk



def build_speaker_mapping(train_paths, data_dir):
    train_speakers = sorted({extract_speaker_id(p, data_dir) for p in train_paths})
    spk2id = {spk: i for i, spk in enumerate(train_speakers)}
    return spk2id



# (선택) 경로에서 감정 라벨 추론 (폴더명에 Anxious/Kind/Dry가 있으면 그걸 사용)
def infer_emotion_from_path(audio_path: str) -> Optional[str]:
    parts = os.path.normpath(audio_path).split(os.sep)
    for p in reversed(parts):
        if p in EMOTION_LABELS:
            return p
    # 폴더명에 없으면 파일명 규칙으로 추론 (기존 함수)
    return get_emotion_from_filename(os.path.basename(audio_path))





