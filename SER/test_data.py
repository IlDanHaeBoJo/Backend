#!/usr/bin/env python3
"""
간단한 데이터 로딩 테스트 스크립트
상대 import 문제 없이 독립적으로 실행 가능
"""

import os
import sys
import re
from typing import List, Tuple, Optional
from collections import Counter

def get_emotion_from_filename(filename: str) -> Optional[str]:
    """
    파일명에서 번호를 추출하여 감정 라벨 반환
    예: F2001_000021.wav → 21 → Anxious
    """
    try:
        # 파일명에서 6자리 번호 추출 (F2001_000021.wav → 000021)
        match = re.search(r'_(\d{6})\.', filename)
        if not match:
            return None
        
        file_num = int(match.group(1))
        
        if 21 <= file_num <= 30:
            return "Anxious"
        elif 31 <= file_num <= 40:
            return "Kind"
        elif 91 <= file_num <= 100:
            return "Dry"
        else:
            return None
    except (ValueError, AttributeError):
        return None

def load_dataset_from_numbered_folders(data_dir: str) -> Tuple[List[str], List[str]]:
    """파일명 번호 기반으로 데이터셋 로드"""
    
    audio_paths = []
    labels = []
    
    if not os.path.exists(data_dir):
        print(f"❌ 데이터 디렉토리를 찾을 수 없습니다: {data_dir}")
        return audio_paths, labels
    
    # 각 person 폴더 탐색 (F2001, F2002, M2001 등)
    person_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    print(f"📁 발견된 person 폴더: {len(person_folders)}개")
    
    if len(person_folders) == 0:
        print(f"❌ person 폴더를 찾을 수 없습니다.")
        return audio_paths, labels
    
    total_files_processed = 0
    emotion_counts = {"Anxious": 0, "Kind": 0, "Dry": 0}
    
    for person_folder in sorted(person_folders):
        person_path = os.path.join(data_dir, person_folder)
        wav_path = os.path.join(person_path, "wav_48000")
        
        # wav_48000 폴더가 존재하는지 확인
        if not os.path.exists(wav_path):
            print(f"⚠️  {person_folder}: wav_48000 폴더를 찾을 수 없습니다")
            continue
        
        person_file_count = {"Anxious": 0, "Kind": 0, "Dry": 0, "Other": 0}
        
        # wav_48000 폴더 내의 모든 오디오 파일 확인
        audio_files = [f for f in os.listdir(wav_path) if f.lower().endswith(('.wav', '.mp3', '.m4a', '.flac'))]
        
        for audio_file in audio_files:
            # 파일명으로 감정 라벨 결정
            emotion_label = get_emotion_from_filename(audio_file)
            
            if emotion_label is not None:
                audio_path = os.path.join(wav_path, audio_file)
                audio_paths.append(audio_path)
                labels.append(emotion_label)
                person_file_count[emotion_label] += 1
                emotion_counts[emotion_label] += 1
            else:
                person_file_count["Other"] += 1
            
            total_files_processed += 1
        
        # 각 person별 통계 출력
        if sum(person_file_count[e] for e in ["Anxious", "Kind", "Dry"]) > 0:
            print(f"📁 {person_folder}: "
                  f"Anxious={person_file_count['Anxious']}, "
                  f"Kind={person_file_count['Kind']}, "
                  f"Dry={person_file_count['Dry']}, "
                  f"기타={person_file_count['Other']}, "
                  f"전체={len(audio_files)}")
    
    # 전체 통계 출력
    print(f"\n{'='*60}")
    print(f"📊 전체 데이터 로딩 완료")
    print(f"{'='*60}")
    print(f"처리된 person 폴더: {len([p for p in person_folders if os.path.exists(os.path.join(data_dir, p, 'wav_48000'))])}개")
    print(f"전체 파일 확인: {total_files_processed}개")
    print(f"사용된 파일: {len(audio_paths)}개")
    
    if len(audio_paths) > 0:
        print(f"\n클래스별 분포:")
        for emotion, count in emotion_counts.items():
            percentage = (count / len(audio_paths) * 100)
            print(f"  {emotion}: {count}개 ({percentage:.1f}%)")
        
        # 클래스 균형 확인
        min_count = min(emotion_counts.values())
        max_count = max(emotion_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        print(f"\n클래스 불균형 비율: {imbalance_ratio:.2f}")
        if imbalance_ratio > 2.0:
            print("⚠️  클래스 불균형이 있습니다. 가중치 조정을 고려해보세요.")
    else:
        print("❌ 조건에 맞는 오디오 파일을 찾을 수 없습니다.")
        print("\n💡 확인 사항:")
        print("   1. 파일명 패턴: [PREFIX]_[6자리숫자].wav")
        print("   2. 예상 번호 범위: 000021-000030(Anxious), 000031-000040(Kind), 000091-000100(Dry)")
        print("   3. 폴더 구조: [data_dir]/[person]/wav_48000/[파일들]")
    
    return audio_paths, labels

def print_sample_files(audio_paths: List[str], labels: List[str], num_samples: int = 10):
    """샘플 파일들 출력"""
    if len(audio_paths) == 0:
        return
    
    print(f"\n📄 샘플 파일들 ({min(num_samples, len(audio_paths))}개):")
    
    # 각 클래스별로 몇 개씩 출력
    samples_by_class = {"Anxious": [], "Kind": [], "Dry": []}
    
    for path, label in zip(audio_paths, labels):
        if len(samples_by_class[label]) < 3:  # 각 클래스별로 최대 3개
            samples_by_class[label].append(path)
    
    for emotion, paths in samples_by_class.items():
        if paths:
            print(f"\n  {emotion}:")
            for i, path in enumerate(paths, 1):
                filename = os.path.basename(path)
                print(f"    {i}. {filename}")

def main():
    """메인 함수"""
    
    print("🧪 데이터 로딩 테스트")
    print("="*50)
    
    # 데이터 경로
    data_dir = "/data/ghdrnjs/SER/small/"
    print(f"📂 데이터 디렉토리: {data_dir}")
    print(f"   파일명 패턴: [PERSON]_[6자리번호].wav")
    print(f"   감정 매핑:")
    print(f"     000021-000030 → Anxious (불안)")
    print(f"     000031-000040 → Kind (친절)")
    print(f"     000091-000100 → Dry (건조)")
    
    # 데이터 로드
    audio_paths, labels = load_dataset_from_numbered_folders(data_dir)
    
    if len(audio_paths) > 0:
        # 샘플 파일 출력
        print_sample_files(audio_paths, labels)
        
        print(f"\n✅ 데이터 로딩 성공!")
        print(f"   총 {len(audio_paths)}개 파일을 사용할 수 있습니다.")
        print(f"   파인튜닝을 시작할 준비가 완료되었습니다.")
    else:
        print(f"\n❌ 사용 가능한 데이터를 찾을 수 없습니다.")
        print(f"   데이터 경로와 구조를 확인해주세요.")

if __name__ == "__main__":
    main()