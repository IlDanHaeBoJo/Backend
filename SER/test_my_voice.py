#!/usr/bin/env python3
"""
학습된 모델로 내 음성 파일 테스트하기
사용법: python test_my_voice.py your_audio_file.wav
"""

import os
import sys
import torch
import numpy as np
import librosa
import argparse
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import warnings
warnings.filterwarnings('ignore')

# 감정 라벨 정의 (학습 시와 동일해야 함)
EMOTION_LABELS = ["Anxious", "Dry", "Kind"]
LABEL2ID = {label: i for i, label in enumerate(EMOTION_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(EMOTION_LABELS)}

# 모델 설정 (학습 시와 동일)
MODEL_NAME = "kresnik/wav2vec2-large-xlsr-korean"
SAMPLE_RATE = 16000
MAX_DURATION = 10.0

def load_trained_model(model_path=None):
    """학습된 모델 로드"""
    
    if model_path and os.path.exists(model_path):
        # 저장된 모델이 있으면 로드
        print(f"🤖 저장된 모델 로드 중: {model_path}")
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
        processor = Wav2Vec2Processor.from_pretrained(model_path)
    else:
        # 기본 모델 로드 (학습 전 상태)
        print(f"⚠️  기본 모델 로드 (학습 전): {MODEL_NAME}")
        print("   학습된 모델을 사용하려면 모델 경로를 지정하세요.")
        
        from transformers import Wav2Vec2Config
        
        config = Wav2Vec2Config.from_pretrained(
            MODEL_NAME,
            num_labels=len(EMOTION_LABELS),
            label2id=LABEL2ID,
            id2label=ID2LABEL,
            finetuning_task="emotion_classification"
        )
        
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            MODEL_NAME,
            config=config,
            ignore_mismatched_sizes=True
        )
        processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    
    return model, processor

def preprocess_audio(file_path, processor):
    """오디오 전처리 (학습 시와 동일한 방식)"""
    try:
        print(f"🎵 오디오 로딩: {file_path}")
        
        # 오디오 로드 (다양한 포맷 지원)
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, res_type='kaiser_fast')
        
        # 길이 정보 출력
        duration = len(audio) / SAMPLE_RATE
        print(f"   - 길이: {duration:.2f}초")
        print(f"   - 샘플링 레이트: {sr}Hz")
        
        # 길이 조정
        target_length = int(SAMPLE_RATE * MAX_DURATION)
        if len(audio) > target_length:
            print(f"   - 오디오가 {MAX_DURATION}초보다 길어서 자릅니다.")
            # 가운데 부분 사용
            start_idx = (len(audio) - target_length) // 2
            audio = audio[start_idx:start_idx + target_length]
        elif len(audio) < target_length:
            print(f"   - 오디오가 {MAX_DURATION}초보다 짧아서 패딩을 추가합니다.")
            pad_length = target_length - len(audio)
            audio = np.pad(audio, (0, pad_length), mode='constant', constant_values=0)
        
        # 정규화
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.8
        
        # Wav2Vec2 processor로 변환
        inputs = processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )
        
        return inputs.input_values.squeeze(0)
        
    except Exception as e:
        print(f"❌ 오디오 전처리 오류: {e}")
        print(f"   파일 경로: {file_path}")
        print(f"   파일 존재 여부: {os.path.exists(file_path)}")
        import traceback
        traceback.print_exc()
        return None

def predict_emotion(model, processor, audio_file, device):
    """감정 예측"""
    
    # 오디오 전처리
    input_values = preprocess_audio(audio_file, processor)
    
    if input_values is None:
        return None, None
    
    # 모델을 평가 모드로
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        # 배치 차원 추가 및 디바이스로 이동
        input_values = input_values.unsqueeze(0).to(device)
        
        # 예측 수행
        outputs = model(input_values=input_values)
        logits = outputs.logits
        
        # 확률 계산
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        probabilities = probabilities.cpu().numpy()[0]
        
        # 최고 확률 클래스
        predicted_class_id = np.argmax(probabilities)
        predicted_emotion = ID2LABEL[predicted_class_id]
        confidence = probabilities[predicted_class_id]
        
        return predicted_emotion, probabilities

def print_results(audio_file, predicted_emotion, probabilities):
    """결과 출력"""
    
    print(f"\n🎯 예측 결과")
    print("=" * 50)
    print(f"📁 파일: {os.path.basename(audio_file)}")
    print(f"🎭 예측 감정: {predicted_emotion}")
    print(f"🎲 확신도: {probabilities[LABEL2ID[predicted_emotion]]:.1%}")
    
    print(f"\n📊 상세 확률:")
    for emotion in EMOTION_LABELS:
        prob = probabilities[LABEL2ID[emotion]]
        bar = "█" * int(prob * 20)  # 0-20 길이의 바
        print(f"  {emotion:8s}: {prob:.1%} {bar}")
    
    # 감정별 설명
    emotion_descriptions = {
        "Anxious": "불안, 초조함",
        "Dry": "건조함, 무미건조함", 
        "Kind": "친절함, 따뜻함"
    }
    
    print(f"\n💭 해석: {emotion_descriptions.get(predicted_emotion, '알 수 없음')}")

def main():
    """메인 함수"""
    
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description="학습된 모델로 음성 감정 분석")
    parser.add_argument("audio_file", help="분석할 오디오 파일 경로")
    parser.add_argument("--model_path", help="학습된 모델 경로 (선택사항)")
    parser.add_argument("--gpu", type=int, default=None, help="사용할 GPU 번호")
    
    args = parser.parse_args()
    
    # 오디오 파일 존재 확인
    if not os.path.exists(args.audio_file):
        print(f"❌ 오디오 파일을 찾을 수 없습니다: {args.audio_file}")
        sys.exit(1)
    
    # GPU 설정
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  사용 디바이스: {device}")
    
    try:
        # 모델 로드
        model, processor = load_trained_model(args.model_path)
        
        # 예측 수행
        print(f"\n🔮 감정 분석 중...")
        predicted_emotion, probabilities = predict_emotion(
            model, processor, args.audio_file, device
        )
        
        if predicted_emotion is None:
            print("❌ 예측에 실패했습니다.")
            sys.exit(1)
        
        # 결과 출력
        print_results(args.audio_file, predicted_emotion, probabilities)
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

def quick_test():
    """빠른 테스트 함수 (인자 없이 실행할 때)"""
    print("🧪 빠른 테스트 모드")
    print("사용법:")
    print("  python test_my_voice.py your_audio.wav")
    print("  python test_my_voice.py your_audio.wav --model_path ./results_3class_simple/final_model")
    print("\n지원 형식: .wav, .mp3, .m4a, .flac")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # 인자 없이 실행하면 사용법 표시
        quick_test()
    else:
        main()