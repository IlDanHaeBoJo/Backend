"""
음성 감정 분석 모델 추론 모듈
"""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Union
import json
import argparse

from .model import SpeechEmotionClassifier
from .config import model_config

class EmotionInference:
    """음성 감정 분석 추론 클래스"""
    
    def __init__(self, model_path: str):
        """
        Args:
            model_path: 훈련된 모델이 저장된 경로
        """
        self.model_path = model_path
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.load_model()
    
    def load_model(self):
        """모델 로드"""
        try:
            print(f"모델 로드 중: {self.model_path}")
            
            if os.path.exists(self.model_path):
                # 저장된 모델에서 로드
                self.model = SpeechEmotionClassifier.from_pretrained(self.model_path)
            else:
                # 기본 모델 로드 (훈련되지 않은 상태)
                print("훈련된 모델을 찾을 수 없습니다. 기본 모델을 로드합니다.")
                self.model = SpeechEmotionClassifier()
            
            self.model.to(self.device)
            self.model.eval()
            
            print(f"모델이 성공적으로 로드되었습니다. 디바이스: {self.device}")
            
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            raise e
    
    def predict_single(self, 
                      audio_path: str, 
                      return_probabilities: bool = True) -> Dict:
        """단일 오디오 파일 감정 예측
        
        Args:
            audio_path: 오디오 파일 경로
            return_probabilities: 확률값 반환 여부
            
        Returns:
            예측 결과 딕셔너리
        """
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {audio_path}")
        
        try:
            # 예측 실행
            if return_probabilities:
                probabilities = self.model.predict(audio_path, return_probabilities=True)
                predicted_emotion = model_config.emotion_labels[np.argmax(probabilities)]
                confidence = float(np.max(probabilities))
                
                # 모든 감정별 확률
                emotion_probs = {
                    emotion: float(prob) 
                    for emotion, prob in zip(model_config.emotion_labels, probabilities[0])
                }
                
                return {
                    'audio_path': audio_path,
                    'predicted_emotion': predicted_emotion,
                    'confidence': confidence,
                    'probabilities': emotion_probs,
                    'status': 'success'
                }
            else:
                predicted_emotion = self.model.predict(audio_path, return_probabilities=False)
                
                return {
                    'audio_path': audio_path,
                    'predicted_emotion': predicted_emotion,
                    'status': 'success'
                }
                
        except Exception as e:
            print(f"예측 실패: {audio_path}, 오류: {e}")
            return {
                'audio_path': audio_path,
                'error': str(e),
                'status': 'error'
            }
    
    def predict_batch(self, 
                     audio_paths: List[str], 
                     return_probabilities: bool = True) -> List[Dict]:
        """여러 오디오 파일 배치 예측
        
        Args:
            audio_paths: 오디오 파일 경로 리스트
            return_probabilities: 확률값 반환 여부
            
        Returns:
            예측 결과 리스트
        """
        
        results = []
        
        for audio_path in audio_paths:
            result = self.predict_single(audio_path, return_probabilities)
            results.append(result)
        
        return results
    
    def predict_directory(self, 
                         directory_path: str,
                         file_extensions: List[str] = None,
                         return_probabilities: bool = True) -> List[Dict]:
        """디렉토리 내 모든 오디오 파일 예측
        
        Args:
            directory_path: 디렉토리 경로
            file_extensions: 처리할 파일 확장자 리스트
            return_probabilities: 확률값 반환 여부
            
        Returns:
            예측 결과 리스트
        """
        
        if file_extensions is None:
            file_extensions = ['.wav', '.mp3', '.m4a', '.flac']
        
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"디렉토리를 찾을 수 없습니다: {directory_path}")
        
        # 오디오 파일 찾기
        audio_files = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in file_extensions):
                    audio_files.append(os.path.join(root, file))
        
        print(f"발견된 오디오 파일: {len(audio_files)}개")
        
        # 배치 예측
        return self.predict_batch(audio_files, return_probabilities)
    
    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        
        return {
            'model_name': model_config.model_name,
            'num_labels': model_config.num_labels,
            'emotion_labels': model_config.emotion_labels,
            'sampling_rate': model_config.sampling_rate,
            'max_duration': model_config.max_duration,
            'device': str(self.device),
            'model_path': self.model_path
        }
    
    def analyze_emotions_distribution(self, results: List[Dict]) -> Dict:
        """감정 분포 분석
        
        Args:
            results: predict_batch 또는 predict_directory 결과
            
        Returns:
            감정 분포 통계
        """
        
        successful_results = [r for r in results if r.get('status') == 'success']
        
        if not successful_results:
            return {'error': '분석할 성공적인 결과가 없습니다.'}
        
        # 감정별 카운트
        emotion_counts = {}
        confidence_scores = []
        
        for result in successful_results:
            emotion = result['predicted_emotion']
            confidence = result.get('confidence', 0)
            
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            confidence_scores.append(confidence)
        
        # 통계 계산
        total_samples = len(successful_results)
        emotion_percentages = {
            emotion: (count / total_samples) * 100 
            for emotion, count in emotion_counts.items()
        }
        
        return {
            'total_samples': total_samples,
            'emotion_counts': emotion_counts,
            'emotion_percentages': emotion_percentages,
            'average_confidence': float(np.mean(confidence_scores)),
            'confidence_std': float(np.std(confidence_scores)),
            'min_confidence': float(np.min(confidence_scores)),
            'max_confidence': float(np.max(confidence_scores))
        }

def create_inference_engine(model_path: str) -> EmotionInference:
    """추론 엔진 생성 팩토리 함수"""
    return EmotionInference(model_path)

def main():
    """CLI 인터페이스"""
    
    parser = argparse.ArgumentParser(description="음성 감정 분석 추론")
    
    parser.add_argument("--model_path", type=str, required=True, help="훈련된 모델 경로")
    parser.add_argument("--audio_path", type=str, help="단일 오디오 파일 경로")
    parser.add_argument("--directory", type=str, help="오디오 파일 디렉토리 경로")
    parser.add_argument("--output_path", type=str, help="결과 저장 경로 (JSON)")
    parser.add_argument("--no_probabilities", action="store_true", help="확률값 제외")
    
    args = parser.parse_args()
    
    # 추론 엔진 생성
    inference = create_inference_engine(args.model_path)
    
    # 모델 정보 출력
    model_info = inference.get_model_info()
    print("모델 정보:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    print()
    
    results = None
    
    if args.audio_path:
        # 단일 파일 예측
        print(f"단일 파일 예측: {args.audio_path}")
        result = inference.predict_single(
            args.audio_path, 
            return_probabilities=not args.no_probabilities
        )
        
        if result['status'] == 'success':
            print(f"예측 감정: {result['predicted_emotion']}")
            if 'confidence' in result:
                print(f"신뢰도: {result['confidence']:.4f}")
        else:
            print(f"예측 실패: {result['error']}")
        
        results = [result]
    
    elif args.directory:
        # 디렉토리 예측
        print(f"디렉토리 예측: {args.directory}")
        results = inference.predict_directory(
            args.directory,
            return_probabilities=not args.no_probabilities
        )
        
        # 감정 분포 분석
        distribution = inference.analyze_emotions_distribution(results)
        print("\n감정 분포 분석:")
        
        if 'error' not in distribution:
            print(f"총 샘플: {distribution['total_samples']}개")
            print(f"평균 신뢰도: {distribution['average_confidence']:.4f}")
            print("\n감정별 분포:")
            
            for emotion, percentage in distribution['emotion_percentages'].items():
                count = distribution['emotion_counts'][emotion]
                print(f"  {emotion}: {count}개 ({percentage:.1f}%)")
        else:
            print(f"  {distribution['error']}")
    
    else:
        print("--audio_path 또는 --directory 중 하나를 지정해주세요.")
        return
    
    # 결과 저장
    if args.output_path and results:
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n결과가 저장되었습니다: {args.output_path}")

if __name__ == "__main__":
    main()