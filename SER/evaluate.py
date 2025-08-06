"""
음성 감정 분석 모델 평가 모듈
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve
)
import json
import argparse

from .inference import EmotionInference
from .data_loader import load_dataset_from_directory, load_dataset_from_csv
from .config import model_config

class ModelEvaluator:
    """모델 평가 클래스"""
    
    def __init__(self, model_path: str):
        """
        Args:
            model_path: 평가할 모델 경로
        """
        self.model_path = model_path
        self.inference_engine = EmotionInference(model_path)
        self.evaluation_results = {}
    
    def evaluate_dataset(self, 
                        audio_paths: List[str], 
                        true_labels: List[str]) -> Dict:
        """데이터셋 평가
        
        Args:
            audio_paths: 오디오 파일 경로 리스트
            true_labels: 실제 감정 라벨 리스트
            
        Returns:
            평가 결과 딕셔너리
        """
        
        print(f"평가 시작: {len(audio_paths)}개 샘플")
        
        # 예측 실행
        predictions = self.inference_engine.predict_batch(
            audio_paths, return_probabilities=True
        )
        
        # 성공적인 예측만 필터링
        successful_predictions = []
        filtered_true_labels = []
        predicted_labels = []
        predicted_probs = []
        
        for i, pred in enumerate(predictions):
            if pred['status'] == 'success':
                successful_predictions.append(pred)
                filtered_true_labels.append(true_labels[i])
                predicted_labels.append(pred['predicted_emotion'])
                
                # 확률 벡터 구성
                prob_vector = [pred['probabilities'][emotion] 
                              for emotion in model_config.emotion_labels]
                predicted_probs.append(prob_vector)
        
        if not successful_predictions:
            return {'error': '성공적인 예측이 없습니다.'}
        
        print(f"성공적인 예측: {len(successful_predictions)}개")
        
        # 라벨을 숫자로 변환
        label_to_id = model_config.label2id
        y_true = [label_to_id[label] for label in filtered_true_labels]
        y_pred = [label_to_id[label] for label in predicted_labels]
        y_probs = np.array(predicted_probs)
        
        # 기본 메트릭 계산
        metrics = self._calculate_metrics(y_true, y_pred, y_probs)
        
        # 클래스별 상세 분석
        class_metrics = self._calculate_class_metrics(y_true, y_pred)
        
        # 혼동 행렬
        cm = confusion_matrix(y_true, y_pred)
        
        # 결과 정리
        evaluation_results = {
            'overall_metrics': metrics,
            'class_metrics': class_metrics,
            'confusion_matrix': cm.tolist(),
            'num_samples': len(successful_predictions),
            'failed_predictions': len(audio_paths) - len(successful_predictions),
            'emotion_labels': model_config.emotion_labels,
            'predictions': successful_predictions[:10]  # 처음 10개 예측 결과 샘플
        }
        
        self.evaluation_results = evaluation_results
        return evaluation_results
    
    def _calculate_metrics(self, y_true: List[int], y_pred: List[int], y_probs: np.ndarray) -> Dict:
        """기본 메트릭 계산"""
        
        metrics = {}
        
        # 정확도
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # F1 스코어
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro')
        
        # Precision, Recall
        precision, recall, _, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        metrics['precision_weighted'] = precision
        metrics['recall_weighted'] = recall
        
        # AUC (다중 클래스의 경우 매크로 평균)
        try:
            # One-vs-Rest 방식으로 AUC 계산
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true, classes=list(range(len(model_config.emotion_labels))))
            
            if y_true_bin.shape[1] > 2:  # 다중 클래스
                auc_scores = []
                for i in range(y_true_bin.shape[1]):
                    if len(np.unique(y_true_bin[:, i])) > 1:  # 해당 클래스가 존재하는 경우
                        auc = roc_auc_score(y_true_bin[:, i], y_probs[:, i])
                        auc_scores.append(auc)
                metrics['auc_macro'] = np.mean(auc_scores) if auc_scores else 0.0
            else:  # 이진 분류
                metrics['auc'] = roc_auc_score(y_true, y_probs[:, 1])
                
        except Exception as e:
            print(f"AUC 계산 중 오류: {e}")
            metrics['auc_macro'] = 0.0
        
        return metrics
    
    def _calculate_class_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict:
        """클래스별 상세 메트릭 계산"""
        
        # 분류 보고서
        report = classification_report(
            y_true, y_pred,
            target_names=model_config.emotion_labels,
            output_dict=True,
            zero_division=0
        )
        
        # 클래스별 메트릭 정리
        class_metrics = {}
        
        for i, emotion in enumerate(model_config.emotion_labels):
            if emotion in report:
                class_metrics[emotion] = {
                    'precision': report[emotion]['precision'],
                    'recall': report[emotion]['recall'],
                    'f1_score': report[emotion]['f1-score'],
                    'support': report[emotion]['support']
                }
        
        return class_metrics
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (12, 10)):
        """혼동 행렬 시각화"""
        
        if 'confusion_matrix' not in self.evaluation_results:
            print("평가 결과가 없습니다. evaluate_dataset을 먼저 실행하세요.")
            return
        
        cm = np.array(self.evaluation_results['confusion_matrix'])
        
        plt.figure(figsize=figsize)
        
        # 정규화된 혼동 행렬
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=model_config.emotion_labels,
            yticklabels=model_config.emotion_labels,
            cbar_kws={'label': 'Normalized Count'}
        )
        
        plt.title('Confusion Matrix (Normalized)', fontsize=16)
        plt.xlabel('Predicted Emotion', fontsize=12)
        plt.ylabel('True Emotion', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"혼동 행렬이 저장되었습니다: {save_path}")
        
        plt.show()
    
    def plot_class_metrics(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (14, 8)):
        """클래스별 메트릭 시각화"""
        
        if 'class_metrics' not in self.evaluation_results:
            print("평가 결과가 없습니다. evaluate_dataset을 먼저 실행하세요.")
            return
        
        class_metrics = self.evaluation_results['class_metrics']
        
        # 데이터 준비
        emotions = list(class_metrics.keys())
        precision = [class_metrics[emotion]['precision'] for emotion in emotions]
        recall = [class_metrics[emotion]['recall'] for emotion in emotions]
        f1 = [class_metrics[emotion]['f1_score'] for emotion in emotions]
        
        # 그래프 생성
        x = np.arange(len(emotions))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=figsize)
        
        bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8)
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Emotions', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Performance Metrics', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(emotions, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.0)
        
        # 값 표시
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom',
                          fontsize=8)
        
        autolabel(bars1)
        autolabel(bars2)
        autolabel(bars3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"클래스별 메트릭 그래프가 저장되었습니다: {save_path}")
        
        plt.show()
    
    def generate_evaluation_report(self, save_path: Optional[str] = None) -> str:
        """평가 보고서 생성"""
        
        if not self.evaluation_results:
            return "평가 결과가 없습니다."
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("음성 감정 분석 모델 평가 보고서")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # 모델 정보
        model_info = self.inference_engine.get_model_info()
        report_lines.append("모델 정보:")
        report_lines.append(f"  - 모델명: {model_info['model_name']}")
        report_lines.append(f"  - 감정 클래스 수: {model_info['num_labels']}")
        report_lines.append(f"  - 감정 라벨: {', '.join(model_info['emotion_labels'])}")
        report_lines.append("")
        
        # 데이터셋 정보
        results = self.evaluation_results
        report_lines.append("데이터셋 정보:")
        report_lines.append(f"  - 전체 샘플 수: {results['num_samples']}")
        report_lines.append(f"  - 실패한 예측: {results['failed_predictions']}")
        report_lines.append("")
        
        # 전체 성능 메트릭
        overall = results['overall_metrics']
        report_lines.append("전체 성능 메트릭:")
        report_lines.append(f"  - 정확도 (Accuracy): {overall['accuracy']:.4f}")
        report_lines.append(f"  - F1-Score (Macro): {overall['f1_macro']:.4f}")
        report_lines.append(f"  - F1-Score (Weighted): {overall['f1_weighted']:.4f}")
        report_lines.append(f"  - Precision (Weighted): {overall['precision_weighted']:.4f}")
        report_lines.append(f"  - Recall (Weighted): {overall['recall_weighted']:.4f}")
        if 'auc_macro' in overall:
            report_lines.append(f"  - AUC (Macro): {overall['auc_macro']:.4f}")
        report_lines.append("")
        
        # 클래스별 성능
        report_lines.append("클래스별 성능:")
        class_metrics = results['class_metrics']
        
        for emotion in model_config.emotion_labels:
            if emotion in class_metrics:
                metrics = class_metrics[emotion]
                report_lines.append(f"  {emotion}:")
                report_lines.append(f"    - Precision: {metrics['precision']:.4f}")
                report_lines.append(f"    - Recall: {metrics['recall']:.4f}")
                report_lines.append(f"    - F1-Score: {metrics['f1_score']:.4f}")
                report_lines.append(f"    - Support: {metrics['support']}")
                report_lines.append("")
        
        # 성능 분석
        report_lines.append("성능 분석:")
        
        # 최고/최저 성능 클래스
        f1_scores = {emotion: class_metrics[emotion]['f1_score'] 
                    for emotion in class_metrics}
        
        if f1_scores:
            best_emotion = max(f1_scores, key=f1_scores.get)
            worst_emotion = min(f1_scores, key=f1_scores.get)
            
            report_lines.append(f"  - 최고 성능 감정: {best_emotion} (F1: {f1_scores[best_emotion]:.4f})")
            report_lines.append(f"  - 최저 성능 감정: {worst_emotion} (F1: {f1_scores[worst_emotion]:.4f})")
            report_lines.append("")
        
        # 혼동 행렬 분석
        cm = np.array(results['confusion_matrix'])
        report_lines.append("혼동 행렬 분석:")
        
        # 대각선 원소 (올바른 예측)
        correct_predictions = np.diag(cm)
        total_per_class = cm.sum(axis=1)
        
        for i, emotion in enumerate(model_config.emotion_labels):
            if i < len(correct_predictions):
                if total_per_class[i] > 0:
                    accuracy = correct_predictions[i] / total_per_class[i]
                    report_lines.append(f"  - {emotion}: {correct_predictions[i]}/{total_per_class[i]} ({accuracy:.4f})")
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"평가 보고서가 저장되었습니다: {save_path}")
        
        return report
    
    def save_results(self, save_path: str):
        """평가 결과를 JSON으로 저장"""
        
        if not self.evaluation_results:
            print("저장할 평가 결과가 없습니다.")
            return
        
        # NumPy 배열을 리스트로 변환
        results_to_save = self.evaluation_results.copy()
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=2, ensure_ascii=False)
        
        print(f"평가 결과가 저장되었습니다: {save_path}")

def main():
    """CLI 인터페이스"""
    
    parser = argparse.ArgumentParser(description="음성 감정 분석 모델 평가")
    
    parser.add_argument("--model_path", type=str, required=True, help="평가할 모델 경로")
    parser.add_argument("--data_dir", type=str, help="테스트 데이터 디렉토리")
    parser.add_argument("--csv_path", type=str, help="테스트 데이터 CSV 파일")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="결과 저장 디렉토리")
    parser.add_argument("--no_plots", action="store_true", help="그래프 생성 안함")
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 평가기 생성
    evaluator = ModelEvaluator(args.model_path)
    
    # 테스트 데이터 로드
    if args.data_dir:
        audio_paths, labels = load_dataset_from_directory(args.data_dir)
    elif args.csv_path:
        audio_paths, labels = load_dataset_from_csv(args.csv_path)
    else:
        print("--data_dir 또는 --csv_path 중 하나를 지정해주세요.")
        return
    
    if not audio_paths:
        print("테스트 데이터를 찾을 수 없습니다.")
        return
    
    # 평가 실행
    print(f"모델 평가 시작...")
    results = evaluator.evaluate_dataset(audio_paths, labels)
    
    if 'error' in results:
        print(f"평가 실패: {results['error']}")
        return
    
    # 결과 출력
    print("\n평가 완료!")
    print(f"정확도: {results['overall_metrics']['accuracy']:.4f}")
    print(f"F1-Score (Weighted): {results['overall_metrics']['f1_weighted']:.4f}")
    
    # 보고서 생성
    report_path = os.path.join(args.output_dir, "evaluation_report.txt")
    report = evaluator.generate_evaluation_report(report_path)
    print(f"\n평가 보고서 미리보기:")
    print(report[:1000] + "..." if len(report) > 1000 else report)
    
    # 결과 저장
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    evaluator.save_results(results_path)
    
    # 그래프 생성
    if not args.no_plots:
        cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
        metrics_path = os.path.join(args.output_dir, "class_metrics.png")
        
        evaluator.plot_confusion_matrix(cm_path)
        evaluator.plot_class_metrics(metrics_path)

if __name__ == "__main__":
    main()