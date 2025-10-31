import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

"""
================================================================================
METRICS CALCULATOR CLASS - 메트릭 계산 클래스
================================================================================
"""
class MetricsCalculator:
    """
    모델 평가를 위한 메트릭 계산 클래스
    Sensitivity, Specificity, F1 등 다양한 지표 계산
    """
    
    @staticmethod
    def calculate_sensitivity_specificity(y_true, y_pred, num_classes):
        """
        클래스별 sensitivity와 specificity 계산
        
        Args:
            y_true: 실제 레이블
            y_pred: 예측 레이블
            num_classes: 클래스 개수
        
        Returns:
            sensitivity: 클래스별 민감도 리스트
            specificity: 클래스별 특이도 리스트
        """
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        
        sensitivity = []
        specificity = []
        
        for i in range(num_classes):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            tn = np.sum(cm) - tp - fn - fp
            
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            sensitivity.append(sens)
            specificity.append(spec)
        
        return sensitivity, specificity
    
    @staticmethod
    def calculate_weighted_metrics(y_true, y_pred, class_indices=None):
        """
        가중 평균 메트릭 계산
        특정 클래스만 선택하여 계산 가능
        
        Args:
            y_true: 실제 레이블
            y_pred: 예측 레이블
            class_indices: 평가할 클래스 인덱스 (None이면 전체)
        
        Returns:
            dict: accuracy, weighted_f1, sensitivity, specificity 등
        """
        if class_indices is not None:
            # 특정 클래스만 필터링
            mask = np.isin(y_true, class_indices)
            y_true_filtered = y_true[mask]
            y_pred_filtered = y_pred[mask]
            
            # 레이블 재매핑 (0, 1, 2, ...)
            label_mapping = {old: new for new, old in enumerate(class_indices)}
            y_true_remapped = np.array([label_mapping[label] for label in y_true_filtered])
            y_pred_remapped = np.array([
                label_mapping[label] if label in label_mapping else -1
                for label in y_pred_filtered
            ])
            
            # 유효한 예측만 사용
            valid_mask = y_pred_remapped != -1
            y_true_final = y_true_remapped[valid_mask]
            y_pred_final = y_pred_remapped[valid_mask]
            num_classes = len(class_indices)
        else:
            y_true_final = y_true
            y_pred_final = y_pred
            num_classes = len(np.unique(y_true))
        
        if len(y_true_final) == 0:
            return {
                'accuracy': 0.0,
                'weighted_f1': 0.0,
                'sensitivity': [0.0] * num_classes,
                'specificity': [0.0] * num_classes,
                'weighted_sensitivity': 0.0,
                'weighted_specificity': 0.0
            }
        
        # 메트릭 계산
        accuracy = np.mean(y_true_final == y_pred_final)
        sensitivity, specificity = MetricsCalculator.calculate_sensitivity_specificity(
            y_true_final, y_pred_final, num_classes
        )
        
        weighted_f1 = f1_score(
            y_true_final, y_pred_final, average='weighted', zero_division=0
        )
        
        # 클래스별 가중치로 가중 평균 계산
        class_counts = np.bincount(y_true_final, minlength=num_classes)
        total_samples = np.sum(class_counts)
        
        if total_samples > 0:
            weights = class_counts / total_samples
            weighted_sensitivity = np.average(sensitivity, weights=weights)
            weighted_specificity = np.average(specificity, weights=weights)
        else:
            weighted_sensitivity = 0.0
            weighted_specificity = 0.0
        
        return {
            'accuracy': accuracy,
            'weighted_f1': weighted_f1,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'weighted_sensitivity': weighted_sensitivity,
            'weighted_specificity': weighted_specificity
        }

