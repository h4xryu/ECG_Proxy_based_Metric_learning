import os
import numpy as np

"""
================================================================================
PREPROCESSING UTILS - 전처리 유틸리티
================================================================================
"""
def label2index(label):
    """
    ECG 비트 타입 문자 레이블을 정수 인덱스로 변환
    N: Normal, S: Supraventricular, V: Ventricular, F: Fusion, Q: Unknown
    
    Args:
        label: 문자 레이블 (N, S, V, F, Q)
    
    Returns:
        정수 인덱스 (0-4)
    """
    mapping = {
        'N': 0,  # Normal beat
        'S': 1,  # Supraventricular ectopic beat
        'V': 2,  # Ventricular ectopic beat
        'F': 3,  # Fusion beat
        'Q': 4   # Unknown beat
    }
    return mapping[label]


def mkdir(path):
    """경로가 존재하지 않으면 디렉토리 생성"""
    if not os.path.exists(path):
        os.makedirs(path)


def normalize_ecg(ecg_data):
    """
    Z-score 정규화를 사용한 ECG 신호 정규화
    평균 0, 표준편차 1로 변환
    
    Args:
        ecg_data: 정규화할 ECG 데이터
    
    Returns:
        정규화된 ECG 데이터
    """
    mean = np.mean(ecg_data, axis=0, keepdims=True)
    std = np.std(ecg_data, axis=0, keepdims=True)
    return (ecg_data - mean) / (std + 1e-8)
