import torch
import torch.nn.functional as F
import numpy as np

"""
================================================================================
PROXY INFERENCE
================================================================================
"""

def proxy_test(features, proxies, temperature=1.0, return_probs=True):
    """
    Proxy 벡터를 이용한 추론 함수
    학습된 proxy 벡터와의 cosine similarity를 기반으로 클래스 예측
    
    Args:
        features: [B, D] 형태의 특징 벡터 (배치 크기 x 특징 차원)
        proxies: [C, D] 형태의 클래스별 proxy 벡터 (클래스 수 x 특징 차원)
        temperature: temperature scaling 파라미터 (기본값 1.0)
        return_probs: 확률값 반환 여부 (현재 미구현)
    
    Returns:
        predictions: [B] 형태의 예측 클래스 인덱스
    """
    
    # L2 정규화로 cosine similarity 계산 준비
    x = F.normalize(features, dim=-1)
    P = F.normalize(proxies, dim=-1)
    
    # 코사인 유사도 계산 및 temperature scaling
    similarities = (x @ P.T) / temperature
    
    # 가장 높은 유사도를 가진 클래스 선택
    predictions = torch.argmax(similarities, dim=1)
    
    return predictions