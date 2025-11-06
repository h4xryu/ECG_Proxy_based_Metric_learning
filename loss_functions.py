"""
Proxy-Anchor Loss Functions
논문 수식 기반 간결한 구현 + 손실 함수 설정 및 계산
"""
import torch
import torch.nn as nn
import torch.nn.functional as F



# ===================== Proxy-Anchor Loss =====================
class ProxyAnchorLoss(nn.Module):
    """
    Proxy-Anchor Loss (CVPR 2020)
    Paper: https://arxiv.org/abs/2003.13911

    """
    def __init__(self, alpha=32.0, delta=0.1):
        super().__init__()
        self.alpha = alpha
        self.delta = delta
    
    def forward(self, features, labels, proxies):
  
        S = F.normalize(features, dim=-1) @ F.normalize(proxies, dim=-1).t()
        
        B, C = S.shape
        P_onehot =  F.one_hot(labels, num_classes=C).float()  # [B, C]
        N_onehot = 1 - P_onehot
        
        pos_exp = torch.exp(-self.alpha * (S - self.delta))
        pos_exp = torch.where(P_onehot == 1, pos_exp, torch.zeros_like(pos_exp))
        
        with_pos = (P_onehot.sum(dim=0) != 0).float()
        num_valid = with_pos.sum().clamp(min=1)
        
        pos_term = torch.log(1 + pos_exp.sum(dim=0))
        pos_loss = (pos_term * with_pos).sum() / num_valid
        
        neg_exp = torch.exp(self.alpha * (S + self.delta))
        neg_exp = torch.where(N_onehot == 1, neg_exp, torch.zeros_like(neg_exp))
        
        neg_term = torch.log(1 + neg_exp.sum(dim=0))
        neg_loss = neg_term.mean()
        
        return pos_loss + neg_loss


# ===================== Focal Style Proxy-Anchor Loss =====================
class FocalStyleProxyAnchorLoss(nn.Module):
    """
    Focal weighting을 적용한 Proxy-Anchor Loss
    어려운 샘플에 더 높은 가중치 부여
    """
    def __init__(self, alpha=32.0, delta=0.1, pos_gamma=2.0, neg_gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.delta = delta
        self.pos_gamma = pos_gamma
        self.neg_gamma = neg_gamma
    
    def forward(self, features, labels, proxies):
        S = F.normalize(features, dim=-1) @ F.normalize(proxies, dim=-1).t()
        
        B, C = S.shape
        P_onehot = F.one_hot(labels, num_classes=C).float()
        N_onehot = 1 - P_onehot
        
        pos_weight = ((1.0 - S).clamp(min=0, max=1) ** self.pos_gamma) * P_onehot
        pos_exp = torch.exp(-self.alpha * (S - self.delta)) * pos_weight
        
        with_pos = (P_onehot.sum(dim=0) != 0).float()
        num_valid = with_pos.sum().clamp(min=1)
        
        pos_term = torch.log(1 + pos_exp.sum(dim=0))
        pos_loss = (pos_term * with_pos).sum() / num_valid
        
        neg_weight = (S.clamp(min=0, max=1) ** self.neg_gamma) * N_onehot
        neg_exp = torch.exp(self.alpha * (S + self.delta)) * neg_weight
        
        neg_term = torch.log(1 + neg_exp.sum(dim=0))
        neg_loss = neg_term.mean()
        
        return pos_loss + neg_loss




# ===================== Focal Loss =====================
class FocalLoss(nn.Module):
    """
    Focal Loss (ICCV 2017)
    FL(p_t) = -α_t * (1-p_t)^γ * log(p_t)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C] model outputs
            targets: [B] class labels
        """
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        
        # Gather target probabilities
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        focal_weight = (1.0 - pt) ** self.gamma

        loss = -self.alpha * focal_weight * log_pt
        
        return loss.mean() if self.reduction == 'mean' else loss.sum() if self.reduction == 'sum' else loss


"""
================================================================================
PROXY INFERENCE
================================================================================
"""

def proxy_test(features, proxies, temperature=1.0, return_probs=True):
    """
    Proxy 기반 추론: 가장 가까운 proxy로 분류
    
    Args:
        features: [B, D] feature vectors
        proxies: [C, D] proxy vectors
        temperature: similarity scaling
    
    Returns:
        predictions: [B] predicted class indices
    """
    x = F.normalize(features, dim=-1)
    P = F.normalize(proxies, dim=-1)
    
    # 코사인 유사도 계산 및 temperature scaling
    similarities = (x @ P.T) / temperature
    
    # 가장 높은 유사도를 가진 클래스 선택
    predictions = torch.argmax(similarities, dim=1)
    
    return predictions


def proxy_test2(features, proxies, temperature=1.0, return_probs=True):
    """
    Proxy 기반 추론: 가장 가까운 proxy로 분류
    
    Args:
        features: [B, D] feature vectors
        proxies: [C, D] proxy vectors
        temperature: similarity scaling
    
    Returns:
        predictions: [B] predicted class indices
    """
    x = F.normalize(features, dim=-1)
    P = F.normalize(proxies, dim=-1)
    
    # 코사인 유사도 계산 및 temperature scaling
    similarities = (x @ P.T) / temperature
        
    return similarities

"""
================================================================================
LOSS SETUP & COMPUTATION
================================================================================
"""

def setup_losses():
    """
    손실 함수 초기화
    
    Returns:
        cross_entropy_loss: CE loss (always initialized)
        proxy_loss: Proxy loss (always initialized, used based on lambda_combined at runtime)
    """
    from train import proxy_type, proxy_alpha, proxy_delta
    
    cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')
    
    # Proxy loss는 항상 초기화 (lambda_combined는 런타임에 체크)
    if proxy_type == 'ProxyAnchorLoss':
        proxy_loss = ProxyAnchorLoss(alpha=proxy_alpha, delta=proxy_delta)
    elif proxy_type == 'FocalStyleProxyAnchorLoss':
        proxy_loss = FocalStyleProxyAnchorLoss(alpha=proxy_alpha, delta=proxy_delta)
    else:
        raise ValueError(f"Unknown proxy_type: {proxy_type}")
    
    return cross_entropy_loss, proxy_loss


def compute_loss(outputs, features, labels, cross_entropy_loss, 
                 proxy_loss, model_proxies, device):
    """
    손실 함수 계산 (λ로 제어)
    
    L = λ*L_CE + (1-λ)*L_Proxy
    
    λ (lambda_combined) ∈ [-1, 0, 1]:
    - λ = -1: Focal Loss only
    - λ = 1.0: CE only
    - λ = 0.0: Proxy only
    - 0 < λ < 1: Combined
    """
    from train import lambda_combined

    if lambda_combined == -1:
        # Focal Loss 인스턴스 생성 후 호출
        focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        loss = focal_loss_fn(outputs, labels)
        
        proxy_loss_val = torch.tensor(0.0, device=device)
        ce_loss = torch.tensor(0.0, device=device)  # 로깅용
        
        total_loss = loss
    
    else:
        # CE Loss (always computed for logging)
        ce_loss = cross_entropy_loss(outputs, labels)
        
        # Proxy Loss (computed if lambda < 1.0)
        proxy_loss_val = torch.tensor(0.0, device=device)
        if lambda_combined < 1.0:
            proxy_loss_val = proxy_loss(features, labels, model_proxies)
        
        # λ*CE + (1-λ)*Proxy
        total_loss = lambda_combined * ce_loss + (1 - lambda_combined) * proxy_loss_val
    
    return total_loss, proxy_loss_val, ce_loss


def get_predictions(outputs, features, model_proxies):
    """
    - λ=1: argmax 사용 (분류기 출력)
    - λ=0: proxy inference 사용 (nearest proxy)
    """
    from train_batch_loss import lambda_combined
    
    # Handle case when model_proxies is None
    if model_proxies is None or lambda_combined == 1.0:
        return torch.argmax(outputs, dim=1)
    elif lambda_combined == 0.0:
        return torch.argmax(proxy_test2(features, model_proxies), dim=1)
    else:
        return torch.argmax(
            lambda_combined * outputs + (1-lambda_combined) * proxy_test2(features, model_proxies), 
            dim=1
        )
