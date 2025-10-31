import torch
import torch.nn as nn
import torch.nn.functional as F


"""
================================================================================
LOSS CLASSES - 손실 함수 클래스들
================================================================================
"""

"""
1. ProxyAnchorLoss
   - 기본 Proxy-Anchor Loss
   - 각 클래스당 하나의 proxy 사용
   - Cosine similarity 기반
"""
class ProxyAnchorLoss(nn.Module):

    def __init__(self, alpha: float = 32.0, delta: float = 0.1,
                 use_cosine: bool = True, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.delta = delta
        self.use_cosine = use_cosine
        self.reduction = reduction

    def forward(self, features: torch.Tensor, target: torch.Tensor, proxies: torch.Tensor):
        S = ProxyLossUtils.compute_similarity_matrix(features, proxies, self.use_cosine)
        B, C = S.shape
        
        onehot = F.one_hot(target, num_classes=C).to(torch.bool)
        
        pos_loss = ProxyLossUtils.compute_positive_loss(S, onehot, self.alpha, self.delta)
        neg_loss = ProxyLossUtils.compute_negative_loss(S, onehot, self.alpha, self.delta)
        
        loss = pos_loss + neg_loss
        
        if self.reduction == "sum":
            return loss * B
        elif self.reduction == "none":
            return torch.stack((pos_loss, neg_loss))
        return loss


"""
2. FocalStyleProxyAnchorLoss
   - Focal weighting을 적용한 Proxy-Anchor Loss
   - 어려운 샘플에 더 높은 가중치
   - Positive/Negative에 각각 다른 gamma 적용 가능
"""
class FocalStyleProxyAnchorLoss(nn.Module):

    def __init__(self, alpha: float = 32.0, delta: float = 0.1,
                 pos_gamma: float = 2.0, neg_gamma: float = 2.0,
                 use_cosine: bool = True, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.delta = delta
        self.pos_gamma = pos_gamma
        self.neg_gamma = neg_gamma
        self.use_cosine = use_cosine
        self.reduction = reduction

    def forward(self, features: torch.Tensor, target: torch.Tensor, proxies: torch.Tensor):
        S = ProxyLossUtils.compute_similarity_matrix(features, proxies, self.use_cosine)
        B, C = S.shape
        device = S.device
        
        onehot = F.one_hot(target, num_classes=C).to(torch.bool)
        
        # Positive term with focal weighting
        pos_scores = -self.alpha * (S - self.delta)
        pos_scores = pos_scores.masked_fill(~onehot, float('-inf'))
        
        pos_weights = ProxyLossUtils.compute_focal_weights(S, onehot, self.pos_gamma, is_positive=True)
        pos_lse = ProxyLossUtils.compute_weighted_logsumexp(pos_scores, pos_weights)
        
        pos_exist = onehot.any(dim=0)
        if pos_exist.any():
            pos_loss = F.softplus(pos_lse[pos_exist]).mean()
        else:
            pos_loss = torch.tensor(0., device=device)
        
        # Negative term with focal weighting
        neg_scores = self.alpha * (S + self.delta)
        neg_scores = neg_scores.masked_fill(onehot, float('-inf'))
        
        neg_weights = ProxyLossUtils.compute_focal_weights(S, onehot, self.neg_gamma, is_positive=False)
        neg_lse = ProxyLossUtils.compute_weighted_logsumexp(neg_scores, neg_weights)
        neg_loss = F.softplus(neg_lse).mean()
        
        loss = pos_loss + neg_loss
        
        if self.reduction == "sum":
            return loss * B
        elif self.reduction == "none":
            return torch.stack((pos_loss, neg_loss))
        return loss


"""
3. FocalLoss
   - Multi-class Focal Loss
   - 클래스 불균형 문제 해결
   - 쉬운 샘플의 가중치를 줄이고 어려운 샘플에 집중
"""
class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma: float = 2.0, 
                 reduction: str = "mean", ignore_index: int = -100):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        
        if alpha is None:
            self.register_buffer("alpha", None)
        else:
            if isinstance(alpha, (list, tuple)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            elif isinstance(alpha, (float, int)):
                alpha = torch.tensor([alpha], dtype=torch.float32)
            self.register_buffer("alpha", alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid = (targets != self.ignore_index)
        if not valid.any():
            return logits.new_tensor(0., requires_grad=True)
        
        logits = logits[valid]
        targets = targets[valid]
        
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        
        target_idx = targets.unsqueeze(1)
        log_p_t = log_probs.gather(1, target_idx).squeeze(1)
        p_t = probs.gather(1, target_idx).squeeze(1)
        
        focal_weight = (1.0 - p_t).clamp(min=0.0) ** self.gamma
        
        if self.alpha is None:
            alpha_t = 1.0
        else:
            if self.alpha.numel() == 1:
                alpha_t = self.alpha.item()
            else:
                alpha_t = self.alpha.to(logits.device).gather(0, targets)
        
        loss = -alpha_t * focal_weight * log_p_t
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


"""
4. MultiProxyAnchorLoss
   - 각 클래스당 여러 개의 proxy 사용
   - 표현력 향상 및 앙상블 효과
   - Top-K hard mining 지원
   - Softplus threshold tuning
"""
class MultiProxyAnchorLoss(nn.Module):

    def __init__(self, alpha: float = 32.0, delta: float = 0.1,
                 use_cosine: bool = True, reduction: str = "mean",
                 num_proxies_per_class: int = 1, eps: float = 1e-14,
                 softplus_threshold: float = 20.0, topk_pos: int = None,
                 topk_neg: int = None, amp_safe_matmul: bool = True):
        super().__init__()
        self.alpha = alpha
        self.delta = delta
        self.use_cosine = use_cosine
        self.reduction = reduction
        self.num_proxies_per_class = num_proxies_per_class
        self.eps = eps
        self.softplus_threshold = softplus_threshold
        self.topk_pos = topk_pos
        self.topk_neg = topk_neg
        self.amp_safe_matmul = amp_safe_matmul
    
    def forward(self, features: torch.Tensor, target: torch.Tensor, proxies: torch.Tensor):
        B = features.size(0)
        
        S, C = ProxyLossUtils.compute_multi_proxy_similarity(
            features, proxies, self.num_proxies_per_class,
            self.use_cosine, self.eps, self.amp_safe_matmul
        )
        
        onehot = F.one_hot(target, num_classes=C).to(torch.bool, non_blocking=True)
        
        pos_loss = ProxyLossUtils.compute_stable_positive_loss(
            S, onehot, self.alpha, self.delta, 
            self.topk_pos, self.softplus_threshold
        )
        neg_loss = ProxyLossUtils.compute_stable_negative_loss(
            S, onehot, self.alpha, self.delta,
            self.topk_neg, self.softplus_threshold
        )
        
        loss = pos_loss + neg_loss
        
        if self.reduction == "sum":
            return loss * B
        elif self.reduction == "none":
            return torch.stack((pos_loss, neg_loss))
        return loss


"""
================================================================================
UTILITY CLASS - 공통 유틸리티 메소드들
================================================================================
"""

class ProxyLossUtils(nn.Module):
    
    @staticmethod
    def compute_similarity_matrix(features, proxies, use_cosine=True):
        """
        특징 벡터와 proxy 벡터 간의 유사도 행렬 계산
        
        Args:
            features: [B, D] 샘플 특징 벡터
            proxies: [C, D] 클래스별 proxy 벡터
            use_cosine: cosine similarity 사용 여부
        
        Returns:
            S: [B, C] 유사도 행렬
        """
        if use_cosine:
            features = F.normalize(features, dim=-1)
            proxies = F.normalize(proxies, dim=-1)
        
        return features @ proxies.t()
    
    @staticmethod
    def compute_positive_loss(similarity_matrix, onehot_labels, alpha, delta):
        """
        Proxy-Anchor Loss의 positive term 계산
        같은 클래스의 샘플들을 proxy에 가깝게 만드는 손실
        
        Args:
            similarity_matrix: [B, C] 유사도 행렬
            onehot_labels: [B, C] one-hot 인코딩된 레이블
            alpha: 스케일링 파라미터
            delta: margin 파라미터
        
        Returns:
            positive 손실값
        """
        device = similarity_matrix.device
        
        pos_scores = -alpha * (similarity_matrix - delta)
        pos_scores = pos_scores.masked_fill(~onehot_labels, float('-inf'))
        
        pos_lse = torch.logsumexp(pos_scores, dim=0)
        
        pos_exist = onehot_labels.any(dim=0)
        if pos_exist.any():
            return F.softplus(pos_lse[pos_exist]).mean()
        else:
            return torch.tensor(0., device=device)
    
    @staticmethod
    def compute_negative_loss(similarity_matrix, onehot_labels, alpha, delta):
        """
        Proxy-Anchor Loss의 negative term 계산
        다른 클래스의 샘플들을 proxy에서 멀게 만드는 손실
        
        Args:
            similarity_matrix: [B, C] 유사도 행렬
            onehot_labels: [B, C] one-hot 인코딩된 레이블
            alpha: 스케일링 파라미터
            delta: margin 파라미터
        
        Returns:
            negative 손실값
        """
        neg_scores = alpha * (similarity_matrix + delta)
        neg_scores = neg_scores.masked_fill(onehot_labels, float('-inf'))
        
        neg_lse = torch.logsumexp(neg_scores, dim=0)
        return F.softplus(neg_lse).mean()
    
    @staticmethod
    def compute_focal_weights(similarity_matrix, onehot_labels, gamma, is_positive=True):
        """
        Focal style 가중치 계산
        어려운 샘플에 높은 가중치를 부여
        
        Args:
            similarity_matrix: [B, C] 유사도 행렬
            onehot_labels: [B, C] one-hot 인코딩된 레이블
            gamma: focusing 파라미터
            is_positive: positive 샘플 가중치인지 여부
        
        Returns:
            weights: [B, C] focal 가중치
        """
        if is_positive:
            weights = ((1.0 - similarity_matrix).clamp(min=0.0, max=1.0)) ** gamma
            weights = weights.masked_fill(~onehot_labels, 0.0)
        else:
            weights = (similarity_matrix.clamp(min=0.0, max=1.0)) ** gamma
            weights = weights.masked_fill(onehot_labels, 0.0)
        
        return weights
    
    @staticmethod
    def compute_weighted_logsumexp(scores, weights):
        """
        가중치가 적용된 Log-Sum-Exp 계산
        수치 안정성을 위해 max value를 빼고 계산
        
        Args:
            scores: [B, C] 점수 행렬
            weights: [B, C] 가중치 행렬
        
        Returns:
            weighted_lse: [C] 각 proxy별 weighted LSE
        """
        C = scores.shape[1]
        device = scores.device
        
        max_scores, _ = torch.max(scores, dim=0, keepdim=True)
        weighted_exp_sum = torch.sum(weights * torch.exp(scores - max_scores), dim=0)
        
        weighted_lse = torch.full((C,), float('-inf'), device=device)
        valid = (weighted_exp_sum > 0)
        weighted_lse[valid] = max_scores.squeeze(0)[valid] + torch.log(weighted_exp_sum[valid])
        
        return weighted_lse
    
    @staticmethod
    def compute_multi_proxy_similarity(features, proxies, num_proxies_per_class, 
                                       use_cosine=True, eps=1e-6, amp_safe=True):
        """
        Multi-proxy similarity calculation
        각 클래스당 여러 개의 proxy를 사용하고 최대값 선택
        
        Args:
            features: [B, D] 샘플 특징 벡터
            proxies: [C*M, D] 전체 proxy (M개/클래스)
            num_proxies_per_class: 클래스당 proxy 개수
            use_cosine: cosine similarity 사용 여부
            eps: normalization epsilon
            amp_safe: FP32 누적 사용 여부
        
        Returns:
            S: [B, C] 클래스별 최대 유사도
            C: 클래스 개수
        """
        B = features.size(0)
        total_proxies = proxies.size(0)
        M = num_proxies_per_class
        C = total_proxies // M
        
        if use_cosine:
            features = F.normalize(features, dim=-1, eps=eps)
            proxies = F.normalize(proxies, dim=-1, eps=eps)
        
        if amp_safe:
            S_all = (features.float() @ proxies.float().t()).to(features.dtype)
        else:
            S_all = features @ proxies.t()
        
        S_all = S_all.clamp(min=-1.0 + eps, max=1.0 - eps)
        S = S_all.reshape(B, C, M).amax(dim=2)
        
        return S, C
    
    @staticmethod
    def apply_topk_mining(scores, onehot_labels, topk, is_positive=True):
        """
        Top-K hard mining
        
        Args:
            scores: [B, C] 점수 행렬
            onehot_labels: [B, C] one-hot 레이블
            topk: 선택할 샘플 수
            is_positive: positive 샘플인지 여부
        
        Returns:
            lse: [C] Top-K에 대한 Log-Sum-Exp
        """
        B = scores.size(0)
        
        if topk is not None:
            k = min(topk, B)
            topk_scores, _ = torch.topk(scores, k=k, dim=0, largest=True)
            lse = torch.logsumexp(topk_scores, dim=0)
        else:
            lse = torch.logsumexp(scores, dim=0)
        
        return lse
    
    @staticmethod
    def compute_stable_positive_loss(similarity_matrix, onehot_labels, alpha, delta, 
                                     topk=None, softplus_threshold=20.0):
        """
        positive term 계산
        Top-K hard mining 및 softplus threshold 적용
        
        Args:
            similarity_matrix: [B, C] 유사도 행렬
            onehot_labels: [B, C] one-hot 레이블
            alpha: 스케일링 파라미터
            delta: margin 파라미터
            topk: hard mining을 위한 top-k
            softplus_threshold: softplus 안정화 임계값
        
        Returns:
            positive 손실값
        """
        device = similarity_matrix.device
        
        pos_scores = -alpha * (similarity_matrix - delta)
        pos_scores = pos_scores.masked_fill(~onehot_labels, float('-inf'))
        
        pos_lse = ProxyLossUtils.apply_topk_mining(pos_scores, onehot_labels, topk, is_positive=True)
        
        pos_exist = onehot_labels.any(dim=0)
        if pos_exist.any():
            return F.softplus(pos_lse[pos_exist], beta=1.0, threshold=softplus_threshold).mean()
        else:
            return torch.tensor(0., device=device)
    
    @staticmethod
    def compute_stable_negative_loss(similarity_matrix, onehot_labels, alpha, delta,
                                     topk=None, softplus_threshold=20.0):
        """
        negative term 계산
        Top-K hard mining 및 softplus threshold 적용
        
        Args:
            similarity_matrix: [B, C] 유사도 행렬
            onehot_labels: [B, C] one-hot 레이블
            alpha: 스케일링 파라미터
            delta: margin 파라미터
            topk: hard mining을 위한 top-k
            softplus_threshold: softplus 안정화 임계값
        
        Returns:
            negative 손실값
        """
        neg_scores = alpha * (similarity_matrix + delta)
        neg_scores = neg_scores.masked_fill(onehot_labels, float('-inf'))
        
        neg_lse = ProxyLossUtils.apply_topk_mining(neg_scores, onehot_labels, topk, is_positive=False)
        neg_lse = torch.nan_to_num(neg_lse, neginf=neg_lse.new_tensor(-float('inf')))
        
        return F.softplus(neg_lse, beta=1.0, threshold=softplus_threshold).mean()
