# Loss Function Guide

## 손실 함수 종류

### 1. CE (Cross Entropy)
```python
L = L_CE
```

**설명**: 표준 Cross Entropy Loss만 사용

**설정**:
```python
loss_type = 'CE'
```

---

### 2. Proxy (Proxy-Anchor Loss)
```python
L = L_Proxy

where:
L_Proxy = (1/|P+|) Σ log(1 + Σ exp(-α(s(x,p)-δ)))
        + (1/|P|)  Σ log(1 + Σ exp(α(s(x,p)+δ)))
```

**설명**: Proxy-Anchor Loss만 사용 (Metric Learning)

**파라미터**:
- `s(x,p)`: cosine similarity between sample x and proxy p
- `α` (alpha): scaling factor (default: 32.0)
- `δ` (delta): margin (default: 0.1)
- `P+`: positive proxies
- `|P|`: all proxies


**설정**:
```python
loss_type = 'proxy'
proxy_type = 'ProxyAnchorLoss'  # or 'FocalStyleProxyAnchorLoss'
proxy_alpha = 32.0
proxy_delta = 0.1
```

---

### 3. Combined 

```python
L = λ*L_CE + (1-λ)*L_Proxy

where λ ∈ [0, 1]
```



**설정**:
```python
loss_type = 'combined'
proxy_type = 'ProxyAnchorLoss'
proxy_alpha = 32.0
proxy_delta = 0.1
lambda_combined = 0.5  # λ
```

---

## 사용 예시

### train.py 설정

#### 예시 1: CE only
```python
# train.py
loss_type = 'CE'
```

#### 예시 2: Proxy-Anchor only
```python
# train.py
loss_type = 'proxy'
proxy_type = 'ProxyAnchorLoss'
proxy_alpha = 32.0
proxy_delta = 0.1
```

#### 예시 3: Combined
```python
# train.py
loss_type = 'combined'
proxy_type = 'ProxyAnchorLoss'
proxy_alpha = 32.0
proxy_delta = 0.1
lambda_combined = 0.5  # 50% CE + 50% Proxy
```

#### 예시 4: Combined 
```python
# train.py
loss_type = 'combined'
proxy_type = 'ProxyAnchorLoss'
proxy_alpha = 32.0
proxy_delta = 0.1
lambda_combined = 0.7  # 70% CE + 30% Proxy
```

#### 예시 5: Focal-weighted Proxy
```python
# train.py
loss_type = 'combined'
proxy_type = 'FocalStyleProxyAnchorLoss'  # Hard example mining
proxy_alpha = 32.0
proxy_delta = 0.1
lambda_combined = 0.5
```

---


### 일반적인 경우
```python
loss_type = 'combined'
proxy_type = 'ProxyAnchorLoss'
lambda_combined = 0.5
```



## 수식 정리

### Cross Entropy Loss
```
L_CE = -Σ y_i log(p_i)

where:
  y_i: true label (one-hot)
  p_i: predicted probability
```

### Proxy-Anchor Loss
```
L_PA = L_pos + L_neg

L_pos = (1/|P+|) Σ_{p∈P+} log(1 + Σ_{x∈X_p^+} exp(-α(s(x,p)-δ)))

L_neg = (1/|P|) Σ_{p∈P} log(1 + Σ_{x∈X_p^-} exp(α(s(x,p)+δ)))

where:
  s(x,p) = (x·p) / (||x|| ||p||)  [cosine similarity]
  X_p^+: positive samples for proxy p
  X_p^-: negative samples for proxy p
  α: temperature scaling
  δ: margin
```

### Combined Loss
```
L_combined = λ*L_CE + (1-λ)*L_PA

where λ ∈ [0,1]
```



## 디버깅

### Loss 값 확인
TensorBoard에서 확인:
- `Loss/Total`: 전체 손실
- `Loss/CE`: Cross Entropy 부분
- `Loss/Proxy`: Proxy-Anchor 부분

### 가중치 확인
학습 시작 시 출력:
```
TRAINING CONFIGURATION
======================================================================
Loss Type: combined
Proxy Type: ProxyAnchorLoss
Proxy α (alpha): 32.0
Proxy δ (delta): 0.1
λ (lambda): 0.50
  → CE weight: 0.50
  → Proxy weight: 0.50
```

