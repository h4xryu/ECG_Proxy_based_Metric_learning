
import os
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, f1_score

from models import SEResNetLSTM, DCRNNModel
from utils import load_test_data

# 전역 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 256
classes = 5
inputs = 1

# 체크포인트 경로 (수정 필요)
checkpoint_path = './logs/20251103_153826/checkpoints/best_model.pth'


def load_model_from_checkpoint(checkpoint_path, device):
    """체크포인트에서 모델 로드"""
    print(f"\nLoading model from: {checkpoint_path}")
    
    # 모델 생성 (train.py와 동일한 모델 사용)
    model = SEResNetLSTM(in_channel=inputs, num_classes=classes).to(device)
    # model = DCRNNModel(in_channel=inputs, num_classes=classes).to(device)
    
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"F1 Score: {checkpoint.get('f1_score', 'N/A'):.4f}")
    
    return model


@torch.no_grad()
def test_model(model, test_loader, device):

    model.eval()
    
    pred_labels, true_labels = [], []
    

    for X, Y in tqdm(test_loader, desc="Test Progress"):
        X = X.float().to(device)
        Y = Y.long().to(device)
        
        # 모델 예측
        pred, _ = model(X, return_features=True)
        pred = torch.argmax(pred, dim=1)
        
        pred_labels.extend(pred.cpu().numpy())
        true_labels.extend(Y.cpu().numpy())
    
    pred_labels = np.array(pred_labels)
    true_labels = np.array(true_labels)
    
    return pred_labels, true_labels


def print_results(pred_labels, true_labels, class_names=['N', 'S', 'V', 'F', 'Q']):

    # Overall accuracy
    overall_acc = np.mean(pred_labels == true_labels)
    weighted_f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(len(class_names))))
    
    # Per-class metrics
    eps = 1e-14
    tp = np.diag(cm).astype(float)
    fn = (cm.sum(axis=1) - tp).astype(float)
    fp = (cm.sum(axis=0) - tp).astype(float)
    tn = (cm.sum() - (tp + fp + fn)).astype(float)
    total = cm.sum().astype(float) + eps
    
    sensitivity = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    ppv = tp / (tp + fp + eps)
    youden = sensitivity + specificity - 1.0
    acc_cls = (tp + tn) / total
    
    # Print results
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    print(f"Overall Accuracy: {overall_acc:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    
    # Confusion Matrix
    print("\n" + "="*80)
    print("CONFUSION MATRIX")
    print("="*80)
    print(f"{'':>10}", end="")
    for name in class_names:
        print(f"{name:>10}", end="")
    print()
    print("-"*80)
    
    for i, name in enumerate(class_names):
        print(f"{name:>10}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i, j]:>10}", end="")
        print()
    
    # Classification Report
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    report = classification_report(
        true_labels, 
        pred_labels, 
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    print(report)
    
    # Custom Metrics
    print("\n" + "-"*80)
    print("DETAILED METRICS")
    print("-"*80)
    print(f"{'Class':<10} {'Acc':<10} {'Sens':<10} {'Spec':<10} {'PPV':<10} {'Youden':<10}")
    print("-"*80)
    
    for i, name in enumerate(class_names):
        print(f"{name:<10} "
              f"{acc_cls[i]:<10.4f} "
              f"{sensitivity[i]:<10.4f} "
              f"{specificity[i]:<10.4f} "
              f"{ppv[i]:<10.4f} "
              f"{youden[i]:<10.4f}")
    
    print("-"*80)
    print(f"{'MACRO AVG':<10} "
          f"{'':<10} "
          f"{np.mean(sensitivity):<10.4f} "
          f"{np.mean(specificity):<10.4f} "
          f"{np.mean(ppv):<10.4f} "
          f"{np.mean(youden):<10.4f}")
    
    print("="*80 + "\n")


def main():
    """메인 함수"""
    print("="*80)
    print("Test Only Script - Load Checkpoint and Evaluate")
    print("="*80)
    
    # 체크포인트 경로 확인
    if not os.path.exists(checkpoint_path):
        print(f"\nError: Checkpoint not found at {checkpoint_path}")
        print("Please update the 'checkpoint_path' variable in this script.")
        return
    
    # 데이터 로드
    print("\nLoading test data...")
    test_loader = load_test_data(batch_size, num_workers=4)
    print(f"Test batches: {len(test_loader)}")
    
    # 모델 로드
    model = load_model_from_checkpoint(checkpoint_path, device)
    
    # 테스트 실행
    pred_labels, true_labels = test_model(model, test_loader, device)
    
    # 결과 출력
    print_results(pred_labels, true_labels)


if __name__ == '__main__':
    main()

