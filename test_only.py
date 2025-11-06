
import os
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from loss_functions import setup_losses, compute_loss, get_predictions
from models import SEResNetLSTM, DCRNNModel, UNet
from utils import *
from train import lambda_combined
import test_analysis 

# 전역 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 256
classes = 5
inputs = 1

# 체크포인트 경로 (수정 필요)
checkpoint_path = './logs/20251104_054038_UNet_rub0_7/checkpoints/best_model.pth'


def load_model_from_checkpoint(checkpoint_path, device):
    """체크포인트에서 모델 로드"""
    print(f"\nLoading model from: {checkpoint_path}")
    
    # 모델 생성 (train.py와 동일한 모델 사용)
    # model = SEResNetLSTM(in_channel=inputs, num_classes=classes, dilation=2).to(device)
    model = UNet(nOUT=classes, in_channels=inputs, rub0_layers=7).to(device)
    # model = DCRNNModel(in_channel=inputs, num_classes=classes).to(device)
    
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"F1 Score: {checkpoint.get('f1_score', 'N/A'):.4f}")
    
    return model


'''
맞춘 샘플 분석
'''
def plot_correct(model, pred_labels, true_labels, all_data, n_samples=10, class_names=['N', 'S', 'V', 'F', 'Q']):
    
    os.makedirs('correct_analysis', exist_ok=True)
  
    
    correct = pred_labels == true_labels
    
    for class_idx in range(len(class_names)):
        class_correct = correct & (true_labels == class_idx)
        correct_indices = np.where(class_correct)[0]
        
        if len(correct_indices) == 0:
            continue
        
        class_dir = f'correct_analysis/{class_names[class_idx]}'
        os.makedirs(class_dir, exist_ok=True)
        
        # n_plot = min(n_samples, len(correct_indices))
        
        for i in range(20):
            idx = correct_indices[i]
            true_label = class_names[true_labels[idx]] # 실제 레이블
            pred_label = class_names[pred_labels[idx]] # 예측 레이블
            
            plt.figure(figsize=(12, 3))
            plt.plot(all_data.dataset[idx][0][0].squeeze(0), label='ECG')
            plt.legend()
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.title(f'ECG : True: {true_label} / Pred: {pred_label}')
            plt.tight_layout()
            plt.savefig(f'{class_dir}/{class_names[class_idx]}_{i+1}_ecg.png', dpi=150, bbox_inches='tight')
            plt.close()
            
'''
틀린 샘플 분석
'''
def plot_misclassified(model, pred_labels, true_labels, all_data, n_samples=10, class_names=['N', 'S', 'V', 'F', 'Q']):
    
    os.makedirs('misclassified_analysis', exist_ok=True)
    """ Temp """




    """ Temp """
    misclassified = pred_labels != true_labels
    
    for class_idx in range(len(class_names)):
        class_wrong = misclassified & (true_labels == class_idx) 
        wrong_indices = np.where(class_wrong)[0]
        
        if len(wrong_indices) == 0:
            continue
        
        class_dir = f'misclassified_analysis/{class_names[class_idx]}'
        os.makedirs(class_dir, exist_ok=True)

        
        for i in range(6):
            idx = wrong_indices[i]

            true_label = class_names[true_labels[idx]] # 실제 레이블
            pred_label = class_names[pred_labels[idx]] # 예측 레이블
            
            plt.figure(figsize=(12, 3))
            plt.plot(all_data.dataset[idx][0][0].squeeze(0), label='ECG')
            plt.legend()
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.title(f'ECG : True: {true_label} / Pred: {pred_label}')
            plt.tight_layout()
            plt.savefig(f'{class_dir}/{class_names[class_idx]}_{i+1}_ecg.png', dpi=150, bbox_inches='tight')
            plt.close()

@torch.no_grad()
def test_model(model, test_loader, device):

    model.eval()
    
    pred_labels, true_labels = [], []
    global lambda_combined

    for X, Y in tqdm(test_loader, desc="Test Progress"):
        X = X.float().to(device)
        Y = Y.long().to(device)
        lambda_combined = 0
        # 모델 예측
        pred, features = model(X, return_features=True)
        # pred = torch.argmax(pred, dim=1)
        pred = get_predictions(pred, features, model.get_proxies() if lambda_combined < 1.0 else None)
        
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

    test_analysis.plot_misclassified(model, pred_labels, true_labels, test_loader, device)
    test_analysis.plot_correct(model, pred_labels, true_labels, test_loader, device)

if __name__ == '__main__':
    main()

