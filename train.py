import os
import random
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from dataloader import ECGDataloader
from models import *
from utils import (
    label2index, load_train_data, load_test_data, mkdir,
    create_log_directory, get_model_dir, create_checkpoint, save_checkpoint, load_best_model,
    print_model_info, print_training_config
)
from loss_functions import setup_losses, compute_loss, get_predictions
from logger import Logger
os.environ["QT_QPA_PLATFORM"] = "offscreen"

"""
================================================================================
전역 변수 - 학습 옵션 설정
================================================================================
"""

# 전역 학습 설정
batch_size = 512
nepoch = 50
lr_initial = 0.0005
decay_epoch = 20
device = 'cuda'

# 모델 기본 설정
classes = 5
inputs = 1
model_path = './checkpoints/best_model.pth'



# Loss 설정

# λ로 Loss function 결합:
# - λ = 1.0: CE only
# - λ = 0.0: Proxy only
# - 0 < λ < 1: Combined (λ*CE + (1-λ)*Proxy),
lambda_combined = 1

# Proxy Loss 파라미터
proxy_type = 'ProxyAnchorLoss'  # 'ProxyAnchorLoss', ' FocalStyleProxyAnchorLoss' 
proxy_alpha = 32.0
proxy_delta = 0.1

# 데이터 전처리 설정 (매번 새로 전처리됨)
segment_seconds = 2.0  # ECG 세그먼트 길이 (초) - 논문 방식: 5.0초 (1800 샘플)
fs_out = 360           # 출력 샘플링 주파수 (Hz)
ecg_leads = ['MLII']   # 사용할 ECG 리드 ['MLII'], ['V1'], ['MLII', 'V1'] 등

# 데이터 경로 설정
path_mitbih_raw = './data/mit-bih-arrhythmia-database-1.0.0'  # Raw MIT-BIH 데이터 경로
path_processed_base = './data/processed'  # 전처리된 데이터 저장 기본 경로



"""
================================================================================
TRAINER CLASS - 학습 클래스
================================================================================
"""


class Trainer:

    
    def __init__(self):

        global device, classes, inputs
        
        self.device = device
        self.class_names = ['N', 'S', 'V', 'F', 'Q']  # ECG 클래스 이름
        
        self._setup_model()
        self._setup_data()
        self._setup_losses()
        self._setup_optimizer()
        self._setup_logger()
        self._init_tracking_vars()

    """================================================================================학습세팅================================================================================"""

 
    def _setup_model(self):
        """모델 초기화"""
        global inputs, classes
        
        # self.model = SEResNetLSTM(in_channel=inputs, num_classes=classes, dilation=2).to(self.device)    
        # self.model = UNet(nOUT=classes, in_channels=inputs, rub0_layers=7).to(self.device)
        self.model = HUnivModel(nOUT=classes, in_channels=inputs).to(self.device)
        # self.model = DCRNNModel(in_channel=inputs, num_classes=classes).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\n{'='*70}")
        print(f"Model Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        
        print(f"{'='*70}\n")
    
    
    def _setup_data(self):
        global batch_size, segment_seconds, fs_out
        

        # Train 데이터 로드 (매번 새로 전처리)
        self.train_loader = load_train_data(batch_size, num_workers=4)
        
        # Test 데이터 로드 (train에서 전처리한 데이터 사용)
        self.valid_loader = load_test_data(batch_size, num_workers=4)
        self.test_loader = self.valid_loader
        
    
    def _setup_optimizer(self):
        """옵티마이저 및 스케줄러 초기화"""
        global lr_initial, decay_epoch
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr_initial,
            weight_decay=0.002,
            betas=(0.9, 0.999)
        )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=decay_epoch,
            gamma=0.1
        )
    
    def _setup_logger(self):
        """로거 초기화"""
        log_dir = create_log_directory()
        self.logger = Logger(log_dir, self.class_names)
    
    def _init_tracking_vars(self):
        """추적 변수 초기화"""
        self.final_metrics = {}

    def train(self):
        """===============================================학습 실행================================================"""
        global nepoch
        
        print_training_config()

        for epoch in range(1, nepoch + 1):
            start_time = time.time()

            # 에포크별 학습 및 검증
            train_loss = self._train_epoch(epoch)
            valid_metrics, valid_loss = self._evaluate()

            # 학습률 스케줄링
            current_lr = self.scheduler.get_last_lr()[0]
            self.scheduler.step()
            
            # 로깅
            epoch_time = time.time() - start_time
            train_metrics = {'loss': train_loss}
            valid_metrics['loss'] = valid_loss
            self.logger.log_epoch(epoch, train_metrics, valid_metrics, current_lr, epoch_time)
            
            # 마지막 에포크 처리
            if epoch == nepoch:
                self.final_metrics = valid_metrics.copy()
                self.final_metrics['epoch'] = epoch
                # 최종 모델 저장
                self._save_final_checkpoint(epoch, valid_metrics['macro_f1'])

        # 50 에포크 학습 완료 후 최종 모델로 테스트
        print('\n' + '='*90)
        print(" " * 27 + "TRAINING COMPLETED")
        print('='*90)
        print(f"\nFinal Epoch [{nepoch}] Validation F1: {self.final_metrics['macro_f1']:.4f}")
        print(f"Testing with final model (Epoch {nepoch})...")
        print('='*90 + '\n')
        
        self._test_final_model()

        # 최종 결과 CSV 저장
        if self.final_metrics:
            self.logger.save_best_results_csv(self.final_metrics, nepoch)
        
        # 실험 정리
        self.logger.finalize_experiment(self.final_metrics)
        self.logger.close()
        
  

    def _compute_loss(self, outputs, features, labels):
        global lambda_combined
        
        return compute_loss(
            outputs=outputs,
            features=features,
            labels=labels,
            cross_entropy_loss=self.cross_entropy_loss,
            proxy_loss=self.proxy_loss,
            model_proxies=self.model.get_proxies() if lambda_combined < 1.0 else None,
            device=self.device
        )
    
    def _train_epoch(self, epoch):
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} Training')
        for X, Y in pbar:
            X, Y = X.float().to(self.device), Y.long().to(self.device)
            
            outputs, features = self.model(X)
            loss, proxy_loss, ce_loss = self._compute_loss(outputs, features, Y)

            l1_loss = l1_regularization(self.model, lambda_l1=0.001)
            self.optimizer.zero_grad()
            loss = loss + l1_loss
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() 
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _evaluate(self, data_loader=None):
        """모델 평가"""
        global lambda_combined
        
        if data_loader is None:
            data_loader = self.valid_loader
        
        self.model.eval()
        pred_labels, true_labels, pred_probas = [], [], []
        total_loss = 0
        
        with torch.no_grad():
            for X, Y in tqdm(data_loader, desc="Evaluation"):
                X, Y = X.float().to(self.device), Y.long().to(self.device)
                
                outputs, features = self.model(X)
                loss, _, _ = self._compute_loss(outputs, features, Y)

                l1_loss = l1_regularization(self.model, lambda_l1=0.001)
                loss = loss + l1_loss
                total_loss += loss.item()

                probas = torch.softmax(outputs, dim=1)
                preds = get_predictions(
                    outputs, features, 
                    self.model.get_proxies() if lambda_combined < 1.0 else None
                )
                pred_labels.extend(preds.cpu().numpy())
                true_labels.extend(Y.cpu().numpy())
                pred_probas.extend(probas.cpu().numpy())
        
        pred_labels = np.array(pred_labels)
        true_labels = np.array(true_labels)
        pred_probas = np.array(pred_probas)
        avg_loss = total_loss / len(data_loader)
        
        metrics = self.logger.calculate_metrics(true_labels, pred_labels, pred_probas)
        
        return metrics, avg_loss

    @torch.no_grad()
    def _test_final_model(self):
  
        global lambda_combined
        
        self.model.eval()
        
        pred_labels, true_labels = [], []
        
        # Prediction loop
        with torch.no_grad():
            for X, Y in tqdm(self.test_loader, desc="Final Test"):
                X = X.float().to(self.device)
                Y = Y.long().to(self.device)
                
                # 모델 출력
                outputs, features = self.model(X)
                
                # lambda_combined에 따라 예측 방식 결정
                preds = get_predictions(
                    outputs, features, 
                    self.model.get_proxies() if lambda_combined < 1.0 else None
                )
                
                pred_labels.extend(preds.cpu().numpy())
                true_labels.extend(Y.cpu().numpy())
        
        pred_labels = np.array(pred_labels)
        true_labels = np.array(true_labels)
        
        # --- Overall accuracy ---
        overall_acc = float(np.mean(pred_labels == true_labels))
        
        # --- Confusion matrix & per-class metrics ---
        num_classes = 5  # N, S, V, F, Q
        class_names = ['N', 'S', 'V', 'F', 'Q']
        
        cm = confusion_matrix(true_labels, pred_labels, labels=list(range(num_classes)))
        eps = 1e-14
        
        tp = np.diag(cm).astype(float)
        fn = (cm.sum(axis=1) - tp).astype(float)
        fp = (cm.sum(axis=0) - tp).astype(float)
        tn = (cm.sum() - (tp + fp + fn)).astype(float)
        total = cm.sum().astype(float) + eps
        
        # Per-class metrics
        sensitivity = tp / (tp + fn + eps)
        specificity = tn / (tn + fp + eps)
        ppv = tp / (tp + fp + eps)
        acc_cls = (tp + tn) / total
        
        # Per-class F1 scores
        per_class_f1 = f1_score(true_labels, pred_labels, labels=list(range(num_classes)), average=None, zero_division=0)
        
        # Macro averages (all classes)
        se_macro = float(np.mean(sensitivity))
        sp_macro = float(np.mean(specificity))
        ppv_macro = float(np.mean(ppv))
        f1_macro = float(np.mean(per_class_f1))
        
        # Package results
        all_class_metrics = {
            'accuracy': overall_acc,
            'weighted_f1': float(f1_score(true_labels, pred_labels, labels=[0, 1, 2, 3, 4], average='weighted', zero_division=0)),
            'macro_f1': f1_macro,
            'sensitivity': sensitivity.tolist(),
            'specificity': specificity.tolist(),
            'ppv': ppv.tolist(),
            'f1': per_class_f1.tolist(),
            'acc_per_class': acc_cls.tolist(),
            'weighted_sensitivity': se_macro,
            'weighted_specificity': sp_macro,
            'weighted_ppv': ppv_macro,
        }
        
        # Print results
        self._print_test_results(
            all_class_metrics, 
            class_names,
            cm,
            true_labels,
            pred_labels
        )
        
        # Log to file if logger exists
        if hasattr(self, 'logger') and hasattr(self.logger, 'log_test_results'):
            test_metrics = {
                'all_classes': all_class_metrics,
            }
            self.logger.log_test_results(test_metrics)


    def _print_test_results(self, all_class_metrics, class_names, cm, true_labels, pred_labels):
        """
        50 에포크 학습 후 최종 성능 결과 출력
        """
        print("\n" + "="*90)
        print(" " * 25 + "TEST RESULTS (Epoch 50)")
        print("="*90)
        
        # 핵심 성능 지표
        print("\n┌─ OVERALL PERFORMANCE " + "─" * 65 + "┐")
        print(f"│  Overall Accuracy : {all_class_metrics['accuracy']*100:6.2f}%")
        print(f"│  Macro F1-Score   : {all_class_metrics['macro_f1']:.4f}")
        print(f"│  Weighted F1-Score: {all_class_metrics['weighted_f1']:.4f}")
        print("└" + "─" * 88 + "┘")
        
        # Per-Class Performance 테이블
        print("\n┌─ PER-CLASS METRICS (N, S, V, F, Q) " + "─" * 51 + "┐")
        print(f"│  {'Class':<8} {'Sens':<8} {'Spec':<8} {'PPV':<8} {'F1':<8} {'Support':<10}")
        print(f"│  {'-'*70}")
        
        for i, name in enumerate(class_names):
            if i < len(all_class_metrics['acc_per_class']):
                support = int(np.sum(true_labels == i))
                print(f"│  {name:<8} "
                    f"{all_class_metrics['sensitivity'][i]:<8.4f} "
                    f"{all_class_metrics['specificity'][i]:<8.4f} "
                    f"{all_class_metrics['ppv'][i]:<8.4f} "
                    f"{all_class_metrics['f1'][i]:<8.4f} "
                    f"{support:<10}")
            else:
                print(f"│  {name:<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'0':<10}")
        
        print(f"│  {'-'*70}")
        print(f"│  {'MACRO':<8} "
            f"{all_class_metrics['weighted_sensitivity']:<8.4f} "
            f"{all_class_metrics['weighted_specificity']:<8.4f} "
            f"{all_class_metrics['weighted_ppv']:<8.4f} "
            f"{all_class_metrics['macro_f1']:<8.4f} "
            f"{len(true_labels):<10}")
        print("└" + "─" * 88 + "┘")
        
        # Confusion Matrix
        print("\n┌─ CONFUSION MATRIX " + "─" * 69 + "┐")
        print("│  " + f"{'Pred →':<10}", end="")
        for name in class_names:
            print(f"{name:>10}", end="")
        print("  │")
        print("│  Actual ↓ " + "-" * 52 + " │")
        
        for i, name in enumerate(class_names):
            print("│  " + f"{name:<10}", end="")
            for j in range(len(class_names)):
                print(f"{cm[i, j]:>10}", end="")
            print("  │")
        print("└" + "─" * 88 + "┘")
        
        # sklearn Classification Report
        print("\n┌─ DETAILED CLASSIFICATION REPORT " + "─" * 54 + "┐")
        report = classification_report(
            true_labels, 
            pred_labels, 
            labels=[0, 1, 2, 3, 4],
            target_names=class_names,
            digits=4,
            zero_division=0
        )
        for line in report.split('\n'):
            if line.strip():
                print(f"│  {line:<84}  │")
        print("└" + "─" * 88 + "┘\n")
        
        print("="*90 + "\n")
    
    def _save_final_checkpoint(self, epoch, f1_score): 
        """
        최종 에포크 체크포인트 저장
        """
        model_dir = get_model_dir(self.logger.log_dir)
        checkpoint = create_checkpoint(
            self.model, self.optimizer, self.scheduler,
            epoch, f1_score, f1_score, self.class_names
        )
        
        # 최종 모델만 저장
        final_path = os.path.join(model_dir, 'final_model.pth')
        torch.save(checkpoint, final_path)
        print(f"Final model saved: {final_path}")

    def _setup_losses(self):
        """손실 함수 초기화"""
        self.cross_entropy_loss, self.proxy_loss = setup_losses()

"""
================================================================================
메인 실행 함수
================================================================================
"""

def set_seed(seed=1234):
    """
    재현성을 위한 시드 고정
    
    Args:
        seed: 랜덤 시드 값 (기본값: 1234)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """
    메인 실행 함수
    ECG 분류 모델 학습 파이프라인 실행
    """
    set_seed()
    
    trainer = Trainer()
    trainer.train()



if __name__ == '__main__':
    main()

