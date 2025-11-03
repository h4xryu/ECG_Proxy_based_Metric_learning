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
batch_size = 256
nepoch = 50
lr_initial = 1e-4
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
# - 0 < λ < 1: Combined (λ*CE + (1-λ)*Proxy)
lambda_combined = 1

# Proxy Loss 파라미터
proxy_type = 'ProxyAnchorLoss'  # 'ProxyAnchorLoss', ' FocalStyleProxyAnchorLoss' 
proxy_alpha = 32.0
proxy_delta = 0.1

# 데이터 전처리 설정 (매번 새로 전처리됨)
segment_seconds = 5.0  # ECG 세그먼트 길이 (초) - 논문 방식: 5.0초 (1800 샘플)
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
        
        # self.model = SEResNetLSTM(in_channel=inputs, num_classes=classes).to(self.device)    
        self.model = UNet(nOUT=classes, in_channels=inputs, rub0_layers=7).to(self.device)
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
            weight_decay=0.001,
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
        self.best_valid_f1 = 0.0
        self.best_metrics = {}
        self.best_model_path = None

    def train(self):
        """===============================================학습 실행================================================"""
        global nepoch
        
        print_training_config()

        for epoch in range(1, nepoch + 1):
            start_time = time.time()

            # 에포크별 학습 및 검증
            train_loss = self._train_epoch(epoch)
            valid_metrics, valid_loss = self._evaluate()
            
            # 최고 성능 모델 추적
            is_best = valid_metrics['macro_f1'] > self.best_valid_f1
            if is_best:
                self.best_valid_f1 = valid_metrics['macro_f1']
                self.best_metrics = valid_metrics.copy()
                self.best_metrics['epoch'] = epoch

            # 체크포인트 저장
            self._save_checkpoint(epoch, valid_metrics['macro_f1'], is_best)

            # 학습률 스케줄링
            current_lr = self.scheduler.get_last_lr()[0]
            self.scheduler.step()
            
            # 로깅
            epoch_time = time.time() - start_time
            train_metrics = {'loss': train_loss}
            valid_metrics['loss'] = valid_loss
            self.logger.log_epoch(epoch, train_metrics, valid_metrics, current_lr, epoch_time)

        # 학습 완료 후 최고 모델로 테스트
        print('\n' + '='*80)
        print('='*80)
        self._test_with_best_model()

        # 베스트 결과 CSV 저장
        if self.best_metrics:
            self.logger.save_best_results_csv(self.best_metrics, self.best_metrics.get('epoch', 0))
        
        # 실험 정리
        self.logger.finalize_experiment(self.best_metrics)
        self.logger.close()
        
        print('\n' + '='*80)
        print('Training and testing completed.')
        print(f"Best validation F1: {self.best_valid_f1:.4f} at epoch {self.best_metrics.get('epoch', 'N/A')}")
        print('='*80 + '\n')

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
            
            outputs, features = self.model(X, return_features=True)
            loss, proxy_loss, ce_loss = self._compute_loss(outputs, features, Y)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.train_loader)

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
                
                outputs, features = self.model(X, return_features=True)
                loss, _, _ = self._compute_loss(outputs, features, Y)
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

    def _load_best_model(self):
        """최고 성능 모델 체크포인트 로드"""
        load_best_model(self.model, self.best_model_path, self.device)
    
    @torch.no_grad()
    def _test_with_best_model(self):
        """
        최고 성능 모델로 테스트 수행
        Proxy Loss 사용 시 거리 기반 예측 적용
        """
        global lambda_combined
        
        self._load_best_model()
        self.model.eval()
        
        pred_labels, true_labels = [], []
        
        # Prediction loop
        with torch.no_grad():
            for X, Y in tqdm(self.test_loader, desc="Final Test"):
                X = X.float().to(self.device)
                Y = Y.long().to(self.device)
                
                # 모델 출력
                outputs, features = self.model(X, return_features=True)
                
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
        테스트 결과 출력 (confusion matrix와 classification report 포함)
        """
        print("\n" + "="*80)
        print("FINAL TEST RESULTS (Best Model Selected by Test Set)")
        print("="*80)
        print(f"Overall Accuracy: {all_class_metrics['accuracy']:.4f}")
        print(f"Macro F1: {all_class_metrics['macro_f1']:.4f}")
        print(f"Weighted F1: {all_class_metrics['weighted_f1']:.4f}")
        
        # Confusion Matrix 출력
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
        
        # Classification Report 출력
        print("\n" + "="*80)
        print("CLASSIFICATION REPORT (sklearn)")
        print("="*80)
        report = classification_report(
            true_labels, 
            pred_labels, 
            labels=[0, 1, 2, 3, 4],  # N, S, V, F, Q 5개 클래스 모두 명시
            target_names=class_names,
            digits=4,
            zero_division=0
        )
        print(report)
        
        # Custom Metrics 출력
        print("\n" + "-"*80)
        print("CUSTOM METRICS (ALL CLASSES: N, S, V, F, Q)")
        print("-"*80)
        print(f"{'Class':<10} {'Acc':<10} {'Sens':<10} {'Spec':<10} {'PPV':<10} {'F1':<10}")
        print("-"*80)
        
        for i, name in enumerate(class_names):
            # 인덱스 범위 체크 (일부 클래스가 데이터에 없을 수 있음)
            if i < len(all_class_metrics['acc_per_class']):
                print(f"{name:<10} "
                    f"{all_class_metrics['acc_per_class'][i]:<10.4f} "
                    f"{all_class_metrics['sensitivity'][i]:<10.4f} "
                    f"{all_class_metrics['specificity'][i]:<10.4f} "
                    f"{all_class_metrics['ppv'][i]:<10.4f} "
                    f"{all_class_metrics['f1'][i]:<10.4f} "
                    )
            else:
                print(f"{name:<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
        print("-"*80)
        print(f"{'MACRO AVG':<10} "
            f"{'':<10} "
            f"{all_class_metrics['weighted_sensitivity']:<10.4f} "
            f"{all_class_metrics['weighted_specificity']:<10.4f} "
            f"{all_class_metrics['weighted_ppv']:<10.4f} "  
            f"{all_class_metrics['macro_f1']:<10.4f} "
            )
        print("\n" + "-"*80)
        print("="*80 + "\n")
    
    def _save_checkpoint(self, epoch, f1_score, is_best): 
        """
        체크포인트 저장

        """
        model_dir = get_model_dir(self.logger.log_dir)
        checkpoint = create_checkpoint(
            self.model, self.optimizer, self.scheduler,
            epoch, f1_score, self.best_valid_f1, self.class_names
        )
        
        best_path = save_checkpoint(checkpoint, model_dir, epoch, is_best)
        if best_path:
            self.best_model_path = best_path

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


def main_receptive_field_experiments():
    """
    Receptive Field 실험 자동화
    
    실험 1-4: UNet rub0_layers [5, 6, 7, 8]
    실험 5-9: SE-ResNet-LSTM dilation [1, 2, 3, 4, 5]
    """
    global lambda_combined, classes, inputs
    
    print("\n" + "="*80)
    print("RECEPTIVE FIELD EXPERIMENTS - CE LOSS ONLY")
    print("="*80)
    print(f"Total Experiments: 9")
    print(f"  - UNet (rub0_layers): 4 experiments")
    print(f"  - SE-ResNet-LSTM (dilation): 5 experiments")
    print(f"  - lambda_combined: 1.0 (CE only)")
    print("="*80 + "\n")
    
    # ===== 실험 1-4: UNet rub0_layers =====
    print("\n" + "="*80)
    print("PART 1: UNet Receptive Field Test")
    print("="*80 + "\n")
    
    rub0_layers_list = [5, 6, 7, 8]
    
    for exp_num, rub0_layers in enumerate(rub0_layers_list, start=1):
        print(f"\n{'='*80}")
        print(f"Experiment {exp_num}/9: UNet with rub0_layers={rub0_layers}")
        print(f"{'='*80}\n")
        
        # CE only로 고정
        globals()['lambda_combined'] = 1.0
        print(f"[DEBUG] lambda_combined set to: {globals()['lambda_combined']}")
        
        set_seed()
        
        # 모델 변경을 위한 Trainer 수정
        class UNetTrainer(Trainer):
            def _setup_model(self):
                self.model = UNet(nOUT=classes, in_channels=inputs, rub0_layers=rub0_layers).to(self.device)
                total_params = sum(p.numel() for p in self.model.parameters())
                print(f"\n{'='*70}")
                print(f"Model: UNet (rub0_layers={rub0_layers})")
                print(f"Model Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
                print(f"{'='*70}\n")
            
            def _setup_logger(self):
                """로거 초기화 with 실험 정보"""
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                exp_name = f"UNet_rub0_{rub0_layers}"
                log_dir = os.path.join(os.getcwd(), 'logs', f'{timestamp}_{exp_name}')
                
                os.makedirs(os.path.join(log_dir, 'tensorboard'), exist_ok=True)
                os.makedirs(os.path.join(log_dir, 'checkpoints'), exist_ok=True)
                
                self.logger = Logger(os.path.join(log_dir, 'tensorboard'), self.class_names)
        
        trainer = UNetTrainer()
        trainer.train()
        
        print(f"\n{'='*80}")
        print(f"Experiment {exp_num}/9 Completed: UNet rub0_layers={rub0_layers}")
        print(f"{'='*80}\n")
    
    # ===== 실험 5-9: SE-ResNet-LSTM dilation =====
    print("\n" + "="*80)
    print("PART 2: SE-ResNet-LSTM Dilation Test")
    print("="*80 + "\n")
    
    dilation_list = [1, 2, 3, 4, 5]
    
    for exp_num, dilation in enumerate(dilation_list, start=5):
        print(f"\n{'='*80}")
        print(f"Experiment {exp_num}/9: SE-ResNet-LSTM with dilation={dilation}")
        print(f"{'='*80}\n")
        
        # CE only로 고정
        globals()['lambda_combined'] = 1.0
        print(f"[DEBUG] lambda_combined set to: {globals()['lambda_combined']}")
        
        set_seed()
        
        # SE-ResNet-LSTM with dilation
        class SEResNetLSTMTrainer(Trainer):
            def _setup_model(self):
                self.model = SEResNetLSTM(
                    in_channel=inputs, 
                    num_classes=classes,
                    dilation=dilation
                ).to(self.device)
                total_params = sum(p.numel() for p in self.model.parameters())
                print(f"\n{'='*70}")
                print(f"Model: SE-ResNet-LSTM (dilation={dilation})")
                print(f"Model Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
                print(f"{'='*70}\n")
            
            def _setup_logger(self):
                """로거 초기화 with 실험 정보"""
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                exp_name = f"SEResNetLSTM_dilation_{dilation}"
                log_dir = os.path.join(os.getcwd(), 'logs', f'{timestamp}_{exp_name}')
                
                os.makedirs(os.path.join(log_dir, 'tensorboard'), exist_ok=True)
                os.makedirs(os.path.join(log_dir, 'checkpoints'), exist_ok=True)
                
                self.logger = Logger(os.path.join(log_dir, 'tensorboard'), self.class_names)
        
        trainer = SEResNetLSTMTrainer()
        trainer.train()
        
        print(f"\n{'='*80}")
        print(f"Experiment {exp_num}/9 Completed: SE-ResNet-LSTM dilation={dilation}")
        print(f"{'='*80}\n")
    
    print("\n" + "="*80)
    print("ALL 9 EXPERIMENTS COMPLETED!")
    print("="*80)
    print("Summary:")
    print("  - UNet (rub0_layers=5,6,7,8): 4 experiments")
    print("  - SE-ResNet-LSTM (dilation=1,2,3,4,5): 5 experiments")
    print("  - Total: 9 experiments")
    print("="*80 + "\n")


# if __name__ == '__main__':
#     main()

if __name__ == '__main__':
    main_receptive_field_experiments()
