"""
로깅 시스템: TensorBoard, Metrics, CSV 저장
"""
import os
import csv
import json
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

class Logger:
    """통합 로거: TensorBoard + 메트릭 계산 + CSV 저장"""
    
    def __init__(self, log_dir, class_names=['N', 'S', 'V', 'F', 'Q']):
        self.log_dir = log_dir
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir)
        
        # 학습 설정 저장
        self._save_config()
        
        # CSV 파일 경로
        parent_dir = os.path.dirname(log_dir)
        self.csv_per_class = os.path.join(parent_dir, 'results_per_class.csv')
        self.csv_overall = os.path.join(parent_dir, 'results_overall.csv')
        
        # 메트릭 히스토리
        self.best_metrics = None
        self.best_epoch = 0
    
    def _save_config(self):
        """학습 설정을 TensorBoard와 JSON 파일에 저장"""
        from train import (batch_size, nepoch, lr_initial, decay_epoch, device,
                          classes, inputs,
                          proxy_type, proxy_alpha, proxy_delta, lambda_combined,
                          segment_seconds, fs_out, ecg_leads)
        
        # lambda 기반 loss mode 결정
        if lambda_combined == 1.0:
            loss_mode = 'CE only'
        elif lambda_combined == 0.0:
            loss_mode = 'Proxy only'
        else:
            loss_mode = 'Combined'
        
        config = {
            'Training': {
                'batch_size': batch_size,
                'epochs': nepoch,
                'lr_initial': lr_initial,
                'decay_epoch': decay_epoch,
                'device': device
            },
            'Model': {
                'classes': classes,
                'inputs': inputs
            },
            'Data': {
                'segment_seconds': segment_seconds,
                'sampling_rate': fs_out,
                'ecg_leads': ecg_leads
            },
            'Loss': {
                'lambda_combined': lambda_combined,
                'mode': loss_mode,
                'ce_weight': lambda_combined,
                'proxy_weight': 1.0 - lambda_combined,
                'proxy_type': proxy_type if lambda_combined < 1.0 else 'N/A',
                'proxy_alpha': proxy_alpha if lambda_combined < 1.0 else 'N/A',
                'proxy_delta': proxy_delta if lambda_combined < 1.0 else 'N/A'
            }
        }
        
        # TensorBoard에 텍스트로 저장
        config_text = json.dumps(config, indent=2)
        self.writer.add_text('Config', f'```\n{config_text}\n```', 0)
        
        # JSON 파일로 저장
        parent_dir = os.path.dirname(self.log_dir)
        config_path = os.path.join(parent_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Config saved to {config_path}")
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """전체 메트릭 계산"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['macro_f1'] = f1_score(y_true, y_pred, labels=list(range(self.num_classes)), average='macro', zero_division=0)
        metrics['weighted_f1'] = f1_score(y_true, y_pred, labels=list(range(self.num_classes)), average='weighted', zero_division=0)
        
        # Confusion matrix 기반 메트릭
        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))
        
        eps = 1e-14
        tp = np.diag(cm).astype(float)
        fn = (cm.sum(axis=1) - tp).astype(float)
        fp = (cm.sum(axis=0) - tp).astype(float)
        tn = (cm.sum() - (tp + fp + fn)).astype(float)
        
        # Per-class metrics
        sensitivity = tp / (tp + fn + eps)
        specificity = tn / (tn + fp + eps)
        ppv = tp / (tp + fp + eps)  # Precision

        
        # Class별 F1
        per_class_f1 = f1_score(y_true, y_pred, labels=list(range(self.num_classes)), average=None, zero_division=0)
        
        # Macro averages
        metrics['macro_sensitivity'] = float(np.mean(sensitivity))
        metrics['macro_specificity'] = float(np.mean(specificity))
        metrics['macro_precision'] = float(np.mean(ppv))

        
        # Weighted averages
        class_counts = cm.sum(axis=1)
        total = class_counts.sum()
        weights = class_counts / (total + eps)
        
        metrics['weighted_sensitivity'] = float(np.sum(sensitivity * weights))
        metrics['weighted_specificity'] = float(np.sum(specificity * weights))
        metrics['weighted_precision'] = float(np.sum(ppv * weights))

        
        # Per-class 저장
        metrics['per_class'] = {
            'sensitivity': sensitivity.tolist(),
            'specificity': specificity.tolist(),
            'precision': ppv.tolist(),
            'f1': per_class_f1.tolist()
        }
        
        return metrics
    
    def log_loss(self, loss, proxy_loss, ce_loss, step):
        """손실 로깅"""
        self.writer.add_scalar('Loss/Total', loss, step)
        self.writer.add_scalar('Loss/Proxy', proxy_loss, step)
        self.writer.add_scalar('Loss/CE', ce_loss, step)
    
    def log_epoch(self, epoch, train_metrics, valid_metrics, learning_rate, epoch_time):
        """에포크 결과 로깅"""
        # TensorBoard
        self.writer.add_scalar('Train/Loss', train_metrics.get('loss', 0), epoch)
        
        self.writer.add_scalar('Valid/Loss', valid_metrics.get('loss', 0), epoch)
        self.writer.add_scalar('Valid/Accuracy', valid_metrics['accuracy'], epoch)
        self.writer.add_scalar('Valid/Macro_F1', valid_metrics['macro_f1'], epoch)
        self.writer.add_scalar('Valid/Weighted_F1', valid_metrics['weighted_f1'], epoch)
        self.writer.add_scalar('Valid/Macro_Sensitivity', valid_metrics['macro_sensitivity'], epoch)
        self.writer.add_scalar('Valid/Macro_Specificity', valid_metrics['macro_specificity'], epoch)
        # 클래스별 sensitivity, specificity, precision, f1 로깅
        for i, class_name in enumerate(self.class_names):
            # 인덱스 범위 체크 (일부 클래스가 데이터에 없을 수 있음)
            if i < len(valid_metrics['per_class']['sensitivity']):
                self.writer.add_scalar(f'Valid/{class_name}_Sensitivity', valid_metrics['per_class']['sensitivity'][i], epoch)
            if i < len(valid_metrics['per_class']['specificity']):
                self.writer.add_scalar(f'Valid/{class_name}_Specificity', valid_metrics['per_class']['specificity'][i], epoch)
            if i < len(valid_metrics['per_class']['precision']):
                self.writer.add_scalar(f'Valid/{class_name}_Precision', valid_metrics['per_class']['precision'][i], epoch)
            if i < len(valid_metrics['per_class']['f1']):
                self.writer.add_scalar(f'Valid/{class_name}_F1', valid_metrics['per_class']['f1'][i], epoch)
        
        self.writer.add_scalar('Learning_Rate', learning_rate, epoch)
        
        # Console
        print(f'\nEPOCH [{epoch}] '
              f'Train Loss: {train_metrics.get("loss", 0):.6f} | '
              f'Valid Loss: {valid_metrics.get("loss", 0):.6f} | '
              f'Acc: {valid_metrics["accuracy"]:.4f} | '
              f'F1: {valid_metrics["macro_f1"]:.4f} | '
              f'LR: {learning_rate:.2e} | '
              f'Time: {epoch_time:.1f}s')
    
    def save_best_results_csv(self, best_metrics, best_epoch):
        """베스트 결과를 CSV로 저장"""
        self.best_metrics = best_metrics
        self.best_epoch = best_epoch
        
        # 1. Per-class CSV
        with open(self.csv_per_class, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Class', 'Sensitivity', 'Precision', 'Specificity', 'F1-Score'])
            
            per_class = best_metrics['per_class']
            for i, class_name in enumerate(self.class_names):
                # 인덱스 범위 체크 (일부 클래스가 데이터에 없을 수 있음)
                if i < len(per_class['sensitivity']):
                    writer.writerow([
                        class_name,
                        f"{per_class['sensitivity'][i]:.4f}",
                        f"{per_class['precision'][i]:.4f}",
                        f"{per_class['specificity'][i]:.4f}",
                        f"{per_class['f1'][i]:.4f}"
                    ])
                else:
                    writer.writerow([
                        class_name,
                        "N/A",
                        "N/A",
                        "N/A",
                        "N/A"
                    ])
        
        print(f"✓ Per-class results saved: {self.csv_per_class}")
        
        # 2. Overall CSV
        with open(self.csv_overall, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric_Type', 'Accuracy', 'F1', 'Sensitivity', 'Precision', 'Specificity'])
            
            writer.writerow([
                'Macro',
                f"{best_metrics['accuracy']:.4f}",
                f"{best_metrics['macro_f1']:.4f}",
                f"{best_metrics['macro_sensitivity']:.4f}",
                f"{best_metrics['macro_precision']:.4f}",
                f"{best_metrics['macro_specificity']:.4f}"
            ])
            
            writer.writerow([
                'Weighted',
                f"{best_metrics['accuracy']:.4f}",
                f"{best_metrics['weighted_f1']:.4f}",
                f"{best_metrics['weighted_sensitivity']:.4f}",
                f"{best_metrics['weighted_precision']:.4f}",
                f"{best_metrics['weighted_specificity']:.4f}"
            ])
        
        print(f"✓ Overall results saved: {self.csv_overall}")
    
    def finalize_experiment(self, best_metrics):
        """실험 종료 시 최종 정리"""
        # TensorBoard에 베스트 메트릭 기록
        self.writer.add_text('Best_Results', f'```\n{json.dumps(best_metrics, indent=2)}\n```', 0)
        
        # Summary 파일 저장
        parent_dir = os.path.dirname(self.log_dir)
        summary_path = os.path.join(parent_dir, 'summary.txt')
        
        with open(summary_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("TRAINING SUMMARY\n")
            f.write("="*70 + "\n")
            f.write(f"Best Epoch: {best_metrics.get('epoch', 'N/A')}\n")
            f.write(f"Best Accuracy: {best_metrics['accuracy']:.4f}\n")
            f.write(f"Best Macro F1: {best_metrics['macro_f1']:.4f}\n")
            f.write(f"Best Weighted F1: {best_metrics['weighted_f1']:.4f}\n")
            f.write(f"Macro Sensitivity: {best_metrics['macro_sensitivity']:.4f}\n")
            f.write(f"Macro Specificity: {best_metrics['macro_specificity']:.4f}\n")
            f.write("="*70 + "\n")
        
        print(f"✓ Summary saved: {summary_path}")
    
    def close(self):
        """로거 종료"""
        self.writer.close()

