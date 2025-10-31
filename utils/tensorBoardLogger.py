import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, average_precision_score, confusion_matrix

"""
================================================================================
TENSORBOARD LOGGER CLASS - TensorBoard 로거 클래스
================================================================================
"""
class EnhancedWriter:
    """Enhanced TensorBoard writer with comprehensive logging capabilities"""
    
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        self.log_dir = log_dir
        
    def log_scalar(self, tag, scalar_value, global_step):
        """Log scalar values to TensorBoard"""
        self.writer.add_scalar(tag, scalar_value, global_step)
        
    def log_metrics(self, metrics_dict, global_step, prefix=''):
        """Log multiple metrics at once"""
        for key, value in metrics_dict.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                tag = f"{prefix}/{key}" if prefix else key
                self.writer.add_scalar(tag, value, global_step)
            
    def log_confusion_matrix(self, cm, class_names, global_step, prefix=''):
        """Log confusion matrix as image"""
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names)
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.title(f'{prefix} Confusion Matrix')
            
            tag = f"{prefix}/confusion_matrix" if prefix else "confusion_matrix"
            self.writer.add_figure(tag, fig, global_step)
            plt.close(fig)
        except Exception as e:
            print(f"Failed to log confusion matrix: {e}")
        
    def log_histogram(self, tag, values, global_step):
        """Log histogram of values"""
        self.writer.add_histogram(tag, values, global_step)
    def log_loss(self, loss, proxy_loss, base_loss, global_step):
        """Log loss values"""
        self.writer.add_scalar('Loss/Total', loss, global_step)
        self.writer.add_scalar('Loss/Proxy', proxy_loss, global_step)
        self.writer.add_scalar('Loss/Base', base_loss, global_step)
    def close(self):
        """Close the writer"""
        self.writer.close()


class ExperimentLogger:
    """Comprehensive experiment logging system"""
    
    def __init__(self, log_dir, opt):
        self.log_dir = log_dir
        self.opt = opt
        self.experiment_info = self._create_experiment_info()
        self.metrics_history = {
            'train': [], 'valid': []
        }
        
        # Create experiment log file
        self.experiment_log_path = os.path.join(log_dir, 'experiment_info.json')
        self.metrics_log_path = os.path.join(log_dir, 'metrics_history.json')
        
        # Save initial experiment info
        self._save_experiment_info()
        
    def _create_experiment_info(self):
        """Create comprehensive experiment information"""
        return {
            'experiment_name': '',
            'timestamp': datetime.now().isoformat(),
            'model_config': {
                'model_type': 'ResUDenseNet',
                'classes': getattr(self.opt, 'classes', None),
                'inputs': getattr(self.opt, 'inputs', None),
                'out_ch': getattr(self.opt, 'out_ch', None),
                'mid_ch': getattr(self.opt, 'mid_ch', None),
                'inconv_size': getattr(self.opt, 'inconv_size', None),
                'growth_rate': getattr(self.opt, 'growth_rate', None),
                'num_heads': getattr(self.opt, 'num_heads', None),
                'r0_layer': getattr(self.opt, 'r0_layer', None)
            },
            'training_config': {
                'epochs': getattr(self.opt, 'nepoch', None),
                'batch_size': getattr(self.opt, 'batch_size', None),
                'initial_lr': getattr(self.opt, 'lr_initial', None),
                'decay_epoch': getattr(self.opt, 'decay_epoch', None),
                'optimizer': 'AdamW',
                'scheduler': 'StepLR',
                'loss_function': 'ZLPR_CE',
                'weight_decay': 0.04
            },
            'data_config': {
                'train_data_path': getattr(self.opt, 'path_train_data', None),
                'train_labels_path': getattr(self.opt, 'path_train_labels', None),
                'val_data_path': getattr(self.opt, 'path_val_data', None),
                'val_labels_path': getattr(self.opt, 'path_val_labels', None),
                'use_kfold': getattr(self.opt, 'use_kfold', False)
            },
            'device': str(getattr(self.opt, 'device', 'cpu')),
            'pretrained': getattr(self.opt, 'pretrained', False)
        }
    
    def _save_experiment_info(self):
        """Save experiment information to JSON file"""
        with open(self.experiment_log_path, 'w') as f:
            json.dump(self.experiment_info, f, indent=2)
    
    def log_epoch_metrics(self, epoch, train_metrics, valid_metrics):
        """Log metrics for current epoch"""
        epoch_data = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'train_metrics': train_metrics,
            'valid_metrics': valid_metrics
        }
        
        self.metrics_history['train'].append({
            'epoch': epoch,
            **train_metrics
        })
        self.metrics_history['valid'].append({
            'epoch': epoch,
            **valid_metrics
        })
        
        # Save updated metrics history
        with open(self.metrics_log_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def finalize_experiment(self, best_metrics):
        """Finalize experiment with best achieved metrics"""
        self.experiment_info['best_metrics'] = best_metrics
        self.experiment_info['completed_at'] = datetime.now().isoformat()
        self._save_experiment_info()


class MetricsCalculator:
    """Calculate comprehensive metrics for model evaluation"""
    
    def __init__(self, class_names):
        self.class_names = class_names
        self.num_classes = len(class_names)
    
    def calculate_comprehensive_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive metrics including AUC-ROC, AUC-PRC, F1 scores"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro')
        metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted')
        
        # Per-class F1 scores
        try:
            per_class_f1 = f1_score(y_true, y_pred, average=None)
            for i, f1_val in enumerate(per_class_f1):
                metrics[f'f1_class_{self.class_names[i]}'] = f1_val
        except Exception as e:
            print(f"Warning: Could not calculate per-class F1 scores: {e}")
        
        # AUC metrics
        try:
            if self.num_classes == 2:
                # Binary classification
                if y_pred_proba.shape[1] >= 2:
                    metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                    metrics['auc_prc'] = average_precision_score(y_true, y_pred_proba[:, 1])
            else:
                # Multi-class classification
                metrics['auc_roc_macro'] = roc_auc_score(y_true, y_pred_proba, 
                                                       multi_class='ovr', average='macro')
                metrics['auc_roc_weighted'] = roc_auc_score(y_true, y_pred_proba, 
                                                          multi_class='ovr', average='weighted')
                
                # Per-class AUC metrics
                y_true_binary = np.zeros((len(y_true), self.num_classes))
                y_true_binary[np.arange(len(y_true)), y_true] = 1
                
                for i in range(self.num_classes):
                    try:
                        if np.sum(y_true_binary[:, i]) > 0:  # Check if class exists in y_true
                            auc_roc = roc_auc_score(y_true_binary[:, i], y_pred_proba[:, i])
                            auc_prc = average_precision_score(y_true_binary[:, i], y_pred_proba[:, i])
                            metrics[f'auc_roc_class_{self.class_names[i]}'] = auc_roc
                            metrics[f'auc_prc_class_{self.class_names[i]}'] = auc_prc
                        else:
                            metrics[f'auc_roc_class_{self.class_names[i]}'] = 0.0
                            metrics[f'auc_prc_class_{self.class_names[i]}'] = 0.0
                    except ValueError:
                        metrics[f'auc_roc_class_{self.class_names[i]}'] = 0.0
                        metrics[f'auc_prc_class_{self.class_names[i]}'] = 0.0
                        
        except ValueError as e:
            print(f"Warning: Could not calculate AUC metrics: {e}")
            metrics['auc_roc'] = 0.0
            metrics['auc_prc'] = 0.0
        
        return metrics
    
    def get_confusion_matrix(self, y_true, y_pred):
        """Get confusion matrix"""
        return confusion_matrix(y_true, y_pred)


class TensorBoardLogger:
    """Main logger class that combines all logging functionality"""
    
    def __init__(self, log_dir, opt, class_names=None):
        self.log_dir = log_dir
        self.opt = opt
        
        # Set default class names if not provided
        if class_names is None:
            num_classes = getattr(opt, 'classes', 5)
            self.class_names = [f'Class_{i}' for i in range(num_classes)]
        else:
            self.class_names = class_names
        
        # Initialize components
        self.writer = EnhancedWriter(log_dir)
        self.experiment_logger = ExperimentLogger(log_dir, opt)
        self.metrics_calculator = MetricsCalculator(self.class_names)
        
        # Log file path
        self.log_file_path = os.path.join(log_dir, 'training_log.txt')
        
        # Initialize log file
        with open(self.log_file_path, 'w') as f:
            f.write(f"Training started at: {datetime.now().isoformat()}\n")
            f.write("="*50 + "\n")
    
    def create_log_directory(self):
        """Create timestamped log directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_log_dir = os.path.join(os.getcwd(), 'logs', f'{timestamp}')
        
        if not os.path.exists(base_log_dir):
            os.makedirs(base_log_dir)
            os.makedirs(os.path.join(base_log_dir, 'tensorboard'))
            os.makedirs(os.path.join(base_log_dir, 'checkpoints'))
        
        return os.path.join(base_log_dir, 'tensorboard')
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive metrics"""
        return self.metrics_calculator.calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba)
    
    def log_loss(self, loss, proxy_loss, base_loss, global_step):
        """Log loss values"""
        self.writer.log_loss(loss, proxy_loss, base_loss, global_step)
    
    def log_epoch(self, epoch, train_metrics, valid_metrics, learning_rate, epoch_time):
        """Log epoch results comprehensively"""
        # Console logging
        self._log_to_console(epoch, train_metrics, valid_metrics, learning_rate, epoch_time)
        
        # TensorBoard logging
        self._log_to_tensorboard(epoch, train_metrics, valid_metrics, learning_rate)
        
        # File logging
        self._log_to_file(epoch, train_metrics, valid_metrics, learning_rate, epoch_time)
        
        # Experiment logging
        self.experiment_logger.log_epoch_metrics(epoch, train_metrics, valid_metrics)
    
    def log_confusion_matrix(self, y_true, y_pred, epoch, prefix='Valid'):
        """Log confusion matrix to TensorBoard"""
        cm = self.metrics_calculator.get_confusion_matrix(y_true, y_pred)
        self.writer.log_confusion_matrix(cm, self.class_names, epoch, prefix)
    
    def _log_to_console(self, epoch, train_metrics, valid_metrics, learning_rate, epoch_time):
        """Log to console"""
        log_message = (
            f'EPOCH[{epoch}] Train Loss: {train_metrics.get("loss", 0):.6f} | '
            f'Valid Acc: {valid_metrics.get("accuracy", 0):.4f} | '
            f'Valid F1: {valid_metrics.get("macro_f1", 0):.4f} | '
            f'Valid Loss: {valid_metrics.get("loss", 0):.6f} | '
            f'LR: {learning_rate:.2e} | '
            f'Time: {epoch_time:.3f}s'
        )
        
        if 'auc_roc_macro' in valid_metrics:
            log_message += f' | AUC-ROC: {valid_metrics["auc_roc_macro"]:.4f}'
        
        print("\n" + log_message)
    
    def _log_to_tensorboard(self, epoch, train_metrics, valid_metrics, learning_rate):
        """Log to TensorBoard"""
        # Training metrics
        self.writer.log_metrics(train_metrics, epoch, 'Train')
        
        # Validation metrics
        self.writer.log_metrics(valid_metrics, epoch, 'Valid')
        
        # Learning rate
        self.writer.log_scalar('Learning_Rate', learning_rate, epoch)
        
        # Per-class metrics
        per_class_metrics = {k: v for k, v in valid_metrics.items() 
                           if k.startswith(('f1_class_', 'auc_roc_class_', 'auc_prc_class_'))}
        if per_class_metrics:
            self.writer.log_metrics(per_class_metrics, epoch, 'Per_Class')
    
    def _log_to_file(self, epoch, train_metrics, valid_metrics, learning_rate, epoch_time):
        """Log to file"""
        log_message = (
            f'EPOCH[{epoch}] Train Loss: {train_metrics.get("loss", 0):.6f} | '
            f'Valid Acc: {valid_metrics.get("accuracy", 0):.4f} | '
            f'Valid F1: {valid_metrics.get("macro_f1", 0):.4f} | '
            f'Valid Loss: {valid_metrics.get("loss", 0):.6f} | '
            f'LR: {learning_rate:.2e} | '
            f'Time: {epoch_time:.3f}s'
        )
        
        with open(self.log_file_path, 'a') as f:
            f.write(log_message + '\n')
    
    def finalize_experiment(self, best_metrics):
        """Finalize experiment logging"""
        self.experiment_logger.finalize_experiment(best_metrics)
        
        # Log final summary to file
        with open(self.log_file_path, 'a') as f:
            f.write("="*50 + "\n")
            f.write(f"Training completed at: {datetime.now().isoformat()}\n")
            f.write(f"Best metrics: {best_metrics}\n")
    
    def close(self):
        """Close all loggers"""
        self.writer.close()


class MetricsCalculator:
    """Calculate comprehensive metrics for model evaluation"""
    
    def __init__(self, class_names):
        self.class_names = class_names
        self.num_classes = len(class_names)
    
    def calculate_comprehensive_metrics(self, y_true, y_pred, outputs):
        """Calculate comprehensive metrics including AUC-ROC, AUC-PRC, F1 scores"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro')
        metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted')
        
        # Per-class F1 scores
        try:
            per_class_f1 = f1_score(y_true, y_pred, average=None)
            for i, f1_val in enumerate(per_class_f1):
                metrics[f'f1_class_{self.class_names[i]}'] = f1_val
        except Exception as e:
            print(f"Warning: Could not calculate per-class F1 scores: {e}")
        
        # AUC metrics
        try:
            if self.num_classes == 2:
                # Binary classification
                if y_pred_proba.shape[1] >= 2:
                    metrics['auc_roc'] = roc_auc_score(y_true, outputs[:, 1])
                    metrics['auc_prc'] = average_precision_score(y_true, outputs[:, 1])
            else:
                # Multi-class classification
                metrics['auc_roc_macro'] = roc_auc_score(y_true, outputs, 
                                                       multi_class='ovr', average='macro')
                metrics['auc_roc_weighted'] = roc_auc_score(y_true, outputs, 
                                                          multi_class='ovr', average='weighted')
                
                # Per-class AUC metrics
                y_true_binary = np.zeros((len(y_true), self.num_classes))
                y_true_binary[np.arange(len(y_true)), y_true] = 1
                
                for i in range(self.num_classes):
                    try:
                        if np.sum(y_true_binary[:, i]) > 0:  # Check if class exists in y_true
                            auc_roc = roc_auc_score(y_true_binary[:, i], outputs[:, i])
                            auc_prc = average_precision_score(y_true_binary[:, i], outputs[:, i])
                            metrics[f'auc_roc_class_{self.class_names[i]}'] = auc_roc
                            metrics[f'auc_prc_class_{self.class_names[i]}'] = auc_prc
                        else:
                            metrics[f'auc_roc_class_{self.class_names[i]}'] = 0.0
                            metrics[f'auc_prc_class_{self.class_names[i]}'] = 0.0
                    except ValueError:
                        metrics[f'auc_roc_class_{self.class_names[i]}'] = 0.0
                        metrics[f'auc_prc_class_{self.class_names[i]}'] = 0.0
                        
        except ValueError as e:
            print(f"Warning: Could not calculate AUC metrics: {e}")
            metrics['auc_roc'] = 0.0
            metrics['auc_prc'] = 0.0
        
        return metrics