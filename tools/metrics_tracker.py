"""
Validation Metrics Tracker Module
Handles tracking of validation metrics across training epochs
"""

import numpy as np
import os
import csv
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    matthews_corrcoef,
    accuracy_score
)
from utils.logger import print_log


class Acc_Metric:
    """Metric wrapper for model selection.

    Supports selecting the best model by any tracked metric:
        'acc', 'balanced_acc', 'f1_macro', 'kappa', 'mcc'

    The selection metric is set via config.selection_metric (default: 'acc').
    """

    # Valid metric names for model selection (higher = better for all)
    VALID_METRICS = ('acc', 'balanced_acc', 'f1_macro', 'kappa', 'mcc')

    def __init__(self, acc=0., selection_metric='acc'):
        if type(acc).__name__ == 'dict':
            self.acc = acc.get('acc', 0.)
        elif type(acc).__name__ == 'Acc_Metric':
            self.acc = acc.acc
        else:
            self.acc = acc
        self.selection_metric = selection_metric
        # Store all metrics when available
        self._metrics = {'acc': self.acc}

    def set_metrics(self, metrics_dict):
        """Store full validation metrics for flexible comparison."""
        self._metrics = dict(metrics_dict)
        self.acc = metrics_dict.get('acc', self.acc)

    def _get_selection_value(self):
        return self._metrics.get(self.selection_metric, self.acc)

    def better_than(self, other):
        return self._get_selection_value() > other._get_selection_value()

    def state_dict(self):
        return dict(self._metrics)


class ValidationMetricsTracker:
    """Tracks comprehensive validation metrics across epochs without using global variables."""

    def __init__(self, num_classes, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names if class_names else [f'class{i}' for i in range(num_classes)]

        # History tracking for plotting
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_balanced_acc': [],
            'val_f1_macro': [],
            'val_kappa': [],
            'val_mcc': [],
        }

        # Metrics at best accuracy model
        self.best_acc = 0.0
        self.best_acc_epoch = 0
        self.balanced_acc_at_best_acc = 0.0
        self.f1_macro_at_best_acc = 0.0
        self.f1_weighted_at_best_acc = 0.0
        self.precision_macro_at_best_acc = 0.0
        self.precision_weighted_at_best_acc = 0.0
        self.kappa_at_best_acc = 0.0
        self.mcc_at_best_acc = 0.0
        self.predictions_at_best_acc = None
        self.labels_at_best_acc = None
        self.per_class_precision_at_best_acc = {}
        self.per_class_recall_at_best_acc = {}
        self.per_class_f1_at_best_acc = {}
        self.per_class_support_at_best_acc = {}
        self.confusion_matrix_at_best_acc = None

    def update(self, epoch, acc, balanced_acc, predictions, labels):
        """Update metrics tracking both best accuracy model and best individual metrics."""

        # Calculate all current metrics
        f1_macro = f1_score(labels, predictions, average='macro', zero_division=0) * 100.
        f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0) * 100.
        precision_macro = precision_score(labels, predictions, average='macro', zero_division=0) * 100.
        precision_weighted = precision_score(labels, predictions, average='weighted', zero_division=0) * 100.
        kappa = cohen_kappa_score(labels, predictions) * 100.
        mcc = matthews_corrcoef(labels, predictions) * 100.

        # Track if this is best accuracy
        is_best_acc = False

        # Update best accuracy model metrics
        if acc > self.best_acc:
            is_best_acc = True
            self.best_acc = acc
            self.best_acc_epoch = epoch
            self.balanced_acc_at_best_acc = balanced_acc
            self.f1_macro_at_best_acc = f1_macro
            self.f1_weighted_at_best_acc = f1_weighted
            self.precision_macro_at_best_acc = precision_macro
            self.precision_weighted_at_best_acc = precision_weighted
            self.kappa_at_best_acc = kappa
            self.mcc_at_best_acc = mcc

            self.predictions_at_best_acc = predictions.copy()
            self.labels_at_best_acc = labels.copy()

            # Generate confusion matrix
            self.confusion_matrix_at_best_acc = confusion_matrix(labels, predictions)

            # Get unique classes present in the data
            unique_classes = np.unique(np.concatenate([labels, predictions]))
            present_class_names = [self.class_names[i] for i in unique_classes if i < len(self.class_names)]

            # Generate detailed classification report only for present classes
            report = classification_report(
                labels,
                predictions,
                labels=unique_classes,
                target_names=present_class_names,
                output_dict=True,
                zero_division=0
            )

            # Extract per-class metrics at best accuracy
            self.per_class_precision_at_best_acc = {}
            self.per_class_recall_at_best_acc = {}
            self.per_class_f1_at_best_acc = {}
            self.per_class_support_at_best_acc = {}

            for class_name in present_class_names:
                if class_name in report:
                    self.per_class_precision_at_best_acc[f"precision_{class_name}"] = (
                        report[class_name]['precision'] * 100.
                    )
                    self.per_class_recall_at_best_acc[f"recall_{class_name}"] = (
                        report[class_name]['recall'] * 100.
                    )
                    self.per_class_f1_at_best_acc[f"f1_{class_name}"] = (
                        report[class_name]['f1-score'] * 100.
                    )
                    self.per_class_support_at_best_acc[f"support_{class_name}"] = (
                        report[class_name]['support']
                    )

        return is_best_acc

    def get_best_metrics_dict(self, trainable_params=None):
        """Return dictionary of metrics at best accuracy model (validation data only)."""
        metrics_dict = {
            # All metrics computed at the epoch with best validation accuracy
            'best_accuracy': self.best_acc,
            'balanced_acc': self.balanced_acc_at_best_acc,
            'f1_macro': self.f1_macro_at_best_acc,
            'f1_weighted': self.f1_weighted_at_best_acc,
            'precision_macro': self.precision_macro_at_best_acc,
            'precision_weighted': self.precision_weighted_at_best_acc,
            'kappa': self.kappa_at_best_acc,
            'mcc': self.mcc_at_best_acc,
            'best_epoch': self.best_acc_epoch,
        }

        if trainable_params is not None:
            metrics_dict['trainable_parameters'] = trainable_params

        # Add per-class metrics at best accuracy
        for key, value in self.per_class_precision_at_best_acc.items():
            metrics_dict[key] = value
        for key, value in self.per_class_recall_at_best_acc.items():
            metrics_dict[key] = value
        for key, value in self.per_class_f1_at_best_acc.items():
            metrics_dict[key] = value
        for key, value in self.per_class_support_at_best_acc.items():
            metrics_dict[key] = value

        return metrics_dict

    def get_confusion_matrix_dict(self):
        """Return confusion matrix as dictionary for logging."""
        if self.confusion_matrix_at_best_acc is None:
            return {}

        cm_dict = {}
        for i, true_class in enumerate(self.class_names):
            for j, pred_class in enumerate(self.class_names):
                key = f"cm_{true_class}_pred_{pred_class}"
                cm_dict[key] = int(self.confusion_matrix_at_best_acc[i, j])

        return cm_dict

    def print_best_metrics(self, logger=None):
        """Print metrics at best accuracy model (validation data only)."""
        print_log('[Best Accuracy Model] Epoch %d:' % self.best_acc_epoch, logger=logger)
        print_log('  Accuracy: %.4f%%, Balanced Acc: %.4f%%' %
                  (self.best_acc, self.balanced_acc_at_best_acc), logger=logger)
        print_log('  F1 (macro): %.4f%%, F1 (weighted): %.4f%%' %
                  (self.f1_macro_at_best_acc, self.f1_weighted_at_best_acc), logger=logger)
        print_log('  Precision (macro): %.4f%%, Precision (weighted): %.4f%%' %
                  (self.precision_macro_at_best_acc, self.precision_weighted_at_best_acc), logger=logger)
        print_log("  Cohen's Kappa: %.4f%%, MCC: %.4f%%" %
                  (self.kappa_at_best_acc, self.mcc_at_best_acc), logger=logger)

    def print_confusion_matrix(self, logger=None):
        """Print confusion matrix in readable format."""
        if self.confusion_matrix_at_best_acc is None:
            return

        print_log('[Confusion Matrix at Best Accuracy Model]:', logger=logger)

        # Header
        header = '      ' + ' '.join([f'{name:>10}' for name in self.class_names])
        print_log(header, logger=logger)

        # Rows
        for i, true_class in enumerate(self.class_names):
            row = f'{true_class:>6}' + ' '.join([
                f'{self.confusion_matrix_at_best_acc[i, j]:>10}'
                for j in range(len(self.class_names))
            ])
            print_log(row, logger=logger)

    def print_per_class_metrics(self, logger=None):
        """Print per-class metrics in readable format."""
        print_log('[Per-Class Metrics at Best Accuracy Model]:', logger=logger)
        print_log(
            f'{"Class":<12} {"Precision":>12} {"Recall":>12} {"F1-Score":>12} {"Support":>12}',
            logger=logger
        )
        print_log('-' * 60, logger=logger)

        for class_name in self.class_names:
            prec = self.per_class_precision_at_best_acc.get(f"precision_{class_name}", 0.0)
            rec = self.per_class_recall_at_best_acc.get(f"recall_{class_name}", 0.0)
            f1 = self.per_class_f1_at_best_acc.get(f"f1_{class_name}", 0.0)
            sup = self.per_class_support_at_best_acc.get(f"support_{class_name}", 0)

            print_log(
                f'{class_name:<12} {prec:>11.4f}% {rec:>11.4f}% {f1:>11.4f}% {sup:>12}',
                logger=logger
            )

    def update_history(self, epoch, train_loss=None, train_acc=None, val_acc=None,
                       val_balanced_acc=None, val_f1_macro=None, val_kappa=None, val_mcc=None):
        """Update training/validation history for plotting."""
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss if train_loss is not None else 0)
        self.history['train_acc'].append(train_acc if train_acc is not None else 0)
        self.history['val_acc'].append(val_acc if val_acc is not None else 0)
        self.history['val_balanced_acc'].append(val_balanced_acc if val_balanced_acc is not None else 0)
        self.history['val_f1_macro'].append(val_f1_macro if val_f1_macro is not None else 0)
        self.history['val_kappa'].append(val_kappa if val_kappa is not None else 0)
        self.history['val_mcc'].append(val_mcc if val_mcc is not None else 0)

    def save_confusion_matrix_pdf(self, save_path, title="Confusion Matrix", verbose=False):
        """Save confusion matrix as a formatted PDF heatmap with publication-quality fonts."""
        if self.confusion_matrix_at_best_acc is None:
            if verbose:
                print_log("No confusion matrix available to save.", logger=None)
            return

        # Set publication-quality font settings
        plt.rcParams.update({
            'font.size': 14,
            'font.weight': 'bold',
            'axes.labelsize': 16,
            'axes.labelweight': 'bold',
            'axes.titlesize': 18,
            'axes.titleweight': 'bold',
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'figure.titlesize': 20,
            'figure.titleweight': 'bold',
        })

        # Create figure with appropriate size based on number of classes
        fig_size = max(10, self.num_classes * 1.5)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        # Create heatmap
        cm = self.confusion_matrix_at_best_acc

        # Normalize confusion matrix for better visualization (show percentages)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero

        # Create annotations with both count and percentage
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = cm[i, j]
                pct = cm_normalized[i, j] * 100
                annot[i, j] = f'{count}\n({pct:.1f}%)'

        # Plot heatmap with larger annotation font
        sns.heatmap(
            cm,
            annot=annot,
            fmt='',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax,
            cbar_kws={'label': 'Count'},
            linewidths=0.5,
            linecolor='gray',
            annot_kws={'size': 12, 'weight': 'bold'}
        )

        ax.set_xlabel('Predicted Label', fontsize=18, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=18, fontweight='bold')
        ax.set_title(f'{title}\n(Best Accuracy: {self.best_acc:.2f}% at Epoch {self.best_acc_epoch})',
                     fontsize=20, fontweight='bold', pad=20)

        # Rotate x-axis labels for better readability with larger font
        plt.xticks(rotation=45, ha='right', fontsize=14, fontweight='bold')
        plt.yticks(rotation=0, fontsize=14, fontweight='bold')

        # Make colorbar label bold
        cbar = ax.collections[0].colorbar
        cbar.ax.set_ylabel('Count', fontsize=14, fontweight='bold')
        cbar.ax.tick_params(labelsize=12)

        # Adjust layout
        plt.tight_layout()

        # Save as PDF
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Reset to default
        plt.rcParams.update(plt.rcParamsDefault)

        if verbose:
            print_log(f"Confusion matrix saved to: {save_path}", logger=None)

    def save_confusion_matrix_latex(self, save_path, title="Confusion Matrix", verbose=False):
        """Save confusion matrix as a LaTeX table."""
        if self.confusion_matrix_at_best_acc is None:
            if verbose:
                print_log("No confusion matrix available to save.", logger=None)
            return

        cm = self.confusion_matrix_at_best_acc
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)

        n_classes = len(self.class_names)

        latex_lines = []
        latex_lines.append("% Confusion Matrix - Auto-generated")
        latex_lines.append("% Requires: \\usepackage{booktabs}, \\usepackage{multirow}, \\usepackage{colortbl}")
        latex_lines.append("")
        latex_lines.append("\\begin{table}[htbp]")
        latex_lines.append("\\centering")
        latex_lines.append("\\scriptsize")
        latex_lines.append(f"\\caption{{{title} (Best Accuracy: {self.best_acc:.2f}\\% at Epoch {self.best_acc_epoch})}}")
        latex_lines.append("\\label{tab:confusion_matrix}")

        # Column format: one for row label + one for each class
        col_format = "l" + "c" * n_classes
        latex_lines.append(f"\\begin{{tabular}}{{{col_format}}}")
        latex_lines.append("\\toprule")

        # Header row with predicted labels
        header = "& \\multicolumn{" + str(n_classes) + "}{c}{\\textbf{Predicted}} \\\\"
        latex_lines.append(header)

        # Class names header
        class_header = "\\textbf{True}"
        for class_name in self.class_names:
            class_header += f" & \\textbf{{{class_name}}}"
        class_header += " \\\\"
        latex_lines.append(class_header)
        latex_lines.append("\\midrule")

        # Data rows
        for i, true_class in enumerate(self.class_names):
            row = f"\\textbf{{{true_class}}}"
            for j in range(n_classes):
                count = cm[i, j]
                pct = cm_normalized[i, j] * 100
                # Highlight diagonal (correct predictions) in bold
                if i == j:
                    row += f" & \\textbf{{{count}}} ({pct:.1f}\\%)"
                else:
                    row += f" & {count} ({pct:.1f}\\%)"
            row += " \\\\"
            latex_lines.append(row)

        latex_lines.append("\\bottomrule")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\end{table}")

        # Write to file
        with open(save_path, 'w') as f:
            f.write('\n'.join(latex_lines))

        if verbose:
            print_log(f"Confusion matrix (LaTeX) saved to: {save_path}", logger=None)

    def save_training_history_plot(self, save_path, title="Training History", verbose=False):
        """Save training/validation accuracy history as separate PDF plots with publication-quality fonts."""
        if not self.history['epoch']:
            if verbose:
                print_log("No history available to plot.", logger=None)
            return

        # Set publication-quality font settings
        plt.rcParams.update({
            'font.size': 16,
            'font.weight': 'bold',
            'axes.labelsize': 18,
            'axes.labelweight': 'bold',
            'axes.titlesize': 20,
            'axes.titleweight': 'bold',
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 14,
            'figure.titlesize': 22,
            'figure.titleweight': 'bold',
            'lines.linewidth': 2.5,
        })

        epochs = self.history['epoch']
        base_path = save_path.replace('.pdf', '')

        # Plot 1: Training and Validation Accuracy
        fig1, ax1 = plt.subplots(figsize=(10, 7))
        if any(self.history['train_acc']):
            ax1.plot(epochs, self.history['train_acc'], 'b-', label='Train Accuracy', linewidth=2.5)
        ax1.plot(epochs, self.history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2.5)
        ax1.axhline(y=self.best_acc, color='g', linestyle='--', linewidth=2,
                    label=f'Best Val Acc: {self.best_acc:.2f}%')
        ax1.set_xlabel('Epoch', fontsize=18, fontweight='bold')
        ax1.set_ylabel('Accuracy (%)', fontsize=18, fontweight='bold')
        ax1.set_title(f'{title}\nTraining & Validation Accuracy', fontsize=20, fontweight='bold', pad=15)
        ax1.legend(loc='lower right', fontsize=14, frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.tick_params(axis='both', which='major', labelsize=14)
        plt.tight_layout()
        plt.savefig(f'{base_path}_accuracy.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.close(fig1)

        # Plot 2: Training Loss
        if any(self.history['train_loss']):
            fig2, ax2 = plt.subplots(figsize=(10, 7))
            ax2.plot(epochs, self.history['train_loss'], 'b-', linewidth=2.5)
            ax2.set_xlabel('Epoch', fontsize=18, fontweight='bold')
            ax2.set_ylabel('Loss', fontsize=18, fontweight='bold')
            ax2.set_title(f'{title}\nTraining Loss', fontsize=20, fontweight='bold', pad=15)
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.tick_params(axis='both', which='major', labelsize=14)
            plt.tight_layout()
            plt.savefig(f'{base_path}_loss.pdf', format='pdf', dpi=300, bbox_inches='tight')
            plt.close(fig2)

        # Plot 3: Validation Metrics (Balanced Acc, F1)
        fig3, ax3 = plt.subplots(figsize=(10, 7))
        ax3.plot(epochs, self.history['val_balanced_acc'], 'g-', label='Balanced Accuracy', linewidth=2.5)
        ax3.plot(epochs, self.history['val_f1_macro'], 'm-', label='F1 Macro', linewidth=2.5)
        ax3.set_xlabel('Epoch', fontsize=18, fontweight='bold')
        ax3.set_ylabel('Score (%)', fontsize=18, fontweight='bold')
        ax3.set_title(f'{title}\nBalanced Accuracy & F1 Score', fontsize=20, fontweight='bold', pad=15)
        ax3.legend(loc='lower right', fontsize=14, frameon=True, fancybox=True, shadow=True)
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.tick_params(axis='both', which='major', labelsize=14)
        plt.tight_layout()
        plt.savefig(f'{base_path}_metrics.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.close(fig3)

        # Plot 4: Kappa and MCC
        fig4, ax4 = plt.subplots(figsize=(10, 7))
        ax4.plot(epochs, self.history['val_kappa'], 'c-', label="Cohen's Kappa", linewidth=2.5)
        ax4.plot(epochs, self.history['val_mcc'], 'orange', label='MCC', linewidth=2.5)
        ax4.set_xlabel('Epoch', fontsize=18, fontweight='bold')
        ax4.set_ylabel('Score (%)', fontsize=18, fontweight='bold')
        ax4.set_title(f'{title}\nKappa & Matthews Correlation Coefficient', fontsize=20, fontweight='bold', pad=15)
        ax4.legend(loc='lower right', fontsize=14, frameon=True, fancybox=True, shadow=True)
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.tick_params(axis='both', which='major', labelsize=14)
        plt.tight_layout()
        plt.savefig(f'{base_path}_kappa_mcc.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.close(fig4)

        # Reset to default
        plt.rcParams.update(plt.rcParamsDefault)

        if verbose:
            print_log(f"Training history plots saved to: {base_path}_*.pdf", logger=None)

    def save_cv_results_csv(self, save_path, model_name, fold=None, k_folds=None,
                            trainable_params=None, total_params=None, extra_info=None, verbose=False):
        """Save model results to CSV file (validation metrics at best accuracy only)."""

        # Prepare data row - only metrics from best accuracy model (validation data)
        row_data = {
            'model_name': model_name,
            'fold': fold if fold is not None else 'N/A',
            'k_folds': k_folds if k_folds is not None else 'N/A',
            'best_accuracy': f'{self.best_acc:.4f}',
            'best_epoch': self.best_acc_epoch,
            'balanced_acc': f'{self.balanced_acc_at_best_acc:.4f}',
            'f1_macro': f'{self.f1_macro_at_best_acc:.4f}',
            'f1_weighted': f'{self.f1_weighted_at_best_acc:.4f}',
            'precision_macro': f'{self.precision_macro_at_best_acc:.4f}',
            'precision_weighted': f'{self.precision_weighted_at_best_acc:.4f}',
            'kappa': f'{self.kappa_at_best_acc:.4f}',
            'mcc': f'{self.mcc_at_best_acc:.4f}',
            'trainable_params': trainable_params if trainable_params else 'N/A',
            'total_params': total_params if total_params else 'N/A',
        }

        # Add per-class metrics
        for class_name in self.class_names:
            prec = self.per_class_precision_at_best_acc.get(f"precision_{class_name}", 0.0)
            rec = self.per_class_recall_at_best_acc.get(f"recall_{class_name}", 0.0)
            f1 = self.per_class_f1_at_best_acc.get(f"f1_{class_name}", 0.0)
            sup = self.per_class_support_at_best_acc.get(f"support_{class_name}", 0)
            row_data[f'precision_{class_name}'] = f'{prec:.4f}'
            row_data[f'recall_{class_name}'] = f'{rec:.4f}'
            row_data[f'f1_{class_name}'] = f'{f1:.4f}'
            row_data[f'support_{class_name}'] = int(sup)

        # Add extra info if provided
        if extra_info:
            row_data.update(extra_info)

        # Check if file exists to determine if we need to write header
        file_exists = os.path.exists(save_path)

        with open(save_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)

        if verbose:
            print_log(f"Results saved to CSV: {save_path}", logger=None)

    def save_history_csv(self, save_path, verbose=False):
        """Save training history to CSV file."""
        if not self.history['epoch']:
            if verbose:
                print_log("No history available to save.", logger=None)
            return

        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.history.keys())
            rows = zip(*self.history.values())
            writer.writerows(rows)

        if verbose:
            print_log(f"Training history saved to: {save_path}", logger=None)
