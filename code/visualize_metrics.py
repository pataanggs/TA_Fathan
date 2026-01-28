"""
Visualization module for Whisper fine-tuning metrics.
Creates comprehensive plots for training metrics, evaluation results,
learning rate schedules, and cross-validation summaries.
Enhanced with detailed data points and annotations.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from datetime import datetime


# Set style for better visibility
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['lines.markersize'] = 8
plt.rcParams['lines.linewidth'] = 2

# Color palette
COLORS = {
    'train_loss': '#2ecc71',      # Green
    'eval_loss': '#e74c3c',       # Red
    'wer': '#3498db',             # Blue
    'cer': '#9b59b6',             # Purple
    'lr': '#f39c12',              # Orange
    'grad_norm': '#1abc9c',       # Teal
    'folds': ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12'],  # For multiple folds
}

# Markers for each fold
MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']


class MetricsVisualizer:
    """
    Visualize training metrics from the metrics logger output.
    """
    
    def __init__(self, metrics_dir: str, output_dir: str = None):
        """
        Initialize visualizer with metrics directory.
        
        Args:
            metrics_dir: Path to the run directory (e.g., outputs/metrics/run_20260122_160012)
            output_dir: Optional output directory for saving plots
        """
        self.metrics_dir = Path(metrics_dir)
        self.output_dir = Path(output_dir) if output_dir else self.metrics_dir / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load cross-validation summary if exists
        self.cv_summary = self._load_json(self.metrics_dir / "cross_validation_summary.json")
        
        # Detect folds
        self.fold_dirs = sorted([
            d for d in self.metrics_dir.iterdir() 
            if d.is_dir() and d.name.startswith("fold_")
        ])
        self.num_folds = len(self.fold_dirs)
        
        print(f"📊 MetricsVisualizer initialized")
        print(f"   Metrics directory: {self.metrics_dir}")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Found {self.num_folds} folds")
    
    def _load_json(self, path: Path) -> Optional[Dict]:
        """Load JSON file if exists."""
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return None
    
    def _load_fold_data(self, fold_idx: int) -> Dict:
        """Load all data for a specific fold."""
        fold_dir = self.metrics_dir / f"fold_{fold_idx}"
        return {
            "training_logs": self._load_json(fold_dir / "training_logs.json") or [],
            "evaluation_logs": self._load_json(fold_dir / "evaluation_logs.json") or [],
            "learning_rate_logs": self._load_json(fold_dir / "learning_rate_logs.json") or [],
            "epoch_summaries": self._load_json(fold_dir / "epoch_summaries.json") or [],
            "sample_predictions": self._load_json(fold_dir / "sample_predictions.json") or [],
            "final_results": self._load_json(fold_dir / "final_results.json") or {},
            "training_config": self._load_json(fold_dir / "training_config.json") or {},
            "model_config": self._load_json(fold_dir / "model_config.json") or {},
        }
    
    def plot_training_loss(self, save: bool = True) -> plt.Figure:
        """Plot training loss over steps for all folds with detailed data points."""
        fig, ax = plt.subplots(figsize=(16, 8))
        
        all_min_loss = float('inf')
        all_max_loss = 0
        
        for fold_idx in range(self.num_folds):
            data = self._load_fold_data(fold_idx)
            training_logs = data["training_logs"]
            
            if training_logs:
                steps = [log["step"] for log in training_logs]
                losses = [log["loss"] for log in training_logs if log["loss"] is not None]
                steps = steps[:len(losses)]
                
                color = COLORS['folds'][fold_idx % len(COLORS['folds'])]
                marker = MARKERS[fold_idx % len(MARKERS)]
                
                # Plot line with markers
                ax.plot(steps, losses, 
                       color=color, alpha=0.8, linewidth=2,
                       marker=marker, markersize=4, markevery=max(1, len(steps)//50),
                       label=f'Fold {fold_idx + 1}')
                
                # Track min/max
                if losses:
                    all_min_loss = min(all_min_loss, min(losses))
                    all_max_loss = max(all_max_loss, max(losses))
                    
                    # Annotate min loss point
                    min_idx = losses.index(min(losses))
                    ax.annotate(f'{losses[min_idx]:.3f}', 
                               xy=(steps[min_idx], losses[min_idx]),
                               xytext=(10, 10), textcoords='offset points',
                               fontsize=9, color=color,
                               arrowprops=dict(arrowstyle='->', color=color, alpha=0.6))
        
        ax.set_xlabel('Training Step', fontsize=14)
        ax.set_ylabel('Loss', fontsize=14)
        ax.set_title('Training Loss Across All Folds (with Data Points)', fontsize=16, fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.4, linestyle='--')
        
        # Add minor gridlines
        ax.minorticks_on()
        ax.grid(True, which='minor', alpha=0.2, linestyle=':')
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / "training_loss_detailed.png"
            fig.savefig(path, dpi=200, bbox_inches='tight')
            print(f"✅ Saved: {path}")
        
        return fig
    
    def plot_evaluation_metrics(self, save: bool = True) -> plt.Figure:
        """Plot WER and CER over evaluation steps for all folds with detailed data points."""
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        
        # Flatten for easier iteration
        ax_wer_all = axes[0, 0]
        ax_cer_all = axes[0, 1]
        ax_wer_detail = axes[1, 0]
        ax_cer_detail = axes[1, 1]
        
        best_wer_overall = float('inf')
        best_cer_overall = float('inf')
        best_wer_fold = 0
        best_cer_fold = 0
        
        for fold_idx in range(self.num_folds):
            data = self._load_fold_data(fold_idx)
            eval_logs = data["evaluation_logs"]
            
            if eval_logs:
                steps = [log["step"] for log in eval_logs]
                wers = [log["wer"] * 100 for log in eval_logs]
                cers = [log["cer"] * 100 for log in eval_logs]
                
                color = COLORS['folds'][fold_idx % len(COLORS['folds'])]
                marker = MARKERS[fold_idx % len(MARKERS)]
                
                # WER plot with all points visible
                ax_wer_all.plot(steps, wers, color=color, alpha=0.9, 
                               linewidth=2, marker=marker, markersize=6,
                               label=f'Fold {fold_idx + 1} (Best: {min(wers):.2f}%)')
                
                # CER plot with all points visible
                ax_cer_all.plot(steps, cers, color=color, alpha=0.9,
                               linewidth=2, marker=marker, markersize=6,
                               label=f'Fold {fold_idx + 1} (Best: {min(cers):.2f}%)')
                
                # Track best values
                if min(wers) < best_wer_overall:
                    best_wer_overall = min(wers)
                    best_wer_fold = fold_idx
                if min(cers) < best_cer_overall:
                    best_cer_overall = min(cers)
                    best_cer_fold = fold_idx
                
                # Add annotations for first, last, and best points
                # Best WER point
                best_wer_idx = wers.index(min(wers))
                ax_wer_all.annotate(f'{wers[best_wer_idx]:.2f}%', 
                                   xy=(steps[best_wer_idx], wers[best_wer_idx]),
                                   xytext=(5, -15), textcoords='offset points',
                                   fontsize=9, color=color, fontweight='bold')
                
                # Best CER point
                best_cer_idx = cers.index(min(cers))
                ax_cer_all.annotate(f'{cers[best_cer_idx]:.2f}%', 
                                   xy=(steps[best_cer_idx], cers[best_cer_idx]),
                                   xytext=(5, -15), textcoords='offset points',
                                   fontsize=9, color=color, fontweight='bold')
        
        # Style WER plot
        ax_wer_all.set_xlabel('Training Step', fontsize=14)
        ax_wer_all.set_ylabel('WER (%)', fontsize=14)
        ax_wer_all.set_title(f'Word Error Rate (WER) - All Folds\nOverall Best: {best_wer_overall:.2f}% (Fold {best_wer_fold + 1})', 
                            fontsize=14, fontweight='bold')
        ax_wer_all.legend(loc='upper right', framealpha=0.9)
        ax_wer_all.grid(True, alpha=0.4, linestyle='--')
        ax_wer_all.minorticks_on()
        ax_wer_all.grid(True, which='minor', alpha=0.2, linestyle=':')
        
        # Style CER plot
        ax_cer_all.set_xlabel('Training Step', fontsize=14)
        ax_cer_all.set_ylabel('CER (%)', fontsize=14)
        ax_cer_all.set_title(f'Character Error Rate (CER) - All Folds\nOverall Best: {best_cer_overall:.2f}% (Fold {best_cer_fold + 1})', 
                            fontsize=14, fontweight='bold')
        ax_cer_all.legend(loc='upper right', framealpha=0.9)
        ax_cer_all.grid(True, alpha=0.4, linestyle='--')
        ax_cer_all.minorticks_on()
        ax_cer_all.grid(True, which='minor', alpha=0.2, linestyle=':')
        
        # Detail plots - show data points with values
        self._plot_metric_with_values(ax_wer_detail, 'wer', 'WER')
        self._plot_metric_with_values(ax_cer_detail, 'cer', 'CER')
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / "evaluation_metrics_detailed.png"
            fig.savefig(path, dpi=200, bbox_inches='tight')
            print(f"✅ Saved: {path}")
        
        return fig
    
    def _plot_metric_with_values(self, ax, metric_key: str, metric_name: str):
        """Helper to plot metric with value annotations for each point."""
        for fold_idx in range(self.num_folds):
            data = self._load_fold_data(fold_idx)
            eval_logs = data["evaluation_logs"]
            
            if eval_logs:
                steps = [log["step"] for log in eval_logs]
                values = [log[metric_key] * 100 for log in eval_logs]
                
                color = COLORS['folds'][fold_idx % len(COLORS['folds'])]
                marker = MARKERS[fold_idx % len(MARKERS)]
                
                # Plot with larger markers
                ax.plot(steps, values, color=color, alpha=0.8,
                       linewidth=1.5, marker=marker, markersize=8,
                       label=f'Fold {fold_idx + 1}')
                
                # Annotate every few points to avoid clutter
                annotate_every = max(1, len(steps) // 8)
                for i in range(0, len(steps), annotate_every):
                    ax.annotate(f'{values[i]:.1f}', 
                               xy=(steps[i], values[i]),
                               xytext=(0, 8), textcoords='offset points',
                               fontsize=8, color=color, ha='center',
                               alpha=0.8)
        
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel(f'{metric_name} (%)', fontsize=12)
        ax.set_title(f'{metric_name} with Values Annotated', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.4, linestyle='--')
    
    def plot_learning_rate_schedule(self, save: bool = True) -> plt.Figure:
        """Plot learning rate schedule with detailed data points."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Just use fold 0 since LR schedule is the same
        data = self._load_fold_data(0)
        lr_logs = data["learning_rate_logs"]
        
        if lr_logs:
            steps = [log["step"] for log in lr_logs]
            lrs = [log["learning_rate"] for log in lr_logs]
            epochs = [log.get("epoch", i/4) for i, log in enumerate(lr_logs)]
            
            # Left plot: LR vs Steps with points
            axes[0].plot(steps, lrs, color=COLORS['lr'], linewidth=2.5, 
                        marker='o', markersize=4, markevery=max(1, len(steps)//30))
            axes[0].fill_between(steps, lrs, alpha=0.3, color=COLORS['lr'])
            
            # Annotate key points
            # Max LR point
            max_lr_idx = lrs.index(max(lrs))
            axes[0].annotate(f'Peak: {lrs[max_lr_idx]:.2e}\nStep {steps[max_lr_idx]}', 
                            xy=(steps[max_lr_idx], lrs[max_lr_idx]),
                            xytext=(30, -20), textcoords='offset points',
                            fontsize=10, color='red', fontweight='bold',
                            arrowprops=dict(arrowstyle='->', color='red'))
            
            # Start and end
            axes[0].annotate(f'Start: {lrs[0]:.2e}', 
                            xy=(steps[0], lrs[0]),
                            xytext=(10, 10), textcoords='offset points',
                            fontsize=9, color=COLORS['lr'])
            axes[0].annotate(f'End: {lrs[-1]:.2e}', 
                            xy=(steps[-1], lrs[-1]),
                            xytext=(-60, 10), textcoords='offset points',
                            fontsize=9, color=COLORS['lr'])
            
            axes[0].set_xlabel('Training Step', fontsize=14)
            axes[0].set_ylabel('Learning Rate', fontsize=14)
            axes[0].set_title('Learning Rate Schedule (Cosine Annealing with Warmup)', 
                             fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.4, linestyle='--')
            axes[0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            axes[0].minorticks_on()
            axes[0].grid(True, which='minor', alpha=0.2, linestyle=':')
            
            # Right plot: LR vs Epoch
            axes[1].plot(epochs, lrs, color=COLORS['lr'], linewidth=2.5,
                        marker='s', markersize=4, markevery=max(1, len(epochs)//30))
            axes[1].fill_between(epochs, lrs, alpha=0.3, color=COLORS['lr'])
            
            axes[1].set_xlabel('Epoch', fontsize=14)
            axes[1].set_ylabel('Learning Rate', fontsize=14)
            axes[1].set_title('Learning Rate vs Epoch', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.4, linestyle='--')
            axes[1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            axes[1].minorticks_on()
            axes[1].grid(True, which='minor', alpha=0.2, linestyle=':')
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / "learning_rate_schedule_detailed.png"
            fig.savefig(path, dpi=200, bbox_inches='tight')
            print(f"✅ Saved: {path}")
        
        return fig
    
    def plot_gradient_norm(self, save: bool = True) -> plt.Figure:
        """Plot gradient norm over training with detailed statistics."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        all_grads_per_fold = []
        
        for fold_idx in range(self.num_folds):
            data = self._load_fold_data(fold_idx)
            training_logs = data["training_logs"]
            
            if training_logs:
                steps = [log["step"] for log in training_logs if log.get("grad_norm")]
                grad_norms = [log["grad_norm"] for log in training_logs if log.get("grad_norm")]
                
                if steps:
                    color = COLORS['folds'][fold_idx % len(COLORS['folds'])]
                    marker = MARKERS[fold_idx % len(MARKERS)]
                    
                    # Left plot: gradient norm over steps
                    axes[0].plot(steps, grad_norms,
                                color=color, alpha=0.8, linewidth=1.5,
                                marker=marker, markersize=3, markevery=max(1, len(steps)//40),
                                label=f'Fold {fold_idx + 1} (μ={np.mean(grad_norms):.2f})')
                    
                    all_grads_per_fold.append((fold_idx, grad_norms))
        
        axes[0].set_xlabel('Training Step', fontsize=14)
        axes[0].set_ylabel('Gradient Norm', fontsize=14)
        axes[0].set_title('Gradient Norm During Training', fontsize=14, fontweight='bold')
        axes[0].legend(loc='upper right', framealpha=0.9)
        axes[0].grid(True, alpha=0.4, linestyle='--')
        axes[0].minorticks_on()
        axes[0].grid(True, which='minor', alpha=0.2, linestyle=':')
        
        # Right plot: Box plot of gradient norms per fold
        if all_grads_per_fold:
            bp = axes[1].boxplot([grads for _, grads in all_grads_per_fold],
                                labels=[f'Fold {idx+1}' for idx, _ in all_grads_per_fold],
                                patch_artist=True)
            
            # Color the boxes
            for patch, (idx, _) in zip(bp['boxes'], all_grads_per_fold):
                patch.set_facecolor(COLORS['folds'][idx % len(COLORS['folds'])])
                patch.set_alpha(0.6)
            
            axes[1].set_ylabel('Gradient Norm', fontsize=14)
            axes[1].set_title('Gradient Norm Distribution by Fold', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.4, linestyle='--', axis='y')
            
            # Add mean annotations
            for i, (idx, grads) in enumerate(all_grads_per_fold):
                axes[1].annotate(f'μ={np.mean(grads):.2f}', 
                                xy=(i+1, np.mean(grads)),
                                xytext=(0, 10), textcoords='offset points',
                                fontsize=9, ha='center', color='red', fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / "gradient_norm_detailed.png"
            fig.savefig(path, dpi=200, bbox_inches='tight')
            print(f"✅ Saved: {path}")
        
        return fig
    
    def plot_loss_comparison(self, save: bool = True) -> plt.Figure:
        """Plot training vs evaluation loss with detailed annotations."""
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        # Left: Combined view
        ax = axes[0]
        for fold_idx in range(self.num_folds):
            data = self._load_fold_data(fold_idx)
            training_logs = data["training_logs"]
            eval_logs = data["evaluation_logs"]
            
            color = COLORS['folds'][fold_idx % len(COLORS['folds'])]
            marker = MARKERS[fold_idx % len(MARKERS)]
            
            if training_logs:
                steps = [log["step"] for log in training_logs]
                losses = [log["loss"] for log in training_logs if log["loss"] is not None]
                ax.plot(steps[:len(losses)], losses, 
                       color=color, alpha=0.4, linewidth=1,
                       label=f'Train F{fold_idx + 1}' if fold_idx == 0 else None)
            
            if eval_logs:
                steps = [log["step"] for log in eval_logs]
                eval_losses = [log["eval_loss"] for log in eval_logs if log.get("eval_loss")]
                ax.plot(steps[:len(eval_losses)], eval_losses,
                       color=color, alpha=0.9, linewidth=2.5, 
                       marker=marker, markersize=7,
                       label=f'Eval Fold {fold_idx + 1}')
                
                # Annotate best eval loss
                if eval_losses:
                    best_idx = eval_losses.index(min(eval_losses))
                    ax.annotate(f'{eval_losses[best_idx]:.3f}', 
                               xy=(steps[best_idx], eval_losses[best_idx]),
                               xytext=(5, 10), textcoords='offset points',
                               fontsize=9, color=color, fontweight='bold')
        
        ax.set_xlabel('Training Step', fontsize=14)
        ax.set_ylabel('Loss', fontsize=14)
        ax.set_title('Training vs Evaluation Loss (All Folds)', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', ncol=2, fontsize=10)
        ax.grid(True, alpha=0.4, linestyle='--')
        ax.minorticks_on()
        ax.grid(True, which='minor', alpha=0.2, linestyle=':')
        
        # Right: Eval loss only with more detail
        ax2 = axes[1]
        for fold_idx in range(self.num_folds):
            data = self._load_fold_data(fold_idx)
            eval_logs = data["evaluation_logs"]
            
            if eval_logs:
                steps = [log["step"] for log in eval_logs]
                eval_losses = [log["eval_loss"] for log in eval_logs if log.get("eval_loss")]
                epochs = [log["epoch"] for log in eval_logs]
                
                color = COLORS['folds'][fold_idx % len(COLORS['folds'])]
                marker = MARKERS[fold_idx % len(MARKERS)]
                
                ax2.plot(epochs[:len(eval_losses)], eval_losses,
                        color=color, alpha=0.9, linewidth=2,
                        marker=marker, markersize=8,
                        label=f'Fold {fold_idx + 1} (Best: {min(eval_losses):.3f})')
                
                # Show all values for small datasets
                if len(eval_losses) <= 20:
                    for i, (e, l) in enumerate(zip(epochs[:len(eval_losses)], eval_losses)):
                        if i % 2 == 0:  # Every other point
                            ax2.annotate(f'{l:.2f}', 
                                        xy=(e, l),
                                        xytext=(0, 8), textcoords='offset points',
                                        fontsize=8, ha='center', color=color, alpha=0.8)
        
        ax2.set_xlabel('Epoch', fontsize=14)
        ax2.set_ylabel('Evaluation Loss', fontsize=14)
        ax2.set_title('Evaluation Loss vs Epoch (Detailed)', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.4, linestyle='--')
        ax2.minorticks_on()
        ax2.grid(True, which='minor', alpha=0.2, linestyle=':')
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / "loss_comparison_detailed.png"
            fig.savefig(path, dpi=200, bbox_inches='tight')
            print(f"✅ Saved: {path}")
        
        return fig
    
    def plot_cross_validation_summary(self, save: bool = True) -> plt.Figure:
        """Plot cross-validation results summary."""
        if not self.cv_summary:
            print("⚠️ No cross-validation summary found")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Get per-fold results
        fold_results = self.cv_summary.get("per_fold_results", [])
        folds = [f"Fold {r['fold'] + 1}" for r in fold_results]
        wers = [r["wer_percent"] for r in fold_results]
        cers = [r["cer_percent"] for r in fold_results]
        
        cv_stats = self.cv_summary.get("cross_validation_summary", {})
        
        # WER Bar Chart
        bars1 = axes[0].bar(folds, wers, color=COLORS['wer'], alpha=0.7, edgecolor='black')
        axes[0].axhline(y=cv_stats.get("mean_wer_percent", 0), color='red', 
                       linestyle='--', linewidth=2, label=f'Mean: {cv_stats.get("mean_wer_percent", 0):.2f}%')
        axes[0].fill_between(
            range(-1, len(folds) + 1),
            cv_stats.get("mean_wer_percent", 0) - cv_stats.get("std_wer_percent", 0),
            cv_stats.get("mean_wer_percent", 0) + cv_stats.get("std_wer_percent", 0),
            alpha=0.2, color='red', label=f'±1 Std: {cv_stats.get("std_wer_percent", 0):.2f}%'
        )
        axes[0].set_ylabel('WER (%)')
        axes[0].set_title('Word Error Rate by Fold')
        axes[0].legend(loc='upper right')
        axes[0].set_xlim(-0.5, len(folds) - 0.5)
        
        # Add value labels on bars
        for bar, val in zip(bars1, wers):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # CER Bar Chart
        bars2 = axes[1].bar(folds, cers, color=COLORS['cer'], alpha=0.7, edgecolor='black')
        axes[1].axhline(y=cv_stats.get("mean_cer_percent", 0), color='red',
                       linestyle='--', linewidth=2, label=f'Mean: {cv_stats.get("mean_cer_percent", 0):.2f}%')
        axes[1].fill_between(
            range(-1, len(folds) + 1),
            cv_stats.get("mean_cer_percent", 0) - cv_stats.get("std_cer_percent", 0),
            cv_stats.get("mean_cer_percent", 0) + cv_stats.get("std_cer_percent", 0),
            alpha=0.2, color='red', label=f'±1 Std: {cv_stats.get("std_cer_percent", 0):.2f}%'
        )
        axes[1].set_ylabel('CER (%)')
        axes[1].set_title('Character Error Rate by Fold')
        axes[1].legend(loc='upper right')
        axes[1].set_xlim(-0.5, len(folds) - 0.5)
        
        # Add value labels on bars
        for bar, val in zip(bars2, cers):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{val:.2f}%', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / "cross_validation_summary.png"
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved: {path}")
        
        return fig
    
    def plot_comprehensive_dashboard(self, save: bool = True) -> plt.Figure:
        """Create a comprehensive dashboard with all key metrics."""
        fig = plt.figure(figsize=(18, 14))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Training Loss (top-left, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        for fold_idx in range(self.num_folds):
            data = self._load_fold_data(fold_idx)
            training_logs = data["training_logs"]
            if training_logs:
                steps = [log["step"] for log in training_logs]
                losses = [log["loss"] for log in training_logs if log["loss"] is not None]
                ax1.plot(steps[:len(losses)], losses,
                        color=COLORS['folds'][fold_idx], alpha=0.7,
                        label=f'Fold {fold_idx + 1}')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. Learning Rate (top-right)
        ax2 = fig.add_subplot(gs[0, 2])
        data = self._load_fold_data(0)
        lr_logs = data["learning_rate_logs"]
        if lr_logs:
            steps = [log["step"] for log in lr_logs]
            lrs = [log["learning_rate"] for log in lr_logs]
            ax2.plot(steps, lrs, color=COLORS['lr'], linewidth=2)
            ax2.fill_between(steps, lrs, alpha=0.3, color=COLORS['lr'])
        ax2.set_xlabel('Step')
        ax2.set_ylabel('LR')
        ax2.set_title('Learning Rate Schedule')
        ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        ax2.grid(True, alpha=0.3)
        
        # 3. WER Progress (middle-left)
        ax3 = fig.add_subplot(gs[1, 0])
        for fold_idx in range(self.num_folds):
            data = self._load_fold_data(fold_idx)
            eval_logs = data["evaluation_logs"]
            if eval_logs:
                steps = [log["step"] for log in eval_logs]
                wers = [log["wer"] * 100 for log in eval_logs]
                ax3.plot(steps, wers, color=COLORS['folds'][fold_idx], 
                        alpha=0.7, marker='o', markersize=2,
                        label=f'Fold {fold_idx + 1}')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('WER (%)')
        ax3.set_title('WER During Training')
        ax3.legend(loc='upper right', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. CER Progress (middle-center)
        ax4 = fig.add_subplot(gs[1, 1])
        for fold_idx in range(self.num_folds):
            data = self._load_fold_data(fold_idx)
            eval_logs = data["evaluation_logs"]
            if eval_logs:
                steps = [log["step"] for log in eval_logs]
                cers = [log["cer"] * 100 for log in eval_logs]
                ax4.plot(steps, cers, color=COLORS['folds'][fold_idx],
                        alpha=0.7, marker='o', markersize=2,
                        label=f'Fold {fold_idx + 1}')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('CER (%)')
        ax4.set_title('CER During Training')
        ax4.legend(loc='upper right', fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # 5. Gradient Norm (middle-right)
        ax5 = fig.add_subplot(gs[1, 2])
        for fold_idx in range(self.num_folds):
            data = self._load_fold_data(fold_idx)
            training_logs = data["training_logs"]
            if training_logs:
                steps = [log["step"] for log in training_logs if log.get("grad_norm")]
                grads = [log["grad_norm"] for log in training_logs if log.get("grad_norm")]
                if steps:
                    ax5.plot(steps, grads, color=COLORS['folds'][fold_idx],
                            alpha=0.5, linewidth=1)
        ax5.set_xlabel('Step')
        ax5.set_ylabel('Grad Norm')
        ax5.set_title('Gradient Norm')
        ax5.grid(True, alpha=0.3)
        
        # 6. Cross-Validation Results (bottom, spans all columns)
        ax6 = fig.add_subplot(gs[2, :])
        if self.cv_summary:
            fold_results = self.cv_summary.get("per_fold_results", [])
            cv_stats = self.cv_summary.get("cross_validation_summary", {})
            
            folds = [f"Fold {r['fold'] + 1}" for r in fold_results]
            wers = [r["wer_percent"] for r in fold_results]
            cers = [r["cer_percent"] for r in fold_results]
            
            x = np.arange(len(folds))
            width = 0.35
            
            bars1 = ax6.bar(x - width/2, wers, width, label='WER (%)', color=COLORS['wer'], alpha=0.7)
            bars2 = ax6.bar(x + width/2, cers, width, label='CER (%)', color=COLORS['cer'], alpha=0.7)
            
            # Add mean lines
            ax6.axhline(y=cv_stats.get("mean_wer_percent", 0), color=COLORS['wer'],
                       linestyle='--', linewidth=2, alpha=0.8)
            ax6.axhline(y=cv_stats.get("mean_cer_percent", 0), color=COLORS['cer'],
                       linestyle='--', linewidth=2, alpha=0.8)
            
            ax6.set_xlabel('Fold')
            ax6.set_ylabel('Error Rate (%)')
            ax6.set_title(f'Cross-Validation Results | Mean WER: {cv_stats.get("mean_wer_percent", 0):.2f}% ± {cv_stats.get("std_wer_percent", 0):.2f}% | Mean CER: {cv_stats.get("mean_cer_percent", 0):.2f}% ± {cv_stats.get("std_cer_percent", 0):.2f}%')
            ax6.set_xticks(x)
            ax6.set_xticklabels(folds)
            ax6.legend(loc='upper right')
            ax6.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, val in zip(bars1, wers):
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f'{val:.1f}', ha='center', fontsize=9)
            for bar, val in zip(bars2, cers):
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f'{val:.1f}', ha='center', fontsize=9)
        
        # Add title
        fig.suptitle('Whisper Fine-tuning Dashboard - Minangkabau Language', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save:
            path = self.output_dir / "comprehensive_dashboard.png"
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved: {path}")
        
        return fig
    
    def plot_single_fold_details(self, fold_idx: int = 0, save: bool = True) -> plt.Figure:
        """Create detailed visualization for a single fold with all data points visible."""
        data = self._load_fold_data(fold_idx)
        
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
        fig.suptitle(f'Fold {fold_idx + 1} Detailed Analysis', fontsize=18, fontweight='bold', y=0.98)
        
        training_logs = data["training_logs"]
        eval_logs = data["evaluation_logs"]
        lr_logs = data["learning_rate_logs"]
        
        # 1. Training Loss with all points
        ax1 = fig.add_subplot(gs[0, 0])
        if training_logs:
            steps = [log["step"] for log in training_logs]
            losses = [log["loss"] for log in training_logs if log["loss"] is not None]
            ax1.plot(steps[:len(losses)], losses, color=COLORS['train_loss'], 
                    linewidth=2, marker='o', markersize=4, markevery=max(1, len(steps)//40))
            
            # Annotate min/max
            if losses:
                min_idx = losses.index(min(losses))
                max_idx = losses.index(max(losses))
                ax1.annotate(f'Min: {losses[min_idx]:.3f}\nStep {steps[min_idx]}', 
                            xy=(steps[min_idx], losses[min_idx]),
                            xytext=(10, 20), textcoords='offset points',
                            fontsize=9, color='green', fontweight='bold',
                            arrowprops=dict(arrowstyle='->', color='green'))
            
            ax1.set_xlabel('Step', fontsize=12)
            ax1.set_ylabel('Loss', fontsize=12)
            ax1.set_title(f'Training Loss\nFinal: {losses[-1]:.4f}', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.4, linestyle='--')
            ax1.minorticks_on()
        
        # 2. Evaluation Loss with all points
        ax2 = fig.add_subplot(gs[0, 1])
        if eval_logs:
            steps = [log["step"] for log in eval_logs]
            eval_losses = [log["eval_loss"] for log in eval_logs]
            ax2.plot(steps, eval_losses, color=COLORS['eval_loss'], 
                    linewidth=2.5, marker='s', markersize=8)
            
            # Annotate each point
            for i, (s, l) in enumerate(zip(steps, eval_losses)):
                ax2.annotate(f'{l:.2f}', xy=(s, l),
                            xytext=(0, 8), textcoords='offset points',
                            fontsize=8, ha='center', color=COLORS['eval_loss'])
            
            ax2.set_xlabel('Step', fontsize=12)
            ax2.set_ylabel('Loss', fontsize=12)
            ax2.set_title(f'Evaluation Loss\nBest: {min(eval_losses):.4f}', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.4, linestyle='--')
        
        # 3. Learning Rate with markers
        ax3 = fig.add_subplot(gs[0, 2])
        if lr_logs:
            steps = [log["step"] for log in lr_logs]
            lrs = [log["learning_rate"] for log in lr_logs]
            ax3.plot(steps, lrs, color=COLORS['lr'], linewidth=2,
                    marker='o', markersize=3, markevery=max(1, len(steps)//30))
            ax3.fill_between(steps, lrs, alpha=0.3, color=COLORS['lr'])
            
            # Mark peak
            max_idx = lrs.index(max(lrs))
            ax3.annotate(f'Peak: {lrs[max_idx]:.2e}', 
                        xy=(steps[max_idx], lrs[max_idx]),
                        xytext=(20, -10), textcoords='offset points',
                        fontsize=10, color='red', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='red'))
            
            ax3.set_xlabel('Step', fontsize=12)
            ax3.set_ylabel('Learning Rate', fontsize=12)
            ax3.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
            ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            ax3.grid(True, alpha=0.4, linestyle='--')
        
        # 4. WER Progress with all points annotated
        ax4 = fig.add_subplot(gs[1, 0])
        if eval_logs:
            steps = [log["step"] for log in eval_logs]
            wers = [log["wer"] * 100 for log in eval_logs]
            ax4.plot(steps, wers, color=COLORS['wer'], linewidth=2.5, 
                    marker='D', markersize=10)
            
            # Annotate all points
            for s, w in zip(steps, wers):
                ax4.annotate(f'{w:.1f}%', xy=(s, w),
                            xytext=(0, 10), textcoords='offset points',
                            fontsize=9, ha='center', color=COLORS['wer'], fontweight='bold')
            
            ax4.set_xlabel('Step', fontsize=12)
            ax4.set_ylabel('WER (%)', fontsize=12)
            ax4.set_title(f'WER Progress\nStart: {wers[0]:.1f}% → Best: {min(wers):.1f}%', 
                         fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.4, linestyle='--')
            
            # Add improvement arrow
            improvement = wers[0] - min(wers)
            ax4.annotate(f'↓ {improvement:.1f}% improvement', 
                        xy=(0.5, 0.95), xycoords='axes fraction',
                        fontsize=11, color='green', fontweight='bold',
                        ha='center')
        
        # 5. CER Progress with all points annotated
        ax5 = fig.add_subplot(gs[1, 1])
        if eval_logs:
            steps = [log["step"] for log in eval_logs]
            cers = [log["cer"] * 100 for log in eval_logs]
            ax5.plot(steps, cers, color=COLORS['cer'], linewidth=2.5,
                    marker='D', markersize=10)
            
            # Annotate all points
            for s, c in zip(steps, cers):
                ax5.annotate(f'{c:.2f}%', xy=(s, c),
                            xytext=(0, 10), textcoords='offset points',
                            fontsize=9, ha='center', color=COLORS['cer'], fontweight='bold')
            
            ax5.set_xlabel('Step', fontsize=12)
            ax5.set_ylabel('CER (%)', fontsize=12)
            ax5.set_title(f'CER Progress\nStart: {cers[0]:.2f}% → Best: {min(cers):.2f}%', 
                         fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.4, linestyle='--')
            
            # Add improvement
            improvement = cers[0] - min(cers)
            ax5.annotate(f'↓ {improvement:.2f}% improvement', 
                        xy=(0.5, 0.95), xycoords='axes fraction',
                        fontsize=11, color='green', fontweight='bold',
                        ha='center')
        
        # 6. Gradient Norm
        ax6 = fig.add_subplot(gs[1, 2])
        if training_logs:
            steps = [log["step"] for log in training_logs if log.get("grad_norm")]
            grads = [log["grad_norm"] for log in training_logs if log.get("grad_norm")]
            if steps:
                ax6.plot(steps, grads, color=COLORS['grad_norm'], linewidth=1.5, 
                        marker='o', markersize=3, markevery=max(1, len(steps)//40), alpha=0.8)
                
                # Add statistics
                ax6.axhline(y=np.mean(grads), color='red', linestyle='--', 
                           linewidth=2, label=f'Mean: {np.mean(grads):.2f}')
                ax6.fill_between(steps, 
                                np.mean(grads) - np.std(grads), 
                                np.mean(grads) + np.std(grads),
                                alpha=0.2, color='red', label=f'±1 Std: {np.std(grads):.2f}')
                
                ax6.set_xlabel('Step', fontsize=12)
                ax6.set_ylabel('Gradient Norm', fontsize=12)
                ax6.set_title('Gradient Norm', fontsize=12, fontweight='bold')
                ax6.legend(loc='upper right', fontsize=9)
                ax6.grid(True, alpha=0.4, linestyle='--')
        
        # 7. Metrics Table (bottom row spans all)
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        # Create summary table
        if eval_logs:
            table_data = []
            headers = ['Eval #', 'Step', 'Epoch', 'Eval Loss', 'WER (%)', 'CER (%)']
            
            for i, log in enumerate(eval_logs):
                table_data.append([
                    i + 1,
                    log['step'],
                    f"{log['epoch']:.1f}",
                    f"{log['eval_loss']:.4f}",
                    f"{log['wer']*100:.2f}%",
                    f"{log['cer']*100:.2f}%"
                ])
            
            # Only show first/last if too many rows
            if len(table_data) > 15:
                table_display = table_data[:6] + [['...', '...', '...', '...', '...', '...']] + table_data[-6:]
            else:
                table_display = table_data
            
            table = ax7.table(cellText=table_display, colLabels=headers,
                             loc='center', cellLoc='center',
                             colColours=['#f0f0f0']*len(headers))
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.8)
            
            # Highlight best WER and CER rows
            wers = [log['wer'] for log in eval_logs]
            cers = [log['cer'] for log in eval_logs]
            best_wer_idx = wers.index(min(wers))
            best_cer_idx = cers.index(min(cers))
            
            ax7.set_title('Evaluation Metrics Table', fontsize=14, fontweight='bold', pad=20)
        
        if save:
            path = self.output_dir / f"fold_{fold_idx}_detailed_analysis.png"
            fig.savefig(path, dpi=200, bbox_inches='tight')
            print(f"✅ Saved: {path}")
        
        return fig
    
    def generate_all_plots(self):
        """Generate all available plots with detailed data points."""
        print("\n📊 Generating all detailed visualizations...\n")
        
        self.plot_training_loss()
        self.plot_evaluation_metrics()
        self.plot_learning_rate_schedule()
        self.plot_gradient_norm()
        self.plot_loss_comparison()
        self.plot_cross_validation_summary()
        self.plot_comprehensive_dashboard()
        
        # Generate detailed evaluation table
        self.plot_evaluation_table()
        
        # Generate per-fold details
        for fold_idx in range(self.num_folds):
            self.plot_single_fold_details(fold_idx)
        
        print(f"\n✅ All detailed visualizations saved to: {self.output_dir}")
    
    def plot_evaluation_table(self, save: bool = True) -> plt.Figure:
        """Create a detailed table showing all evaluation metrics for all folds."""
        fig, ax = plt.subplots(figsize=(20, 12))
        ax.axis('off')
        
        # Collect all data
        all_data = []
        for fold_idx in range(self.num_folds):
            data = self._load_fold_data(fold_idx)
            eval_logs = data["evaluation_logs"]
            
            for log in eval_logs:
                all_data.append({
                    'Fold': fold_idx + 1,
                    'Step': log['step'],
                    'Epoch': log['epoch'],
                    'Eval Loss': log['eval_loss'],
                    'WER (%)': log['wer'] * 100,
                    'CER (%)': log['cer'] * 100,
                })
        
        if not all_data:
            return fig
        
        # Create DataFrame for display
        df = pd.DataFrame(all_data)
        
        # Format numbers
        df['Eval Loss'] = df['Eval Loss'].apply(lambda x: f'{x:.4f}')
        df['WER (%)'] = df['WER (%)'].apply(lambda x: f'{x:.2f}')
        df['CER (%)'] = df['CER (%)'].apply(lambda x: f'{x:.2f}')
        df['Epoch'] = df['Epoch'].apply(lambda x: f'{x:.1f}')
        
        # Create table
        table = ax.table(cellText=df.values, colLabels=df.columns,
                        loc='center', cellLoc='center',
                        colColours=['#4a90d9']*len(df.columns))
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Style header
        for i in range(len(df.columns)):
            table[(0, i)].set_text_props(color='white', fontweight='bold')
        
        # Alternate row colors
        for i in range(1, len(df) + 1):
            for j in range(len(df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
                else:
                    table[(i, j)].set_facecolor('white')
        
        ax.set_title('Complete Evaluation Metrics Across All Folds', 
                    fontsize=16, fontweight='bold', pad=20)
        
        if save:
            path = self.output_dir / "evaluation_table_complete.png"
            fig.savefig(path, dpi=200, bbox_inches='tight')
            print(f"✅ Saved: {path}")
        
        return fig
    
    def print_summary(self):
        """Print a text summary of the results."""
        if not self.cv_summary:
            print("⚠️ No cross-validation summary available")
            return
        
        cv_stats = self.cv_summary.get("cross_validation_summary", {})
        config = self.cv_summary.get("training_config", {})
        
        print("\n" + "="*60)
        print("📊 TRAINING RESULTS SUMMARY")
        print("="*60)
        
        print(f"\n📈 Cross-Validation Results ({self.num_folds} Folds):")
        print("-"*40)
        print(f"  WER: {cv_stats.get('mean_wer_percent', 0):.2f}% ± {cv_stats.get('std_wer_percent', 0):.2f}%")
        print(f"  CER: {cv_stats.get('mean_cer_percent', 0):.2f}% ± {cv_stats.get('std_cer_percent', 0):.2f}%")
        print(f"  Best WER: {cv_stats.get('min_wer', 0)*100:.2f}%")
        print(f"  Best CER: {cv_stats.get('min_cer', 0)*100:.2f}%")
        
        print(f"\n⚙️ Training Configuration:")
        print("-"*40)
        print(f"  Learning Rate: {config.get('learning_rate', 'N/A')}")
        print(f"  Batch Size: {config.get('batch_size', 'N/A')}")
        print(f"  Max Steps: {config.get('max_steps', 'N/A')}")
        print(f"  Warmup Ratio: {config.get('warmup_ratio', 'N/A')}")
        print(f"  Weight Decay: {config.get('weight_decay', 'N/A')}")
        print(f"  LR Scheduler: {config.get('lr_scheduler_type', 'N/A')}")
        
        print("\n" + "="*60)


def visualize_run(run_dir: str, output_dir: str = None):
    """
    Convenience function to visualize a training run.
    
    Args:
        run_dir: Path to the run directory
        output_dir: Optional custom output directory
    """
    visualizer = MetricsVisualizer(run_dir, output_dir)
    visualizer.print_summary()
    visualizer.generate_all_plots()
    return visualizer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize Whisper fine-tuning metrics")
    parser.add_argument(
        "--run-dir", "-r",
        type=str,
        default="outputs/metrics/run_20260122_160012",
        help="Path to the run directory containing metrics"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Optional output directory for plots"
    )
    
    args = parser.parse_args()
    
    visualize_run(args.run_dir, args.output_dir)
