"""
Visualization script for Whisper fine-tuning training results.
Supports visualization from specific WandB runs or checkpoint trainer states.
Generates train_metrics.png, eval_metrics.png, and fold_comparison.png
"""

import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import glob
from datetime import datetime
import argparse


def load_trainer_state(checkpoint_dir: str) -> Dict:
    """Load trainer state from checkpoint."""
    state_file = Path(checkpoint_dir) / "trainer_state.json"
    if state_file.exists():
        with open(state_file, 'r') as f:
            return json.load(f)
    return {}


def load_wandb_summary(run_dir: str) -> Dict:
    """Load WandB summary from run directory."""
    summary_file = Path(run_dir) / "files" / "wandb-summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            return json.load(f)
    return {}


def find_all_trainer_states(base_dir: str = "outputs/checkpoints") -> List[Dict]:
    """Find all trainer states from all folds."""
    states = []
    for fold_dir in sorted(glob.glob(f"{base_dir}/fold_*")):
        checkpoints = sorted(glob.glob(f"{fold_dir}/checkpoint-*"))
        if checkpoints:
            state = load_trainer_state(checkpoints[-1])
            if state:
                state['fold'] = int(Path(fold_dir).name.split('_')[1])
                states.append(state)
    return states


def extract_training_logs(trainer_state: Dict) -> pd.DataFrame:
    """Extract training logs from trainer state."""
    log_history = trainer_state.get('log_history', [])
    if not log_history:
        return pd.DataFrame()
    return pd.DataFrame(log_history)


def create_run_folder(base_dir: str = "visualizations", run_name: str = None) -> Path:
    """Create a new folder for this visualization run."""
    if run_name:
        run_folder = Path(base_dir) / run_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = Path(base_dir) / f"run_{timestamp}"
    run_folder.mkdir(parents=True, exist_ok=True)
    return run_folder


def plot_train_metrics(trainer_states: List[Dict], save_dir: Path, run_name: str = ""):
    """Plot training metrics in 2x3 grid."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    colors = plt.cm.tab10(np.linspace(0, 1, len(trainer_states)))
    
    for i, state in enumerate(trainer_states):
        fold = state.get('fold', i)
        df = extract_training_logs(state)
        if df.empty:
            continue
        
        train_df = df[df['loss'].notna()]
        
        # train/epoch
        if 'epoch' in train_df.columns and 'step' in train_df.columns:
            axes[0, 0].plot(train_df['step'], train_df['epoch'], 
                          label=f'Fold {fold}', color=colors[i], marker='.', markersize=2, alpha=0.8)
        
        # train/global_step
        if 'step' in train_df.columns:
            axes[0, 1].plot(range(len(train_df)), train_df['step'], 
                          label=f'Fold {fold}', color=colors[i], marker='.', markersize=2, alpha=0.8)
        
        # train/grad_norm
        if 'grad_norm' in train_df.columns:
            axes[0, 2].plot(train_df['step'], train_df['grad_norm'], 
                          label=f'Fold {fold}', color=colors[i], marker='.', markersize=2, alpha=0.8)
        
        # train/learning_rate
        if 'learning_rate' in train_df.columns:
            axes[1, 0].plot(train_df['step'], train_df['learning_rate'] * 1e5, 
                          label=f'Fold {fold}', color=colors[i], marker='.', markersize=2, alpha=0.8)
        
        # train/loss
        if 'loss' in train_df.columns:
            axes[1, 1].plot(train_df['step'], train_df['loss'], 
                          label=f'Fold {fold}', color=colors[i], marker='.', markersize=2, alpha=0.8)
    
    # Configure axes
    axes[0, 0].set_title('train/epoch', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Epoch')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('train/global_step', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Index')
    axes[0, 1].set_ylabel('Step')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].set_title('train/grad_norm', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Step')
    axes[0, 2].set_ylabel('Gradient Norm')
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('train/learning_rate', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('LR (×10⁻⁵)')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('train/loss', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Learning Rate per Epoch in last subplot
    for i, state in enumerate(trainer_states):
        fold = state.get('fold', i)
        df = extract_training_logs(state)
        if df.empty or 'learning_rate' not in df.columns:
            continue
        train_df = df[df['loss'].notna()]
        if 'epoch' in train_df.columns and 'learning_rate' in train_df.columns:
            axes[1, 2].plot(train_df['epoch'], train_df['learning_rate'] * 1e5, 
                          label=f'Fold {fold}', color=colors[i], marker='.', markersize=2, alpha=0.8)
    
    axes[1, 2].set_title('train/learning_rate per Epoch', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('LR (×10⁻⁵)')
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(True, alpha=0.3)
    
    title = f'Training Metrics (5-Fold CV)'
    if run_name:
        title += f' - {run_name}'
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = save_dir / "train_metrics.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


def plot_eval_metrics(trainer_states: List[Dict], save_dir: Path, run_name: str = ""):
    """Plot evaluation metrics in 2x3 grid."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    colors = plt.cm.tab10(np.linspace(0, 1, len(trainer_states)))
    
    for i, state in enumerate(trainer_states):
        fold = state.get('fold', i)
        df = extract_training_logs(state)
        if df.empty:
            continue
        
        eval_df = df[df['eval_loss'].notna()]
        
        # eval/cer
        if 'eval_cer' in eval_df.columns:
            axes[0, 0].plot(eval_df['step'], eval_df['eval_cer'] * 100, 
                          label=f'Fold {fold}', color=colors[i], marker='o', markersize=4, alpha=0.8)
        
        # eval/loss
        if 'eval_loss' in eval_df.columns:
            axes[0, 1].plot(eval_df['step'], eval_df['eval_loss'], 
                          label=f'Fold {fold}', color=colors[i], marker='o', markersize=4, alpha=0.8)
        
        # eval/runtime
        if 'eval_runtime' in eval_df.columns:
            axes[0, 2].plot(eval_df['step'], eval_df['eval_runtime'], 
                          label=f'Fold {fold}', color=colors[i], marker='o', markersize=4, alpha=0.8)
        
        # eval/samples_per_second
        if 'eval_samples_per_second' in eval_df.columns:
            axes[1, 0].plot(eval_df['step'], eval_df['eval_samples_per_second'], 
                          label=f'Fold {fold}', color=colors[i], marker='o', markersize=4, alpha=0.8)
        
        # eval/steps_per_second
        if 'eval_steps_per_second' in eval_df.columns:
            axes[1, 1].plot(eval_df['step'], eval_df['eval_steps_per_second'], 
                          label=f'Fold {fold}', color=colors[i], marker='o', markersize=4, alpha=0.8)
        
        # eval/wer
        if 'eval_wer' in eval_df.columns:
            axes[1, 2].plot(eval_df['step'], eval_df['eval_wer'] * 100, 
                          label=f'Fold {fold}', color=colors[i], marker='o', markersize=4, alpha=0.8)
    
    # Configure axes
    axes[0, 0].set_title('eval/cer', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('CER (%)')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('eval/loss', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].set_title('eval/runtime', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Step')
    axes[0, 2].set_ylabel('Runtime (s)')
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('eval/samples_per_second', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Samples/s')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('eval/steps_per_second', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Steps/s')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].set_title('eval/wer', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Step')
    axes[1, 2].set_ylabel('WER (%)')
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(True, alpha=0.3)
    
    title = f'Evaluation Metrics (5-Fold CV)'
    if run_name:
        title += f' - {run_name}'
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = save_dir / "eval_metrics.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


def plot_fold_comparison(trainer_states: List[Dict], save_dir: Path, run_name: str = ""):
    """Plot final WER and CER comparison across folds."""
    folds, wer_scores, cer_scores = [], [], []
    
    for state in trainer_states:
        df = extract_training_logs(state)
        if df.empty:
            continue
        eval_df = df[df['eval_wer'].notna()]
        if not eval_df.empty:
            folds.append(f"Fold {state.get('fold', len(folds))}")
            wer_scores.append(eval_df['eval_wer'].iloc[-1] * 100)
            cer_scores.append(eval_df['eval_cer'].iloc[-1] * 100)
    
    if not folds:
        print("No evaluation data found for fold comparison")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(folds))
    width = 0.6
    
    bars1 = axes[0].bar(x, wer_scores, width, color='steelblue', edgecolor='black')
    axes[0].axhline(y=np.mean(wer_scores), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(wer_scores):.2f}% ± {np.std(wer_scores):.2f}%')
    axes[0].set_ylabel('WER (%)', fontsize=12)
    axes[0].set_title('Word Error Rate by Fold', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(folds)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars1, wer_scores):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                    f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    bars2 = axes[1].bar(x, cer_scores, width, color='coral', edgecolor='black')
    axes[1].axhline(y=np.mean(cer_scores), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(cer_scores):.2f}% ± {np.std(cer_scores):.2f}%')
    axes[1].set_ylabel('CER (%)', fontsize=12)
    axes[1].set_title('Character Error Rate by Fold', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(folds)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars2, cer_scores):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    title = f'5-Fold Cross-Validation Results'
    if run_name:
        title += f' - {run_name}'
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_path = save_dir / "fold_comparison.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


def plot_validation_summary(trainer_states: List[Dict], save_dir: Path, run_name: str = ""):
    """Plot validation/image summary showing WER/CER progression."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(trainer_states)))
    
    for i, state in enumerate(trainer_states):
        fold = state.get('fold', i)
        df = extract_training_logs(state)
        if df.empty:
            continue
        
        eval_df = df[df['eval_wer'].notna()]
        if eval_df.empty:
            continue
        
        # WER over epochs
        if 'epoch' in eval_df.columns:
            axes[0].plot(eval_df['epoch'], eval_df['eval_wer'] * 100, 
                        label=f'Fold {fold}', color=colors[i], marker='o', markersize=5, linewidth=2)
        
        # CER over epochs
        if 'epoch' in eval_df.columns:
            axes[1].plot(eval_df['epoch'], eval_df['eval_cer'] * 100, 
                        label=f'Fold {fold}', color=colors[i], marker='o', markersize=5, linewidth=2)
    
    axes[0].set_title('WER Validation per Epoch', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('WER (%)', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title('CER Validation per Epoch', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('CER (%)', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    title = 'Validation Metrics per Epoch'
    if run_name:
        title += f' - {run_name}'
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_path = save_dir / "validation_summary.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


def plot_lr_per_epoch(trainer_states: List[Dict], save_dir: Path, run_name: str = ""):
    """Plot learning rate per epoch."""
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(trainer_states)))
    
    for i, state in enumerate(trainer_states):
        fold = state.get('fold', i)
        df = extract_training_logs(state)
        if df.empty or 'learning_rate' not in df.columns:
            continue
        
        train_df = df[df['loss'].notna()]
        if 'epoch' in train_df.columns:
            ax.plot(train_df['epoch'], train_df['learning_rate'] * 1e5, 
                   label=f'Fold {fold}', color=colors[i], marker='.', markersize=3, linewidth=1.5, alpha=0.8)
    
    ax.set_title('Learning Rate Schedule per Epoch', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate (×10⁻⁵)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = save_dir / "learning_rate_per_epoch.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


def save_summary(trainer_states: List[Dict], save_dir: Path) -> Dict:
    """Save and return summary statistics."""
    wer_scores, cer_scores = [], []
    summary_lines = []
    
    summary_lines.append("=" * 60)
    summary_lines.append("TRAINING SUMMARY")
    summary_lines.append("=" * 60)
    
    for state in trainer_states:
        df = extract_training_logs(state)
        if df.empty:
            continue
        
        fold = state.get('fold', len(wer_scores))
        eval_df = df[df['eval_wer'].notna()]
        
        if not eval_df.empty:
            final_wer = eval_df['eval_wer'].iloc[-1] * 100
            final_cer = eval_df['eval_cer'].iloc[-1] * 100
            best_wer = eval_df['eval_wer'].min() * 100
            best_cer = eval_df['eval_cer'].min() * 100
            
            wer_scores.append(final_wer)
            cer_scores.append(final_cer)
            
            summary_lines.append(f"\nFold {fold}:")
            summary_lines.append(f"  Final WER: {final_wer:.2f}% | Best WER: {best_wer:.2f}%")
            summary_lines.append(f"  Final CER: {final_cer:.2f}% | Best CER: {best_cer:.2f}%")
    
    if wer_scores:
        summary_lines.append("\n" + "-" * 60)
        summary_lines.append("OVERALL RESULTS (5-Fold CV)")
        summary_lines.append("-" * 60)
        summary_lines.append(f"WER: {np.mean(wer_scores):.2f}% ± {np.std(wer_scores):.2f}%")
        summary_lines.append(f"CER: {np.mean(cer_scores):.2f}% ± {np.std(cer_scores):.2f}%")
        summary_lines.append("=" * 60)
    
    for line in summary_lines:
        print(line)
    
    summary_path = save_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("\n".join(summary_lines))
    print(f"\n✅ Saved: {summary_path}")
    
    return {"wer_mean": np.mean(wer_scores) if wer_scores else 0, 
            "wer_std": np.std(wer_scores) if wer_scores else 0,
            "cer_mean": np.mean(cer_scores) if cer_scores else 0, 
            "cer_std": np.std(cer_scores) if cer_scores else 0}


def save_run_config(save_dir: Path, trainer_states: List[Dict], summary: Dict, run_name: str = ""):
    """Save run configuration as JSON."""
    config = {
        "timestamp": datetime.now().isoformat(), 
        "run_name": run_name,
        "num_folds": len(trainer_states), 
        "results": summary
    }
    with open(save_dir / "run_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ Saved: {save_dir / 'run_config.json'}")


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description='Visualize Whisper fine-tuning results')
    parser.add_argument('--name', type=str, default='', help='Run name for output folder')
    parser.add_argument('--checkpoint-dir', type=str, default='outputs/checkpoints', help='Checkpoint directory')
    args = parser.parse_args()
    
    print("=" * 60)
    print("WHISPER FINE-TUNING VISUALIZATION")
    print("=" * 60)
    
    trainer_states = find_all_trainer_states(args.checkpoint_dir)
    
    if not trainer_states:
        print(f"❌ No training data found in {args.checkpoint_dir}/")
        return
    
    print(f"Found {len(trainer_states)} fold(s)")
    
    run_folder = create_run_folder(run_name=args.name if args.name else None)
    print(f"\n📁 Saving visualizations to: {run_folder}")
    
    summary = save_summary(trainer_states, run_folder)
    
    print("\nGenerating visualizations...")
    plot_train_metrics(trainer_states, run_folder, args.name)
    plot_eval_metrics(trainer_states, run_folder, args.name)
    plot_fold_comparison(trainer_states, run_folder, args.name)
    plot_validation_summary(trainer_states, run_folder, args.name)
    plot_lr_per_epoch(trainer_states, run_folder, args.name)
    
    save_run_config(run_folder, trainer_states, summary, args.name)
    
    print(f"\n✅ Visualization complete! All files saved to: {run_folder}")


if __name__ == "__main__":
    main()
