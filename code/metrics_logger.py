"""
Comprehensive Metrics Logger for Whisper Fine-tuning.
Saves all training metrics both locally (JSON/CSV) and to WandB.
Includes: learning rate, loss, WER, CER, sample predictions, and more.
"""

import os
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import wandb
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments


class MetricsLogger:
    """
    Centralized metrics logger that saves to both local files and WandB.
    """
    
    def __init__(
        self, 
        output_dir: str,
        fold_idx: int = 0,
        experiment_name: str = None,
        save_locally: bool = True,
        log_to_wandb: bool = True
    ):
        self.fold_idx = fold_idx
        self.save_locally = save_locally
        self.log_to_wandb = log_to_wandb
        
        # Create timestamped experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name or f"run_{timestamp}"
        
        # Setup directories
        self.output_dir = Path(output_dir)
        self.metrics_dir = self.output_dir / "metrics" / self.experiment_name / f"fold_{fold_idx}"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage for different metric types
        self.training_logs: List[Dict[str, Any]] = []
        self.evaluation_logs: List[Dict[str, Any]] = []
        self.learning_rate_logs: List[Dict[str, float]] = []
        self.sample_predictions: List[Dict[str, Any]] = []
        self.epoch_summaries: List[Dict[str, Any]] = []
        
        # File paths
        self.training_log_path = self.metrics_dir / "training_logs.json"
        self.evaluation_log_path = self.metrics_dir / "evaluation_logs.json"
        self.lr_log_path = self.metrics_dir / "learning_rate_logs.json"
        self.predictions_path = self.metrics_dir / "sample_predictions.json"
        self.epoch_summary_path = self.metrics_dir / "epoch_summaries.json"
        self.training_csv_path = self.metrics_dir / "training_metrics.csv"
        self.evaluation_csv_path = self.metrics_dir / "evaluation_metrics.csv"
        
        # CSV headers
        self._init_csv_files()
        
        print(f"📊 Metrics Logger initialized")
        print(f"   Local save directory: {self.metrics_dir}")
        print(f"   Save locally: {self.save_locally}")
        print(f"   Log to WandB: {self.log_to_wandb}")
    
    def _init_csv_files(self):
        """Initialize CSV files with headers."""
        if self.save_locally:
            # Training CSV
            with open(self.training_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'step', 'epoch', 'loss', 'learning_rate',
                    'grad_norm', 'samples_per_second', 'steps_per_second'
                ])
            
            # Evaluation CSV
            with open(self.evaluation_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'step', 'epoch', 'eval_loss', 'wer', 'cer',
                    'eval_samples_per_second', 'eval_steps_per_second', 'eval_runtime'
                ])
    
    def log_training_step(
        self,
        step: int,
        epoch: float,
        loss: float,
        learning_rate: float,
        grad_norm: float = None,
        samples_per_second: float = None,
        steps_per_second: float = None,
        **kwargs
    ):
        """Log metrics for a single training step."""
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            "timestamp": timestamp,
            "step": step,
            "epoch": round(epoch, 4),
            "loss": round(loss, 6) if loss else None,
            "learning_rate": learning_rate,
            "grad_norm": round(grad_norm, 6) if grad_norm else None,
            "samples_per_second": round(samples_per_second, 2) if samples_per_second else None,
            "steps_per_second": round(steps_per_second, 4) if steps_per_second else None,
            **kwargs
        }
        
        self.training_logs.append(log_entry)
        self.learning_rate_logs.append({
            "step": step,
            "epoch": round(epoch, 4),
            "learning_rate": learning_rate
        })
        
        # Save to local files
        if self.save_locally:
            self._append_to_csv(self.training_csv_path, [
                timestamp, step, round(epoch, 4), loss, learning_rate,
                grad_norm, samples_per_second, steps_per_second
            ])
            self._save_json(self.training_log_path, self.training_logs)
            self._save_json(self.lr_log_path, self.learning_rate_logs)
        
        # NOTE: We don't log to WandB here because HuggingFace Trainer already does it
        # This avoids step conflicts and duplicate logging
    
    def log_evaluation(
        self,
        step: int,
        epoch: float,
        eval_loss: float,
        wer: float,
        cer: float,
        eval_runtime: float = None,
        eval_samples_per_second: float = None,
        eval_steps_per_second: float = None,
        **kwargs
    ):
        """Log evaluation metrics."""
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            "timestamp": timestamp,
            "step": step,
            "epoch": round(epoch, 4),
            "eval_loss": round(eval_loss, 6) if eval_loss else None,
            "wer": round(wer, 6),
            "cer": round(cer, 6),
            "wer_percent": round(wer * 100, 2),
            "cer_percent": round(cer * 100, 2),
            "eval_runtime": round(eval_runtime, 2) if eval_runtime else None,
            "eval_samples_per_second": round(eval_samples_per_second, 2) if eval_samples_per_second else None,
            "eval_steps_per_second": round(eval_steps_per_second, 4) if eval_steps_per_second else None,
            **kwargs
        }
        
        self.evaluation_logs.append(log_entry)
        
        # Save to local files
        if self.save_locally:
            self._append_to_csv(self.evaluation_csv_path, [
                timestamp, step, round(epoch, 4), eval_loss, wer, cer,
                eval_samples_per_second, eval_steps_per_second, eval_runtime
            ])
            self._save_json(self.evaluation_log_path, self.evaluation_logs)
        
        # NOTE: We don't log to WandB here because HuggingFace Trainer already does it
        # This avoids step conflicts and duplicate logging
    
    def log_sample_predictions(
        self,
        step: int,
        epoch: float,
        predictions: List[str],
        references: List[str],
        audio_ids: List[str] = None,
        wer_per_sample: List[float] = None,
        cer_per_sample: List[float] = None
    ):
        """Log sample predictions for qualitative analysis."""
        timestamp = datetime.now().isoformat()
        
        samples = []
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            sample = {
                "index": i,
                "prediction": pred,
                "reference": ref,
                "audio_id": audio_ids[i] if audio_ids else None,
                "wer": wer_per_sample[i] if wer_per_sample else None,
                "cer": cer_per_sample[i] if cer_per_sample else None,
            }
            samples.append(sample)
        
        prediction_entry = {
            "timestamp": timestamp,
            "step": step,
            "epoch": round(epoch, 4),
            "samples": samples
        }
        
        self.sample_predictions.append(prediction_entry)
        
        # Save locally
        if self.save_locally:
            self._save_json(self.predictions_path, self.sample_predictions)
        
        # NOTE: Prediction tables can still be logged to WandB without step conflicts
    
    def log_epoch_summary(
        self,
        epoch: int,
        train_loss_avg: float,
        eval_loss: float,
        wer: float,
        cer: float,
        learning_rate_start: float,
        learning_rate_end: float,
        best_wer_so_far: float,
        best_cer_so_far: float,
        **kwargs
    ):
        """Log epoch-level summary."""
        timestamp = datetime.now().isoformat()
        
        summary = {
            "timestamp": timestamp,
            "epoch": epoch,
            "train_loss_avg": round(train_loss_avg, 6) if train_loss_avg else None,
            "eval_loss": round(eval_loss, 6) if eval_loss else None,
            "wer": round(wer, 6),
            "cer": round(cer, 6),
            "wer_percent": round(wer * 100, 2),
            "cer_percent": round(cer * 100, 2),
            "learning_rate_start": learning_rate_start,
            "learning_rate_end": learning_rate_end,
            "best_wer_so_far": round(best_wer_so_far, 6),
            "best_cer_so_far": round(best_cer_so_far, 6),
            **kwargs
        }
        
        self.epoch_summaries.append(summary)
        
        # Save locally
        if self.save_locally:
            self._save_json(self.epoch_summary_path, self.epoch_summaries)
        
        # NOTE: Epoch summary logging to WandB is handled by HF Trainer
    
    def log_model_config(self, config: Dict[str, Any]):
        """Log model configuration."""
        config_path = self.metrics_dir / "model_config.json"
        
        if self.save_locally:
            self._save_json(config_path, config)
        
        if self.log_to_wandb and wandb.run is not None:
            wandb.config.update(config, allow_val_change=True)
    
    def log_training_config(self, config: Dict[str, Any]):
        """Log training configuration."""
        config_path = self.metrics_dir / "training_config.json"
        
        if self.save_locally:
            self._save_json(config_path, config)
        
        if self.log_to_wandb and wandb.run is not None:
            wandb.config.update(config, allow_val_change=True)
    
    def log_dataset_info(
        self,
        train_size: int,
        eval_size: int,
        total_audio_duration_hours: float = None,
        **kwargs
    ):
        """Log dataset information."""
        info = {
            "train_size": train_size,
            "eval_size": eval_size,
            "total_audio_duration_hours": total_audio_duration_hours,
            **kwargs
        }
        
        info_path = self.metrics_dir / "dataset_info.json"
        
        if self.save_locally:
            self._save_json(info_path, info)
        
        # Only update config (not log) to avoid step conflicts
        if self.log_to_wandb and wandb.run is not None:
            wandb.config.update(info, allow_val_change=True)
    
    def log_final_results(
        self,
        final_wer: float,
        final_cer: float,
        best_wer: float,
        best_cer: float,
        total_training_time: float,
        total_steps: int,
        **kwargs
    ):
        """Log final training results."""
        results = {
            "final_wer": round(final_wer, 6),
            "final_cer": round(final_cer, 6),
            "best_wer": round(best_wer, 6),
            "best_cer": round(best_cer, 6),
            "final_wer_percent": round(final_wer * 100, 2),
            "final_cer_percent": round(final_cer * 100, 2),
            "best_wer_percent": round(best_wer * 100, 2),
            "best_cer_percent": round(best_cer * 100, 2),
            "total_training_time_seconds": round(total_training_time, 2),
            "total_training_time_minutes": round(total_training_time / 60, 2),
            "total_steps": total_steps,
            **kwargs
        }
        
        results_path = self.metrics_dir / "final_results.json"
        
        if self.save_locally:
            self._save_json(results_path, results)
        
        # Only update summary (not log) to avoid step conflicts
        if self.log_to_wandb and wandb.run is not None:
            wandb.summary.update(results)
    
    def create_summary_report(self) -> Dict[str, Any]:
        """Create a comprehensive summary report."""
        report = {
            "experiment_name": self.experiment_name,
            "fold_idx": self.fold_idx,
            "total_training_steps": len(self.training_logs),
            "total_evaluations": len(self.evaluation_logs),
        }
        
        if self.training_logs:
            losses = [l["loss"] for l in self.training_logs if l["loss"] is not None]
            lrs = [l["learning_rate"] for l in self.training_logs if l["learning_rate"] is not None]
            report["training_summary"] = {
                "initial_loss": losses[0] if losses else None,
                "final_loss": losses[-1] if losses else None,
                "min_loss": min(losses) if losses else None,
                "max_loss": max(losses) if losses else None,
                "avg_loss": np.mean(losses) if losses else None,
                "initial_lr": lrs[0] if lrs else None,
                "final_lr": lrs[-1] if lrs else None,
                "min_lr": min(lrs) if lrs else None,
                "max_lr": max(lrs) if lrs else None,
            }
        
        if self.evaluation_logs:
            wers = [e["wer"] for e in self.evaluation_logs]
            cers = [e["cer"] for e in self.evaluation_logs]
            report["evaluation_summary"] = {
                "initial_wer": wers[0],
                "final_wer": wers[-1],
                "best_wer": min(wers),
                "worst_wer": max(wers),
                "initial_cer": cers[0],
                "final_cer": cers[-1],
                "best_cer": min(cers),
                "worst_cer": max(cers),
            }
        
        summary_path = self.metrics_dir / "summary_report.json"
        if self.save_locally:
            self._save_json(summary_path, report)
        
        return report
    
    def _save_json(self, path: Path, data: Any):
        """Save data to JSON file."""
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _append_to_csv(self, path: Path, row: List[Any]):
        """Append a row to CSV file."""
        with open(path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)


class ComprehensiveMetricsCallback(TrainerCallback):
    """
    Custom callback that logs all metrics during training.
    Integrates with MetricsLogger for dual local + WandB logging.
    """
    
    def __init__(
        self, 
        metrics_logger: MetricsLogger,
        processor = None,
        log_predictions_every_n_evals: int = 1,
        num_prediction_samples: int = 5
    ):
        self.metrics_logger = metrics_logger
        self.processor = processor
        self.log_predictions_every_n_evals = log_predictions_every_n_evals
        self.num_prediction_samples = num_prediction_samples
        
        # Track state
        self.current_epoch_losses: List[float] = []
        self.best_wer = float('inf')
        self.best_cer = float('inf')
        self.epoch_start_lr = None
        self.eval_count = 0
        self.training_start_time = None
    
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the beginning of training."""
        import time
        self.training_start_time = time.time()
        
        # Log training configuration
        training_config = {
            "learning_rate": args.learning_rate,
            "batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
            "max_steps": args.max_steps,
            "num_train_epochs": args.num_train_epochs,
            "warmup_ratio": args.warmup_ratio,
            "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay,
            "lr_scheduler_type": str(args.lr_scheduler_type),
            "eval_steps": args.eval_steps,
            "save_steps": args.save_steps,
            "logging_steps": args.logging_steps,
            "bf16": args.bf16,
            "fp16": args.fp16,
        }
        self.metrics_logger.log_training_config(training_config)
        
        print(f"🚀 Training started - logging all metrics locally and to WandB")
    
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Dict[str, float] = None, **kwargs):
        """Called when logging metrics."""
        if logs is None:
            return
        
        step = state.global_step
        epoch = state.epoch or 0
        
        # Get learning rate from logs or optimizer
        learning_rate = logs.get("learning_rate", 0)
        
        # Track epoch start LR
        if self.epoch_start_lr is None:
            self.epoch_start_lr = learning_rate
        
        # Log training step metrics
        if "loss" in logs:
            loss = logs.get("loss")
            self.current_epoch_losses.append(loss)
            
            self.metrics_logger.log_training_step(
                step=step,
                epoch=epoch,
                loss=loss,
                learning_rate=learning_rate,
                grad_norm=logs.get("grad_norm"),
                samples_per_second=logs.get("train_samples_per_second"),
                steps_per_second=logs.get("train_steps_per_second"),
            )
        
        # Log evaluation metrics
        if "eval_loss" in logs:
            wer = logs.get("eval_wer", 0)
            cer = logs.get("eval_cer", 0)
            
            # Update best metrics
            if wer < self.best_wer:
                self.best_wer = wer
            if cer < self.best_cer:
                self.best_cer = cer
            
            self.metrics_logger.log_evaluation(
                step=step,
                epoch=epoch,
                eval_loss=logs.get("eval_loss"),
                wer=wer,
                cer=cer,
                eval_runtime=logs.get("eval_runtime"),
                eval_samples_per_second=logs.get("eval_samples_per_second"),
                eval_steps_per_second=logs.get("eval_steps_per_second"),
            )
            
            self.eval_count += 1
    
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: Dict[str, float] = None, **kwargs):
        """Called after evaluation."""
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: Dict[str, float] = None, **kwargs):
        """Called after evaluation."""
        # NOTE: Charts logging removed to avoid step conflicts
        # HuggingFace Trainer handles all WandB logging properly
        pass
    
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of each epoch."""
        epoch = int(state.epoch) if state.epoch else 0
        
        # Calculate epoch summary
        if self.current_epoch_losses:
            avg_loss = np.mean(self.current_epoch_losses)
        else:
            avg_loss = None
        
        # Get latest evaluation metrics
        if self.metrics_logger.evaluation_logs:
            latest_eval = self.metrics_logger.evaluation_logs[-1]
            eval_loss = latest_eval.get("eval_loss")
            wer = latest_eval.get("wer", 0)
            cer = latest_eval.get("cer", 0)
        else:
            eval_loss = None
            wer = 0
            cer = 0
        
        # Get current learning rate
        current_lr = 0
        if self.metrics_logger.learning_rate_logs:
            current_lr = self.metrics_logger.learning_rate_logs[-1]["learning_rate"]
        
        self.metrics_logger.log_epoch_summary(
            epoch=epoch,
            train_loss_avg=avg_loss,
            eval_loss=eval_loss,
            wer=wer,
            cer=cer,
            learning_rate_start=self.epoch_start_lr or 0,
            learning_rate_end=current_lr,
            best_wer_so_far=self.best_wer,
            best_cer_so_far=self.best_cer,
        )
        
        # Reset for next epoch
        self.current_epoch_losses = []
        self.epoch_start_lr = current_lr
        
        avg_loss_str = f"{avg_loss:.4f}" if avg_loss else "N/A"
        print(f"Epoch {epoch} Summary: Avg Loss={avg_loss_str}, WER={wer:.4f}, CER={cer:.4f}, Best WER={self.best_wer:.4f}")
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of training."""
        import time
        total_time = time.time() - self.training_start_time if self.training_start_time else 0
        
        # Get final metrics
        final_wer = self.metrics_logger.evaluation_logs[-1]["wer"] if self.metrics_logger.evaluation_logs else 0
        final_cer = self.metrics_logger.evaluation_logs[-1]["cer"] if self.metrics_logger.evaluation_logs else 0
        
        self.metrics_logger.log_final_results(
            final_wer=final_wer,
            final_cer=final_cer,
            best_wer=self.best_wer,
            best_cer=self.best_cer,
            total_training_time=total_time,
            total_steps=state.global_step,
        )
        
        # Create summary report
        report = self.metrics_logger.create_summary_report()
        
        print(f"\n✅ Training complete!")
        print(f"   Total time: {total_time/60:.2f} minutes")
        print(f"   Total steps: {state.global_step}")
        print(f"   Best WER: {self.best_wer:.4f} ({self.best_wer*100:.2f}%)")
        print(f"   Best CER: {self.best_cer:.4f} ({self.best_cer*100:.2f}%)")
        print(f"   Metrics saved to: {self.metrics_logger.metrics_dir}")


class PredictionLoggingCallback(TrainerCallback):
    """
    Callback to log sample predictions during evaluation.
    """
    
    def __init__(
        self,
        metrics_logger: MetricsLogger,
        processor,
        eval_dataset,
        num_samples: int = 5,
        log_every_n_evals: int = 1
    ):
        self.metrics_logger = metrics_logger
        self.processor = processor
        self.eval_dataset = eval_dataset
        self.num_samples = num_samples
        self.log_every_n_evals = log_every_n_evals
        self.eval_count = 0
    
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        """Log sample predictions after evaluation."""
        self.eval_count += 1
        
        if self.eval_count % self.log_every_n_evals != 0:
            return
        
        if model is None:
            return
        
        try:
            import evaluate
            wer_metric = evaluate.load("wer")
            cer_metric = evaluate.load("cer")
            
            # Get sample predictions
            predictions = []
            references = []
            wer_per_sample = []
            cer_per_sample = []
            
            model.eval()
            device = next(model.parameters()).device
            
            # Sample from eval dataset
            indices = list(range(min(self.num_samples, len(self.eval_dataset))))
            
            for idx in indices:
                sample = self.eval_dataset[idx]
                
                # Get input features
                input_features = torch.tensor(sample["input_features"]).unsqueeze(0).to(device)
                
                # Create attention mask (1 for non-padded, assuming all valid input)
                attention_mask = torch.ones(input_features.shape[:2], dtype=torch.long, device=device)
                
                # Generate prediction with attention mask
                with torch.no_grad():
                    generated_ids = model.generate(
                        input_features,
                        attention_mask=attention_mask,
                    )
                
                # Decode
                pred_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                # Get reference
                labels = sample["labels"]
                labels = [l if l != -100 else self.processor.tokenizer.pad_token_id for l in labels]
                ref_text = self.processor.tokenizer.decode(labels, skip_special_tokens=True)
                
                predictions.append(pred_text)
                references.append(ref_text)
                
                # Calculate per-sample metrics
                if pred_text and ref_text:
                    sample_wer = wer_metric.compute(predictions=[pred_text], references=[ref_text])
                    sample_cer = cer_metric.compute(predictions=[pred_text], references=[ref_text])
                else:
                    sample_wer = 1.0
                    sample_cer = 1.0
                
                wer_per_sample.append(sample_wer)
                cer_per_sample.append(sample_cer)
            
            # Log predictions
            self.metrics_logger.log_sample_predictions(
                step=state.global_step,
                epoch=state.epoch or 0,
                predictions=predictions,
                references=references,
                wer_per_sample=wer_per_sample,
                cer_per_sample=cer_per_sample
            )
            
        except Exception as e:
            print(f"Warning: Could not log predictions: {e}")


def create_metrics_logger(
    output_dir: str,
    fold_idx: int = 0,
    experiment_name: str = None,
    save_locally: bool = True,
    log_to_wandb: bool = True
) -> MetricsLogger:
    """
    Factory function to create a MetricsLogger instance.
    """
    return MetricsLogger(
        output_dir=output_dir,
        fold_idx=fold_idx,
        experiment_name=experiment_name,
        save_locally=save_locally,
        log_to_wandb=log_to_wandb
    )


def create_metrics_callbacks(
    metrics_logger: MetricsLogger,
    processor = None,
    eval_dataset = None,
    log_predictions: bool = True,
    num_prediction_samples: int = 5
) -> List[TrainerCallback]:
    """
    Create all metrics-related callbacks.
    
    Returns:
        List of callbacks to add to trainer
    """
    callbacks = [
        ComprehensiveMetricsCallback(
            metrics_logger=metrics_logger,
            processor=processor,
        )
    ]
    
    if log_predictions and processor is not None and eval_dataset is not None:
        callbacks.append(
            PredictionLoggingCallback(
                metrics_logger=metrics_logger,
                processor=processor,
                eval_dataset=eval_dataset,
                num_samples=num_prediction_samples,
            )
        )
    
    return callbacks
