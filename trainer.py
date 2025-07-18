import os
import gc
import wandb
import numpy as np
import torch
from typing import Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass

try:
    from transformers import (
        AutoModelForAudioClassification,
        WhisperForAudioClassification,
        Wav2Vec2ForSequenceClassification,  # Try this first
        HubertForSequenceClassification,    # Try this first
        TrainingArguments,
        Trainer,
        TrainerCallback,
        EarlyStoppingCallback
    )
except ImportError:
    # Fallback if sequence classification models are not available
    from transformers import (
        AutoModelForAudioClassification,
        WhisperForAudioClassification,
        TrainingArguments,
        Trainer,
        TrainerCallback,
        EarlyStoppingCallback
    )
    # Set placeholders for unavailable models
    Wav2Vec2ForSequenceClassification = None
    HubertForSequenceClassification = None

from transformers.trainer_utils import get_last_checkpoint
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import evaluate


@dataclass
class FinetuningConfig:
    """Configuration for finetuning parameters"""
    model_name: str = "audio-classifier"
    num_train_epochs: int = 10
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    fp16: bool = True
    dataloader_num_workers: int = 4
    save_total_limit: int = 2
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 500
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_accuracy"
    greater_is_better: bool = True
    freeze_ratio: float = 0.5  # Fraction of layers to freeze
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_private_repo: bool = False
    use_wandb: bool = True
    wandb_project: str = "audio-classification"
    wandb_entity: Optional[str] = None
    seed: int = 42
    
    

class WandbCallback(TrainerCallback):
    """Custom callback for enhanced W&B logging"""
    
    def __init__(self, config: FinetuningConfig):
        self.config = config
        
    def on_train_begin(self, args, state, control, **kwargs):
        if self.config.use_wandb and not wandb.run:
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                config=self.config.__dict__,
                name=self.config.model_name,
                tags=["audio-classification", "finetuning"]
            )
    
    def on_evaluate(self, args, state, control, **kwargs):
        if self.config.use_wandb and wandb.run:
            # Log model architecture info
            model = kwargs.get('model')
            if model and state.epoch == 1:  # Log only once
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                wandb.log({
                    "model_info/total_parameters": total_params,
                    "model_info/trainable_parameters": trainable_params,
                    "model_info/trainable_percentage": (trainable_params / total_params) * 100,
                    "model_info/frozen_percentage": ((total_params - trainable_params) / total_params) * 100
                })

class DetailedMetricsCallback(TrainerCallback):
    """Callback for detailed metrics calculation"""
    
    def __init__(self, trainer, id2label: Dict[int, str], config: FinetuningConfig):
        self._trainer = trainer
        self.id2label = id2label
        self.config = config
        self.best_metrics = {}
        
    def on_evaluate(self, args, state, control, **kwargs):
        if control.should_evaluate:
            # Get predictions for detailed metrics
            eval_dataset = self._trainer.eval_dataset
            predictions = self._trainer.predict(eval_dataset)
            
            y_true = predictions.label_ids
            y_pred = np.argmax(predictions.predictions, axis=1)
            
            # Calculate detailed metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='weighted'
            )
            
            # Per-class metrics
            precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
                y_true, y_pred, average=None
            )
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Calculate specificity and NPV for each class
            specificity_per_class = []
            npv_per_class = []
            
            for i in range(len(self.id2label)):
                TP = cm[i, i]
                FP = cm[:, i].sum() - TP
                FN = cm[i, :].sum() - TP
                TN = cm.sum() - (TP + FP + FN)
                
                spec = TN / (TN + FP) if (TN + FP) != 0 else 0
                npv = TN / (TN + FN) if (TN + FN) != 0 else 0
                
                specificity_per_class.append(spec)
                npv_per_class.append(npv)
            
            # Log metrics
            metrics = {
                "eval_precision_weighted": precision,
                "eval_recall_weighted": recall,
                "eval_f1_weighted": f1,
                "eval_specificity_weighted": np.mean(specificity_per_class),
                "eval_npv_weighted": np.mean(npv_per_class)
            }
            
            # Log per-class metrics
            for i, label in self.id2label.items():
                metrics.update({
                    f"eval_precision_{label}": precision_per_class[i],
                    f"eval_recall_{label}": recall_per_class[i],
                    f"eval_f1_{label}": f1_per_class[i],
                    f"eval_specificity_{label}": specificity_per_class[i],
                    f"eval_npv_{label}": npv_per_class[i]
                })
            
            if self.config.use_wandb and wandb.run:
                wandb.log(metrics, step=state.global_step)
                
                # Log confusion matrix as heatmap
                wandb.log({
                    "confusion_matrix": wandb.plot.confusion_matrix(
                        y_true=y_true,
                        preds=y_pred,
                        class_names=list(self.id2label.values())
                    )
                }, step=state.global_step)
            
            # Update best metrics
            current_accuracy = kwargs.get('logs', {}).get('eval_accuracy', 0)
            if current_accuracy > self.best_metrics.get('best_accuracy', 0):
                self.best_metrics = {
                    'best_accuracy': current_accuracy,
                    'best_precision': precision,
                    'best_recall': recall,
                    'best_f1': f1,
                    'best_epoch': state.epoch
                }
                
                if self.config.use_wandb and wandb.run:
                    wandb.log(self.best_metrics, step=state.global_step)
                    
def setup_model_for_finetuning(
    model: Any,  # Use Any since model classes might not be available
    config: FinetuningConfig
) -> None:
    """Setup model for finetuning with layer freezing"""
    
    # Get base model parameters
    if hasattr(model, 'base_model'):
        base_params = list(model.base_model.parameters())
    elif hasattr(model, 'wav2vec2'):
        base_params = list(model.wav2vec2.parameters())
    elif hasattr(model, 'hubert'):
        base_params = list(model.hubert.parameters())
    else:
        # Fallback: get all model parameters except classifier
        all_params = list(model.parameters())
        # Try to exclude classifier parameters
        classifier_params = []
        if hasattr(model, 'classifier'):
            classifier_params = list(model.classifier.parameters())
        elif hasattr(model, 'projector'):
            classifier_params = list(model.projector.parameters())
        
        base_params = [p for p in all_params if p not in classifier_params]
    
    # Calculate number of layers to freeze
    num_layers_to_freeze = int(len(base_params) * config.freeze_ratio)
    
    # Freeze specified layers
    for i in range(num_layers_to_freeze):
        base_params[i].requires_grad = False
    
    # Keep remaining layers trainable
    for i in range(num_layers_to_freeze, len(base_params)):
        base_params[i].requires_grad = True
    
    # Always keep classifier head trainable
    if hasattr(model, 'classifier'):
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif hasattr(model, 'projector'):
        for param in model.projector.parameters():
            param.requires_grad = True

def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute metrics for evaluation"""
    metric = evaluate.load("accuracy")
    predictions = np.argmax(eval_pred.predictions, axis=1)
    accuracy = metric.compute(predictions=predictions, references=eval_pred.label_ids)
    return accuracy

def finetune_audio_model(
    dataset: Dict,
    feature_extractor: Any,
    label_mappings: Tuple[Dict[str, int], Dict[int, str]],
    model_id: str,
    config: FinetuningConfig,
    resume_from_checkpoint: Optional[str] = None
) -> Tuple[Any, Any, Dict]:
    """
    Universal finetuning function for audio classification models.
    
    Args:
        dataset: Dictionary containing 'train' and 'validation' datasets
        feature_extractor: Feature extractor for the model
        label_mappings: Tuple of (label2id, id2label) mappings
        model_id: Hugging Face model ID
        config: Finetuning configuration
        resume_from_checkpoint: Path to checkpoint to resume from
        
    Returns:
        Tuple of (trained_model, trainer, training_results)
    """
    
    # Set seed for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Extract label mappings
    label2id, id2label = label_mappings
    num_labels = len(id2label)
    
    # Load appropriate model
    model = None
    try:
        if "whisper" in model_id.lower():
            model = WhisperForAudioClassification.from_pretrained(
                model_id,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True
            )
            # Clean up Whisper-specific config issues
            if hasattr(model.config, 'max_length'):
                delattr(model.config, 'max_length')
            if hasattr(model.config, 'suppress_tokens'):
                delattr(model.config, 'suppress_tokens')
            if hasattr(model.config, 'begin_suppress_tokens'):
                delattr(model.config, 'begin_suppress_tokens')
                
        elif "wav2vec2" in model_id.lower() and Wav2Vec2ForSequenceClassification is not None:
            model = Wav2Vec2ForSequenceClassification.from_pretrained(
                model_id,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True
            )
        elif "hubert" in model_id.lower() and HubertForSequenceClassification is not None:
            model = HubertForSequenceClassification.from_pretrained(
                model_id,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True
            )
        
        # If no specific model was loaded, try AutoModel
        if model is None:
            print(f"Using AutoModelForAudioClassification for {model_id}")
            model = AutoModelForAudioClassification.from_pretrained(
                model_id,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True
            )
            
    except Exception as e:
        print(f"Error loading specific model: {e}")
        print("Trying AutoModelForAudioClassification...")
        model = AutoModelForAudioClassification.from_pretrained(
            model_id,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
    
    # Setup model for finetuning
    setup_model_for_finetuning(model, config)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔧 Model loaded: {model_id}")
    print(f"📊 Total parameters: {total_params:,}")
    print(f"🎯 Trainable parameters: {trainable_params:,}")
    print(f"🔒 Frozen parameters: {total_params - trainable_params:,}")
    print(f"📈 Trainable percentage: {(trainable_params / total_params) * 100:.2f}%")
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=config.model_name,
        eval_strategy=config.eval_strategy,
        save_strategy=config.save_strategy,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        num_train_epochs=config.num_train_epochs,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        fp16=config.fp16,
        dataloader_num_workers=config.dataloader_num_workers,
        save_total_limit=config.save_total_limit,
        lr_scheduler_type=config.lr_scheduler_type,
        report_to="wandb" if config.use_wandb else None,
        run_name=config.model_name,
        push_to_hub=config.push_to_hub,
        hub_model_id=config.hub_model_id,
        hub_private_repo=config.hub_private_repo,
        seed=config.seed,
        remove_unused_columns=False,  # Important for audio data
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
        callbacks=[
            WandbCallback(config),
            DetailedMetricsCallback(None, id2label, config),  # Will be set after trainer init
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience,
                early_stopping_threshold=config.early_stopping_threshold
            )
        ]
    )
    
    # Set trainer reference in callback
    for callback in trainer.callback_handler.callbacks:
        if isinstance(callback, DetailedMetricsCallback):
            callback._trainer = trainer
    
    # Check for existing checkpoints
    checkpoint = None
    if resume_from_checkpoint:
        checkpoint = resume_from_checkpoint
    else:
        checkpoint = get_last_checkpoint(config.model_name)
        if checkpoint:
            print(f"🔄 Resuming from checkpoint: {checkpoint}")
    
    # Train the model
    print(f"🚀 Starting training...")
    try:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
        # Save the final model
        trainer.save_model()
        feature_extractor.save_pretrained(config.model_name)
        
        # Get training results
        training_results = {
            'train_runtime': train_result.metrics['train_runtime'],
            'train_samples_per_second': train_result.metrics['train_samples_per_second'],
            'train_steps_per_second': train_result.metrics['train_steps_per_second'],
            'total_flos': train_result.metrics['total_flos'],
            'train_loss': train_result.metrics['train_loss'],
        }
        
        # Final evaluation
        print("📊 Running final evaluation...")
        final_metrics = trainer.evaluate()
        training_results.update(final_metrics)
        
        # Log final results
        if config.use_wandb and wandb.run:
            wandb.log({"final_metrics": training_results})
            
            # Log model artifacts
            wandb.log_artifact(config.model_name, name=f"{config.model_name}_model", type="model")
        
        # Push to Hub if requested
        if config.push_to_hub:
            print("🔄 Pushing to Hugging Face Hub...")
            trainer.push_to_hub()
            feature_extractor.push_to_hub(config.hub_model_id or config.model_name)
            print("✅ Model pushed to Hub successfully!")
        
        print("🎉 Training completed successfully!")
        return model, trainer, training_results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        if config.use_wandb and wandb.run:
            wandb.finish(exit_code=1)
        raise e
    finally:
        # Clean up
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        
from dataset_preprocessor import preprocess_for_whisper

dataset_whisper, fe_whisper, (l2i_w, i2l_w), info_w = preprocess_for_whisper(
            dataset_name='zaghloul2012/egyptian-dialect-audio',
            model_id="Seyfelislem/whisper-medium-arabic",
            max_duration=10  # Limit duration for faster processing
        )


# Example usage function
def example_usage():
    """Example of how to use the finetuning function"""
    
    # Configure finetuning
    config = FinetuningConfig(
        model_name="whisper-egyptian-dialect-v1",
        num_train_epochs=1,
        learning_rate=2e-5,
        per_device_train_batch_size=4,  # Reduced batch size to avoid memory issues
        gradient_accumulation_steps=4,  # Increased to maintain effective batch size
        freeze_ratio=0.7,  # Freeze more layers to speed up training
        use_wandb=False,  # Disable W&B for now
        wandb_project="egyptian-dialect-classification",
        push_to_hub=False,
        hub_model_id="zaghloul2012/whisper-egyptian-dialect-v1",
        hub_private_repo=False,
        early_stopping_patience=5
    )
    
    
    # Fine-tune the model
    model, trainer, results = finetune_audio_model(
        dataset=dataset_whisper,
        feature_extractor=fe_whisper,
        label_mappings=(l2i_w, i2l_w),
        model_id="Seyfelislem/whisper-medium-arabic",
        config=config
    )
    
    print(f"Training completed! Final accuracy: {results.get('eval_accuracy', 'N/A')}")
    
    return model, trainer, results

if __name__ == "__main__":
    # Run the example
    example_usage()