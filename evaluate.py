import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import json
import torch
from pathlib import Path

# Visualization and metrics imports
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Transformers imports
from transformers import pipeline, AutoModelForAudioClassification, AutoFeatureExtractor
import librosa
import soundfile as sf
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Optional W&B import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("W&B not available. Logging to W&B will be disabled.")


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation"""
    model_path: str
    test_dataset: Any = None
    test_data_path: Optional[str] = None
    output_dir: str = "evaluation_results"
    save_predictions: bool = True
    save_plots: bool = True
    save_metrics: bool = True
    plot_format: str = "png"  # png, pdf, svg
    dpi: int = 300
    figsize: Tuple[int, int] = (12, 8)
    use_wandb: bool = False
    wandb_project: str = "model-evaluation"
    wandb_run_name: Optional[str] = None
    device: str = "auto"  # auto, cpu, cuda
    batch_size: int = 32
    num_workers: int = 4
    show_plots: bool = False
    detailed_analysis: bool = True
    include_per_sample_analysis: bool = False


class ModelEvaluator:
    """Comprehensive model evaluation class for audio classification"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.results = {}
        self.predictions_df = None
        self.model = None
        self.feature_extractor = None
        self.pipeline = None
        self.id2label = None
        self.label2id = None
        
        # Setup output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Setup device
        if self.config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device
            
        print(f"🔧 Using device: {self.device}")
        
        # Initialize W&B if requested
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name or f"eval_{Path(self.config.model_path).name}",
                config=self.config.__dict__
            )
    
    def load_model(self) -> None:
        """Load the trained model and feature extractor"""
        try:
            print(f"📂 Loading model from: {self.config.model_path}")
            
            # Load model
            self.model = AutoModelForAudioClassification.from_pretrained(self.config.model_path)
            
            # Load feature extractor
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.config.model_path)
            
            # Get label mappings
            self.id2label = self.model.config.id2label
            self.label2id = self.model.config.label2id
            
            # Create pipeline for easy inference
            self.pipeline = pipeline(
                "audio-classification",
                model=self.model,
                feature_extractor=self.feature_extractor,
                device=0 if self.device == "cuda" else -1
            )
            
            print(f"✅ Model loaded successfully!")
            print(f"📊 Number of classes: {len(self.id2label)}")
            print(f"🏷️ Classes: {list(self.id2label.values())}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e
    
    def prepare_test_data(self, test_dataset: Any) -> List[Dict]:
        """Prepare test data for evaluation"""
        test_data = []
        
        print(f"📊 Preparing {len(test_dataset)} test samples...")
        
        for idx, sample in enumerate(tqdm(test_dataset, desc="Preparing data")):
            try:
                test_data.append({
                    'idx': idx,
                    'audio': sample['audio'],
                    'true_label': sample['label'],
                    'audio_url': sample.get('audio_url', f'sample_{idx}')
                })
            except Exception as e:
                print(f"⚠️ Error processing sample {idx}: {e}")
                continue
        
        print(f"✅ Prepared {len(test_data)} samples for evaluation")
        return test_data
    
    def run_inference(self, test_data: List[Dict]) -> pd.DataFrame:
        """Run inference on test data and collect predictions"""
        predictions = []
        
        print("🚀 Running inference...")
        
        for sample in tqdm(test_data, desc="Inference"):
            try:
                # Run prediction
                result = self.pipeline(sample['audio'])
                
                # Get top prediction
                top_pred = result[0]
                pred_label = top_pred['label']
                pred_confidence = top_pred['score']
                
                # Get all class scores
                all_scores = {item['label']: item['score'] for item in result}
                
                # Convert numerical labels if needed
                if pred_label.startswith('LABEL_'):
                    pred_label_idx = int(pred_label.replace('LABEL_', ''))
                    pred_label = self.id2label[pred_label_idx]
                
                true_label = self.id2label[sample['true_label']] if isinstance(sample['true_label'], int) else sample['true_label']
                
                prediction = {
                    'sample_idx': sample['idx'],
                    'audio_url': sample['audio_url'],
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'pred_confidence': pred_confidence,
                    'correct': true_label == pred_label,
                    **{f'score_{label}': all_scores.get(f'LABEL_{self.label2id[label]}', all_scores.get(label, 0.0)) 
                       for label in self.id2label.values()}
                }
                
                predictions.append(prediction)
                
            except Exception as e:
                print(f"⚠️ Error processing sample {sample['idx']}: {e}")
                continue
        
        self.predictions_df = pd.DataFrame(predictions)
        print(f"✅ Completed inference on {len(self.predictions_df)} samples")
        
        return self.predictions_df
    
    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        if self.predictions_df is None:
            raise ValueError("No predictions available. Run inference first.")
        
        print("📊 Calculating metrics...")
        
        y_true = self.predictions_df['true_label'].values
        y_pred = self.predictions_df['pred_label'].values
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=list(self.id2label.values())
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=list(self.id2label.values()))
        
        # Calculate specificity and NPV for each class
        specificity_per_class = []
        npv_per_class = []
        
        for i, label in enumerate(self.id2label.values()):
            TP = cm[i, i]
            FP = cm[:, i].sum() - TP
            FN = cm[i, :].sum() - TP
            TN = cm.sum() - (TP + FP + FN)
            
            spec = TN / (TN + FP) if (TN + FP) != 0 else 0
            npv = TN / (TN + FN) if (TN + FN) != 0 else 0
            
            specificity_per_class.append(spec)
            npv_per_class.append(npv)
        
        # Store results
        self.results = {
            'overall_metrics': {
                'accuracy': accuracy,
                'precision_weighted': precision,
                'recall_weighted': recall,
                'f1_weighted': f1,
                'specificity_weighted': np.mean(specificity_per_class),
                'npv_weighted': np.mean(npv_per_class),
                'total_samples': len(y_true),
                'correct_predictions': sum(self.predictions_df['correct']),
            },
            'per_class_metrics': {},
            'confusion_matrix': cm.tolist(),
            'class_names': list(self.id2label.values())
        }
        
        # Per-class metrics
        for i, label in enumerate(self.id2label.values()):
            self.results['per_class_metrics'][label] = {
                'precision': precision_per_class[i],
                'recall': recall_per_class[i],
                'f1': f1_per_class[i],
                'specificity': specificity_per_class[i],
                'npv': npv_per_class[i],
                'support': int(support_per_class[i])
            }
        
        print("✅ Metrics calculated successfully!")
        return self.results
    
    def plot_confusion_matrix(self) -> plt.Figure:
        """Create confusion matrix heatmap"""
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        cm = np.array(self.results['confusion_matrix'])
        class_names = self.results['class_names']
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create heatmap
        sns.heatmap(
            cm_percent,
            annot=True,
            fmt='.1f',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )
        
        ax.set_title('Confusion Matrix (%)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        if self.config.save_plots:
            plt.savefig(
                os.path.join(self.config.output_dir, f'confusion_matrix.{self.config.plot_format}'),
                dpi=self.config.dpi, bbox_inches='tight'
            )
        
        if self.config.show_plots:
            plt.show()
        
        return fig
    
    def plot_metrics_comparison(self) -> plt.Figure:
        """Create metrics comparison bar chart"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        class_names = list(self.results['per_class_metrics'].keys())
        metrics_to_plot = ['precision', 'recall', 'f1', 'specificity']
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            values = [self.results['per_class_metrics'][cls][metric] for cls in class_names]
            
            bars = ax.bar(class_names, values, alpha=0.8, color=plt.cm.Set3(np.linspace(0, 1, len(class_names))))
            ax.set_title(f'{metric.capitalize()} by Class', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric.capitalize())
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
            
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if self.config.save_plots:
            plt.savefig(
                os.path.join(self.config.output_dir, f'metrics_comparison.{self.config.plot_format}'),
                dpi=self.config.dpi, bbox_inches='tight'
            )
        
        if self.config.show_plots:
            plt.show()
        
        return fig
    
    def plot_roc_curves(self) -> plt.Figure:
        """Create ROC curves for multiclass classification"""
        if self.predictions_df is None:
            raise ValueError("No predictions available for ROC analysis")
        
        # Prepare data for ROC analysis
        class_names = list(self.id2label.values())
        y_true = self.predictions_df['true_label'].values
        
        # Get prediction scores for each class
        y_scores = np.array([
            [self.predictions_df[f'score_{cls}'].values for cls in class_names]
        ]).squeeze().T
        
        # Binarize true labels
        y_true_bin = label_binarize(y_true, classes=class_names)
        if y_true_bin.shape[1] == 1:  # Handle binary case
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
        
        # Calculate ROC curve for each class
        fig, ax = plt.subplots(figsize=self.config.figsize)
        colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
        
        for i, (class_name, color) in enumerate(zip(class_names, colors)):
            if i < y_true_bin.shape[1] and i < y_scores.shape[1]:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
                roc_auc = auc(fpr, tpr)
                
                ax.plot(
                    fpr, tpr, color=color, lw=2,
                    label=f'{class_name} (AUC = {roc_auc:.3f})'
                )
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.8)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - One vs Rest', fontsize=16, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.config.save_plots:
            plt.savefig(
                os.path.join(self.config.output_dir, f'roc_curves.{self.config.plot_format}'),
                dpi=self.config.dpi, bbox_inches='tight'
            )
        
        if self.config.show_plots:
            plt.show()
        
        return fig
    
    def plot_confidence_distribution(self) -> plt.Figure:
        """Plot confidence score distributions"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Overall confidence distribution
        axes[0].hist(
            self.predictions_df['pred_confidence'],
            bins=50, alpha=0.7, color='skyblue', edgecolor='black'
        )
        axes[0].set_title('Overall Confidence Distribution', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Confidence Score')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
        
        # Confidence by correctness
        correct_conf = self.predictions_df[self.predictions_df['correct']]['pred_confidence']
        incorrect_conf = self.predictions_df[~self.predictions_df['correct']]['pred_confidence']
        
        axes[1].hist(
            correct_conf, bins=30, alpha=0.7, label='Correct', color='green', edgecolor='black'
        )
        axes[1].hist(
            incorrect_conf, bins=30, alpha=0.7, label='Incorrect', color='red', edgecolor='black'
        )
        axes[1].set_title('Confidence Distribution by Correctness', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Confidence Score')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.config.save_plots:
            plt.savefig(
                os.path.join(self.config.output_dir, f'confidence_distribution.{self.config.plot_format}'),
                dpi=self.config.dpi, bbox_inches='tight'
            )
        
        if self.config.show_plots:
            plt.show()
        
        return fig
    
    def analyze_errors(self) -> Dict:
        """Analyze prediction errors in detail"""
        errors = self.predictions_df[~self.predictions_df['correct']].copy()
        
        if len(errors) == 0:
            print("🎉 Perfect predictions! No errors to analyze.")
            return {}
        
        print(f"🔍 Analyzing {len(errors)} prediction errors...")
        
        error_analysis = {
            'total_errors': len(errors),
            'error_rate': len(errors) / len(self.predictions_df),
            'errors_by_true_class': errors.groupby('true_label').size().to_dict(),
            'errors_by_pred_class': errors.groupby('pred_label').size().to_dict(),
            'common_confusions': errors.groupby(['true_label', 'pred_label']).size().head(10).to_dict(),
            'low_confidence_errors': len(errors[errors['pred_confidence'] < 0.5]),
            'high_confidence_errors': len(errors[errors['pred_confidence'] >= 0.8]),
        }
        
        # Find most confused pairs
        confusion_pairs = errors.groupby(['true_label', 'pred_label']).size().sort_values(ascending=False)
        error_analysis['most_confused_pairs'] = [
            f"{true_label} → {pred_label} ({count} errors)"
            for (true_label, pred_label), count in confusion_pairs.head(5).items()
        ]
        
        return error_analysis
    
    def save_results(self) -> None:
        """Save all evaluation results"""
        if not self.config.save_metrics:
            return
        
        print("💾 Saving evaluation results...")
        
        # Save metrics as JSON
        with open(os.path.join(self.config.output_dir, 'metrics.json'), 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save predictions as CSV
        if self.config.save_predictions and self.predictions_df is not None:
            self.predictions_df.to_csv(
                os.path.join(self.config.output_dir, 'predictions.csv'),
                index=False
            )
        
        # Save detailed report as text
        report_path = os.path.join(self.config.output_dir, 'evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write("="*50 + "\n")
            f.write("MODEL EVALUATION REPORT\n")
            f.write("="*50 + "\n\n")
            
            f.write("OVERALL METRICS:\n")
            f.write("-" * 20 + "\n")
            for metric, value in self.results['overall_metrics'].items():
                f.write(f"{metric.replace('_', ' ').title()}: {value:.4f}\n")
            
            f.write("\nPER-CLASS METRICS:\n")
            f.write("-" * 20 + "\n")
            for class_name, metrics in self.results['per_class_metrics'].items():
                f.write(f"\n{class_name}:\n")
                for metric, value in metrics.items():
                    if metric != 'support':
                        f.write(f"  {metric.capitalize()}: {value:.4f}\n")
                    else:
                        f.write(f"  {metric.capitalize()}: {value}\n")
            
            # Add error analysis if available
            if hasattr(self, 'error_analysis'):
                f.write("\nERROR ANALYSIS:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Total Errors: {self.error_analysis['total_errors']}\n")
                f.write(f"Error Rate: {self.error_analysis['error_rate']:.4f}\n")
                f.write(f"Low Confidence Errors: {self.error_analysis['low_confidence_errors']}\n")
                f.write(f"High Confidence Errors: {self.error_analysis['high_confidence_errors']}\n")
                
                f.write("\nMost Confused Pairs:\n")
                for pair in self.error_analysis['most_confused_pairs']:
                    f.write(f"  {pair}\n")
        
        print(f"✅ Results saved to: {self.config.output_dir}")
    
    def log_to_wandb(self) -> None:
        """Log results to Weights & Biases"""
        if not (self.config.use_wandb and WANDB_AVAILABLE and wandb.run):
            return
        
        print("📊 Logging to W&B...")
        
        # Log overall metrics
        wandb.log(self.results['overall_metrics'])
        
        # Log per-class metrics
        for class_name, metrics in self.results['per_class_metrics'].items():
            for metric, value in metrics.items():
                wandb.log({f"{class_name}_{metric}": value})
        
        # Log confusion matrix as heatmap
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                y_true=self.predictions_df['true_label'].values,
                preds=self.predictions_df['pred_label'].values,
                class_names=self.results['class_names']
            )
        })
        
        # Log plots if they exist
        plot_files = [
            'confusion_matrix', 'metrics_comparison', 
            'roc_curves', 'confidence_distribution'
        ]
        
        for plot_name in plot_files:
            plot_path = os.path.join(self.config.output_dir, f'{plot_name}.{self.config.plot_format}')
            if os.path.exists(plot_path):
                wandb.log({plot_name: wandb.Image(plot_path)})
        
        print("✅ Logged to W&B successfully!")
    
    def evaluate(self, test_dataset: Any = None) -> Dict:
        """Run complete evaluation pipeline"""
        print("🚀 Starting model evaluation...")
        
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
        
        # Use provided dataset or config dataset
        dataset = test_dataset or self.config.test_dataset
        if dataset is None:
            raise ValueError("No test dataset provided")
        
        # Prepare test data
        test_data = self.prepare_test_data(dataset)
        
        # Run inference
        self.run_inference(test_data)
        
        # Calculate metrics
        self.calculate_metrics()
        
        # Generate plots
        print("📊 Generating visualizations...")
        self.plot_confusion_matrix()
        self.plot_metrics_comparison()
        self.plot_roc_curves()
        self.plot_confidence_distribution()
        
        # Analyze errors
        self.error_analysis = self.analyze_errors()
        
        # Save results
        self.save_results()
        
        # Log to W&B
        self.log_to_wandb()
        
        print("✅ Evaluation completed successfully!")
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"📊 Total Samples: {self.results['overall_metrics']['total_samples']}")
        print(f"✅ Correct Predictions: {self.results['overall_metrics']['correct_predictions']}")
        print(f"🎯 Accuracy: {self.results['overall_metrics']['accuracy']:.4f}")
        print(f"📈 F1 Score (Weighted): {self.results['overall_metrics']['f1_weighted']:.4f}")
        print(f"🔍 Precision (Weighted): {self.results['overall_metrics']['precision_weighted']:.4f}")
        print(f"📋 Recall (Weighted): {self.results['overall_metrics']['recall_weighted']:.4f}")
        
        if self.error_analysis:
            print(f"❌ Error Rate: {self.error_analysis['error_rate']:.4f}")
        
        print(f"💾 Results saved to: {self.config.output_dir}")
        
        return self.results


def evaluate_model(
    model_path: str,
    test_dataset: Any,
    output_dir: str = "evaluation_results",
    config_overrides: Optional[Dict] = None
) -> Dict:
    """
    Convenience function to evaluate a model with default settings.
    
    Args:
        model_path: Path to the trained model
        test_dataset: Test dataset
        output_dir: Directory to save results
        config_overrides: Optional config overrides
    
    Returns:
        Evaluation results dictionary
    """
    
    config = EvaluationConfig(
        model_path=model_path,
        test_dataset=test_dataset,
        output_dir=output_dir
    )
    
    # Apply any overrides
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    evaluator = ModelEvaluator(config)
    results = evaluator.evaluate()
    
    return results


# Integration function for training script
def evaluate_after_training(
    trainer: Any,
    test_dataset: Any,
    model_path: str,
    output_dir: str = None,
    use_wandb: bool = False,
    wandb_project: str = "model-evaluation"
) -> Dict:
    """
    Function to integrate evaluation into training pipeline.
    
    Args:
        trainer: Hugging Face trainer object
        test_dataset: Test dataset
        model_path: Path where model is saved
        output_dir: Directory to save evaluation results
        use_wandb: Whether to log to W&B
        wandb_project: W&B project name
    
    Returns:
        Evaluation results
    """
    
    if output_dir is None:
        output_dir = os.path.join(model_path, "evaluation")
    
    config = EvaluationConfig(
        model_path=model_path,
        test_dataset=test_dataset,
        output_dir=output_dir,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        save_plots=True,
        save_predictions=True,
        save_metrics=True
    )
    
    evaluator = ModelEvaluator(config)
    results = evaluator.evaluate()
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Model Evaluator - Example Usage")
    print("This script provides comprehensive evaluation for audio classification models")
    print("Use the evaluate_model() function or ModelEvaluator class directly")