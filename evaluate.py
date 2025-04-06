import os
import numpy as np
import torch
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import pytorch_lightning as pl_lightning
from transformers import Wav2Vec2FeatureExtractor
from huggingface_hub import login
import wandb

from config import CONFIG, CACHE_SENSITIVE_CONFIG, set_seeds, logger
from models import LitModel
from data import VaaniDataset, ResumableDataLoader, load_datasets

def plot_confusion_matrix(y_true, y_pred, labels, output_dir):
    """Plot and save confusion matrix."""
    logger.info(f"Shape of y_true: {y_true.shape}")
    logger.info(f"Shape of y_pred: {y_pred.shape}")
    logger.info(f"Number of unique true labels: {len(np.unique(y_true))}")
    logger.info(f"Number of unique predicted labels: {len(np.unique(y_pred))}")
    logger.info(f"Labels provided: {labels}")
    
    if len(y_true) == 0 or len(y_pred) == 0:
        logger.error("Empty arrays provided to confusion matrix")
        return
        
    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"Confusion matrix shape: {cm.shape}")
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def analyze_errors(dataset, predictions, true_labels, label_mapping, output_dir):
    """Analyze misclassified examples."""
    errors = []
    for idx, (pred, true) in enumerate(zip(predictions, true_labels)):
        if pred != true:
            errors.append({
                'Index': idx,
                'True_Label': label_mapping[true],
                'Predicted_Label': label_mapping[pred],
                'Speaker_ID': dataset[idx]['speakerID'],
            })
    
    # Create Polars DataFrame
    if errors:
        error_df = pl.DataFrame(errors)
        error_df.write_parquet(os.path.join(output_dir, 'error_analysis.parquet'))
    else:
        logger.info("No errors found in the evaluation set")
        error_df = pl.DataFrame(schema={
            'Index': pl.Int64,
            'True_Label': pl.Utf8,
            'Predicted_Label': pl.Utf8,
            'Speaker_ID': pl.Utf8
        })
    
    return error_df

def evaluate_model(checkpoint_path):
    """Comprehensive model evaluation."""
    # Initial setup
    set_seeds()
    login(CONFIG["hf_token"])
    logger.info("Starting model evaluation")
    
    # Setup wandb logging for evaluation
    wandb_api_key = os.environ.get("WANDB_KEY_K")
    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key
        logger.info("Found WANDB API key, enabling wandb logging for evaluation")
        # Extract version from checkpoint filename
        version = os.path.basename(checkpoint_path).split("-v")[-1].split(".ckpt")[0]
        wandb.init(
            project=f"{CONFIG['name']}-{CACHE_SENSITIVE_CONFIG['task_type']}-eval",
            name=f"{CONFIG['model']['name'].split('/')[-1]}-eval-v{version}",
            config={**CONFIG, **CACHE_SENSITIVE_CONFIG},
        )
    
    # Load datasets
    _, _, val_ds, labels = load_datasets(CONFIG, CACHE_SENSITIVE_CONFIG)
    logger.info(f"Successfully loaded datasets")
    
    # Setup model and processor
    label_to_id = {label: i for i, label in enumerate(labels)}
    id_to_label = {i: label for i, label in enumerate(labels)}
    
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        CONFIG["model"]["feature_extractor"]
    )
    logger.info(f"Successfully loaded feature extractor")
    
    # Load trained model
    model = LitModel.load_from_checkpoint(
        checkpoint_path,
    )
    model.eval()
    logger.info(f"Successfully loaded model")
    
    # Create validation dataset and dataloader
    val_dataset = VaaniDataset(val_ds, feature_extractor, label_to_id, is_train=False, config=CONFIG)
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    val_loader = ResumableDataLoader(
        val_dataset,
        batch_size=CONFIG["training"]["per_device_train_batch_size"],
        num_workers=CACHE_SENSITIVE_CONFIG["num_proc"],
        pin_memory=True,
        persistent_workers=True
    )
    logger.info(f"Successfully created validation dataloader with {len(val_loader)} batches")
    
    # Initialize trainer for validation
    trainer = pl_lightning.Trainer(
        accelerator="auto",
        devices=[1],
        precision="16-mixed",
        enable_checkpointing=False,  # Disable checkpointing during evaluation
        logger=False  # Disable logging during evaluation
    )
    logger.info(f"Successfully initialized trainer")
    
    # Reset test step outputs
    model.test_step_outputs = []
    
    # Test model and get predictions
    results = trainer.test(model, val_loader)[0]
    logger.info(f"Test results: {results}")
    logger.info(f"Successfully validated model")
    
    # Extract version from checkpoint filename
    version = os.path.basename(checkpoint_path).split("-v")[-1].split(".ckpt")[0]
    
    # Create versioned evaluation directory
    eval_dir = os.path.join(os.path.dirname(checkpoint_path), f'evaluation_v{version}')
    os.makedirs(eval_dir, exist_ok=True)
    logger.info(f"Saving evaluation results to {eval_dir}")
    
    # Get predictions and labels from test step outputs
    all_preds = []
    all_labels = []
    logger.info(f"Number of test step outputs: {len(model.test_step_outputs)}")
    for i, output in enumerate(model.test_step_outputs):
        all_preds.extend(output['preds'].cpu().numpy())
        all_labels.extend(output['labels'].cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    logger.info(f"Successfully got all predictions and labels")
    
    # Generate and save confusion matrix
    plot_confusion_matrix(all_labels, all_preds, labels, eval_dir)
    
    # If wandb is active, log the confusion matrix
    if wandb_api_key and wandb.run is not None:
        cm_fig = plt.figure(figsize=(12, 10))
        sns.heatmap(
            confusion_matrix(all_labels, all_preds),
            annot=True,
            fmt='d',
            xticklabels=labels,
            yticklabels=labels
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        wandb.log({"confusion_matrix": wandb.Image(cm_fig)})
        plt.close(cm_fig)
    
    # Generate and save classification report
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=labels,
        output_dict=True
    )
    
    # Convert classification report to Polars DataFrame
    report_df = pl.DataFrame(
        {k: v if isinstance(v, dict) else {'precision': v, 'recall': v, 'f1-score': v, 'support': v} 
         for k, v in report.items()}
    ).transpose()
    
    report_df.write_parquet(os.path.join(eval_dir, 'classification_report.parquet'))
    
    # Analyze errors
    error_df = analyze_errors(val_ds, all_preds, all_labels, id_to_label, eval_dir)
    
    # Print summary
    logger.info("\nEvaluation Results:")
    logger.info(f"Validation Loss: {results['test_loss']:.4f}")
    logger.info(f"Validation Accuracy: {results['test_acc']:.4f}")
    logger.info(f"Validation F1 Score (Macro): {results['test_f1']:.4f}")
    logger.info(f"Validation Precision (Macro): {results['test_precision']:.4f}")
    logger.info(f"Validation Recall (Macro): {results['test_recall']:.4f}")
    
    # Log to wandb if active
    if wandb_api_key and wandb.run is not None:
        wandb.log({
            "val_loss": results['test_loss'],
            "val_accuracy": results['test_acc'],
            "val_f1": results['test_f1'],
            "val_precision": results['test_precision'],
            "val_recall": results['test_recall']
        })
        
        # Close wandb
        wandb.finish()
    
    logger.info(f"\nDetailed results saved in: {eval_dir}")
    logger.info(f"- Confusion Matrix: {os.path.join(eval_dir, 'confusion_matrix.png')}")
    logger.info(f"- Classification Report: {os.path.join(eval_dir, 'classification_report.parquet')}")
    logger.info(f"- Error Analysis: {os.path.join(eval_dir, 'error_analysis.parquet')}")

if __name__ == "__main__":
    # Get the latest checkpoint
    model_dir = CONFIG["output_dir"]
    checkpoint_path = os.path.join(model_dir, "best-checkpoint.ckpt")
    
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"No checkpoint found at {checkpoint_path}")
    
    evaluate_model(checkpoint_path)