import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import pytorch_lightning as pl_lightning
from transformers import Wav2Vec2FeatureExtractor, AutoFeatureExtractor, AutoTokenizer
from huggingface_hub import login
import wandb
from dataclasses import asdict
from typing import List, Dict, Any, Tuple
import datasets as dts

from config import get_config, set_seeds, logger, Config, CacheConfig
from models_vaani import LitModel
from data import VaaniDataset, load_datasets

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str], output_dir: str) -> None:
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

def analyze_errors(dataset: dts.Dataset, predictions: np.ndarray, true_labels: np.ndarray, label_mapping: Dict[int, str], output_dir: str, cache_config: CacheConfig) -> pl.DataFrame:
    """Analyze misclassified examples."""
    errors = []
    split_key = cache_config.split_key 
    for idx, (pred, true) in enumerate(zip(predictions, true_labels)):
        if pred != true:
            errors.append({
                'Index': idx,
                'True_Label': label_mapping[true],
                'Predicted_Label': label_mapping[pred],
                split_key: dataset[idx][split_key],
            })
    
    
    if errors:
        error_df = pl.DataFrame(errors)
        error_df.write_parquet(os.path.join(output_dir, 'error_analysis.parquet'))
    else:
        logger.info("No errors found in the evaluation set")
        error_df = pl.DataFrame(schema={
            'Index': pl.Int64,
            'True_Label': pl.Utf8,
            'Predicted_Label': pl.Utf8,
            split_key: pl.Utf8
        })
    
    return error_df

def collate_fn(batch: List[Dict[str, Any]], feature_extractor: Any, config: Config) -> Dict[str, torch.Tensor]:
    """
    Custom collate function that handles preprocessing and feature extraction for classification.
    """
    audio_samples = [item["audio"]["array"] for item in batch]
    labels = [item["label"] for item in batch]
    max_length = 16000 * config.model_config.num_seconds 
    processed_inputs = feature_extractor(
        audio_samples,
        sampling_rate=16000,
        padding="longest",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=True,
    )
    labels = torch.tensor(labels, dtype=torch.long)
    if "input_features" in processed_inputs:
        return {
            "input_values": processed_inputs.input_features,
            "attention_mask": processed_inputs.attention_mask,
            "label": labels,
        }
    else:
        return {
            "input_values": processed_inputs.input_values,
            "attention_mask": processed_inputs.attention_mask,
            "label": labels,
        }

def evaluate_model(checkpoint_path: str) -> Dict[str, float]:
    config, cache_config = get_config('classification')
    set_seeds(cache_config.random_seed)
    login(config.hf_token)
    logger.info("Starting model evaluation")

    wandb_api_key = os.environ.get("WANDB_KEY_K")
    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key
        logger.info("Found WANDB API key, enabling wandb logging for evaluation")
        version = os.path.basename(checkpoint_path).split("-v")[-1].split(".ckpt")[0]
        wandb.init(
            project=f"{config.name}-{cache_config.task_type}-eval",
            name=f"{config.model.name.split('/')[-1]}-eval-v{version}",
            config={**asdict(config), **asdict(cache_config)},
        )

    train_ds, test_ds, val_ds = load_datasets(config, cache_config)
    logger.info(f"Successfully loaded datasets")

    col = cache_config.task_type
    all_labels = (
        set(train_ds.unique(col)) | set(test_ds.unique(col)) | set(val_ds.unique(col))
    )
    labels = sorted(all_labels)
    label_to_id = {label: i for i, label in enumerate(labels)}
    id_to_label = {i: label for i, label in enumerate(labels)}

    if config.model.name == "facebook/w2v-bert-2.0":
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            config.model.feature_extractor
        )
    else:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            config.model.feature_extractor
        )
    logger.info(f"Loaded feature extractor: {config.model.feature_extractor}")

    model = LitModel.load_from_checkpoint(
        checkpoint_path,
        num_labels=len(labels),
        model_name=config.model.name,
        model_config=config.model_config,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        classifier_only=config.training.classifier_only,
    )
    model.eval()
    logger.info(f"Successfully loaded model")

    val_dataset = VaaniDataset(
        val_ds,
        feature_extractor,
        label_to_id,
        is_train=False,
        config=config,
        cache_config=cache_config,
    )
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.per_device_train_batch_size,
        num_workers=cache_config.num_proc,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=lambda batch: collate_fn(batch, feature_extractor, config),
    )
    logger.info(f"Successfully created validation dataloader with {len(val_loader)} batches")

    trainer = pl_lightning.Trainer(
        accelerator="auto",
        devices=[1],
        precision="16-mixed",
        enable_checkpointing=False,
        logger=False
    )
    logger.info(f"Successfully initialized trainer")

    model.test_step_outputs = []

    results = trainer.test(model, val_loader)[0]
    logger.info(f"Test results: {results}")
    logger.info(f"Successfully validated model")

    version = os.path.basename(checkpoint_path).split("-v")[-1].split(".ckpt")[0]
    eval_dir = os.path.join(os.path.dirname(checkpoint_path), f'evaluation_v{version}')
    os.makedirs(eval_dir, exist_ok=True)
    logger.info(f"Saving evaluation results to {eval_dir}")

    all_preds = []
    all_labels_arr = []
    logger.info(f"Number of test step outputs: {len(model.test_step_outputs)}")
    for output in model.test_step_outputs:
        all_preds.extend(output['preds'].cpu().numpy())
        all_labels_arr.extend(output['labels'].cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels_arr = np.array(all_labels_arr)
    logger.info(f"Successfully got all predictions and labels")

    plot_confusion_matrix(all_labels_arr, all_preds, labels, eval_dir)

    if wandb_api_key and wandb.run is not None:
        cm_fig = plt.figure(figsize=(12, 10))
        sns.heatmap(
            confusion_matrix(all_labels_arr, all_preds),
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

    report = classification_report(
        all_labels_arr,
        all_preds,
        target_names=labels,
        output_dict=True
    )
    report_df = pl.DataFrame(
        {k: v if isinstance(v, dict) else {'precision': v, 'recall': v, 'f1-score': v, 'support': v}
         for k, v in report.items()}
    ).transpose()
    report_df.write_parquet(os.path.join(eval_dir, 'classification_report.parquet'))

    error_df = analyze_errors(val_ds, all_preds, all_labels_arr, id_to_label, eval_dir, cache_config)

    logger.info("\nEvaluation Results:")
    logger.info(f"Validation Loss: {results['test_loss']:.4f}")
    logger.info(f"Validation Accuracy: {results['test_acc']:.4f}")
    logger.info(f"Validation F1 Score (Macro): {results['test_f1']:.4f}")
    logger.info(f"Validation Precision (Macro): {results['test_precision']:.4f}")
    logger.info(f"Validation Recall (Macro): {results['test_recall']:.4f}")

    if wandb_api_key and wandb.run is not None:
        wandb.log({
            "val_loss": results['test_loss'],
            "val_accuracy": results['test_acc'],
            "val_f1": results['test_f1'],
            "val_precision": results['test_precision'],
            "val_recall": results['test_recall']
        })
        wandb.finish()

    logger.info(f"\nDetailed results saved in: {eval_dir}")
    logger.info(f"- Confusion Matrix: {os.path.join(eval_dir, 'confusion_matrix.png')}")
    logger.info(f"- Classification Report: {os.path.join(eval_dir, 'classification_report.parquet')}")
    logger.info(f"- Error Analysis: {os.path.join(eval_dir, 'error_analysis.parquet')}")
    return results

if __name__ == "__main__":
    config: Config
    cache_config: CacheConfig
    config, cache_config = get_config('classification')
    model_dir = config.output_dir
    ckpt_list = sorted(glob.glob(os.path.join(model_dir, "best-checkpoint-v*.ckpt")))
    if not ckpt_list:
        raise ValueError(f"No checkpoint found in {model_dir} matching best-checkpoint-v*.ckpt")
    checkpoint_path = ckpt_list[-1] 

    logger.info(f"Using checkpoint: {checkpoint_path}")
    evaluate_model(checkpoint_path)