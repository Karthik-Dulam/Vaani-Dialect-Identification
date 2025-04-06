import os
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import jiwer
import json
import re
from collections import defaultdict

import pytorch_lightning as pl
from transformers import Wav2Vec2Processor
from huggingface_hub import login

from models import TranscriptionLitModel
from data import VaaniDataset, ResumableDataLoader, load_datasets
from config import TRANSCRIPTION_CONFIG, TRANSCRIPTION_CACHE_CONFIG, set_seeds, logger

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a trained ASR model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for evaluation')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to evaluate on (test or validation)')
    return parser.parse_args()

def extract_dialect(text):
    """Extract dialect tag from text if it exists."""
    match = re.search(r'<([^>]+)>', text)
    return match.group(1) if match else None

def remove_dialect(text):
    """Remove dialect tag from text."""
    return re.sub(r'<[^>]+>\s*', '', text)

def evaluate_model(checkpoint_path, output_dir=None, batch_size=16, split='test'):
    """Evaluate the transcription model."""
    # Set the device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Initial setup
    set_seeds()
    login(TRANSCRIPTION_CONFIG["hf_token"])
    
    # If output_dir is not specified, use the checkpoint directory
    if output_dir is None:
        output_dir = os.path.dirname(checkpoint_path)
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the processor
    processor_path = os.path.dirname(checkpoint_path)
    processor = Wav2Vec2Processor.from_pretrained(processor_path)
    
    # Load model from checkpoint
    model = TranscriptionLitModel.load_from_checkpoint(
        checkpoint_path,
        processor=processor
    )
    model.to(device)
    model.eval()
    
    # Load dataset
    _, test_ds, val_ds = load_datasets(TRANSCRIPTION_CONFIG, TRANSCRIPTION_CACHE_CONFIG)
    
    # Use appropriate split
    eval_ds = test_ds if split == 'test' else val_ds
    
    # Get all unique dialects
    all_dialects = sorted(set(eval_ds["dialect"]))
    label_to_id = {dialect: i for i, dialect in enumerate(all_dialects)}
    
    # Create evaluation dataset
    eval_dataset = VaaniDataset(
        eval_ds,
        processor,
        label_to_id,
        is_train=False,
        config=TRANSCRIPTION_CONFIG,
        cache_config=TRANSCRIPTION_CACHE_CONFIG,
        task="transcription"
    )
    
    # Create dataloader
    eval_loader = ResumableDataLoader(
        eval_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Initialize metrics
    all_wer = []
    all_cer = []
    dialect_predictions = []
    dialect_ground_truth = []
    all_predictions = []
    all_ground_truth = []
    dialect_metrics = defaultdict(lambda: {'correct': 0, 'total': 0, 'wer': []})
    
    # Evaluate model
    logger.info(f"Evaluating model on {len(eval_dataset)} examples from {split} set")
    
    for batch in tqdm(eval_loader, desc="Evaluating"):
        with torch.no_grad():
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Get model predictions
            outputs = model.model(
                input_values=batch["input_values"],
                attention_mask=batch.get("attention_mask", None)
            )
            
            # Get predicted token IDs
            predicted_ids = torch.argmax(outputs.logits, dim=-1)
            
            # Convert predictions to text
            pred_strings = processor.batch_decode(predicted_ids)
            
            # Convert labels to text
            labels = batch["labels"].detach().cpu().numpy()
            labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
            label_strings = processor.batch_decode(labels)
            
            # Extract dialects from predictions and ground truth
            for pred, label in zip(pred_strings, label_strings):
                pred_dialect = extract_dialect(pred)
                label_dialect = extract_dialect(label)
                
                # Process only if we have both dialect and transcript
                if label_dialect:
                    # Remove dialect token for WER/CER calculation
                    pred_text = remove_dialect(pred)
                    label_text = remove_dialect(label)
                    
                    # Calculate WER and CER
                    wer = jiwer.wer(label_text, pred_text)
                    cer = jiwer.cer(label_text, pred_text)
                    
                    all_wer.append(wer)
                    all_cer.append(cer)
                    
                    # Track dialect predictions
                    dialect_predictions.append(pred_dialect if pred_dialect else "unknown")
                    dialect_ground_truth.append(label_dialect)
                    
                    # Store full predictions and ground truth
                    all_predictions.append(pred)
                    all_ground_truth.append(label)
                    
                    # Update dialect-specific metrics
                    dialect = label_dialect
                    dialect_metrics[dialect]['total'] += 1
                    if pred_dialect == label_dialect:
                        dialect_metrics[dialect]['correct'] += 1
                    dialect_metrics[dialect]['wer'].append(wer)
    
    # Calculate overall metrics
    overall_wer = sum(all_wer) / len(all_wer) if all_wer else 0
    overall_cer = sum(all_cer) / len(all_cer) if all_cer else 0
    
    # Calculate overall dialect accuracy
    correct_dialects = sum(1 for p, l in zip(dialect_predictions, dialect_ground_truth) if p == l)
    dialect_accuracy = correct_dialects / len(dialect_ground_truth) if dialect_ground_truth else 0
    
    # Calculate dialect-specific metrics
    dialect_results = {}
    for dialect, metrics in dialect_metrics.items():
        accuracy = metrics['correct'] / metrics['total'] if metrics['total'] > 0 else 0
        avg_wer = sum(metrics['wer']) / len(metrics['wer']) if metrics['wer'] else 0
        dialect_results[dialect] = {
            'accuracy': accuracy,
            'samples': metrics['total'],
            'wer': avg_wer
        }
    
    # Print summary
    logger.info(f"Overall WER: {overall_wer:.4f}")
    logger.info(f"Overall CER: {overall_cer:.4f}")
    logger.info(f"Overall Dialect Accuracy: {dialect_accuracy:.4f}")
    
    # Create confusion matrix for dialect prediction
    dialect_labels = sorted(set(dialect_ground_truth))
    cm = np.zeros((len(dialect_labels), len(dialect_labels)), dtype=int)
    
    # Populate confusion matrix
    dialect_to_idx = {dialect: i for i, dialect in enumerate(dialect_labels)}
    for true, pred in zip(dialect_ground_truth, dialect_predictions):
        if pred in dialect_to_idx:  # Handle the case where prediction is "unknown"
            cm[dialect_to_idx[true], dialect_to_idx[pred]] += 1
    
    # Plot and save confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=dialect_labels, yticklabels=dialect_labels)
    plt.title('Dialect Prediction Confusion Matrix')
    plt.ylabel('True Dialect')
    plt.xlabel('Predicted Dialect')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dialect_confusion_matrix.png'))
    
    # Create and save a results DataFrame
    results_df = pd.DataFrame({
        'prediction': all_predictions,
        'ground_truth': all_ground_truth,
        'pred_dialect': dialect_predictions,
        'true_dialect': dialect_ground_truth,
        'wer': all_wer,
        'cer': all_cer,
        'dialect_correct': [p == l for p, l in zip(dialect_predictions, dialect_ground_truth)]
    })
    
    results_df.to_csv(os.path.join(output_dir, f'transcription_results_{split}.csv'), index=False)
    
    # Save dialect-specific results
    dialect_df = pd.DataFrame(
        [{'dialect': k, **v} for k, v in dialect_results.items()]
    ).sort_values('dialect')
    
    dialect_df.to_csv(os.path.join(output_dir, f'dialect_results_{split}.csv'), index=False)
    
    # Save summary metrics
    metrics = {
        'overall_wer': overall_wer,
        'overall_cer': overall_cer,
        'dialect_accuracy': dialect_accuracy,
        'num_samples': len(all_predictions),
        'dialect_breakdown': dialect_results
    }
    
    with open(os.path.join(output_dir, f'metrics_summary_{split}.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Plot dialect-specific WER and accuracy
    plt.figure(figsize=(12, 8))
    sns.barplot(x='dialect', y='wer', data=dialect_df)
    plt.title('Word Error Rate by Dialect')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'dialect_wer_{split}.png'))
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='dialect', y='accuracy', data=dialect_df)
    plt.title('Dialect Recognition Accuracy by Dialect')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'dialect_accuracy_{split}.png'))
    
    # Return metrics for external use
    return metrics

def collate_fn(batch):
    """
    Custom collate function for handling variable length inputs.
    """
    # Get input values and labels
    input_values = [item["input_values"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    # Pad input values and create attention mask
    input_lengths = [len(x) for x in input_values]
    max_input_length = max(input_lengths)
    
    padded_input_values = []
    attention_mask = []
    
    for item, length in zip(input_values, input_lengths):
        # Pad input values
        padded = torch.nn.functional.pad(
            item, 
            (0, max_input_length - length), 
            mode='constant', 
            value=0
        )
        padded_input_values.append(padded)
        
        # Create attention mask
        mask = torch.ones(length)
        mask = torch.nn.functional.pad(
            mask, 
            (0, max_input_length - length), 
            mode='constant', 
            value=0
        )
        attention_mask.append(mask)
    
    # Stack tensors
    input_values = torch.stack(padded_input_values)
    attention_mask = torch.stack(attention_mask)
    
    # Pad labels
    label_lengths = [len(x) for x in labels]
    max_label_length = max(label_lengths)
    
    padded_labels = []
    
    for item, length in zip(labels, label_lengths):
        # Pad labels
        padded = torch.nn.functional.pad(
            item, 
            (0, max_label_length - length), 
            mode='constant', 
            value=-100  # Ignore padding in loss computation
        )
        padded_labels.append(padded)
    
    labels = torch.stack(padded_labels)
    
    return {
        "input_values": input_values,
        "attention_mask": attention_mask,
        "labels": labels
    }

if __name__ == "__main__":
    args = parse_args()
    evaluate_model(
        args.checkpoint, 
        args.output_dir, 
        args.batch_size, 
        args.split
    ) 