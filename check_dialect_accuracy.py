#!/usr/bin/env python
"""
Quick script to check dialect accuracy of a trained transcription model.
"""
import os
import argparse
import torch
import re
from tqdm import tqdm
from transformers import Wav2Vec2Processor
from models import TranscriptionLitModel
from data import VaaniDataset, ResumableDataLoader, load_datasets
from config import TRANSCRIPTION_CONFIG, TRANSCRIPTION_CACHE_CONFIG, set_seeds, logger

def extract_dialect(text):
    """Extract dialect tag from text if it exists."""
    match = re.search(r'<([^>]+)>', text)
    return match.group(1) if match else None

def main():
    parser = argparse.ArgumentParser(description='Check dialect accuracy of a trained ASR model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to evaluate')
    parser.add_argument('--split', type=str, default='validation', choices=['test', 'validation'], 
                        help='Dataset split to evaluate on')
    args = parser.parse_args()
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set random seed
    set_seeds()
    
    # Load processor
    processor_path = os.path.dirname(args.checkpoint)
    processor = Wav2Vec2Processor.from_pretrained(processor_path)
    
    # Load model
    model = TranscriptionLitModel.load_from_checkpoint(
        args.checkpoint,
        processor=processor
    )
    model.to(device)
    model.eval()
    
    # Load dataset
    _, test_ds, val_ds = load_datasets(TRANSCRIPTION_CONFIG, TRANSCRIPTION_CACHE_CONFIG)
    eval_ds = test_ds if args.split == 'test' else val_ds
    
    # Get dialects
    all_dialects = sorted(set(eval_ds["dialect"]))
    label_to_id = {dialect: i for i, dialect in enumerate(all_dialects)}
    
    # Create dataset
    eval_dataset = VaaniDataset(
        eval_ds,
        processor,
        label_to_id,
        is_train=False,
        config=TRANSCRIPTION_CONFIG,
        cache_config=TRANSCRIPTION_CACHE_CONFIG,
        task="transcription"
    )
    
    # Limit to specified number of samples
    if args.num_samples < len(eval_dataset):
        indices = torch.randperm(len(eval_dataset))[:args.num_samples].tolist()
        subset = torch.utils.data.Subset(eval_dataset, indices)
    else:
        subset = eval_dataset
    
    # Create dataloader
    eval_loader = torch.utils.data.DataLoader(
        subset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Evaluate
    dialect_correct = 0
    total = 0
    wer_total = 0
    
    for batch in tqdm(eval_loader, desc=f"Evaluating {args.split} set"):
        with torch.no_grad():
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Get predictions
            outputs = model.model(
                input_values=batch["input_values"],
                attention_mask=batch.get("attention_mask", None)
            )
            
            # Get predicted tokens
            predicted_ids = torch.argmax(outputs.logits, dim=-1)
            
            # Decode predictions
            pred_strings = processor.batch_decode(predicted_ids)
            
            # Decode labels
            labels = batch["labels"].detach().cpu().numpy()
            import numpy as np
            labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
            label_strings = processor.batch_decode(labels)
            
            # Calculate dialect accuracy
            for pred, label in zip(pred_strings, label_strings):
                pred_dialect = extract_dialect(pred)
                label_dialect = extract_dialect(label)
                
                if label_dialect:
                    total += 1
                    if pred_dialect == label_dialect:
                        dialect_correct += 1
                    
                    # Print examples
                    print(f"\nTrue: '{label}'")
                    print(f"Pred: '{pred}'")
                    print(f"Dialect correct: {pred_dialect == label_dialect} (True: {label_dialect}, Pred: {pred_dialect or 'None'})")
                    
                    # Calculate WER for this sample
                    import jiwer
                    pred_text = re.sub(r'<[^>]+>\s*', '', pred)
                    label_text = re.sub(r'<[^>]+>\s*', '', label)
                    wer = jiwer.wer(label_text, pred_text)
                    wer_total += wer
                    print(f"WER: {wer:.4f}")
                    
                    # Print separator
                    print("-" * 80)
    
    # Print summary
    dialect_accuracy = dialect_correct / total if total > 0 else 0
    avg_wer = wer_total / total if total > 0 else 0
    
    print(f"\n{'=' * 40} SUMMARY {'=' * 40}")
    print(f"Evaluated {total} samples from {args.split} set")
    print(f"Dialect Accuracy: {dialect_accuracy:.4f} ({dialect_correct}/{total})")
    print(f"Average WER: {avg_wer:.4f}")

def collate_fn(batch):
    """Collate function for variable length inputs."""
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
    main() 