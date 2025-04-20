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
from transformers import (
    Wav2Vec2Processor,
    WhisperProcessor,
    WhisperTokenizer,
    Wav2Vec2FeatureExtractor,
)
from huggingface_hub import login

from models_vaani import TranscriptionLitModel
from data import VaaniDataset, ResumableDataLoader, load_datasets
from config import get_config, set_seeds, logger


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained ASR model")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the checkpoint file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use for evaluation"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate on (test or validation)",
    )
    return parser.parse_args()


def extract_dialect(text):
    match = re.search(r"<([^>]+)>", text)
    return match.group(1) if match else None


def remove_dialect(text):
    return re.sub(r"<[^>]+>\s*", "", text)


def evaluate_model(checkpoint_path, output_dir=None, batch_size=16, split="test"):
    """Evaluate the transcription model."""
    config_updates = {"output_dir": output_dir} if output_dir else {}
    config, cache_config = get_config("transcription", config_updates)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seeds(cache_config.random_seed)
    login(config.hf_token)

    processor_path = os.path.dirname(checkpoint_path)
    try:
        processor = WhisperProcessor.from_pretrained(processor_path)
        logger.info("Loaded WhisperProcessor")
    except:
        try:
            processor = {
                "tokenizer": WhisperTokenizer.from_pretrained(processor_path),
                "feature_extractor": Wav2Vec2FeatureExtractor.from_pretrained(
                    processor_path
                ),
            }
            logger.info("Loaded Wav2Vec2 processor components")
        except Exception as e:
            logger.error(f"Failed to load processor: {e}")
            raise

    model = TranscriptionLitModel.load_from_checkpoint(
        checkpoint_path, processor=processor
    )
    model.to(device)
    model.eval()

    _, test_ds, val_ds = load_datasets(config, cache_config)

    eval_ds = test_ds if split == "test" else val_ds

    all_dialects = sorted(set(eval_ds["dialect"]))
    label_to_id = {dialect: i for i, dialect in enumerate(all_dialects)}

    eval_dataset = VaaniDataset(
        eval_ds,
        processor,
        label_to_id,
        is_train=False,
        config=config,
        cache_config=cache_config,
        task="transcription",
    )

    eval_loader = ResumableDataLoader(
        eval_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    all_wer = []
    all_cer = []
    dialect_predictions = []
    dialect_ground_truth = []
    all_predictions = []
    all_ground_truth = []
    dialect_metrics = defaultdict(lambda: {"correct": 0, "total": 0, "wer": []})

    logger.info(f"Evaluating model on {len(eval_dataset)} examples from {split} set")

    for batch in tqdm(eval_loader, desc="Evaluating"):
        with torch.no_grad():
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            outputs = model.model(
                input_values=batch["input_values"],
                attention_mask=batch.get("attention_mask", None),
            )

            if isinstance(processor, WhisperProcessor):
                generated_ids = model.model.whisper.generate(
                    input_features=batch["input_values"], max_length=225
                )
                pred_strings = processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )

                label_strings = processor.batch_decode(
                    batch["labels"], skip_special_tokens=True
                )
            else:
                logits = outputs.logits
                predicted_ids = torch.argmax(logits, dim=-1)
                pred_strings = processor["tokenizer"].batch_decode(predicted_ids)

                labels = batch["labels"].detach().cpu().numpy()
                labels = np.where(
                    labels != -100, labels, processor["tokenizer"].pad_token_id
                )
                label_strings = processor["tokenizer"].batch_decode(labels)

            for pred, label in zip(pred_strings, label_strings):
                pred_dialect = extract_dialect(pred)
                label_dialect = extract_dialect(label)

                if label_dialect:
                    pred_text = remove_dialect(pred)
                    label_text = remove_dialect(label)

                    wer = jiwer.wer(label_text, pred_text)
                    cer = jiwer.cer(label_text, pred_text)

                    all_wer.append(wer)
                    all_cer.append(cer)

                    dialect_predictions.append(
                        pred_dialect if pred_dialect else "unknown"
                    )
                    dialect_ground_truth.append(label_dialect)

                    all_predictions.append(pred)
                    all_ground_truth.append(label)

                    dialect = label_dialect
                    dialect_metrics[dialect]["total"] += 1
                    if pred_dialect == label_dialect:
                        dialect_metrics[dialect]["correct"] += 1
                    dialect_metrics[dialect]["wer"].append(wer)

    overall_wer = sum(all_wer) / len(all_wer) if all_wer else 0
    overall_cer = sum(all_cer) / len(all_cer) if all_cer else 0

    correct_dialects = sum(
        1 for p, l in zip(dialect_predictions, dialect_ground_truth) if p == l
    )
    dialect_accuracy = (
        correct_dialects / len(dialect_ground_truth) if dialect_ground_truth else 0
    )

    dialect_results = {}
    for dialect, metrics in dialect_metrics.items():
        accuracy = metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0
        avg_wer = sum(metrics["wer"]) / len(metrics["wer"]) if metrics["wer"] else 0
        dialect_results[dialect] = {
            "accuracy": accuracy,
            "samples": metrics["total"],
            "wer": avg_wer,
        }

    logger.info(f"Overall WER: {overall_wer:.4f}")
    logger.info(f"Overall CER: {overall_cer:.4f}")
    logger.info(f"Overall Dialect Accuracy: {dialect_accuracy:.4f}")

    dialect_labels = sorted(set(dialect_ground_truth))
    cm = np.zeros((len(dialect_labels), len(dialect_labels)), dtype=int)

    dialect_to_idx = {dialect: i for i, dialect in enumerate(dialect_labels)}
    for true, pred in zip(dialect_ground_truth, dialect_predictions):
        if pred in dialect_to_idx:
            cm[dialect_to_idx[true], dialect_to_idx[pred]] += 1

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", xticklabels=dialect_labels, yticklabels=dialect_labels
    )
    plt.title("Dialect Prediction Confusion Matrix")
    plt.ylabel("True Dialect")
    plt.xlabel("Predicted Dialect")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dialect_confusion_matrix.png"))

    results_df = pd.DataFrame(
        {
            "prediction": all_predictions,
            "ground_truth": all_ground_truth,
            "pred_dialect": dialect_predictions,
            "true_dialect": dialect_ground_truth,
            "wer": all_wer,
            "cer": all_cer,
            "dialect_correct": [
                p == l for p, l in zip(dialect_predictions, dialect_ground_truth)
            ],
        }
    )

    results_df.to_csv(
        os.path.join(output_dir, f"transcription_results_{split}.csv"), index=False
    )

    dialect_df = pd.DataFrame(
        [{"dialect": k, **v} for k, v in dialect_results.items()]
    ).sort_values("dialect")

    dialect_df.to_csv(
        os.path.join(output_dir, f"dialect_results_{split}.csv"), index=False
    )

    metrics = {
        "overall_wer": overall_wer,
        "overall_cer": overall_cer,
        "dialect_accuracy": dialect_accuracy,
        "num_samples": len(all_predictions),
        "dialect_breakdown": dialect_results,
    }

    with open(os.path.join(output_dir, f"metrics_summary_{split}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    plt.figure(figsize=(12, 8))
    sns.barplot(x="dialect", y="wer", data=dialect_df)
    plt.title("Word Error Rate by Dialect")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"dialect_wer_{split}.png"))

    plt.figure(figsize=(12, 8))
    sns.barplot(x="dialect", y="accuracy", data=dialect_df)
    plt.title("Dialect Recognition Accuracy by Dialect")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"dialect_accuracy_{split}.png"))

    return metrics


def collate_fn(batch):
    """
    Custom collate function for handling variable length inputs.
    """
    input_values = [item["input_values"] for item in batch]
    labels = [item["labels"] for item in batch]

    input_lengths = [len(x) for x in input_values]
    max_input_length = max(input_lengths)

    padded_input_values = []
    attention_mask = []

    for item, length in zip(input_values, input_lengths):
        padded = torch.nn.functional.pad(
            item, (0, max_input_length - length), mode="constant", value=0
        )
        padded_input_values.append(padded)

        mask = torch.ones(length)
        mask = torch.nn.functional.pad(
            mask, (0, max_input_length - length), mode="constant", value=0
        )
        attention_mask.append(mask)

    input_values = torch.stack(padded_input_values)
    attention_mask = torch.stack(attention_mask)

    label_lengths = [len(x) for x in labels]
    max_label_length = max(label_lengths)

    padded_labels = []

    for item, length in zip(labels, label_lengths):
        padded = torch.nn.functional.pad(
            item, (0, max_label_length - length), mode="constant", value=-100
        )
        padded_labels.append(padded)

    labels = torch.stack(padded_labels)

    return {
        "input_values": input_values,
        "attention_mask": attention_mask,
        "labels": labels,
    }


if __name__ == "__main__":
    args = parse_args()
    evaluate_model(args.checkpoint, args.output_dir, args.batch_size, args.split)
