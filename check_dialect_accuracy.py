import os
import argparse
import torch
import re
import numpy as np
from tqdm import tqdm
from transformers import (
    Wav2Vec2Processor,
    WhisperProcessor,
    WhisperTokenizer,
    Wav2Vec2FeatureExtractor,
)
from models_vaani import TranscriptionLitModel
from data import VaaniDataset, ResumableDataLoader, load_datasets
from config import get_config, set_seeds, logger


def extract_dialect(text):
    match = re.search(r"<([^>]+)>", text)
    return match.group(1) if match else None


def evaluate_dialect_accuracy(
    config, cache_config, checkpoint_path, num_samples=100, split="validation"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seeds(cache_config.random_seed)
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
    label_to_id = {d: i for i, d in enumerate(all_dialects)}
    eval_dataset = VaaniDataset(
        eval_ds,
        processor,
        label_to_id,
        is_train=False,
        config=config,
        cache_config=cache_config,
        task="transcription",
    )
    if num_samples < len(eval_dataset):
        indices = torch.randperm(len(eval_dataset))[:num_samples].tolist()
        subset = torch.utils.data.Subset(eval_dataset, indices)
    else:
        subset = eval_dataset
    eval_loader = torch.utils.data.DataLoader(
        subset, batch_size=16, shuffle=False, collate_fn=collate_fn
    )
    dialect_correct = total = wer_total = 0
    for batch in tqdm(eval_loader, desc=f"Evaluating {split} set"):
        with torch.no_grad():
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            outputs = model.model(
                input_values=batch["input_values"],
                attention_mask=batch.get("attention_mask"),
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
                    total += 1
                    if pred_dialect == label_dialect:
                        dialect_correct += 1

                    print(f"\nTrue: '{label}'")
                    print(f"Pred: '{pred}'")
                    print(
                        f"Dialect correct: {pred_dialect == label_dialect} (True: {label_dialect}, Pred: {pred_dialect or 'None'})"
                    )

                    import jiwer

                    pred_text = re.sub(r"<[^>]+>\s*", "", pred)
                    label_text = re.sub(r"<[^>]+>\s*", "", label)
                    wer = jiwer.wer(label_text, pred_text)
                    wer_total += wer
                    print(f"WER: {wer:.4f}")

                    print("-" * 80)

    dialect_accuracy = dialect_correct / total if total > 0 else 0
    avg_wer = wer_total / total if total > 0 else 0

    print(f"\n{'=' * 40} SUMMARY {'=' * 40}")
    print(f"Evaluated {total} samples from {split} set")
    print(f"Dialect Accuracy: {dialect_accuracy:.4f} ({dialect_correct}/{total})")
    print(f"Average WER: {avg_wer:.4f}")


def collate_fn(batch):
    """Collate function for variable length inputs."""
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
            item,
            (0, max_label_length - length),
            mode="constant",
            value=-100,  # Ignore padding in loss computation
        )
        padded_labels.append(padded)

    labels = torch.stack(padded_labels)

    return {
        "input_values": input_values,
        "attention_mask": attention_mask,
        "labels": labels,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check dialect accuracy of a trained ASR model"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument(
        "--split", type=str, default="validation", choices=["test", "validation"]
    )
    args = parser.parse_args()
    config, cache_config = get_config("transcription")
    evaluate_dialect_accuracy(
        config, cache_config, args.checkpoint, args.num_samples, args.split
    )
