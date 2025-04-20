import os
import re
import datasets as dts
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm
import logging
import json
from typing import Dict, Any, List, Optional, Iterator

from config import get_config, logger, CacheConfig


def remove_special_characters(
    batch: Dict[str, List[Optional[str]]], chars_to_remove_regex: str
) -> Dict[str, List[str]]:
    """Cleans transcripts by removing special characters and tags."""
    clean_transcripts = []
    for transcript in batch["transcript"]:
        # Skip None values
        if transcript is None:
            clean_transcripts.append("")
            continue

        # Remove all content within angle brackets < > (potential noise tags)
        transcript = re.sub(r"<[^>]*>", "", transcript)

        # Remove all content within square brackets [ ] (potential noise tags)
        transcript = re.sub(r"\[[^\]]*\]", "", transcript)

        # Handle English code switching: remove braces but keep the English text
        # Example: {okay} -> okay
        transcript = re.sub(r"\{([^\}]*)\}", r"\1", transcript)

        # Remove specified punctuation and special characters
        transcript = re.sub(chars_to_remove_regex, "", transcript)

        # Clean up extra whitespace (multiple spaces -> single space)
        transcript = re.sub(r"\s+", " ", transcript).strip()

        clean_transcripts.append(transcript)

    batch["transcript"] = clean_transcripts
    return batch


def get_training_corpus(
    dataset: dts.Dataset, cache_config: CacheConfig
) -> Iterator[List[str]]:
    """Yields cleaned transcripts from the dataset for tokenizer training."""
    chars_to_remove_regex = r"[\,\?\.\!\-\;\:\"\“\%\‘\”\\'\(\)]"
    logger.info("Cleaning transcripts for tokenizer training...")
    cleaned_dataset = dataset.map(
        lambda batch: remove_special_characters(batch, chars_to_remove_regex),
        batched=True,
        num_proc=cache_config.num_proc,
        desc="Cleaning transcripts",
    )
    logger.info("Yielding cleaned transcripts...")
    for item in tqdm(cleaned_dataset, desc="Iterating cleaned transcripts"):
        if item["transcript"]:
            yield item["transcript"]


def main():
    logger.info("--- Starting BPE Tokenizer Training ---")

    config, cache_config = get_config("transcription")
    output_dir = config.output_dir
    tokenizer_save_path = os.path.join(output_dir, "bpe_tokenizer")
    os.makedirs(tokenizer_save_path, exist_ok=True)
    logger.info(f"Tokenizer will be saved to: {tokenizer_save_path}")

    dataset_name = config.dataset.name
    dataset_config = config.dataset.configs
    logger.info(f"Loading dataset: {dataset_name} ({dataset_config})")
    try:
        ds = dts.load_dataset(
            dataset_name,
            dataset_config,
            split="train",
            cache_dir=config.cache_dir,
            num_proc=cache_config.num_proc,
            revision=config.dataset.revision,
        )
        logger.info(f"Loaded {len(ds)} training samples.")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    if "district" in ds.column_names:
        logger.info("Renaming 'district' column to 'dialect'")
        ds = ds.rename_column("district", "dialect")

    logger.info("Extracting unique dialects from training split...")
    try:
        if "dialect" not in ds.column_names:
            logger.error("Column 'dialect' not found in the dataset.")
            return
        unique_dialects = set(ds.unique("dialect"))
        unique_dialects = {str(d) for d in unique_dialects if d}
        if not unique_dialects:
            logger.warning(
                "No valid dialects found in the training split. Proceeding without dialect tokens."
            )
        else:
            logger.info(
                f"Found {len(unique_dialects)} unique dialects in train split: {sorted(list(unique_dialects))}"
            )
    except Exception as e:
        logger.error(f"Error extracting dialects: {e}")
        unique_dialects = set()

    dialect_tokens = [f"<{dialect}>" for dialect in sorted(list(unique_dialects))]
    special_tokens = ["[PAD]", "[UNK]", "|"] + dialect_tokens
    logger.info(f"Defined {len(special_tokens)} special tokens: {special_tokens}")

    vocab_size = 900
    tokenizer = ByteLevelBPETokenizer()

    logger.info(f"Training BPE tokenizer with target vocab_size={vocab_size}...")
    try:
        tokenizer.train_from_iterator(
            get_training_corpus(ds, cache_config),
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=special_tokens,
            show_progress=True,
        )
        logger.info("Tokenizer training complete.")
        logger.info(f"Actual trained vocab size: {tokenizer.get_vocab_size()}")
    except Exception as e:
        logger.error(f"Tokenizer training failed: {e}")
        return

    try:
        tokenizer.save_model(tokenizer_save_path)
        logger.info(
            f"Tokenizer model (vocab.json, merges.txt) saved to {tokenizer_save_path}"
        )

        special_tokens_file = os.path.join(tokenizer_save_path, "special_tokens.json")
        with open(special_tokens_file, "w", encoding="utf-8") as f:
            json.dump(special_tokens, f, indent=2, ensure_ascii=False)
        logger.info(f"Special tokens list saved to {special_tokens_file}")

    except Exception as e:
        logger.error(f"Failed to save tokenizer: {e}")

    logger.info("--- BPE Tokenizer Training Finished ---")


if __name__ == "__main__":
    main()
