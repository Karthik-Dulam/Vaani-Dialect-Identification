import os
import re
import datasets as dts
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm
import logging
import json

# Assuming config.py is in the same directory or accessible via PYTHONPATH
# Adjust the import path if necessary
try:
    from config import TRANSCRIPTION_CONFIG, TRANSCRIPTION_CACHE_CONFIG, logger
except ImportError:
    print("Error: Could not import config. Ensure config.py is accessible.")
    # Fallback basic logger if config import fails
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # Define fallback configs if needed, or exit
    exit("Config import failed. Please check PYTHONPATH or script location.")


# --- Copy necessary function from data.py ---
# (Ensure this function stays in sync with data.py if it changes there)
def remove_special_characters(batch, chars_to_remove_regex):
    """Cleans transcripts by removing special characters and tags."""
    clean_transcripts = []
    for transcript in batch["transcript"]:
        # Skip None values
        if transcript is None:
            clean_transcripts.append("")
            continue

        # Remove all content within angle brackets < > (potential noise tags)
        transcript = re.sub(r'<[^>]*>', '', transcript)

        # Remove all content within square brackets [ ] (potential noise tags)
        transcript = re.sub(r'\[[^\]]*\]', '', transcript)

        # Handle English code switching: remove braces but keep the English text
        # Example: {okay} -> okay
        transcript = re.sub(r'\{([^\}]*)\}', r'\1', transcript)

        # Remove specified punctuation and special characters
        transcript = re.sub(chars_to_remove_regex, "", transcript)

        # Clean up extra whitespace (multiple spaces -> single space)
        transcript = re.sub(r'\s+', ' ', transcript).strip()

        clean_transcripts.append(transcript)

    batch["transcript"] = clean_transcripts
    return batch
# --- End copied function ---

def get_training_corpus(dataset):
    """Yields cleaned transcripts from the dataset for tokenizer training."""
    # Define the regex based on data.py's usage
    # Ensure this matches the cleaning done before model training
    chars_to_remove_regex = r"[\,\?\.\!\-\;\:\"\“\%\‘\”\\'\(\)]" # Added parentheses
    logger.info("Cleaning transcripts for tokenizer training...")
    # Process in batches for efficiency
    # Use dataset.map for potentially faster parallel processing if num_proc > 1
    cleaned_dataset = dataset.map(
        lambda batch: remove_special_characters(batch, chars_to_remove_regex),
        batched=True,
        num_proc=TRANSCRIPTION_CACHE_CONFIG.get("num_proc", 1), # Use num_proc from config
        desc="Cleaning transcripts"
    )
    logger.info("Yielding cleaned transcripts...")
    # Yield transcripts one by one
    for item in tqdm(cleaned_dataset, desc="Iterating cleaned transcripts"):
        # Ensure transcript is not None before yielding
        if item["transcript"]:
            yield item["transcript"]


def main():
    logger.info("--- Starting BPE Tokenizer Training ---")

    # 1. Load Config & Setup Output Dir
    output_dir = TRANSCRIPTION_CONFIG["output_dir"]
    tokenizer_save_path = os.path.join(output_dir, "bpe_tokenizer")
    os.makedirs(tokenizer_save_path, exist_ok=True)
    logger.info(f"Tokenizer will be saved to: {tokenizer_save_path}")

    # 2. Load Data (Train split only)
    dataset_name = TRANSCRIPTION_CONFIG["dataset"]["name"]
    dataset_config = TRANSCRIPTION_CONFIG["dataset"]["configs"]
    logger.info(f"Loading dataset: {dataset_name} ({dataset_config})")
    try:
        ds = dts.load_dataset(
            dataset_name,
            dataset_config,
            split="train", # Use only train split for tokenizer training
            cache_dir=TRANSCRIPTION_CONFIG.get("cache_dir"), # Use .get for safety
            num_proc=TRANSCRIPTION_CACHE_CONFIG.get("num_proc", 1),
            revision=TRANSCRIPTION_CONFIG["dataset"].get("revision"),
        )
        logger.info(f"Loaded {len(ds)} training samples.")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return # Exit if dataset loading fails

    # Rename 'district' to 'dialect' if necessary (as done in data.py)
    if "district" in ds.column_names:
         logger.info("Renaming 'district' column to 'dialect'")
         ds = ds.rename_column("district", "dialect")

    # 3. Clean Transcripts (Handled by get_training_corpus)

    # 4. Extract Dialects (from train split)
    logger.info("Extracting unique dialects from training split...")
    try:
        # Ensure 'dialect' column exists
        if "dialect" not in ds.column_names:
             logger.error("Column 'dialect' not found in the dataset.")
             return
        unique_dialects = set(ds.unique("dialect"))
        # Filter out potential None or empty strings
        unique_dialects = {str(d) for d in unique_dialects if d}
        if not unique_dialects:
             logger.warning("No valid dialects found in the training split. Proceeding without dialect tokens.")
        else:
             logger.info(f"Found {len(unique_dialects)} unique dialects in train split: {sorted(list(unique_dialects))}")
    except Exception as e:
        logger.error(f"Error extracting dialects: {e}")
        unique_dialects = set() # Default to empty set on error


    # 5. Define Special Tokens
    dialect_tokens = [f"<{dialect}>" for dialect in sorted(list(unique_dialects))]
    # Using standard HF special tokens + word delimiter used by Wav2Vec2CTCTokenizer
    # Ensure PAD is first (often index 0)
    special_tokens = ["[PAD]", "[UNK]", "|"] + dialect_tokens # Removed CLS, SEP, MASK as they might not be needed for CTC
    logger.info(f"Defined {len(special_tokens)} special tokens: {special_tokens}")

    # 6. Initialize & Train Tokenizer
    vocab_size = 900 # As requested
    tokenizer = ByteLevelBPETokenizer()

    logger.info(f"Training BPE tokenizer with target vocab_size={vocab_size}...")
    try:
        tokenizer.train_from_iterator(
            get_training_corpus(ds),
            vocab_size=vocab_size,
            min_frequency=2, # Standard default, prevents rare tokens
            special_tokens=special_tokens,
            show_progress=True,
        )
        logger.info("Tokenizer training complete.")
        logger.info(f"Actual trained vocab size: {tokenizer.get_vocab_size()}")
    except Exception as e:
        logger.error(f"Tokenizer training failed: {e}")
        return # Exit if training fails

    # 7. Save Tokenizer
    try:
        tokenizer.save_model(tokenizer_save_path) # Saves vocab.json and merges.txt
        logger.info(f"Tokenizer model (vocab.json, merges.txt) saved to {tokenizer_save_path}")

        # Optional: Save special tokens list separately for reference
        special_tokens_file = os.path.join(tokenizer_save_path, "special_tokens.json")
        with open(special_tokens_file, 'w', encoding='utf-8') as f:
            json.dump(special_tokens, f, indent=2, ensure_ascii=False)
        logger.info(f"Special tokens list saved to {special_tokens_file}")

    except Exception as e:
        logger.error(f"Failed to save tokenizer: {e}")

    logger.info("--- BPE Tokenizer Training Finished ---")


if __name__ == "__main__":
    # Setup basic logging if running as script
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
