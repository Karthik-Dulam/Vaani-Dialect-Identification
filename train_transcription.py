import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
)
from huggingface_hub import login
import copy
import wandb
import json

from models import TranscriptionLitModel
from data import VaaniDataset, ResumableDataLoader, load_datasets
from config import TRANSCRIPTION_CONFIG, TRANSCRIPTION_CACHE_CONFIG, set_seeds, logger

# Enable Tensor Core optimization
torch.set_float32_matmul_precision("high")

# Create output directory if it doesn't exist
os.makedirs(TRANSCRIPTION_CONFIG["output_dir"], exist_ok=True)


def main():
    # Initial setup
    set_seeds()
    login(TRANSCRIPTION_CONFIG["hf_token"])

    # Set up wandb logging
    wandb_api_key = os.environ.get("WANDB_KEY_K")
    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key
        logger.info("Found WANDB API key, enabling wandb logging")
    else:
        logger.warning(
            "WANDB_KEY_K environment variable not found. wandb logging will be disabled."
        )

    logger.info(
        "Starting transcription training with configuration: %s",
        {**TRANSCRIPTION_CONFIG, **TRANSCRIPTION_CACHE_CONFIG},
    )

    # Create output directory
    os.makedirs(TRANSCRIPTION_CONFIG["output_dir"], exist_ok=True)

    # Load datasets
    train_ds, test_ds, val_ds, vocab_dict, dialects = load_datasets(
        TRANSCRIPTION_CONFIG, TRANSCRIPTION_CACHE_CONFIG
    )

    label_to_id = {label: i for i, label in enumerate(dialects)}



    vocab_file = os.path.join(TRANSCRIPTION_CONFIG["output_dir"], "vocab.json")
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(
            vocab_dict, f, ensure_ascii=False, indent=2
        )  

    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=vocab_file,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
        encoding="utf-8",
    )

    TRANSCRIPTION_CONFIG["model_config"]["vocab_size"] = len(tokenizer)
    TRANSCRIPTION_CONFIG["model_config"]["pad_token_id"] = tokenizer.pad_token_id


    added_tokens = [token for token in vocab_dict if token not in tokenizer.get_vocab()]
    if added_tokens:
        logger.warning(
            f"The following tokens might not have been added correctly by loading the vocab file: {added_tokens}"
        )
        logger.warning(
            "Consider using tokenizer.add_tokens() and model.resize_token_embeddings() if issues arise."
        )

    print(tokenizer.get_vocab())

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )

    # Create datasets
    train_dataset = VaaniDataset(
        train_ds,
        processor,
        label_to_id,
        is_train=True,
        config=TRANSCRIPTION_CONFIG,
        cache_config=TRANSCRIPTION_CACHE_CONFIG,
        task="transcription",
    )

    val_dataset = VaaniDataset(
        val_ds,
        processor,
        label_to_id,
        is_train=False,
        config=TRANSCRIPTION_CONFIG,
        cache_config=TRANSCRIPTION_CACHE_CONFIG,
        task="transcription",
    )

    test_dataset = VaaniDataset(
        test_ds,
        processor,
        label_to_id,
        is_train=False,
        config=TRANSCRIPTION_CONFIG,
        cache_config=TRANSCRIPTION_CACHE_CONFIG,
        task="transcription",
    )

    # Create dataloaders
    train_loader = ResumableDataLoader(
        train_dataset,
        batch_size=TRANSCRIPTION_CONFIG["training"]["per_device_train_batch_size"],
        shuffle=True,
        num_workers=TRANSCRIPTION_CACHE_CONFIG["num_proc"],
        pin_memory=True,
        persistent_workers=True,
        collate_fn=lambda batch: collate_fn(batch, processor),
    )

    val_loader = ResumableDataLoader(
        val_dataset,
        batch_size=TRANSCRIPTION_CONFIG["training"]["per_device_train_batch_size"],
        num_workers=TRANSCRIPTION_CACHE_CONFIG["num_proc"],
        pin_memory=True,
        persistent_workers=True,
        collate_fn=lambda batch: collate_fn(batch, processor),
    )

    test_loader = ResumableDataLoader(
        test_dataset,
        batch_size=TRANSCRIPTION_CONFIG["training"]["per_device_train_batch_size"],
        num_workers=TRANSCRIPTION_CACHE_CONFIG["num_proc"],
        pin_memory=True,
        persistent_workers=True,
        collate_fn=lambda batch: collate_fn(batch, processor),
    )

    # Setup loggers
    tb_logger = TensorBoardLogger(
        TRANSCRIPTION_CONFIG["output_dir"], name="lightning_logs"
    )

    # Ensure the log directory exists
    os.makedirs(tb_logger.log_dir, exist_ok=True)

    loggers = [tb_logger]

    # Add wandb logger if API key is available
    if wandb_api_key:
        wandb_logger = WandbLogger(
            project=f"{TRANSCRIPTION_CONFIG['name']}",
            name=f"{TRANSCRIPTION_CONFIG['model']['name'].split('/')[-1]}-v{tb_logger.version}",
            log_model=True,
            save_dir=TRANSCRIPTION_CONFIG["output_dir"],
        )
        # Log hyperparameters to wandb
        wandb_logger.log_hyperparams(
            {**TRANSCRIPTION_CONFIG, **TRANSCRIPTION_CACHE_CONFIG}
        )
        loggers.append(wandb_logger)

    # Add callbacks
    checkpoint_wer_callback = ModelCheckpoint(
        dirpath=TRANSCRIPTION_CONFIG["output_dir"],
        filename=f"best-wer-checkpoint-v{tb_logger.version}",
        save_top_k=2,
        verbose=True,
        monitor="val_wer",
        mode="min",
    )

    checkpoint_dialect_callback = ModelCheckpoint(
        dirpath=TRANSCRIPTION_CONFIG["output_dir"],
        filename=f"best-dialect-checkpoint-v{tb_logger.version}",
        save_top_k=2,
        verbose=True,
        monitor="val_dialect_acc",
        mode="max",
    )


    # early_stopping_callback = EarlyStopping(
    #     monitor="val_dialect_acc", patience=3, verbose=True, mode="max"
    # )

    # Save configurations
    config_path = os.path.join(tb_logger.log_dir, f"config_v{tb_logger.version}.json")
    config_to_save = copy.deepcopy(TRANSCRIPTION_CONFIG)
    config_to_save["version"] = tb_logger.version

    with open(config_path, "w") as f:
        json.dump(
            {
                "training_config": config_to_save,
                "cache_sensitive_config": TRANSCRIPTION_CACHE_CONFIG,
                "tensorboard_version": tb_logger.version,
            },
            f,
            indent=4,
        )
    logger.info(f"Saved configurations to {config_path}")

    logger.info(
        "Initializing model with vocab_size=%d", len(processor.tokenizer.get_vocab())
    )
    logger.info("Model config: %s", TRANSCRIPTION_CONFIG["model_config"])

    lightning_model = TranscriptionLitModel(
        model_name=TRANSCRIPTION_CONFIG["model"]["name"],
        model_config=TRANSCRIPTION_CONFIG["model_config"],
        processor=processor,
        learning_rate=TRANSCRIPTION_CONFIG["training"]["learning_rate"],
        weight_decay=TRANSCRIPTION_CONFIG["training"]["weight_decay"],
    )

    # Save processor for later use
    processor.save_pretrained(TRANSCRIPTION_CONFIG["output_dir"])

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=TRANSCRIPTION_CONFIG["training"]["num_train_epochs"],
        accelerator="auto",
        devices=[0,1],
        logger=loggers,
        callbacks=[
            checkpoint_wer_callback,
            checkpoint_dialect_callback,
            # early_stopping_callback,
        ],
        precision="16-mixed",
        val_check_interval=TRANSCRIPTION_CONFIG["training"]["eval_steps"],
        log_every_n_steps=TRANSCRIPTION_CONFIG["training"]["logging_steps"],
        gradient_clip_val=1.0,  # Add gradient clipping for stability
    )

    # Check for existing checkpoints
    if TRANSCRIPTION_CONFIG["resume"]:
        ckpt_path = os.path.join(
            TRANSCRIPTION_CONFIG["output_dir"],
            f"best-dialect-checkpoint-v{tb_logger.version}.ckpt",
        )
        if os.path.exists(ckpt_path):
            logger.info(f"Resuming training from checkpoint: {ckpt_path}")
        else:
            # Try wer checkpoint
            wer_ckpt_path = os.path.join(
                TRANSCRIPTION_CONFIG["output_dir"],
                f"best-wer-checkpoint-v{tb_logger.version}.ckpt",
            )
            if os.path.exists(wer_ckpt_path):
                logger.info(f"Resuming training from WER checkpoint: {wer_ckpt_path}")
                ckpt_path = wer_ckpt_path
            else:
                logger.warning(f"No checkpoints found - starting from scratch")
                ckpt_path = None
    else:
        ckpt_path = None

    # Train model
    trainer.fit(lightning_model, train_loader, val_loader, ckpt_path=ckpt_path)

    logger.info("Training complete. Evaluating model...")

    # Evaluate on test set
    trainer.test(lightning_model, test_loader)

    logger.info("Evaluation complete.")

    # Clean up wandb if it was used
    if wandb_api_key and wandb.run is not None:
        wandb.finish()


def collate_fn(batch, processor):
    """
    Custom collate function that handles preprocessing and tokenization.
    """

    # Extract audio and transcripts from batch
    audio_samples = [item["audio"]["array"] for item in batch]
    transcripts = [item["transcript"] if item["transcript"] is not None else "" for item in batch]
    dialects = [item["dialect"] for item in batch]
    sampling_rates = [item["audio"]["sampling_rate"] for item in batch]


    # Process audio samples with the processor
    processed_inputs = processor(
        audio_samples,
        sampling_rate=16_000,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    )

    # Process transcripts with the tokenizer
    target_transcriptions = [f"<{dialect}> {transcript}".strip() for dialect, transcript in zip(dialects, transcripts)]
    tokenizer_output = processor.tokenizer(
        text=target_transcriptions,
        padding="longest",
        return_tensors="pt",
    )

    labels = tokenizer_output.input_ids

    # Replace padding token id with -100 for loss calculation
    # labels = torch.where(labels == processor.tokenizer.pad_token_id, -100, labels)

    
    return {
        "input_values": processed_inputs.input_values,
        "attention_mask": processed_inputs.attention_mask,
        "labels": labels
    }


if __name__ == "__main__":
    main()
