import os
import torch
import time
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    WhisperProcessor,
    WhisperTokenizer,
)
from huggingface_hub import login
import copy
import wandb
import json
from torch.utils.data import DataLoader
from dataclasses import asdict

from models_vaani import TranscriptionLitModel
from data import VaaniDataset, load_datasets
from config import get_config, set_seeds, logger

torch.set_float32_matmul_precision("high")

config, cache_config = get_config('transcription')

def main():
    set_seeds(cache_config.random_seed)
    login(config.hf_token)

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
        {**asdict(config), **asdict(cache_config)},
    )

    os.makedirs(config.output_dir, exist_ok=True)

    train_ds, test_ds, val_ds, vocab_dict, dialects = load_datasets(
        config, cache_config
    )

    label_to_id = {label: i for i, label in enumerate(dialects)}

    is_whisper = "whisper" in config.model.name.lower()

    whisper_model_name = config.model.tokenizer
    text_tokenizer = WhisperTokenizer.from_pretrained(whisper_model_name)
    logger.info(f"Loaded Whisper tokenizer for text from: {whisper_model_name}")

    dialect_tokens = [f"<{d}>" for d in dialects]
    text_tokenizer.add_tokens(dialect_tokens, special_tokens=True)
    logger.info(f"Added {len(dialect_tokens)} dialect tokens to Whisper tokenizer")

    if is_whisper:
        whisper_processor = WhisperProcessor.from_pretrained(
            config.model.name
        )
        audio_processor = whisper_processor.feature_extractor
        whisper_processor.tokenizer = text_tokenizer
        processor_to_save = whisper_processor
        logger.info(
            f"Loaded Whisper feature extractor from {config.model.name}"
        )
    else:
        audio_processor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True,
        )
        processor_to_save = {"tokenizer": text_tokenizer, "feature_extractor": audio_processor}
        logger.info(f"Using Wav2Vec2 feature extractor with Whisper tokenizer")

    setattr(config.model_config, 'vocab_size', len(text_tokenizer))
    if not is_whisper:
        setattr(config.model_config, 'pad_token_id', text_tokenizer.pad_token_id)
        setattr(config.model_config, 'bos_token_id', text_tokenizer.bos_token_id)
        setattr(config.model_config, 'eos_token_id', text_tokenizer.eos_token_id)

    train_dataset = VaaniDataset(
        train_ds,
        processor_to_save,
        label_to_id,
        is_train=True,
        config=config,
        cache_config=cache_config,
        task="transcription",
    )

    val_dataset = VaaniDataset(
        val_ds,
        processor_to_save,
        label_to_id,
        is_train=False,
        config=config,
        cache_config=cache_config,
        task="transcription",
    )

    test_dataset = VaaniDataset(
        test_ds,
        processor_to_save,
        label_to_id,
        is_train=False,
        config=config,
        cache_config=cache_config,
        task="transcription",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.per_device_train_batch_size,
        shuffle=True,
        num_workers=cache_config.num_proc,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=lambda batch: collate_fn(batch, processor_to_save),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.per_device_train_batch_size,
        num_workers=cache_config.num_proc,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=lambda batch: collate_fn(batch, processor_to_save),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.per_device_train_batch_size * 4,
        num_workers=cache_config.num_proc,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=lambda batch: collate_fn(batch, processor_to_save),
    )

    loggers = []
    ts = time.strftime("%Y%m%d-%H%M%S")
    if wandb_api_key:
        wandb_logger = WandbLogger(
            project=f"{config.name}",
            name=f"{config.model.name.split('/')[-1]}-{ts}",
            log_model=True,
            save_dir=config.output_dir,
        )
        wandb_logger.log_hyperparams(
            {**asdict(config), **asdict(cache_config)}
        )
        loggers.append(wandb_logger)

    checkpoint_wer_callback = ModelCheckpoint(
        dirpath=config.output_dir,
        filename=f"best-wer-checkpoint-{ts}",
        save_top_k=2,
        verbose=True,
        monitor="val_wer",
        mode="min",
    )

    checkpoint_dialect_callback = ModelCheckpoint(
        dirpath=config.output_dir,
        filename=f"best-dialect-checkpoint-{ts}",
        save_top_k=2,
        verbose=True,
        monitor="val_dialect_acc",
        mode="max",
    )

    # early_stopping_callback = EarlyStopping(
    #     monitor="val_dialect_acc", patience=3, verbose=True, mode="max"
    # )

    config_path = os.path.join(config.output_dir, "config.json")
    config_to_save = copy.deepcopy(config)
    with open(config_path, "w") as f:
        json.dump(
            {
                "training_config": asdict(config_to_save),
                "cache_sensitive_config": asdict(cache_config),
            },
            f,
            indent=4,
        )
    logger.info(f"Saved configurations to {config_path}")

    logger.info(
        "Initializing model with vocab_size=%d", len(text_tokenizer.get_vocab())
    )
    logger.info("Model config: %s", config.model_config)

    lightning_model = TranscriptionLitModel(
        model_name=config.model.name,
        model_config=config.model_config,
        processor=processor_to_save,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    if isinstance(processor_to_save, WhisperProcessor):
        processor_to_save.save_pretrained(config.output_dir)
    else:
        text_tokenizer.save_pretrained(config.output_dir)
        audio_processor.save_pretrained(config.output_dir)

    trainer = pl.Trainer(
        max_epochs=config.training.num_train_epochs,
        accelerator="auto",
        devices=[3, 6, 7],
        strategy="ddp_find_unused_parameters_true",
        logger=loggers,
        callbacks=[
            checkpoint_wer_callback,
            checkpoint_dialect_callback,
            # early_stopping_callback,
        ],
        precision="bf16-true",
        val_check_interval=config.training.val_check_interval,
        log_every_n_steps=config.training.logging_steps,
        gradient_clip_val=1.0,
        # limit_train_batches=32,
        # limit_val_batches=64
    )

    if config.resume:
        ckpt_path = os.path.join(config.output_dir, f"best-dialect-checkpoint-{ts}.ckpt")
        if os.path.exists(ckpt_path):
            logger.info(f"Resuming training from checkpoint: {ckpt_path}")
        else:
            wer_ckpt_path = os.path.join(
                config.output_dir,
                f"best-wer-checkpoint-{ts}.ckpt",
            )
            if os.path.exists(wer_ckpt_path):
                logger.info(f"Resuming training from WER checkpoint: {wer_ckpt_path}")
                ckpt_path = wer_ckpt_path
            else:
                logger.warning(f"No checkpoints found - starting from scratch")
                ckpt_path = None
    else:
        ckpt_path = None

    trainer.fit(lightning_model, train_loader, test_loader, ckpt_path=ckpt_path)

    logger.info("Training complete. Evaluating model...")

    trainer.test(lightning_model, test_loader)

    logger.info("Evaluation complete.")

    if wandb_api_key and wandb.run is not None:
        wandb.finish()


def collate_fn(batch, processor):
    """
    Custom collate function that handles preprocessing and tokenization.
    Expects processor to be either a WhisperProcessor or a dict with {'tokenizer', 'feature_extractor'}
    """
    audio_samples = [item["audio"]["array"] for item in batch]
    transcripts = [item["transcript"] if item["transcript"] is not None else "" for item in batch]
    dialects = [item["dialect"] for item in batch]

    is_whisper = isinstance(processor, WhisperProcessor)

    if is_whisper:
        processed_inputs = processor.feature_extractor(
            audio_samples,
            sampling_rate=16_000,
            return_tensors="pt",
        )
        input_features = processed_inputs.input_features
        text_processor = processor.tokenizer
    else:
        processed_inputs = processor["feature_extractor"](
            audio_samples,
            sampling_rate=16_000,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features = processed_inputs.input_values
        attention_mask = processed_inputs.attention_mask
        text_processor = processor["tokenizer"]

    target_transcriptions = [
        f"<{dialect}> {transcript}".strip()
        for dialect, transcript in zip(dialects, transcripts)
    ]
    tokenizer_output = text_processor(
        text=target_transcriptions,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=448,
    )

    labels = tokenizer_output.input_ids
    labels[labels == text_processor.pad_token_id] = -100

    if is_whisper:
        return {
            "input_values": input_features,
            "labels": labels,
        }
    else:
        return {
            "input_values": input_features,
            "attention_mask": attention_mask,
            "labels": labels,
        }


if __name__ == "__main__":
    main()
