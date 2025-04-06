import os
from config import CONFIG, CACHE_SENSITIVE_CONFIG, set_seeds, logger
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint 
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from transformers import Wav2Vec2FeatureExtractor, AutoFeatureExtractor
from huggingface_hub import login
import copy  # Added import
import polars
import wandb

from models import LitModel
from data import VaaniDataset, ResumableDataLoader, load_datasets
from evaluate import evaluate_model
import json

# Enable Tensor Core optimization
torch.set_float32_matmul_precision("high")


def collate_fn(batch, feature_extractor):
    """
    Custom collate function that handles preprocessing and feature extraction for classification.
    """
    # Extract audio samples and labels from batch
    audio_samples = [item["audio"]["array"] for item in batch]
    labels = [item["label"] for item in batch]
    
    # Process audio samples with the feature extractor
    max_length = 16000 * CONFIG["model_config"]["num_seconds"]
    processed_inputs = feature_extractor(
        audio_samples,
        sampling_rate=16000,
        padding="longest",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=True,
    )
    
    # Convert labels to tensor
    labels = torch.tensor(labels, dtype=torch.long)
    
    # Return features based on the model type
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


def main():
    # Initial setup
    set_seeds()
    login(CONFIG["hf_token"])
    
    # Set up wandb logging
    wandb_api_key = os.environ.get("WANDB_KEY_K")
    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key
        logger.info("Found WANDB API key, enabling wandb logging")
    else:
        logger.warning("WANDB_KEY_K environment variable not found. wandb logging will be disabled.")
    
    logger.info(
        "Starting training with configuration: %s", {**CONFIG, **CACHE_SENSITIVE_CONFIG}
    )

    # Create base output directory
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # Load datasets
    train_ds, test_ds, val_ds = load_datasets(CONFIG, CACHE_SENSITIVE_CONFIG)

    col = CACHE_SENSITIVE_CONFIG["task_type"]

    all_labels = (
        set(train_ds.unique(col)) | set(test_ds.unique(col)) | set(val_ds.unique(col))
    )
    labels = sorted(all_labels)

    label_to_id = {label: i for i, label in enumerate(labels)}
    id_to_label = {i: label for i, label in enumerate(labels)}

    # print the distribution of the labels using pandas in training, validation and test datasets
    train_df = polars.Series(train_ds[col])
    val_df = polars.Series(val_ds[col])
    test_df = polars.Series(test_ds[col])

    logger.info(f"Training dataset distribution: \n{train_df.value_counts()}")
    logger.info(f"Validation dataset distribution: \n{val_df.value_counts()}")
    logger.info(f"Test dataset distribution: \n{test_df.value_counts()}")

    if CONFIG["model"]["name"] == "facebook/w2v-bert-2.0":
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            CONFIG["model"]["feature_extractor"]
        )
    else:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            CONFIG["model"]["feature_extractor"]
        )

    logger.info(f"Loaded feature extractor: {CONFIG['model']['feature_extractor']}")

    # Initialize Lightning module with custom model
    lightning_model = LitModel(
        num_labels=len(labels),
        model_name=CONFIG["model"]["name"],
        model_config=CONFIG["model_config"],
        learning_rate=CONFIG["training"]["learning_rate"],
        weight_decay=CONFIG["training"]["weight_decay"],
        classifier_only=CONFIG["training"]["classifier_only"],
    )

    logger.info(f"Loaded model: {CONFIG['model']['name']}")

    # Create datasets
    train_dataset = VaaniDataset(
        train_ds,
        feature_extractor,
        label_to_id,
        is_train=True,
        config=CONFIG,
        cache_config=CACHE_SENSITIVE_CONFIG,
    )
    val_dataset = VaaniDataset(
        val_ds,
        feature_extractor,
        label_to_id,
        is_train=False,
        config=CONFIG,
        cache_config=CACHE_SENSITIVE_CONFIG,
    )
    test_dataset = VaaniDataset(
        test_ds,
        feature_extractor,
        label_to_id,
        is_train=False,
        config=CONFIG,
        cache_config=CACHE_SENSITIVE_CONFIG,
    )

    # Create dataloaders
    train_loader = ResumableDataLoader(
        train_dataset,
        batch_size=CONFIG["training"]["per_device_train_batch_size"],
        shuffle=True,
        num_workers=CACHE_SENSITIVE_CONFIG["num_proc"],
        pin_memory=True,
        persistent_workers=True,
        collate_fn=lambda batch: collate_fn(batch, feature_extractor),
    )
    val_loader = ResumableDataLoader(
        val_dataset,
        batch_size=CONFIG["training"]["per_device_train_batch_size"],
        num_workers=CACHE_SENSITIVE_CONFIG["num_proc"],
        pin_memory=True,
        persistent_workers=True,
        collate_fn=lambda batch: collate_fn(batch, feature_extractor),
    )
    test_loader = ResumableDataLoader(
        test_dataset,
        batch_size=CONFIG["training"]["per_device_train_batch_size"],
        num_workers=CACHE_SENSITIVE_CONFIG["num_proc"],
        pin_memory=True,
        persistent_workers=True,
        collate_fn=lambda batch: collate_fn(batch, feature_extractor),
    )

    # Setup loggers
    tb_logger = TensorBoardLogger(CONFIG["output_dir"], name="lightning_logs")
    
    # Ensure the log directory exists
    os.makedirs(tb_logger.log_dir, exist_ok=True)
    
    loggers = [tb_logger]
    
    # Add wandb logger if API key is available
    if wandb_api_key:
        wandb_logger = WandbLogger(
            project=f"{CONFIG['name']}-{CACHE_SENSITIVE_CONFIG['task_type']}",
            name=f"{CONFIG['model']['name'].split('/')[-1]}-v{tb_logger.version}",
            log_model=True,
            save_dir=CONFIG["output_dir"]
        )
        # Log hyperparameters to wandb
        wandb_logger.log_hyperparams({
            **CONFIG,
            **CACHE_SENSITIVE_CONFIG
        })
        loggers.append(wandb_logger)

    checkpoint_callback = ModelCheckpoint(
        dirpath=CONFIG["output_dir"],
        filename=f"best-checkpoint-v{tb_logger.version}",
        save_top_k=1,
        verbose=True,
        monitor="val_acc",
        mode="max",
    )

    # Save configurations in the versioned log directory
    config_path = os.path.join(tb_logger.log_dir, f"config_v{tb_logger.version}.json")
    config_to_save = copy.deepcopy(CONFIG)
    config_to_save["version"] = tb_logger.version  # Add version to config

    with open(config_path, "w") as f:
        json.dump(
            {
                "training_config": config_to_save,
                "cache_sensitive_config": CACHE_SENSITIVE_CONFIG,
                "tensorboard_version": tb_logger.version,  # Add version at top level too
            },
            f,
            indent=4,
        )
    logger.info(f"Saved configurations to {config_path}")

    # Initialize trainer with multiple loggers
    trainer = pl.Trainer(
        max_epochs=CONFIG["training"]["num_train_epochs"],
        accelerator="auto",
        devices=[0],
        logger=loggers,  # Use both loggers
        callbacks=[checkpoint_callback],
        precision="16-mixed",
        val_check_interval=CONFIG["training"]["eval_steps"],
        log_every_n_steps=CONFIG["training"]["logging_steps"],
    )

    # Check for existing checkpoints
    if CONFIG["resume"]:
        ckpt_path = os.path.join(
            CONFIG["output_dir"], f"best-checkpoint-v{tb_logger.version}.ckpt"
        )
        if os.path.exists(ckpt_path):
            logger.info(f"Resuming training from checkpoint: {ckpt_path}")
        else:
            logger.warning(f"Checkpoint not found: {ckpt_path} - starting from scratch")
            ckpt_path = None
    else:
        ckpt_path = None

    # Train model
    trainer.fit(lightning_model, train_loader, val_loader, ckpt_path=ckpt_path)

    logger.info("Training complete. Evaluating model...")

    # Evaluate model using evaluate.py instead of trainer.test
    checkpoint_path = os.path.join(
        CONFIG["output_dir"], f"best-checkpoint-v{tb_logger.version}.ckpt"
    )
    evaluate_model(checkpoint_path)

    logger.info("Evaluation complete.")

    # Clean up wandb if it was used
    if wandb_api_key and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
