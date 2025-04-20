import os
import time
from config import get_config, set_seeds, logger, generate_output_dir, Config, CacheConfig
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint 
from pytorch_lightning.loggers import WandbLogger
from transformers import Wav2Vec2FeatureExtractor, AutoFeatureExtractor
from huggingface_hub import login
import copy
import polars
import wandb
from dataclasses import asdict  # Added for dataclass serialization
from typing import Dict, Any, List, Optional

from models_vaani import LitModel
from data import VaaniDataset, load_datasets
import json

torch.set_float32_matmul_precision("high")


def collate_fn(batch: List[Dict[str, Any]], feature_extractor: Any, config: Config) -> Dict[str, torch.Tensor]:
    """
    Custom collate function that handles preprocessing and feature extraction for classification.
    config: configuration dict containing model_config
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
    input_vals = getattr(processed_inputs, "input_features", processed_inputs.input_values)
    return {"input_values": input_vals, "attention_mask": processed_inputs.attention_mask, "label": labels}


def prepare_loader(raw_dataset: Any, feature_extractor: Any, label_to_id: Dict[str, int], config: Config, cache_config: CacheConfig, is_train: bool = False) -> DataLoader:
    ds = VaaniDataset(
        raw_dataset, feature_extractor, label_to_id,
        is_train=is_train, config=config, cache_config=cache_config
    )
    return DataLoader(
        ds,
        batch_size=config.training.per_device_train_batch_size,
        shuffle=is_train,
        num_workers=cache_config.num_proc,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=lambda b: collate_fn(b, feature_extractor, config),
    )


def run_experiment(config: Config, cache_config: CacheConfig, wandb_project: Optional[str] = None, run_name: Optional[str] = None) -> Dict[str, float]:
    """
    Run model training and evaluation given config and cache_config.
    Optionally log to specified WandB project/run.
    Returns evaluation metrics dict.
    """
    config = copy.deepcopy(config)
    cache_config = copy.deepcopy(cache_config)

    if config.output_dir is None:
        config.output_dir = generate_output_dir(config, cache_config)
    os.makedirs(config.output_dir, exist_ok=True)

    set_seeds(cache_config.random_seed)
    login(config.hf_token)
    ts = time.strftime("%Y%m%d-%H%M%S")

    loggers = []
    api_key = os.environ.get("WANDB_KEY_K")
    if (api_key):
        os.environ["WANDB_API_KEY"] = api_key
        wb_name = run_name or f"{config.model.name.split('/')[-1]}-{ts}"
        wb_prj = wandb_project or f"{config.name}-{cache_config.task_type}"
        wb_logger = WandbLogger(project=wb_prj, name=wb_name, log_model=True, save_dir=config.output_dir)
        wb_logger.log_hyperparams({**asdict(config), **asdict(cache_config)})
        loggers.append(wb_logger)
    
    train_ds, test_ds, val_ds = load_datasets(config, cache_config)

    col = cache_config.task_type

    all_labels = (
        set(train_ds.unique(col)) | set(test_ds.unique(col)) | set(val_ds.unique(col))
    )
    labels = sorted(all_labels)

    label_to_id = {label: i for i, label in enumerate(labels)}

    train_df = polars.Series(train_ds[col])
    val_df = polars.Series(val_ds[col])
    test_df = polars.Series(test_ds[col])

    for name, series in [("Training", train_df), ("Validation", val_df), ("Test", test_df)]:
        logger.info(f"{name} dataset distribution:\n{series.value_counts()}")

    Extractor = AutoFeatureExtractor if config.model.name == "facebook/w2v-bert-2.0" else Wav2Vec2FeatureExtractor
    feature_extractor = Extractor.from_pretrained(config.model.feature_extractor)

    logger.info(f"Loaded feature extractor: {config.model.feature_extractor}")

    lightning_model = LitModel(
        num_labels=len(labels),
        model_name=config.model.name,
        model_config=config.model_config,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        classifier_only=config.training.classifier_only,
    )

    logger.info(f"Loaded model: {config.model.name}")

    train_loader = prepare_loader(train_ds, feature_extractor, label_to_id, config, cache_config, is_train=True)
    val_loader   = prepare_loader(val_ds,   feature_extractor, label_to_id, config, cache_config)
    test_loader  = prepare_loader(test_ds,  feature_extractor, label_to_id, config, cache_config)

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.output_dir,
        filename=f"best-checkpoint-{ts}",
        save_top_k=1,
        verbose=True,
        monitor="val_acc",
        mode="max",
    )

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

    trainer = pl.Trainer(
        max_epochs=config.training.num_train_epochs,
        accelerator="auto",
        devices=[3,6,7],
        strategy="ddp_find_unused_parameters_true",
        logger=loggers,
        callbacks=[checkpoint_callback],
        precision="16-mixed",
        val_check_interval=config.training.eval_steps,
        log_every_n_steps=config.training.logging_steps,
    )

    if config.resume:
        ckpt_path = os.path.join(config.output_dir, f"best-checkpoint-{ts}.ckpt")
        if os.path.exists(ckpt_path):
            logger.info(f"Resuming training from checkpoint: {ckpt_path}")
        else:
            logger.warning(f"Checkpoint not found: {ckpt_path} - starting from scratch")
            ckpt_path = None
    else:
        ckpt_path = None

    result = trainer.fit(lightning_model, train_loader, val_loader, ckpt_path=ckpt_path)

    logger.info("Training complete.")


    metrics = trainer.callback_metrics
    logger.info(f"Obtained metrics from training: {metrics}")

    if api_key and wandb.run is not None:
        wandb.finish()
    return metrics


def main() -> None:
    config: Config
    cache_config: CacheConfig
    config, cache_config = get_config('classification')
    run_experiment(config, cache_config)

if __name__ == "__main__":
    main()
