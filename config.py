import os
import logging
import pytorch_lightning as pl
import torch
import copy
import yaml  # Added for YAML support
from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Dict, List, Any, Optional, Union

hf_token = os.environ.get("HF_TOKEN_K")

hindi_districts = [
    # Bihar
    "Bihar_Araria",
    "Bihar_Begusarai",
    "Bihar_Bhagalpur",
    "Bihar_Darbhanga",
    "Bihar_EastChamparan",
    "Bihar_Gaya",
    "Bihar_Gopalganj",
    "Bihar_Jahanabad",
    "Bihar_Jamui",
    "Bihar_Kishanganj",
    "Bihar_Lakhisarai",
    "Bihar_Madhepura",
    "Bihar_Muzaffarpur",
    "Bihar_Purnia",
    "Bihar_Saharsa",
    "Bihar_Samastipur",
    "Bihar_Saran",
    "Bihar_Sitamarhi",
    "Bihar_Supaul",
    "Bihar_Vaishali",
    # Uttar Pradesh
    "UttarPradesh_Budaun",
    "UttarPradesh_Deoria",
    "UttarPradesh_Etah",
    "UttarPradesh_Ghazipur",
    "UttarPradesh_Gorakhpur",
    "UttarPradesh_Hamirpur",
    "UttarPradesh_Jalaun",
    "UttarPradesh_JyotibaPhuleNagar",
    "UttarPradesh_Muzzaffarnagar",
    "UttarPradesh_Varanasi",
    # Rajasthan
    "Rajasthan_Churu",
    "Rajasthan_Nagaur",
    # Uttarakhand
    "Uttarakhand_TehriGarhwal",
    "Uttarakhand_Uttarkashi",
    # Chhattisgarh
    "Chhattisgarh_Balrampur",
    "Chhattisgarh_Bastar",
    "Chhattisgarh_Bilaspur",
    "Chhattisgarh_Jashpur",
    "Chhattisgarh_Kabirdham",
    "Chhattisgarh_Korba",
    "Chhattisgarh_Raigarh",
    "Chhattisgarh_Rajnandgaon",
    "Chhattisgarh_Sarguja",
    "Chhattisgarh_Sukma",
    # Jharkhand
    "Jharkhand_Jamtara",
    "Jharkhand_Sahebganj",
]

selected_districts_per_state = [
    "AndhraPradesh_Anantpur",
    "Bihar_Araria",
    "Chhattisgarh_Balrampur",
    "Goa_NorthSouthGoa",
    "Jharkhand_Jamtara",
    "Karnataka_Belgaum",
    "Maharashtra_Aurangabad",
    "Rajasthan_Churu",
    "Telangana_Karimnagar",
    "UttarPradesh_Budaun",
    "Uttarakhand_TehriGarhwal",
    "WestBengal_DakshinDinajpur",
]

hindi_districts_per_state = [
    "Bihar_Purnia",
    "Maharashtra_Solapur",
    "UttarPradesh_Varanasi",
    # "Rajasthan_Nagaur",
    "Uttarakhand_Uttarkashi",
    "Karnataka_Bijapur",
    "Chhattisgarh_Bastar",
]

andhra_pradesh_districts = [
    "AndhraPradesh_Anantpur",
    "AndhraPradesh_Chittoor",
    "AndhraPradesh_Guntur",
    "AndhraPradesh_Krishna",
    "AndhraPradesh_Srikakulam",
    "AndhraPradesh_Vishakapattanam",
]


@dataclass
class DatasetConfig:
    name: str
    configs: List[str]
    revision: Optional[str] = None


@dataclass
class ModelConfig:
    name: str
    processor: str
    feature_extractor: str
    tokenizer: Optional[str] = None


@dataclass
class ModelArchConfig:
    attention_dropout: float
    hidden_dropout: float
    feat_proj_dropout: float
    mask_time_prob: float
    layerdrop: float
    num_seconds: int
    mean_pool: bool = True
    ctc_loss_reduction: Optional[str] = None
    ctc_zero_infinity: Optional[bool] = None
    vocab_size: Optional[int] = None
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None


@dataclass
class AugmentationConfig:
    noise_factor: float
    pitch_shift_steps: List[int]
    prob_noise: float
    prob_pitch: float


@dataclass
class TrainingConfig:
    learning_rate: float
    per_device_train_batch_size: int
    num_train_epochs: int
    weight_decay: float
    classifier_only: bool
    eval_steps: Optional[int] = None
    save_steps: Optional[int] = None
    logging_steps: int = 500
    accumulate_grad_batches: int = 1
    val_check_interval: Optional[float] = None


@dataclass
class TuningConfig:
    learning_rate_min: float
    learning_rate_max: float
    weight_decay_min: float
    weight_decay_max: float
    dropout_min: float
    dropout_max: float


@dataclass
class Config:
    data_dir: str
    cache_dir: str
    base_dir: str
    hf_token: str
    batch_size: int
    name: str
    resume: bool
    dataset: DatasetConfig
    model: ModelConfig
    model_config: ModelArchConfig
    augmentation: AugmentationConfig
    training: TrainingConfig
    tuning: TuningConfig
    output_dir: Optional[str] = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class CacheConfig:
    random_seed: int
    language: str
    num_proc: int
    min_seconds: int
    task_type: str
    split_key: str
    min_stay: int

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


# Base configurations as dataclass instances
CONFIG = Config(
    data_dir="/scratch/yasaswini/data",
    cache_dir="/scratch/yasaswini/data",
    base_dir="/scratch/karthik/models",
    hf_token=hf_token,
    batch_size=64,
    name="Vaani",
    resume=False,
    dataset=DatasetConfig(
        name="ARTPARK-IISC/Vaani",
        configs=hindi_districts_per_state,
        revision="0816973e005ca2377c8ffa65323c2bb866e24fbf",
    ),
    model=ModelConfig(
        name="facebook/wav2vec2-xls-r-300m",
        processor="facebook/wav2vec2-base",
        feature_extractor="facebook/wav2vec2-base",
    ),
    model_config=ModelArchConfig(
        attention_dropout=0.01,
        hidden_dropout=0.01,
        feat_proj_dropout=0.01,
        mask_time_prob=0.01,
        layerdrop=0.01,
        num_seconds=10,
        mean_pool=True,
    ),
    augmentation=AugmentationConfig(
        noise_factor=0.03,
        pitch_shift_steps=[-2, -1, 1, 2],
        prob_noise=0.5,
        prob_pitch=0.5,
    ),
    training=TrainingConfig(
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        num_train_epochs=5,
        weight_decay=1e-4,
        classifier_only=False,
        eval_steps=500,
        save_steps=500,
        logging_steps=500,
        accumulate_grad_batches=8,
    ),
    tuning=TuningConfig(
        learning_rate_min=1e-6,
        learning_rate_max=1e-4,
        weight_decay_min=1e-5,
        weight_decay_max=1e-2,
        dropout_min=0.0,
        dropout_max=0.5,
    ),
)  # Added missing closing parenthesis

CACHE_SENSITIVE_CONFIG = CacheConfig(
    random_seed=41,
    language="Hindi",
    num_proc=10,
    min_seconds=5,
    task_type="dialect",
    split_key="pincode",
    min_stay=0,
)

TRANSCRIPTION_CONFIG = Config(
    data_dir=CONFIG.data_dir,
    cache_dir=CONFIG.cache_dir,
    base_dir=CONFIG.base_dir,
    hf_token=CONFIG.hf_token,
    batch_size=16,
    name="Vaani-Transcription",
    resume=False,
    dataset=DatasetConfig(
        name="ARTPARK-IISc/Vaani-transcription-part", configs="Telugu", revision=None
    ),
    model=ModelConfig(
        name="facebook/wav2vec2-xls-r-300m",
        feature_extractor="facebook/wav2vec2-xls-r-300m",
        processor="facebook/wav2vec2-xls-r-300m",
        tokenizer="vasista22/whisper-telugu-medium",
    ),
    model_config=ModelArchConfig(
        attention_dropout=0.05,
        hidden_dropout=0.05,
        feat_proj_dropout=0.05,
        mask_time_prob=0.05,
        layerdrop=0.05,
        num_seconds=10,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        vocab_size=None,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
    ),
    augmentation=AugmentationConfig(
        noise_factor=0.0,
        pitch_shift_steps=[-2, -1, 1, 2],
        prob_noise=0.0,
        prob_pitch=0.0,
    ),
    training=TrainingConfig(
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        num_train_epochs=6,
        weight_decay=1e-4,
        classifier_only=False,
        logging_steps=50,
        val_check_interval=0.25,
        accumulate_grad_batches=1,
    ),
    tuning=TuningConfig(  # Add tuning config for transcription if needed, using defaults for now
        learning_rate_min=1e-6,
        learning_rate_max=1e-4,
        weight_decay_min=1e-5,
        weight_decay_max=1e-2,
        dropout_min=0.0,
        dropout_max=0.5,
    ),
)

TRANSCRIPTION_CACHE_CONFIG = CacheConfig(
    random_seed=41,
    language="Hindi",
    num_proc=10,
    min_seconds=3,
    task_type="transcription",
    split_key="pincode",
    min_stay=0,
)


def generate_output_dir(config: Config, cache_config: CacheConfig) -> str:
    """Generate concise output directory name."""
    base = config.base_dir
    name_parts = [
        config.model.name.split("/")[-1],
        cache_config.task_type,
        config.name,
    ]
    dir_name = "_".join(name_parts)
    return os.path.join(base, dir_name)


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    pl.seed_everything(seed)
    torch.set_num_threads(1)


def deep_update_dataclass(orig: Any, updates: Dict[str, Any]) -> None:
    """Recursively update orig (dict or dataclass instance) with values from updates dict."""
    if is_dataclass(orig):
        for key, val in updates.items():
            if hasattr(orig, key):
                orig_val = getattr(orig, key)
                if (
                    is_dataclass(orig_val) or isinstance(orig_val, dict)
                ) and isinstance(val, dict):
                    deep_update_dataclass(orig_val, val)
                else:
                    setattr(orig, key, val)
    elif isinstance(orig, dict):
        for key, val in updates.items():
            if key in orig:
                orig_val = orig[key]
                if (
                    is_dataclass(orig_val) or isinstance(orig_val, dict)
                ) and isinstance(val, dict):
                    deep_update_dataclass(orig_val, val)
                else:
                    orig[key] = val
            else:
                orig[key] = val


def load_config(
    task: str = "classification", yaml_path: Optional[str] = None
) -> tuple[Config, CacheConfig]:
    """
    Load base configuration for a task and apply updates from a YAML file.
    task: 'classification' or 'transcription'
    yaml_path: Optional path to a YAML file containing config updates.
              The YAML can contain top-level keys matching Config fields,
              and a specific key 'cache_config_updates' for CacheConfig fields.
    Returns the base config and cache_config potentially updated by YAML.
    """
    if task == "classification":
        base, cache = CONFIG, CACHE_SENSITIVE_CONFIG
    elif task == "transcription":
        base, cache = TRANSCRIPTION_CONFIG, TRANSCRIPTION_CACHE_CONFIG
    else:
        raise ValueError(f"Unknown task: {task}")

    config = copy.deepcopy(base)
    cache_config = copy.deepcopy(cache)

    # Load updates from YAML file
    if yaml_path and os.path.exists(yaml_path):
        try:
            with open(yaml_path, "r") as f:
                yaml_data = yaml.safe_load(f) or {}
            logger.info(f"Loaded config updates from YAML: {yaml_path}")

            # Separate updates for Config and CacheConfig
            config_yaml_updates = {
                k: v for k, v in yaml_data.items() if k != "cache_config_updates"
            }
            cache_yaml_updates = yaml_data.get("cache_config_updates", {})

            # Apply updates
            if config_yaml_updates:
                deep_update_dataclass(config, config_yaml_updates)
                logger.info(f"Applied Config updates from YAML.")
            if cache_yaml_updates:
                deep_update_dataclass(cache_config, cache_yaml_updates)
                logger.info(f"Applied CacheConfig updates from YAML.")

        except Exception as e:
            logger.error(f"Error loading or applying YAML file {yaml_path}: {e}")

    config.output_dir = generate_output_dir(config, cache_config)
    os.makedirs(config.output_dir, exist_ok=True)

    os.environ["HF_HOME"] = config.data_dir
    os.environ["HF_DATASETS_CACHE"] = config.cache_dir
    return config, cache_config


def update_config(
    config: Config,
    cache_config: CacheConfig,
    config_updates: Optional[Dict[str, Any]] = None,
    cache_updates: Optional[Dict[str, Any]] = None,
) -> tuple[Config, CacheConfig]:
    """
    Apply explicit dictionary updates to config objects, generate output dir, and set env vars.
    config: The Config object (potentially already updated by YAML).
    cache_config: The CacheConfig object.
    config_updates: Explicit dict of overrides for config.
    cache_updates: Explicit dict of overrides for cache_config.
    Side-effects: creates output_dir, sets HF_HOME and HF_DATASETS_CACHE env vars.
    Returns the updated config and cache_config.
    """
    if config_updates:
        deep_update_dataclass(config, config_updates)
    if cache_updates:
        deep_update_dataclass(cache_config, cache_updates)

    config.output_dir = generate_output_dir(config, cache_config)
    os.makedirs(config.output_dir, exist_ok=True)

    os.environ["HF_HOME"] = config.data_dir
    os.environ["HF_DATASETS_CACHE"] = config.cache_dir
    return config, cache_config


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
