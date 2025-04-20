import os
import logging
import pytorch_lightning as pl
import torch
import copy
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
    max_target_position: Optional[int] = None
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
        revision="0816973e005ca2377c8ffa65323c2bb866e24fbf"
    ),
    model=ModelConfig(
        name="facebook/wav2vec2-xls-r-300m",
        processor="facebook/wav2vec2-base",
        feature_extractor="facebook/wav2vec2-base"
    ),
    model_config=ModelArchConfig(
        attention_dropout=0.01,
        hidden_dropout=0.01,
        feat_proj_dropout=0.01,
        mask_time_prob=0.01,
        layerdrop=0.01,
        num_seconds=10,
        mean_pool=True
    ),
    augmentation=AugmentationConfig(
        noise_factor=0.03,
        pitch_shift_steps=[-2, -1, 1, 2],
        prob_noise=0.5,
        prob_pitch=0.5
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
        accumulate_grad_batches=8
    )
)

CACHE_SENSITIVE_CONFIG = CacheConfig(
    random_seed=41,
    language="Hindi",
    num_proc=10,
    min_seconds=5,
    task_type="dialect",
    split_key="pincode",
    min_stay=0
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
        name="ARTPARK-IISc/Vaani-transcription-part",
        configs="Telugu",
        revision=None
    ),
    model=ModelConfig(
        name="facebook/wav2vec2-xls-r-300m",
        feature_extractor="facebook/wav2vec2-xls-r-300m",
        processor="facebook/wav2vec2-xls-r-300m",
        tokenizer="vasista22/whisper-telugu-medium"
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
        max_target_position=448,
        vocab_size=None,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None
    ),
    augmentation=AugmentationConfig(
        noise_factor=0.0,
        pitch_shift_steps=[-2, -1, 1, 2],
        prob_noise=0.0,
        prob_pitch=0.0
    ),
    training=TrainingConfig(
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        num_train_epochs=6,
        weight_decay=1e-4,
        classifier_only=False,
        logging_steps=50,
        val_check_interval=0.25,
        accumulate_grad_batches=1
    )
)

TRANSCRIPTION_CACHE_CONFIG = CacheConfig(
    random_seed=41,
    language="Hindi",
    num_proc=10,
    min_seconds=3,
    task_type="transcription",
    split_key="pincode",
    min_stay=0
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
                if (is_dataclass(orig_val) or isinstance(orig_val, dict)) and isinstance(val, dict):
                    deep_update_dataclass(orig_val, val)
                else:
                    setattr(orig, key, val)
    elif isinstance(orig, dict):
         for key, val in updates.items():
            if key in orig:
                orig_val = orig[key]
                if (is_dataclass(orig_val) or isinstance(orig_val, dict)) and isinstance(val, dict):
                    deep_update_dataclass(orig_val, val)
                else:
                    orig[key] = val
            else:
                 orig[key] = val

def get_config(task: str = 'classification', config_updates: Optional[Dict[str, Any]] = None, cache_updates: Optional[Dict[str, Any]] = None) -> tuple[Config, CacheConfig]:
    """
    Return a pair (config, cache_config), optionally updating defaults.
    task: 'classification' or 'transcription'
    config_updates: dict of overrides for config
    cache_updates: dict of overrides for cache_config
    Side-effects: creates output_dir, sets HF_HOME and HF_DATASETS_CACHE env vars.
    """
    if task == 'classification':
        base, cache = CONFIG, CACHE_SENSITIVE_CONFIG
    elif task == 'transcription':
        base, cache = TRANSCRIPTION_CONFIG, TRANSCRIPTION_CACHE_CONFIG
    else:
        raise ValueError(f"Unknown task: {task}")

    config = copy.deepcopy(base)
    cache_config = copy.deepcopy(cache)
    if config_updates:
        deep_update_dataclass(config, config_updates)
    if cache_updates:
        deep_update_dataclass(cache_config, cache_updates)
    config.output_dir = generate_output_dir(config, cache_config)
    os.makedirs(config.output_dir, exist_ok=True) 
    os.environ['HF_HOME'] = config.data_dir 
    os.environ['HF_DATASETS_CACHE'] = config.cache_dir 
    return config, cache_config

# leave logger configured at module import if desired
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
