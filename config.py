import os
import logging
import pytorch_lightning as pl
import torch

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

andhra_pradesh_districts = [
    "AndhraPradesh_Anantpur",
    "AndhraPradesh_Chittoor",
    "AndhraPradesh_Guntur",
    "AndhraPradesh_Krishna",
    "AndhraPradesh_Srikakulam",
    "AndhraPradesh_Vishakapattanam",
]
# Configuration
CACHE_SENSITIVE_CONFIG = {
    "random_seed": 41,
    "language": "Telugu",
    "num_proc": 10,
    "min_seconds": 0,
    "task_type": "dialect",
    "split_key": "pincode",
    "min_stay": 0,
}

CONFIG = {
    "data_dir": "/mnt/c1e1833e-4df6-4c4c-88aa-8cd3d7d3932b/vaani/data",
    "cache_dir": "/mnt/c1e1833e-4df6-4c4c-88aa-8cd3d7d3932b/vaani/data",
    "base_dir": "/mnt/c1e1833e-4df6-4c4c-88aa-8cd3d7d3932b/vaani/models",
    "hf_token": hf_token,
    "batch_size": 64,
    "name": "Vaani",
    "resume": False,
    # "dataset": {
    #    "name": "Respin",
    # },
    "dataset": {
        "name": "ARTPARK-IISC/Vaani",
        "configs": andhra_pradesh_districts,
        "revision": "33434cee57f8f96fd7c46641c9394310888d2f97",
        # "revision": "0816973e005ca2377c8ffa65323c2bb866e24fbf",
    },
    "model": {
        # "name": "facebook/mms-300m",
        # "name": "facebook/wav2vec2-xls-r-300m",
        # "processor": "facebook/wav2vec2-base",
        "feature_extractor": "facebook/wav2vec2-base",
        "name": "facebook/wav2vec2-conformer-rope-large-960h-ft", 
        "processor": "facebook/wav2vec2-conformer-rope-large-960h-ft",
        # "name": "facebook/w2v-bert-2.0",
        # "feature_extractor": "facebook/w2v-bert-2.0",
    },
    "model_config": {
        "attention_dropout": 0.01,
        "hidden_dropout": 0.01,
        "feat_proj_dropout": 0.01,
        "mask_time_prob": 0.01,
        "layerdrop": 0.01,
        "num_seconds" : 10,
        "mean_pool": True,
    },
    "augmentation": {
        "noise_factor": 0.03,
        "pitch_shift_steps": [-2, -1, 1, 2],
        "prob_noise": 0.5,
        "prob_pitch": 0.5,
    },
    "training": {
        "learning_rate": 5e-5,
        "per_device_train_batch_size": 8,
        "num_train_epochs": 6,
        "weight_decay": 1e-4,
        "classifier_only": False,
        "eval_steps": 500,
        "save_steps": 500,
        "logging_steps": 500,
        "accumulate_grad_batches": 8,
    },
}

# Configuration for transcription tasks
TRANSCRIPTION_CACHE_CONFIG = {
    "random_seed": 41,
    "language": "Telugu",  # Change according to your needs
    "num_proc": 10,
    "min_seconds": 3,
    "task_type": "transcription",
    "split_key": "pincode",
    "min_stay": 0,
}

TRANSCRIPTION_CONFIG = {
    "data_dir": CONFIG["data_dir"],
    "cache_dir": CONFIG["cache_dir"],
    "base_dir": CONFIG["base_dir"],
    "hf_token": CONFIG["hf_token"],
    "batch_size": 16,  # Smaller batch size for longer sequences
    "name": "Vaani-Transcription",
    "resume": False,
    "dataset": {
        "name": "ARTPARK-IISc/Vaani-transcription-part",
        "configs": "Telugu",
        "revision": None,
    },
    "model": {
        "name": "facebook/wav2vec2-xls-r-300m",  # Good multilingual model
        "feature_extractor": "facebook/wav2vec2-xls-r-300m",
        "processor": "facebook/wav2vec2-xls-r-300m",
    },
    "model_config": {
        "attention_dropout": 0.0,
        "hidden_dropout": 0.0,
        "feat_proj_dropout": 0.0,
        "mask_time_prob": 0.0,
        "layerdrop": 0.0,
        "ctc_loss_reduction": "mean",
        "ctc_zero_infinity": True,
    },
    "augmentation": {
        "noise_factor": 0.0,
        "pitch_shift_steps": [-2, -1, 1, 2],
        "prob_noise": 0.0,
        "prob_pitch": 0.0,
    },
    "training": {
        "learning_rate": 5e-6,
        "per_device_train_batch_size": 16,
        "num_train_epochs": 20,
        "weight_decay": 1e-4,
        "classifier_only": False,
        "eval_steps": 500,
        "save_steps": 500,
        "logging_steps": 100,
        "accumulate_grad_batches": 1,
    },
}

def generate_output_dir():
    """Generate output directory name based on configuration."""
    base_dir = CONFIG["base_dir"]
    model_name = (
        f"{CONFIG['model']['name'].split('/')[-1]}"
        f"_frozen_{CONFIG['training']['classifier_only']}"
        f"_task_{CACHE_SENSITIVE_CONFIG['task_type']}"
        f"_{CONFIG['name']}"
        f"_{CACHE_SENSITIVE_CONFIG['language']}"
        f"_lr_{CONFIG['training']['learning_rate']}"
        f"_dropouts_{CONFIG['model_config']['attention_dropout']}"
        f"_{CONFIG['model_config']['hidden_dropout']}"
        f"_{CONFIG['model_config']['feat_proj_dropout']}"
        f"_{CONFIG['model_config']['mask_time_prob']}"
        f"_{CONFIG['model_config']['layerdrop']}"
        f"_split_by_{CACHE_SENSITIVE_CONFIG['split_key']}"
    )

    if not CONFIG["model_config"]["mean_pool"]:
        model_name += f"_cnn"

    if CONFIG["augmentation"]["prob_noise"] > 0:
        model_name += f"_noisy_{CONFIG['augmentation']['prob_noise']}"
    if CONFIG["augmentation"]["prob_pitch"] > 0:
        model_name += f"_pitch_shifted_{CONFIG['augmentation']['prob_pitch']}"

    return os.path.join(base_dir, model_name)

# Set output directory
CONFIG["output_dir"] = generate_output_dir()

# Create output directory
os.makedirs(CONFIG["output_dir"], exist_ok=True)

# Set environment variables
os.environ["HF_HOME"] = CONFIG["data_dir"]
os.environ["HF_DATASETS_CACHE"] = CONFIG["cache_dir"]

# Set transcription output directory
TRANSCRIPTION_CONFIG["output_dir"] = os.path.join(
    TRANSCRIPTION_CONFIG["base_dir"],
    f"{TRANSCRIPTION_CONFIG['model']['name'].split('/')[-1]}_transcription_{TRANSCRIPTION_CACHE_CONFIG['language']}"
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(CONFIG["output_dir"], "training.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def set_seeds(seed=None):
    """Set random seeds for reproducibility."""
    if seed is None:
        seed = CACHE_SENSITIVE_CONFIG["random_seed"]
    pl.seed_everything(seed)
    torch.set_num_threads(1)
