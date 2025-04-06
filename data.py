import torch
import numpy as np
import librosa
import datasets as dts
from torch.utils.data import DataLoader
from tqdm import tqdm
import polars
import pathlib
import json
import logging
import re

logger = logging.getLogger(__name__)


def extract_years(stay_str):
    """Extract the first multi-digit number from a string."""
    if not stay_str or stay_str == "NA":
        return 0
    match = re.search(r"\d+", str(stay_str))
    return int(match.group(0)) if match else 0


class VaaniDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        processor,
        label_to_id,
        is_train=False,
        config=None,
        cache_config=None,
        task="classification",
    ):
        self.dataset = dataset
        self.processor = processor
        self.label_to_id = label_to_id
        self.is_train = is_train
        self.config = config
        self.cache_config = cache_config
        self.current_index = 0
        self.task = task

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        self.current_index = idx
        item = self.dataset[idx]
        audio = item["audio"]["array"]
        sampling_rate = item["audio"]["sampling_rate"]

        # Resample if needed
        if sampling_rate != 16000:
            audio = librosa.resample(y=audio, orig_sr=sampling_rate, target_sr=16000)
            sampling_rate = 16000

        if self.is_train and self.config:
            # Apply augmentations (for both classification and transcription)
            if np.random.random() < self.config["augmentation"]["prob_noise"]:
                noise = np.random.randn(len(audio))
                audio = audio + self.config["augmentation"]["noise_factor"] * noise
                audio = audio.astype(type(audio[0]))

            if np.random.random() < self.config["augmentation"]["prob_pitch"]:
                n_steps = np.random.choice(
                    self.config["augmentation"]["pitch_shift_steps"]
                )
                audio = librosa.effects.pitch_shift(y=audio, sr=16000, n_steps=n_steps)

        if self.task == "transcription":
            # Return raw audio and metadata for batch processing in collate_fn
            return {
                "audio": {"array": audio, "sampling_rate": 16000},
                "transcript": item["transcript"] or "",
                "dialect": item["dialect"],
            }

        elif self.task == "classification":
            # Return raw audio and metadata for batch processing in collate_fn
            return {
                "audio": {"array": audio, "sampling_rate": 16000},
                "label": self.label_to_id[item[self.cache_config["task_type"]]],
            }
        else:
            raise ValueError(f"Unsupported task type: {self.task}")

    def state_dict(self):
        return {"current_index": self.current_index}

    def load_state_dict(self, state_dict):
        self.current_index = state_dict["current_index"]


class ResumableDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._iterator = None
        self.epoch = 0
        self._length = None

    def __len__(self):
        if self._length is None:
            self._length = len(self.dataset) // self.batch_size
            if len(self.dataset) % self.batch_size != 0:
                self._length += 1
        return self._length

    def state_dict(self):
        if self._iterator is None:
            return {"epoch": self.epoch}
        return {
            "epoch": self.epoch,
            "dataset_state": (
                self.dataset.state_dict()
                if hasattr(self.dataset, "state_dict")
                else None
            ),
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict["epoch"]
        if "dataset_state" in state_dict and hasattr(self.dataset, "load_state_dict"):
            self.dataset.load_state_dict(state_dict["dataset_state"])


def _load_vaani_dataset(config, cache_config):
    """Loads and prepares the ARTPARK-IISC/Vaani dataset."""
    trains, tests, vals = [], [], []
    confs = config["dataset"]["configs"]
    for conf in tqdm(confs):
        ds = dts.load_dataset(
            config["dataset"]["name"],
            conf,
            split="train",
            cache_dir=config["cache_dir"],
            num_proc=cache_config["num_proc"],
            revision=config["dataset"]["revision"],
        )

        # logger.info(f"{conf} size before duration filtering: {len(ds)} samples")
        # ds = ds.filter(
        #     lambda r: len(r["audio"]["array"]) / 16000 >= cache_config["min_seconds"],
        #     num_proc=cache_config["num_proc"],
        # )
        # logger.info(f"{conf} size after duration filtering: {len(ds)} samples")

        ds = ds.rename_column("district", "dialect")

        spIDs = ds[cache_config["split_key"]]
        unq_spIDs = np.unique(spIDs)
        test_spIDs = np.random.choice(
            unq_spIDs, size=int(0.1 * len(unq_spIDs)), replace=False
        )
        val_spIDs = np.random.choice(
            [spID for spID in unq_spIDs if spID not in test_spIDs],
            size=int(0.1 * len(unq_spIDs)),
            replace=False,
        )

        # For dialect task, filter by language. For language task, include all languages
        # Added stay duration filter for training set
        print(ds.unique("dialect"))

        def train_filter(r):
            return (
                r[cache_config["split_key"]] not in test_spIDs
                and r[cache_config["split_key"]] not in val_spIDs
                and extract_years(r.get("stay(years)")) >= cache_config["min_stay"]
                and (
                    r["language"] == cache_config["language"]
                    if cache_config["task_type"] == "dialect"
                    else True
                )
            )

        # Test and validation filters remain unchanged
        def test_filter(r):
            return (
                (r[cache_config["split_key"]] in test_spIDs)
                and (
                    r["language"] == cache_config["language"]
                    if cache_config["task_type"] == "dialect"
                    else True
                )
                and (len(r["audio"]["array"]) / 16000 >= cache_config["min_seconds"])
            )

        def val_filter(r):
            (r[cache_config["split_key"]] in val_spIDs) and (
                r["language"] == cache_config["language"]
                if cache_config["task_type"] == "dialect"
                else True
            ) and (len(r["audio"]["array"]) / 16000 >= cache_config["min_seconds"])

        train_ds = ds.filter(train_filter, num_proc=cache_config["num_proc"])
        test_ds = ds.filter(test_filter, num_proc=cache_config["num_proc"])
        val_ds = ds.filter(val_filter, num_proc=cache_config["num_proc"])

        # Log dataset sizes after filtering
        logger.info(f"{conf} sizes after filtering:")
        logger.info(f"Train set: {len(train_ds)} samples")
        logger.info(f"Test set: {len(test_ds)} samples")
        logger.info(f"Validation set: {len(val_ds)} samples")

        trains.append(train_ds)
        tests.append(test_ds)
        vals.append(val_ds)

    return (
        dts.interleave_datasets(trains),
        dts.interleave_datasets(tests),
        dts.interleave_datasets(vals),
    )


def remove_special_characters(batch, chars_to_remove_regex):
    # Process each transcript in the batch
    clean_transcripts = []
    for transcript in batch["transcript"]:
        # Skip None values
        if transcript is None:
            clean_transcripts.append("")
            continue

        # Remove all content within angle brackets < >
        transcript = re.sub(r"<[^>]*>", "", transcript)

        # Remove all content within square brackets [ ]
        transcript = re.sub(r"\[[^\]]*\]", "", transcript)

        # Remove all content within curly brackets { }
        transcript = re.sub(r"\{[^}]*\}", "", transcript)

        # Remove punctuation and special characters
        transcript = re.sub(chars_to_remove_regex, "", transcript)

        # Clean up extra whitespace
        transcript = re.sub(r"\s+", " ", transcript).strip()

        # Replace space " " with |
        transcript = re.sub(r" ", "|", transcript)

        clean_transcripts.append(transcript)

    batch["transcript"] = clean_transcripts
    return batch


def _load_vaani_transcripts_dataset(config, cache_config):
    """Loads and prepares the ARTPARK-IISC/Vaani dataset."""
    conf = config["dataset"]["configs"]
    ds = dts.load_dataset(
        config["dataset"]["name"],
        conf,
        cache_dir=config["cache_dir"],
        num_proc=cache_config["num_proc"],
        revision=config["dataset"]["revision"],
    )

    num_rows = sum(ds.num_rows.values())
    logger.info(f"{conf} size before duration filtering: {num_rows} samples")
    ds = ds.filter(
        lambda r: len(r["audio"]["array"]) / 16000 >= cache_config["min_seconds"],
        num_proc=cache_config["num_proc"],
    )

    ds = ds.rename_column("district", "dialect")

    logger.info(
        f"{conf} size after duration filtering: {sum(ds.num_rows.values())} samples"
    )

    chars_to_remove_regex = r"[\,\?\.\!\-\;\:\"\“\%\‘\”\�']"
    ds = ds.map(
        lambda x: remove_special_characters(x, chars_to_remove_regex),
        batch_size=100,
        batched=True,
        num_proc=cache_config["num_proc"],
    )

    # Create vocabulary from the training data
    logger.info("Creating vocabulary from training data...")

    # Get all unique dialects
    unique_dialects = ds.unique("dialect")
    dialects = set()
    for dialect in unique_dialects.values():
        dialects.update(dialect)

    logger.info(f"Found {len(dialects)} unique dialects: {unique_dialects}")

    total_transcipt = (
        " ".join(ds["train"]["transcript"])
        + " ".join(ds["validation"]["transcript"])
        + " ".join(ds["test"]["transcript"])
    )

    vocab_set = set(total_transcipt)

    vocab_dict = {
        # "<pad>": 0,
        # "<s>": 1,
        # "</s>": 2,
        # "<unk>": 3,
        "|": 0,
    }
    current_id = 1

    for dialect in dialects:
        dialect_token = f"<{dialect}>"
        vocab_dict[dialect_token] = current_id
        current_id += 1

    # Add characters from transcripts to vocabulary
    for char in sorted(list(vocab_set)):
        vocab_dict[char] = current_id
        current_id += 1

    logger.info(f"Train set: {ds.num_rows['train']} samples")
    logger.info(f"Validation set: {ds.num_rows['validation']} samples")
    logger.info(f"Test set: {ds.num_rows['test']} samples")

    return (
        ds["train"],
        ds["test"],
        ds["test"],
        vocab_dict,
        dialects,
    )  # since the validation set is a joke


def _load_respin_dataset(config, cache_config):
    """Loads and prepares the Respin dataset."""
    # Load training data
    with open("data/respin/train/meta_train_te_small.json") as f:
        meta_train = json.load(f)

    meta_train_list = []
    for k, v in meta_train.items():
        v.update({"id": k})
        meta_train_list.append(v)

    train_ds = dts.Dataset.from_polars(polars.from_dicts(meta_train_list))

    def prepend_path(base_path, relative_path):
        full_path = pathlib.Path(base_path) / relative_path
        return str(full_path)

    train_ds = train_ds.map(
        lambda x: {
            "wav_path": prepend_path(
                "/mnt/c1e1833e-4df6-4c4c-88aa-8cd3d7d3932b/vaani/data/respin/train/",
                x["wav_path"],
            )
        },
        num_proc=cache_config["num_proc"],
    )
    train_ds = train_ds.cast_column("wav_path", dts.Value("string")).cast_column(
        "wav_path", dts.Audio()
    )
    train_ds = train_ds.rename_column("wav_path", "audio")

    # Load validation data
    with open("data/respin/val/meta_dev_te.json") as f:
        meta_val = json.load(f)

    meta_val_list = []
    for k, v in meta_val.items():
        v.update({"id": k})
        meta_val_list.append(v)

    val_ds = dts.Dataset.from_polars(polars.from_dicts(meta_val_list))
    val_ds = val_ds.map(
        lambda x: {
            "wav_path": prepend_path(
                "/mnt/c1e1833e-4df6-4c4c-88aa-8cd3d7d3932b/vaani/data/respin/val/",
                x["wav_path"],
            )
        },
        num_proc=cache_config["num_proc"],
    )
    val_ds = val_ds.cast_column("wav_path", dts.Value("string")).cast_column(
        "wav_path", dts.Audio()
    )
    val_ds = val_ds.rename_column("wav_path", "audio")

    # Create an empty test dataset with the same schema
    test_ds = dts.Dataset.from_dict(train_ds[:0])

    logger.info(f"Train set: {len(train_ds)} samples")
    logger.info(f"Validation set: {len(val_ds)} samples")
    logger.info(f"Test set: {len(test_ds)} samples")

    return train_ds, test_ds, val_ds


def load_datasets(config, cache_config):
    """Load and prepare datasets based on the configuration."""

    dataset_name = config["dataset"]["name"]

    if dataset_name == "ARTPARK-IISC/Vaani":
        return _load_vaani_dataset(config, cache_config)

    elif dataset_name == "Respin":
        return _load_respin_dataset(config, cache_config)

    elif dataset_name == "ARTPARK-IISc/Vaani-transcription-part":
        return _load_vaani_transcripts_dataset(config, cache_config)

    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")
