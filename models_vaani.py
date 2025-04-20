import torch
import transformers
import pytorch_lightning as pl
from sklearn.metrics import precision_score, recall_score, f1_score
import logging
import torch.nn.functional as F
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from transformers.modeling_outputs import CausalLMOutput, SequenceClassifierOutput
import jiwer
import numpy as np
import re
from dataclasses import asdict, dataclass
from typing import Dict, Any, List, Optional, Tuple, Union
from config import ModelArchConfig

logger = logging.getLogger(__name__)


class Wav2Vec2Classifier(torch.nn.Module):
    def __init__(
        self,
        num_labels: int,
        model_name: str,
        model_config: ModelArchConfig,
        classifier_only: bool = False,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.model_config: Dict[str, Any] = asdict(model_config)
        if "num_seconds" in self.model_config:
            del self.model_config["num_seconds"]
        self.mean_pool = (
            self.model_config.pop("mean_pool")
            if "mean_pool" in self.model_config
            else True
        )

        if model_name == "facebook/w2v-bert-2.0":
            self.wav2vec2 = transformers.Wav2Vec2BertModel.from_pretrained(
                model_name, **self.model_config
            )
        elif model_name == "facebook/wav2vec2-conformer-rope-large-960h-ft":
            self.wav2vec2 = transformers.Wav2Vec2ConformerModel.from_pretrained(
                model_name, **self.model_config
            )
        else:
            self.wav2vec2 = transformers.Wav2Vec2Model.from_pretrained(
                model_name, **self.model_config
            )

        if classifier_only:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False
            self.wav2vec2.eval()

        hidden_size = self.wav2vec2.config.hidden_size

        if not self.mean_pool:
            # Instead of flattening, use a CNN for classification
            self.cnn_classifier = torch.nn.Sequential(
                # Input shape: (batch_size, hidden_size, sequence_length)
                torch.nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
                # Shape: (batch_size, hidden_size, sequence_length)
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(2),
                # Shape: (batch_size, hidden_size, sequence_length/2)
                torch.nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=3, padding=1),
                # Shape: (batch_size, hidden_size*2, sequence_length/2)
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(2),
                # Shape: (batch_size, hidden_size*2, sequence_length/4)
                torch.nn.Conv1d(
                    hidden_size * 2, hidden_size * 2, kernel_size=3, padding=1
                ),
                # Shape: (batch_size, hidden_size*2, sequence_length/4)
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool1d(1),  # Global average pooling to fixed size
                # Shape: (batch_size, hidden_size*2, 1)
                torch.nn.Flatten(),  # Flatten for final linear layers
                # Shape: (batch_size, hidden_size*2)
            )
            hidden_size *= 2  # Account for channel expansion in CNN

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size, num_labels),
        )

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.train()

    def forward(
        self, input_values: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> SequenceClassifierOutput:
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state

        if self.mean_pool:
            pooled = torch.mean(hidden_states, dim=1)
        else:
            # Process through CNN
            # Transpose to (batch, channels, sequence) for Conv1d
            hidden_states = hidden_states.transpose(1, 2)
            pooled = self.cnn_classifier(hidden_states)

        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return transformers.modeling_outputs.SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )


class LitModel(pl.LightningModule):
    def __init__(
        self,
        num_labels: int,
        model_name: str,
        model_config: ModelArchConfig,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.001,
        classifier_only: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = Wav2Vec2Classifier(
            num_labels, model_name, model_config, classifier_only
        )
        self.validation_step_outputs: List[Dict[str, torch.Tensor]] = []
        self.test_step_outputs: List[Dict[str, torch.Tensor]] = []
        self.training_step_outputs: List[Dict[str, torch.Tensor]] = []

    def forward(self, input_values: torch.Tensor) -> SequenceClassifierOutput:
        return self.model(input_values=input_values)

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        outputs = self.model(input_values=batch["input_values"], labels=batch["label"])
        self.log("train_loss", outputs.loss, prog_bar=True)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        self.log(
            "train_acc", (predictions == batch["label"]).float().mean(), prog_bar=True
        )
        self.training_step_outputs.append(
            {"preds": predictions, "labels": batch["label"]}
        )
        return outputs.loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        outputs = self.model(input_values=batch["input_values"], labels=batch["label"])
        self.log("val_loss", outputs.loss, prog_bar=True)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        self.log(
            "val_acc",
            (predictions == batch["label"]).float().mean(),
            prog_bar=True,
        )
        self.validation_step_outputs.append(
            {"preds": predictions, "labels": batch["label"]}
        )
        return {
            "val_loss": outputs.loss,
            "preds": predictions,
            "labels": batch["label"],
        }

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        outputs = self.model(input_values=batch["input_values"], labels=batch["label"])
        self.log("test_loss", outputs.loss, prog_bar=True)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        self.log(
            "test_acc", (predictions == batch["label"]).float().mean(), prog_bar=True
        )
        output_dict = {"preds": predictions, "labels": batch["label"]}
        self.test_step_outputs.append(output_dict)
        return output_dict

    def _compute_metrics(
        self, outputs: List[Dict[str, torch.Tensor]], prefix: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        preds = torch.cat([x["preds"] for x in outputs]).cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).cpu().numpy()

        metrics = {
            f"{prefix}_precision": precision_score(
                labels, preds, average="macro", zero_division=0
            ),
            f"{prefix}_recall": recall_score(
                labels, preds, average="macro", zero_division=0
            ),
            f"{prefix}_f1": f1_score(labels, preds, average="macro", zero_division=0),
        }
        self.log_dict(metrics, prog_bar=True)
        return preds, labels

    def on_validation_epoch_end(self) -> None:
        self._compute_metrics(self.validation_step_outputs, "val")
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self) -> Dict[str, np.ndarray]:
        logger.info(f"Test epoch end: collected {len(self.test_step_outputs)} outputs")
        preds, labels = self._compute_metrics(self.test_step_outputs, "test")
        logger.info(f"Test epoch end: computed metrics on {len(preds)} predictions")
        return {"preds": preds, "labels": labels}

    def on_train_epoch_end(self) -> None:
        self._compute_metrics(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )


class Wav2Vec2ForASR(torch.nn.Module):
    def __init__(self, model_name: str, model_config: ModelArchConfig, vocab_size: int):
        super().__init__()

        model_config_copy: Dict[str, Any] = asdict(model_config)
        keys_to_remove = ["num_seconds", "mean_pool", "max_target_position"]
        for key in keys_to_remove:
            if key in model_config_copy:
                del model_config_copy[key]

        self.wav2vec2 = Wav2Vec2ForCTC.from_pretrained(model_name, **model_config_copy)

        logger.info(f"Resized Wav2Vec2 token embeddings to vocab_size: {vocab_size}")

    def forward(
        self,
        input_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> transformers.modeling_outputs.BaseModelOutput:
        if labels is not None and (labels == -100).all():
            labels = None

        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        return outputs


class WhisperForASR(torch.nn.Module):
    def __init__(self, model_name: str, model_config: Optional[ModelArchConfig] = None):
        super().__init__()

        self.whisper: WhisperForConditionalGeneration = (
            WhisperForConditionalGeneration.from_pretrained(model_name)
        )

    def forward(
        self,
        input_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Union[CausalLMOutput, transformers.modeling_outputs.Seq2SeqLMOutput]:
        if labels is not None:
            outputs = self.whisper(
                input_features=input_values,
                decoder_input_ids=None,
                labels=labels,
                return_dict=True,
            )
        else:
            outputs = self.whisper(input_features=input_values, return_dict=True)

        return outputs


class TranscriptionLitModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        model_config: ModelArchConfig,
        processor: Union[Wav2Vec2Processor, WhisperProcessor],
        learning_rate: float = 5e-5,
        weight_decay: float = 0.001,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["processor"])
        self.processor = processor

        self.is_whisper = "whisper" in model_name.lower()

        if self.is_whisper:
            self.model = WhisperForASR(model_name, model_config)
            logger.info(f"Using Whisper model: {model_name}")
            vocab_size = len(self.processor.get_vocab())
        else:
            vocab_size = len(self.processor["tokenizer"].get_vocab())
            self.model = Wav2Vec2ForASR(model_name, model_config, vocab_size)
            logger.info(f"Using Wav2Vec2 model: {model_name}")

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.training_step_outputs: List[Dict[str, torch.Tensor]] = []
        self.validation_step_outputs: List[Dict[str, Any]] = []
        self.test_step_outputs: List[Dict[str, Any]] = []

        # Only extract dialect tokens if we're not using Whisper
        if not self.is_whisper:
            self.dialect_tokens = self._get_dialect_tokens_from_vocab()
            logger.info(
                f"Identified {len(self.dialect_tokens)} dialect tokens for extraction."
            )
        else:
            self.dialect_tokens = []

    def _get_dialect_tokens_from_vocab(self) -> List[str]:
        """Extracts potential dialect tokens (e.g., '<Dialect>') from the tokenizer's vocab."""
        # Skip for Whisper models as they handle vocabulary differently
        if self.is_whisper:
            return []

        vocab = self.processor["tokenizer"].get_vocab()
        # Basic check: starts with '<', ends with '>', isn't a known special token
        special_tokens = ["<pad>", "<s>", "</s>", "<unk>", "|"]
        dialect_tokens = [
            token
            for token in vocab
            if token.startswith("<")
            and token.endswith(">")
            and token not in special_tokens
        ]
        dialect_tokens.sort(key=len, reverse=True)
        return dialect_tokens

    def forward(
        self,
        input_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Any:
        return self.model(
            input_values=input_values, labels=labels, attention_mask=attention_mask
        )

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        outputs = self.model(
            input_values=batch["input_values"],
            attention_mask=batch.get("attention_mask", None),
            labels=batch["labels"],
        )

        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)
        self.training_step_outputs.append({"loss": loss})
        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        with torch.no_grad():
            outputs = self.model(
                input_values=batch["input_values"],
                attention_mask=batch.get("attention_mask", None),
                labels=batch["labels"],
            )

            loss = outputs.loss
            self.log("val_loss", loss, prog_bar=True)

            if self.is_whisper:
                generated_ids = self.model.whisper.generate(
                    input_features=batch["input_values"], max_length=225
                )

                pred_strings = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )

                label_strings = self.processor.batch_decode(
                    batch["labels"], skip_special_tokens=True
                )
            else:
                logits = outputs.logits
                predicted_ids = torch.argmax(logits, dim=-1)
                pred_strings = self.processor["tokenizer"].batch_decode(predicted_ids)

                labels = batch["labels"].detach().cpu().numpy()
                labels = np.where(
                    labels != -100, labels, self.processor["tokenizer"].pad_token_id
                )
                label_strings = self.processor["tokenizer"].batch_decode(labels)

            wer = jiwer.wer(label_strings, pred_strings)

            dialect_correct = 0
            total = 0

            for pred, label in zip(pred_strings, label_strings):
                pred_dialect = self._extract_dialect(pred)
                label_dialect = self._extract_dialect(label)

                if label_dialect:
                    total += 1
                    if pred_dialect == label_dialect:
                        dialect_correct += 1

            dialect_accuracy = dialect_correct / total if total > 0 else 0

            self.log("val_wer", wer, prog_bar=True, on_step=True)
            self.log("val_dialect_acc", dialect_accuracy, prog_bar=True, on_step=True)

            self.validation_step_outputs.append(
                {
                    "val_loss": loss,
                    "wer": wer,
                    "dialect_accuracy": dialect_accuracy,
                    "preds": pred_strings,
                    "labels": label_strings,
                }
            )

            return {
                "val_loss": loss,
                "wer": wer,
                "dialect_accuracy": dialect_accuracy,
                "preds": pred_strings,
                "labels": label_strings,
            }

    def _extract_dialect(self, text: str) -> Optional[str]:
        """Extract dialect tag from text if it exists."""
        match = re.search(r"<([^>]+)>", text)
        return match.group(1) if match else None

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        return self.validation_step(batch, batch_idx)

    def on_train_epoch_end(self) -> None:
        avg_loss = torch.stack([x["loss"] for x in self.training_step_outputs]).mean()
        self.log("train_loss_epoch", avg_loss)
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        avg_loss = torch.stack(
            [x["val_loss"] for x in self.validation_step_outputs]
        ).mean()
        avg_wer = (
            sum([x["wer"] for x in self.validation_step_outputs])
            / len(self.validation_step_outputs)
            if self.validation_step_outputs
            else 0
        )
        avg_dialect_acc = (
            sum([x["dialect_accuracy"] for x in self.validation_step_outputs])
            / len(self.validation_step_outputs)
            if self.validation_step_outputs
            else 0
        )

        self.log("val_loss_epoch", avg_loss, prog_bar=True, on_epoch=True)
        self.log("val_wer_epoch", avg_wer, prog_bar=True, on_epoch=True)
        self.log("val_dialect_acc_epoch", avg_dialect_acc, prog_bar=True, on_epoch=True)

        if self.validation_step_outputs:
            examples = self.validation_step_outputs[0]
            for i in range(min(3, len(examples["preds"]))):
                logger.info(f"Example {i}:")
                logger.info(f"  Pred: {examples['preds'][i]}")
                logger.info(f"  Label: {examples['labels'][i]}")

        self.validation_step_outputs.clear()

    def on_test_epoch_end(self) -> None:
        avg_wer = (
            sum([x["wer"] for x in self.test_step_outputs])
            / len(self.test_step_outputs)
            if self.test_step_outputs
            else 0
        )
        avg_dialect_acc = (
            sum([x["dialect_accuracy"] for x in self.test_step_outputs])
            / len(self.test_step_outputs)
            if self.test_step_outputs
            else 0
        )

        self.log("test_wer", avg_wer)
        self.log("test_dialect_acc", avg_dialect_acc)

        self.test_step_outputs.clear()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
