import torch
import transformers
import pytorch_lightning as pl
from sklearn.metrics import precision_score, recall_score, f1_score
import logging
import torch.nn.functional as F
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers.modeling_outputs import CausalLMOutput
import jiwer
import numpy as np
import re

logger = logging.getLogger(__name__)


class Wav2Vec2Classifier(torch.nn.Module):
    def __init__(self, num_labels, model_name, model_config, classifier_only=False):
        super().__init__()
        self.num_labels = num_labels
        self.model_config = model_config.copy()
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
            # Load the base wav2vec2 model
            self.wav2vec2 = transformers.Wav2Vec2Model.from_pretrained(
                model_name, **self.model_config
            )

        if classifier_only:
            # Freeze the wav2vec2 model
            for param in self.wav2vec2.parameters():
                param.requires_grad = False
            self.wav2vec2.eval()

        # Get the output dimension of wav2vec2
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

        # Add custom classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size, num_labels),
        )

        # Loss function
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # Set all modules to train mode
        self.train()

    def forward(self, input_values, labels=None):
        # Get wav2vec2 features
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state

        if self.mean_pool:
            # Pool the output (take mean of all tokens)
            pooled = torch.mean(hidden_states, dim=1)
        else:
            # Process through CNN
            # Transpose to (batch, channels, sequence) for Conv1d
            hidden_states = hidden_states.transpose(1, 2)
            pooled = self.cnn_classifier(hidden_states)

        # Pass through classifier
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
        num_labels,
        model_name,
        model_config,
        learning_rate=1e-5,
        weight_decay=0.001,
        classifier_only=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = Wav2Vec2Classifier(
            num_labels, model_name, model_config, classifier_only
        )
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.training_step_outputs = []

    def forward(self, input_values):
        return self.model(input_values=input_values)

    def training_step(self, batch, batch_idx):
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

    def validation_step(self, batch, batch_idx):
        outputs = self.model(input_values=batch["input_values"], labels=batch["label"])
        self.log("val_loss", outputs.loss, prog_bar=True)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        self.log(
            "val_acc", (predictions == batch["label"]).float().mean(), prog_bar=True
        )
        self.validation_step_outputs.append(
            {"preds": predictions, "labels": batch["label"]}
        )
        return {
            "val_loss": outputs.loss,
            "preds": predictions,
            "labels": batch["label"],
        }

    def test_step(self, batch, batch_idx):
        outputs = self.model(input_values=batch["input_values"], labels=batch["label"])
        self.log("test_loss", outputs.loss, prog_bar=True)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        self.log(
            "test_acc", (predictions == batch["label"]).float().mean(), prog_bar=True
        )
        # Store predictions and labels
        output_dict = {"preds": predictions, "labels": batch["label"]}
        self.test_step_outputs.append(output_dict)
        return output_dict

    def _compute_metrics(self, outputs, prefix):
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

    def on_validation_epoch_end(self):
        self._compute_metrics(self.validation_step_outputs, "val")
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        logger.info(f"Test epoch end: collected {len(self.test_step_outputs)} outputs")
        preds, labels = self._compute_metrics(self.test_step_outputs, "test")
        logger.info(f"Test epoch end: computed metrics on {len(preds)} predictions")
        # Don't clear test_step_outputs here as we need them for confusion matrix
        return {"preds": preds, "labels": labels}

    def on_train_epoch_end(self):
        self._compute_metrics(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )


class Wav2Vec2ForASR(torch.nn.Module):
    def __init__(self, model_name, model_config, vocab_size):
        super().__init__()

        # Copy model config and remove parameters that would conflict
        model_config_copy = model_config.copy()
        keys_to_remove = ["num_seconds", "mean_pool"]
        for key in keys_to_remove:
            if key in model_config_copy:
                del model_config_copy[key]

        # Load the wav2vec2 model with CTC head
        self.wav2vec2 = Wav2Vec2ForCTC.from_pretrained(model_name, **model_config_copy)

    def forward(self, input_values, labels=None, attention_mask=None):
        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        return outputs


class TranscriptionLitModel(pl.LightningModule):
    def __init__(
        self,
        model_name,
        model_config,
        processor,
        learning_rate=5e-5,
        weight_decay=0.001,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["processor"])
        self.processor = processor
        vocab_size = len(processor.tokenizer.get_vocab())
        self.model = Wav2Vec2ForASR(model_name, model_config, vocab_size)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.dialect_tokens = self._get_dialect_tokens_from_vocab()
        logger.info(
            f"Identified {len(self.dialect_tokens)} dialect tokens for extraction."
        )

    def _get_dialect_tokens_from_vocab(self):
        """Extracts potential dialect tokens (e.g., '<Dialect>') from the tokenizer's vocab."""
        vocab = self.processor.tokenizer.get_vocab()
        # Basic check: starts with '<', ends with '>', isn't a known special token
        special_tokens = ["<pad>", "<s>", "</s>", "<unk>", "|"]
        dialect_tokens = [
            token
            for token in vocab
            if token.startswith("<")
            and token.endswith(">")
            and token not in special_tokens
        ]
        # Sort by length descending to match longest tokens first (e.g., <State_District> before <State>)
        dialect_tokens.sort(key=len, reverse=True)
        return dialect_tokens

    def forward(self, input_values, labels=None, attention_mask=None):
        return self.model(
            input_values=input_values, labels=None, attention_mask=attention_mask
        )

    def training_step(self, batch, batch_idx):
        outputs = self.model(
            input_values=batch["input_values"],
            attention_mask=batch.get("attention_mask", None),
            labels=batch["labels"],
        )

        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)
        self.training_step_outputs.append({"loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            outputs = self.model(
                input_values=batch["input_values"],
                attention_mask=batch.get("attention_mask", None),
                labels=batch["labels"],
            )

            loss = outputs.loss
            self.log("val_loss", loss, prog_bar=True)

            logits = outputs.logits
            predicted_ids = torch.argmax(logits, dim=-1)

            # Convert predictions to text
            pred_strings = self.processor.batch_decode(predicted_ids)

            # Convert labels to text - handle padding correctly
            labels = batch["labels"].detach().cpu().numpy()
            # Replace -100 (padding token) with pad_token_id
            # labels = np.where(
            #     labels != -100, labels, self.processor.tokenizer.pad_token_id
            # )
            label_strings = self.processor.batch_decode(labels)

            # Calculate WER
            wer = jiwer.wer(label_strings, pred_strings)

            # Calculate dialect prediction accuracy
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

        self.log("val_wer", wer, prog_bar=True)
        self.log("val_dialect_acc", dialect_accuracy, prog_bar=True)

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

    def _extract_dialect(self, text):
        """Extract dialect tag from text if it exists."""
        match = re.search(r"<([^>]+)>", text)
        return match.group(1) if match else None

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_train_epoch_end(self):
        avg_loss = torch.stack([x["loss"] for x in self.training_step_outputs]).mean()
        self.log("train_loss_epoch", avg_loss)
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
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

        self.log("val_loss_epoch", avg_loss)
        self.log("val_wer_epoch", avg_wer)
        self.log("val_dialect_acc_epoch", avg_dialect_acc)

        # Log a few examples
        if self.validation_step_outputs:
            examples = self.validation_step_outputs[0]
            for i in range(min(3, len(examples["preds"]))):
                logger.info(f"Example {i}:")
                logger.info(f"  Pred: {examples['preds'][i]}")
                logger.info(f"  Label: {examples['labels'][i]}")

        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
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

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
