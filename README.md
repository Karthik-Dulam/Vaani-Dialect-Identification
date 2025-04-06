# Vaani: ASR & Dialect Recognition

Vaani is a project for training and evaluating automatic speech recognition models with a focus on dialect detection using state-of-the-art models (e.g., Wav2Vec2). It contains scripts for both classification and transcription tasks, as well as utilities for data processing and tokenizer training.

## File Structure

- **train_lightning.py**: Main training script for audio classification.
- **train_transcription.py**: Training script for transcription models.
- PyTorch Lightning-based training implementation.
- **evaluate.py / check_dialect_accuracy.py**: Scripts for evaluating model performance and accuracy on dialect predictions.
- **models.py**: Contains model definitions, including custom classifier and ASR model implementations.
- **data.py**: Data loader and preprocessing utilities.
- **config.py**: Project configuration, dataset, model settings, and environment variables.
- **train_tokenizer.py**: Script for training a ByteLevel BPE tokenizer.

## Setup & Dependencies

1. Install the necessary Python packages (e.g., `torch`, `transformers`, `pytorch-lightning`, `datasets`, `jiwer`, `polars`, etc.).
2. Set the required environment variables (e.g., HF token, WANDB API key) as described in `config.py`.
3. Ensure you have access to the required datasets referenced in `config.py`.

## Usage

### Training
- **Classification Training**:  
  Run `python train_lightning.py` to start training a classification model.
  
- **Transcription Training**:  
  Use `python train_transcription.py` to train the transcription model.

### Evaluation
- **Model Evaluation**:  
  Run `python evaluate.py` or use the provided Jupyter notebook (`eval.ipynb`) to evaluate trained models.
  
- **Dialect Accuracy Check**:  
  Use `python check_dialect_accuracy.py --checkpoint <path-to-checkpoint>` to view detailed dialect and transcription accuracy.

### Tokenizer Training
To train the tokenizer for transcription, run:
```
python train_tokenizer.py
```

## Configuration

Review and modify `config.py` to adjust:
- Data paths and caching parameters.
- Model and training hyperparameters.
- Dataset settings and augmentation options.

## Notes

- The project supports both classification and transcription tasks.
- Custom data collate functions and data augmentation are implemented in `data.py`.
- Model checkpointing, logging (TensorBoard, wandb), and evaluation metrics are set up in the training scripts.

For further details, please refer to the comments in each source file.
