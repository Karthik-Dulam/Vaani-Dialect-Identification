# ASR & Dialect Recognition With Vaani and Respin Datasets

## Overview

This project focuses on training and evaluating Automatic Speech Recognition (ASR) and Dialect Recognition models, primarily using the ARTPARK-IISC/Vaani dataset and other relevant speech corpora. It leverages state-of-the-art transformer models like Wav2Vec2 and Whisper, integrated within a PyTorch Lightning framework for efficient training and evaluation.

## Features

*   **Flexible Model Architecture**: Supports various Wav2Vec2-based models (Base, XLS-R, Conformer) and Whisper models.
*   **Data Augmentation**: Includes options for noise injection and pitch shifting during training.
*   **Efficient Data Handling**: Uses the `datasets` library for loading and processing large datasets, including caching and multiprocessing.
*   **Structured Configuration**: Centralized configuration management using Python dataclasses in `config.py`, with overrides possible via YAML files.
*   **Experiment Tracking**: Integrates with Weights & Biases (Wandb) for logging metrics and results (optional).
*   **Reproducibility**: Seed setting for consistent results.
*   **Evaluation Suite**: Scripts for detailed evaluation, including classification reports, confusion matrices, Word Error Rate (WER), and error analysis.
*   **Hyperparameter Tuning**: Includes support for Optuna-based hyperparameter optimization.

## Datasets

This project is designed to work with the following datasets:

1.  **ARTPARK-IISC/Vaani**:
    *   **Description**: A large-scale Indian speech dataset featuring diverse languages and dialects. Used for both dialect classification and ASR. Contains transcribed and untranscribed speech.
    *   **Access**: Requires a Hugging Face account and agreement to the dataset's terms. Configure access using your Hugging Face token (`HF_TOKEN_K`).
    *   **Citation**: If you use this data, please cite the Vaani project:
        ```
        @misc{vaani2025,
          author       = {VAANI Team},
          title        = {VAANI: Capturing the Language Landscape for an Inclusive Digital India (Phase 1)},
          howpublished = {\url{https://vaani.iisc.ac.in/}},
          year         = {2025}
        }
        ```
    *   **Contact**: `vaanicontact@gmail.com`

2.  **ARTPARK-IISc/Vaani-transcription-part**:
    *   **Description**: A subset of the Vaani dataset specifically prepared for ASR tasks.
    *   **Access**: Available on the Hugging Face Hub.

3.  **Respin**:
    *   **Description**: An unpublished dataset prepared by SPIRE LAB at IISc, contains audio samples with transcriptions of various indic languages along with a dialect label.
    *   **Access**: Please contact SPIRE Lab @ IISc.

## Setup

1.  **Clone the Repository**:
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Create Environment (Recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Environment Variables**: Set the following environment variables (e.g., in your shell profile like `.bashrc` or `.zshrc`, or export them before running scripts):
    *   `HF_TOKEN_K`: Your Hugging Face API token (required for accessing private/gated datasets like Vaani).
    *   `WANDB_KEY_K`: Your Weights & Biases API key (optional, for experiment logging).

5.  **Data Configuration**:
    *   Verify and potentially update the `data_dir` and `cache_dir` paths in `config.py` (these are the base defaults).
    *   Ensure the Respin dataset files are accessible at the paths specified in `data.py` if you intend to use it.
    *   Review the default settings in `config.py` for `CONFIG`, `CACHE_SENSITIVE_CONFIG`, `TRANSCRIPTION_CONFIG`, and `TRANSCRIPTION_CACHE_CONFIG`.
    *   Modify `classification_config.yaml` or `transcription_config.yaml` to override specific default parameters for your experiments.

## Configuration (`config.py` and YAML files)

Configuration is managed through a combination of `config.py` and YAML files:

*   **`config.py`**: Defines the base dataclass structures (`Config`, `CacheConfig`, etc.) and provides default configurations (`CONFIG`, `CACHE_SENSITIVE_CONFIG`, `TRANSCRIPTION_CONFIG`, `TRANSCRIPTION_CACHE_CONFIG`). Key sections within the dataclasses include:
    *   **Base Paths**: `data_dir`, `cache_dir`, `base_dir` (for model outputs).
    *   **DatasetConfig**: Define which dataset (`name`), specific configurations (`configs` - e.g., languages, districts), and dataset revision (`revision`) to use.
    *   **ModelConfig**: Specify the Hugging Face model identifier (`name`), processor, feature extractor, and tokenizer.
    *   **ModelArchConfig**: Control model hyperparameters like dropout rates, pooling strategy, and CTC/attention settings.
    *   **AugmentationConfig**: Enable/disable and configure data augmentation (noise, pitch shift).
    *   **TrainingConfig**: Set training parameters like learning rate, batch size, epochs, weight decay, gradient accumulation, and evaluation frequency.
    *   **CacheConfig**: Parameters affecting data loading and preprocessing, such as language filtering, minimum audio duration (`min_seconds`), task type (`dialect`, `transcription`), and data splitting strategy (`split_key`).
    *   **TuningConfig**: Defines ranges for hyperparameter tuning.
*   **YAML Files (`classification_config.yaml`, `transcription_config.yaml`)**: These files allow you to override the default settings defined in `config.py` without modifying the Python code directly. Specify the parameters you want to change in the appropriate YAML file. Use the key `cache_config_updates` within the YAML to target `CacheConfig` parameters.
*   **`load_config` function**: Used by the training/tuning scripts. It loads the appropriate base configuration from `config.py` based on the task (`classification` or `transcription`) and then applies any overrides found in the specified YAML file (`--config_yaml` argument).
*   **`update_config` function**: Used internally to apply further updates (e.g., from hyperparameter tuning trials) after the initial configuration is loaded.

The order of precedence is: Default (`config.py`) < YAML Override (`*.yaml`) < Script/Tuning Override.

## Usage

All training and tuning scripts require specifying the configuration YAML file and the devices (GPU IDs) to use.

### Training

*   **Dialect Classification**:
    ```bash
    python train_lightning.py --config_yaml classification_config.yaml --devices 0
    ```
    *(Loads base classification config, applies overrides from `classification_config.yaml`, runs on GPU 0)*

*   **Automatic Speech Recognition (ASR)**:
    ```bash
    python train_transcription.py --config_yaml transcription_config.yaml --devices 0,1
    ```
    *(Loads base transcription config, applies overrides from `transcription_config.yaml`, runs on GPUs 0 and 1)*

*   **Custom Tokenizer Training** (if needed for ASR):
    ```bash
    python train_tokenizer.py 
    ```
    *(Run this before `train_transcription.py` if you are creating a new tokenizer based on your specific dataset subset)*

### Evaluation

*   **Classification Model Evaluation**: Evaluates the *latest* best checkpoint found in the configured output directory (determined by the config used for training).
    ```bash
    python evaluate.py # Note: This script might need updates to load config via YAML/args.
    ```
    *(Generates confusion matrix, classification report, and error analysis)*

*   **Transcription Model Evaluation**:
    ```bash
    python evaluate_transcription.py --checkpoint <path_to_asr_checkpoint.ckpt> --output_dir <directory_to_save_results>
    ```
    *(Calculates WER and saves predicted vs. actual transcripts)*

*   **Quick Dialect Accuracy Check (for ASR models)**: Checks dialect tag prediction accuracy embedded in ASR output.
    ```bash
    python check_dialect_accuracy.py --checkpoint <path_to_asr_checkpoint.ckpt> --num_samples 100 --split validation
    ```

### Hyperparameter Tuning

*   **Run Optuna Search (for Classification)**:
    ```bash
    python hyperparameter_tuning.py --config_yaml classification_config.yaml --devices 0,1,2,3
    ```
    *(Uses `classification_config.yaml` for base settings, searches for optimal hyperparameters on GPUs 0-3 based on the objective defined in the script)*

## Project Structure

*   `config.py`: Central configuration file using dataclasses, defines defaults.
*   `classification_config.yaml`: YAML file for overriding classification defaults.
*   `transcription_config.yaml`: YAML file for overriding transcription defaults.
*   `data.py`: Handles dataset loading, preprocessing, augmentation, and splitting logic. Defines `VaaniDataset`.
*   `models_vaani.py`: Defines the model architectures (`Wav2Vec2Classifier`, `Wav2Vec2ForASR`, `WhisperForASR`) and the PyTorch Lightning modules (`LitModel`, `TranscriptionLitModel`).
*   `train_lightning.py`: Main script for training classification models.

This project relies heavily on the following open-source libraries and datasets:

*   [Hugging Face Transformers](https://github.com/huggingface/transformers)
*   [Hugging Face Datasets](https://github.com/huggingface/datasets)
*   [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers)
*   `requirements.txt`: List of Python dependencies.
*   `README.md`: This file.
*   `models/`: Default directory for saving trained model checkpoints and outputs (can be changed in `config.py`).
*   `wandb/`: Directory created by Weights & Biases for run logs.

## Acknowledgements

This project relies heavily on the following open-source libraries and datasets:

*   [Hugging Face Transformers](https://github.com/huggingface/transformers)
*   [Hugging Face Datasets](https://github.com/huggingface/datasets)
*   [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers)
*   [PyTorch](https://pytorch.org/)
*   [PyTorch Lightning](https://www.pytorchlightning.ai/)
*   [ARTPARK-IISC/Vaani Dataset](https://huggingface.co/datasets/ARTPARK-IISC/Vaani) (Project Vaani by ARTPARK and IISc)
*   [Weights & Biases](https://wandb.ai/)
*   [Optuna](https://optuna.org/)
*   [Librosa](https://librosa.org/)
*   [JiWER](https://github.com/jitsi/jiwer)
*   [Polars](https://pola.rs/)
*   [PyYAML](https://pyyaml.org/)

Special thanks to the creators and maintainers of these resources.

## License

This project is licensed under the MIT License.

```text
MIT License

Copyright (c) 2024 <Your Name or Organization>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
