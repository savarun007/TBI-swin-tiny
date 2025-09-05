# TBI Classification with Advanced Deep Learning Models

This repository contains the complete code and methodology for developing and evaluating state-of-the-art deep learning models for the multi-class classification of Traumatic Brain Hemorrhages from head CT scan images. The project focuses on the `Swin Transformer` and `EfficientNetV2-S` architectures, employing advanced techniques like synthetic data balancing and multi-stage training to achieve high accuracy.

## Features

* **Data Balancing:** Automated generation of a large, perfectly balanced dataset (60,000 images) from an imbalanced source.
* **Advanced Model Training:** Flexible training scripts for both single-stage fine-tuning and multi-stage progressive resizing.
* **Rigorous Evaluation:** Automated generation of detailed classification reports and prediction files.
* **Statistical Validation:** A full suite of scripts for K-Fold Cross-Validation, McNemar's Test for model comparison, and Z-tests for significance.
* **Explainability (XAI):** Generation of Grad-CAM++ visualizations to interpret model predictions.
* **Publication-Ready Figures:** Standalone scripts to generate high-quality plots for research papers.

## Step-by-Step Workflow to Replicate

### Step 1: Setup

1.  **Create Environment:** Create and activate a Python virtual environment.
2.  **Install Dependencies:** Place the `requirements.txt` file in your project root and run:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Install PyTorch for GPU:** Go to the [official PyTorch website](https://pytorch.org/get-started/locally/) and install the version of PyTorch that matches your system's CUDA version. This is critical for GPU training.
4.  **Place Data:** Put your original, unbalanced dataset folders (e.g., `epidural`, `ne`) inside a `data/raw/` directory.

### Step 2: Data Preparation

1.  **Split the Dataset:** This takes the balanced data and splits it into `train`, `val`, and `test` sets inside `data/processed`.
    ```bash
    python -m src.data_utils.prepare_dataset --synthetic_data_dir data/synthetic
    ```

### Step 3: Model Training

Train your desired model. For the best-performing `Swin Transformer`:
```bash
python -m src.engine.train --model_name swin_tiny --epochs 80 --image_size 224 --learning_rate 1e-4 --batch_size 32 --weight_decay 0.05
