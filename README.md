TBI Classification with Swin Transformer
This project develops and evaluates a state-of-the-art Swin Transformer model for the multi-class classification of Traumatic Brain Hemorrhages from head CT scan images. This guide details the complete pipeline for replicating the best-performing model, swin_tiny, which achieved 82.02% test accuracy.

Features
Data Balancing: Automated generation of a large, perfectly balanced dataset (60,000 images) from an imbalanced source.

Advanced Model Training: A flexible training script to fine-tune the swin_tiny model.

Rigorous Evaluation: Automated generation of detailed classification reports.

Statistical Validation: Scripts to perform 5-Fold Cross-Validation and McNemar's Test to validate model stability and significance.

Explainability (XAI): Generation of Grad-CAM++ visualizations to interpret model predictions.

Step-by-Step Workflow
Step 1: Setup
Create Environment: Create and activate a Python virtual environment.

Install Dependencies: Place the provided requirements.txt in your project root and run:

pip install -r requirements.txt

Install PyTorch for GPU: Go to the official PyTorch website and install the version of PyTorch that matches your system's CUDA version. This is critical for GPU training.

Place Data: Put your original, unbalanced dataset folders (e.g., epidural, ne) inside the data/raw/ directory.

Step 2: Data Preparation
This two-step process creates the balanced dataset used for training.

Generate Synthetic Data: This script creates a new data/synthetic folder with 10,000 images per class. This will take a significant amount of time and disk space.

python -m src.data_utils.generate_synthetic_data --raw_data_dir data/raw

Split the Dataset: This script takes the balanced synthetic data and splits it into train, val, and test sets inside data/processed.

python -m src.data_utils.prepare_dataset --synthetic_data_dir data/synthetic

Step 3: Model Training
This command trains the swin_tiny model for 40 epochs on the balanced dataset. The best-performing checkpoint will be saved to outputs/checkpoints/swin_tiny_best.pth.

python -m src.engine.train --model_name swin_tiny --epochs 40 --image_size 224 --learning_rate 1e-4 --batch_size 32 --weight_decay 0.05

Step 4: Evaluation and Analysis
After training is complete, run these scripts to get the final performance metrics.

Generate Performance Report: This calculates the final test accuracy and saves the predictions needed for other scripts.

python -m src.engine.evaluate --model_name swin_tiny --image_size 224

Run Statistical Analysis: This performs a 5-Fold Cross-Validation (on a subset, for speed) and McNemar's Test to compare against the efficientnetv2_s model.

# (Optional but recommended) Run 5-Fold Cross-Validation
python -m src.engine.kfold_validation --model_name swin_tiny --epochs 8 --k_folds 5 --sample_fraction 0.3

# (Requires efficientnetv2_s_predictions.csv to exist)
python -m src.engine.statistical_comparison --model1 swin_tiny --model2 efficientnetv2_s

Generate XAI Visualizations: This creates Grad-CAM++ heatmaps for sample test images.

python -m src.xai.run_xai --model_name swin_tiny --image_size 224
