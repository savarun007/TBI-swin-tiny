import os, glob, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from src.data_utils.dataset import HemorrhageDataset, get_transforms
from src.models.get_model import get_model

def run_kfold_cv(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_paths = glob.glob(os.path.join(args.data_dir, '**', '*.png'), recursive=True)
    all_labels_str = [os.path.basename(os.path.dirname(p)) for p in all_paths]
    class_names, class_to_idx = sorted(list(set(all_labels_str))), {name: i for i, name in enumerate(sorted(list(set(all_labels_str))))}
    all_labels = [class_to_idx[name] for name in all_labels_str]

    _, sample_paths, _, sample_labels = train_test_split(all_paths, all_labels, test_size=args.sample_fraction, stratify=all_labels, random_state=42)
    
    full_dataset = HemorrhageDataset(sample_paths, [class_names[i] for i in sample_labels], class_to_idx, get_transforms('train', args.image_size))
    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(sample_paths, sample_labels)):
        train_subset, val_subset = Subset(full_dataset, train_idx), Subset(full_dataset, val_idx)
        val_subset.dataset.transforms = get_transforms('val', args.image_size)
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        model = get_model(args.model_name, num_classes=len(class_names), pretrained=True).to(device)
        optimizer, scheduler = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay), optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion, scaler = nn.CrossEntropyLoss(), torch.amp.GradScaler('cuda')

        for epoch in range(args.epochs):
            model.train()
            for images, labels in tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{args.epochs}"):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'): outputs = model(images); loss = criterion(outputs, labels)
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            scheduler.step()
        
        model.eval()
        all_preds, all_labels_fold = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images.to(device))
                all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy()); all_labels_fold.extend(labels.cpu().numpy())
        fold_accuracies.append(accuracy_score(all_labels_fold, all_preds))
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=fold_accuracies, color='skyblue'); sns.stripplot(y=fold_accuracies, color='black', jitter=0.1, size=8)
    plt.title(f'{args.k_folds}-Fold CV Accuracy for {args.model_name}')
    plt.savefig(os.path.join(args.output_dir, f"{args.model_name}_{args.k_folds}-fold_cv_results.png"), dpi=300)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="K-Fold Cross-Validation Script.")
    parser.add_argument('--model_name', type=str, required=True, choices=['efficientnetv2_s', 'swin_tiny', 'convnext_tiny'])
    parser.add_argument('--data_dir', type=str, default='data/synthetic')
    parser.add_argument('--output_dir', type=str, default='outputs/plots')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--k_folds', type=int, default=10)
    parser.add_argument('--sample_fraction', type=float, default=0.3)
    args = parser.parse_args()
    run_kfold_cv(args)