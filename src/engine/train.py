import os, glob, argparse, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
import pandas as pd
from timm.data.mixup import Mixup
from src.data_utils.dataset import HemorrhageDataset, get_transforms
from src.models.get_model import get_model

def run_one_epoch(model, loader, criterion, optimizer, scaler, device, mixup_fn, is_training, epoch_desc):
    model.train() if is_training else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    progress_bar = tqdm(loader, desc=epoch_desc)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        if is_training:
            if mixup_fn: images, labels = mixup_fn(images, labels)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(images); loss = criterion(outputs, labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total_loss += loss.item(); progress_bar.set_postfix(loss=loss.item())
        else:
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    outputs = model(images); loss = nn.CrossEntropyLoss()(outputs, labels)
                total_loss += loss.item(); all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy()); all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader)
    val_f1 = f1_score(all_labels, all_preds, average='macro') if not is_training else 0.0
    return avg_loss, val_f1

def get_data_loaders(data_dir, image_size, batch_size, class_to_idx):
    train_paths = glob.glob(os.path.join(data_dir, 'train', '**', '*.png'), recursive=True)
    val_paths = glob.glob(os.path.join(data_dir, 'val', '**', '*.png'), recursive=True)
    train_labels, val_labels = [os.path.basename(os.path.dirname(p)) for p in train_paths], [os.path.basename(os.path.dirname(p)) for p in val_paths]
    train_dataset = HemorrhageDataset(train_paths, train_labels, class_to_idx, get_transforms('train', image_size))
    val_dataset = HemorrhageDataset(val_paths, val_labels, class_to_idx, get_transforms('val', image_size))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    class_names = sorted(os.listdir(os.path.join(args.data_dir, 'train')))
    num_classes, class_to_idx = len(class_names), {name: i for i, name in enumerate(class_names)}
    model = get_model(args.model_name, num_classes=num_classes).to(device)
    scaler = torch.amp.GradScaler('cuda')
    best_f1, patience_counter, training_log = 0.0, 0, []
    if not args.progressive_resizing:
        train_loader, val_loader = get_data_loaders(args.data_dir, args.image_size, args.batch_size, class_to_idx)
        optimizer, scheduler = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay), optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        mixup_fn = Mixup(mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha, label_smoothing=args.label_smoothing, num_classes=num_classes) if args.mixup_alpha > 0 or args.cutmix_alpha > 0 else None
        for epoch in range(args.epochs):
            train_loss, _ = run_one_epoch(model, train_loader, criterion, optimizer, scaler, device, mixup_fn, True, f"Epoch {epoch+1}/{args.epochs}")
            val_loss, val_f1 = run_one_epoch(model, val_loader, criterion, None, None, device, None, False, f"Epoch {epoch+1}/{args.epochs}")
            scheduler.step()
            training_log.append({'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss, 'val_f1': val_f1})
            if val_f1 > best_f1: best_f1 = val_f1; torch.save(model.state_dict(), os.path.join(args.output_dir, f"{args.model_name}_best.pth"))
    # (Progressive resizing logic is available in previous turns but omitted here for brevity)
    os.makedirs("outputs/logs", exist_ok=True)
    pd.DataFrame(training_log).to_csv(os.path.join("outputs/logs", f"{args.model_name}_training_log.csv"), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Advanced training for TBI classification.")
    parser.add_argument('--model_name', type=str, required=True, choices=['efficientnetv2_s', 'swin_tiny', 'convnext_tiny'])
    parser.add_argument('--data_dir', type=str, default='data/processed'); parser.add_argument('--output_dir', type=str, default='outputs/checkpoints')
    parser.add_argument('--weight_decay', type=float, default=0.05); parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--mixup_alpha', type=float, default=0.0); parser.add_argument('--cutmix_alpha', type=float, default=0.0)
    parser.add_argument('--patience', type=int, default=7); parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--image_size', type=int, default=224); parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32); parser.add_argument('--progressive_resizing', action='store_true')
    parser.add_argument('--stage1_epochs', type=int, default=20); parser.add_argument('--image_size_s1', type=int, default=224)
    parser.add_argument('--stage1_lr', type=float, default=1e-3); parser.add_argument('--freeze_epochs', type=int, default=5)
    parser.add_argument('--stage2_epochs', type=int, default=15); parser.add_argument('--image_size_s2', type=int, default=260)
    parser.add_argument('--stage2_lr', type=float, default=5e-5)
    args = parser.parse_args()
    train(args)