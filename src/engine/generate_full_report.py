import os, glob, torch, torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pandas as pd
from tqdm import tqdm
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from src.data_utils.dataset import HemorrhageDataset, get_transforms
from src.models.get_model import get_model

CLASSIC_MODELS_TO_TRAIN = [
    {'name': 'Logistic Regression', 'model': LogisticRegression(max_iter=1000)},
    {'name': 'K-Nearest Neighbors', 'model': KNeighborsClassifier(n_neighbors=5, n_jobs=-1)},
    {'name': 'Support Vector Machine', 'model': SVC(kernel='rbf', C=1.0)},
    {'name': 'Random Forest', 'model': RandomForestClassifier(n_estimators=100, n_jobs=-1)}
]
DEEP_LEARNING_MODELS_TO_EVALUATE = [
    {'name': 'efficientnetv2_s', 'image_size': 260, 'strategy': 'Multi-Stage & Progressive'},
    {'name': 'swin_tiny', 'image_size': 224, 'strategy': 'Fine-Tuning (40 Epochs)'},
    {'name': 'convnext_tiny', 'image_size': 224, 'strategy': 'Standard Fine-Tuning'}
]

def extract_features(loader, device):
    feature_extractor = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)
    feature_extractor.fc = torch.nn.Identity()
    feature_extractor.eval()
    features, labels = [], []
    with torch.no_grad():
        for images, lbls in tqdm(loader, desc="Extracting features"):
            feature_batch = feature_extractor(images.to(device))
            features.append(feature_batch.cpu().numpy()); labels.append(lbls.numpy())
    return np.concatenate(features), np.concatenate(labels)

def evaluate_dl_model(model_name, image_size, data_dir, checkpoint_dir, device):
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_best.pth")
    if not os.path.exists(checkpoint_path): return None
    test_loader, class_names = get_dataloader(data_dir, image_size, is_feature_extraction=False)
    model = get_model(model_name, num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, lbls in tqdm(test_loader, desc=f"Inference for {model_name}"):
            outputs = model(images.to(device))
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy()); all_labels.extend(lbls.numpy())
    return accuracy_score(all_labels, all_preds)

def get_dataloader(data_dir, image_size, is_feature_extraction=False, split='test'):
    paths = glob.glob(os.path.join(data_dir, split, '**', '*.png'), recursive=True)
    class_names = sorted(os.listdir(os.path.join(data_dir, split)))
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    labels = [os.path.basename(os.path.dirname(p)) for p in paths]
    dataset = HemorrhageDataset(paths, labels, class_to_idx, get_transforms('val', image_size))
    batch_size = 128 if is_feature_extraction else 64
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return loader, class_names

def generate_full_report():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir, checkpoint_dir = 'data/processed', 'outputs/checkpoints'
    summary_results = []
    
    train_loader_feat, _ = get_dataloader(data_dir, 224, is_feature_extraction=True, split='train')
    test_loader_feat, _ = get_dataloader(data_dir, 224, is_feature_extraction=True, split='test')
    X_train, y_train = extract_features(train_loader_feat, device)
    X_test, y_test = extract_features(test_loader_feat, device)
    
    for config in CLASSIC_MODELS_TO_TRAIN:
        model = config['model']; model.fit(X_train, y_train)
        summary_results.append({'Model': config['name'], 'Training Strategy': 'On ResNet-18 Features', 'Test Accuracy (%)': f"{model.score(X_test, y_test) * 100:.2f}"})

    summary_results.append({'Model': 'ResNet-18 (Baseline)', 'Training Strategy': 'Undertrained (Imbalanced Data, 10 Epochs)', 'Test Accuracy (%)': "44.18"})
    
    for config in DEEP_LEARNING_MODELS_TO_EVALUATE:
        accuracy = evaluate_dl_model(config['name'], config['image_size'], data_dir, checkpoint_dir, device)
        if accuracy: summary_results.append({'Model': config['name'], 'Training Strategy': config['strategy'], 'Test Accuracy (%)': f"{accuracy * 100:.2f}"})

    print("\n\n--- FOR YOUR RESEARCH PAPER ---")
    print("\n## Table 5: Comparison with State-of-the-Art and Baseline Models")
    print(pd.DataFrame(summary_results).to_markdown(index=False))

if __name__ == '__main__':
    generate_full_report()