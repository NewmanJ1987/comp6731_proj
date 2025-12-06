import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from training_utilitites import train_epoch_ce, train_epoch_dmml, eval_accuracy_loss_ce, eval_accuracy_loss_dmml, dmml_gaussian
from functools import partial

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))



# ============================================================
# 1. DATASET LOADING 
# ============================================================

class HeartDiseaseDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values if hasattr(y, 'values') else y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def preprocess_heart_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates()
    X = df.drop(columns=['target'])
    y = df['target']

    categorical_cols = ["cp", "restecg", "slope", "ca", "thal"]
        
    X_cat = pd.get_dummies(X[categorical_cols].astype(str))
    X_num = X.drop(columns=categorical_cols)

    X = np.hstack([X_num.values, X_cat.values])

    return X, y


def load_heart_disease(path="heart.csv", val_size=0.2):
    X, y = preprocess_heart_data(path)
    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    return X_train, X_val, y_train, y_val


# ============================================================
# 2. MODEL: SIMPLE MLP CLASSIFIER
# ============================================================

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, return_features=False):
        feats = self.net(x)
        logits = self.classifier(feats)
        if return_features:
            return logits, feats
        return logits

# ============================================================
# 3. VISUALIZATION UTILITIES
# ============================================================

def visualize_accuracy(acc_list, label="CE Acc"):
    plt.figure(figsize=(12,5))
    plt.plot(acc_list, label=label)
    plt.legend()
    plt.title("Validation Accuracy")
    plt.show()

def visualize_pca_features_heart_disease(X_tensor, labels):
    X_np = X_tensor.cpu().numpy()

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_np)

    plt.figure(figsize=(6,6))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap="coolwarm", alpha=0.7)
    plt.title("Raw Input Features (PCA Before Training)")
    plt.show()

def visualize_tsne_embedding_heart_disease(model_ce, X_tensor, labels, model_name="CE"):
    with torch.no_grad():
        _, feats = model_ce(X_tensor, return_features=True)

    feats = feats.cpu().numpy()

    X_vis = TSNE(n_components=2, learning_rate="auto").fit_transform(feats)

    

    plt.scatter(X_vis[:,0], X_vis[:,1], c=labels, cmap="coolwarm")
    plt.title(f"t-SNE Embedding of Heart Disease Features ({model_name})")
    plt.show()


# ============================================================
# 4. MAIN EXPERIMENT
# ============================================================

def main():
    # Load dataset path from environment variable
    heart_csv_path = os.getenv('HEART_DATASET_CSV')
    if not heart_csv_path:
        raise ValueError("HEART_DATASET_CSV environment variable not set in .env file")
    
    X_train, X_val, y_train, y_val = load_heart_disease(heart_csv_path)

    train_ds = HeartDiseaseDataset(X_train, y_train)
    val_ds   = HeartDiseaseDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=128)

    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    # Lists for plotting: 
    # ce_train stores CE training loss per epoch, ce_acc stores CE validation accuracy per epoch
    # dmm_g_train stores DMML-G training loss per epoch, dmm_g_acc stores DMML-G validation accuracy per epoch
    ce_train, ce_acc = [], []
    dmm_g_train, dmm_g_acc = [], []

    # ====================================================
    # 1. CROSS ENTROPY
    # ====================================================
    print("\nTraining CE...")
    model_ce = MLPClassifier(input_dim, hidden_dim=64, num_classes=num_classes).to(device)
    opt_ce = torch.optim.Adam(model_ce.parameters(), lr=1e-3)

    best_val_loss = float('inf')
    best_val_acc = 0
    patience = 5
    patience_counter = 0
    for epoch in range(1, 100):
        tl = train_epoch_ce(model_ce, train_loader, opt_ce, device)
        acc, loss = eval_accuracy_loss_ce(model_ce, val_loader, device)

        ce_train.append(tl)
        ce_acc.append(acc)
        if loss < best_val_loss:
            best_val_loss = loss
            best_val_acc = acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'The best validation loss: {best_val_loss:.4f}, the best validation accuracy: {best_val_acc:.4f}')
                print('Early Stopping!')
                break

        print(f"[CE] Epoch {epoch:02d}  Loss={tl:.4f}  Acc={acc:.4f}")

            

    # ====================================================
    # 3. DMML - GAUSSIAN
    # ====================================================
    print("\nTraining DMML (Gaussian)...")
    model_dmm_g = MLPClassifier(input_dim, hidden_dim=64, num_classes=num_classes).to(device)
    opt_dmm_g = torch.optim.Adam(model_dmm_g.parameters(), lr=1e-3)

    best_val_loss = float('inf')
    best_val_acc = 0
    patience = 5
    patience_counter = 0

    for epoch in range(1, 100):
        partial_dmml_gaussian = partial(dmml_gaussian, ce_weight=1.0, mm_weight=1.0, var_weight=0.3, beta=3, sigma=1)
        tl = train_epoch_dmml(model_dmm_g, train_loader, opt_dmm_g, device, loss_fn=partial_dmml_gaussian)
        acc, loss = eval_accuracy_loss_dmml(model_dmm_g, val_loader, device, loss_fn=partial_dmml_gaussian)

        dmm_g_train.append(tl)
        dmm_g_acc.append(acc)
        if loss < best_val_loss:
            best_val_loss = loss
            best_val_acc = acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'The best validation loss: {best_val_loss:.4f}, the best validation accuracy: {best_val_acc:.4f}')
                print('Early Stopping!')
                break

        print(f"[DMML-G] Epoch {epoch:02d}  Loss={tl:.4f}  Acc={acc:.4f}")

    
    visualize_accuracy(ce_acc, label="CE Acc")
    visualize_accuracy(dmm_g_acc, label="DMML-G Acc")


    X, y = preprocess_heart_data(heart_csv_path)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    labels = y.values


    visualize_pca_features_heart_disease(X_tensor, labels)    
    visualize_tsne_embedding_heart_disease(model_dmm_g, X_tensor, labels, model_name="DMML-G")
    visualize_tsne_embedding_heart_disease(model_ce, X_tensor, labels, model_name="CE")




if __name__ == "__main__":
    main()
