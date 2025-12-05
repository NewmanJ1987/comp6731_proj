import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# ============================================================
# 1. DATASET LOADING (DERMATOLOGY)
# ============================================================

class HeartDiseaseDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values if hasattr(y, 'values') else y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_heart_disease(path="heart.csv", val_size=0.2):
    """
    Loads the dermatology dataset from Kaggle.
    Expects last column to be the class label (1–6).
    Some versions of the dataset may have missing values encoded as '?'.
    """

    df = pd.read_csv(path)

    X = df.drop(columns=['target'])
    y = df['target']

    categorical_cols = ["cp", "restecg", "slope", "ca", "thal"]
        
    X_cat = pd.get_dummies(X[categorical_cols].astype(str))
    X_num = X.drop(columns=categorical_cols)

    X = np.hstack([X_num.values, X_cat.values])

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
    def __init__(self, input_dim, hidden_dim=32, num_classes=6):
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



def dmml_gaussian(features, logits, labels, classifier,
                  ce_weight=1.0, mm_weight=1.0, var_weight=0.3,
                  beta=3, sigma=1):
    """
    DMML using Gaussian similarities.
    """
    device = features.device
    N, D = features.shape
    num_classes = logits.shape[1]

    ce_loss = nn.functional.cross_entropy(logits, labels)

    W = classifier.weight  # [C, D]

    # Squared distances and Gaussian similarity
    diff = features.unsqueeze(1) - W.unsqueeze(0)
    dist2 = (diff ** 2).sum(dim=2)
    sim = torch.exp(-dist2 / (2 * sigma**2))

    s_pos = sim[torch.arange(N), labels]

    # Mask out the positive class
    mask = torch.zeros_like(sim, dtype=torch.bool)
    mask[torch.arange(N), labels] = True
    neg_sim = sim.masked_fill(mask, float("-inf"))

    s_neg, _ = neg_sim.max(dim=1)

    # Margin in similarity space: s_pos >= s_neg + beta
    violation = beta + s_neg - s_pos
    mm_loss = (torch.relu(violation)**2).mean()

    # Variance term
    var_loss = 0.0
    classes_present = 0
    for c in range(num_classes):
        group = features[labels == c]
        if len(group) > 1:
            mu = group.mean(dim=0, keepdim=True)
            var_loss += ((group - mu)**2).sum(dim=1).mean()
            classes_present += 1

    if classes_present > 0:
        var_loss /= classes_present

    total = ce_weight * ce_loss + mm_weight * mm_loss + var_weight * var_loss
    return total


# ============================================================
# 4. TRAINING UTILITIES
# ============================================================

def train_epoch_ce(model, loader, opt, device):
    model.train()
    total_loss = 0
    total = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        opt.zero_grad()
        logits = model(X)
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()
        opt.step()

        total_loss += loss.item() * len(X)
        total += len(X)

    return total_loss / total


def train_epoch_dmml(model, loader, opt, device, loss_fn):
    model.train()
    total_loss = 0
    total = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        opt.zero_grad()
        logits, feats = model(X, return_features=True)
        loss = loss_fn(feats, logits, y, model.classifier)
        loss.backward()
        opt.step()

        total_loss += loss.item() * len(X)
        total += len(X)

    return total_loss / total


def eval_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds = model(X).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(X)
    return correct / total


# ============================================================
# 5. MAIN EXPERIMENT
# ============================================================

def main():
    X_train, X_val, y_train, y_val = load_heart_disease("/Users/n_thurai/workspace/comp_6731/project/healthcare/heart.csv")

    train_ds = HeartDiseaseDataset(X_train, y_train)
    val_ds   = HeartDiseaseDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=128)

    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    # Lists for plotting
    ce_train, ce_acc = [], []
    dmm_g_train, dmm_g_acc = [], []

    # ====================================================
    # 1. CROSS ENTROPY
    # ====================================================
    print("\nTraining CE...")
    model_ce = MLPClassifier(input_dim, hidden_dim=32, num_classes=num_classes).to(device)
    opt_ce = torch.optim.Adam(model_ce.parameters(), lr=1e-3)

    for epoch in range(1, 100):
        tl = train_epoch_ce(model_ce, train_loader, opt_ce, device)
        acc = eval_accuracy(model_ce, val_loader, device)

        ce_train.append(tl)
        ce_acc.append(acc)

        print(f"[CE] Epoch {epoch:02d}  Loss={tl:.4f}  Acc={acc:.4f}")


    # ====================================================
    # 3. DMML - GAUSSIAN
    # ====================================================
    print("\nTraining DMML (Gaussian)...")
    model_dmm_g = MLPClassifier(input_dim, hidden_dim=32, num_classes=num_classes).to(device)
    opt_dmm_g = torch.optim.Adam(model_dmm_g.parameters(), lr=1e-3)

    for epoch in range(1, 100):
        tl = train_epoch_dmml(model_dmm_g, train_loader, opt_dmm_g, device, loss_fn=dmml_gaussian)
        acc = eval_accuracy(model_dmm_g, val_loader, device)

        dmm_g_train.append(tl)
        dmm_g_acc.append(acc)

        print(f"[DMML-G] Epoch {epoch:02d}  Loss={tl:.4f}  Acc={acc:.4f}")

    # ====================================================
    # OPTIONAL: PLOTTING
    # ====================================================
    try:
        

        epochs = range(1, 100)

        plt.figure(figsize=(12,5))
        plt.plot(epochs, ce_train, label="CE Loss")
        plt.plot(epochs, dmm_g_train, label="DMML-G Loss")
        plt.legend(); plt.title("Training Loss"); plt.show()

        plt.figure(figsize=(12,5))
        plt.plot(epochs, ce_acc, label="CE Acc")
        plt.plot(epochs, dmm_g_acc, label="DMML-G Acc")
        plt.legend(); plt.title("Validation Accuracy"); plt.show()
    except:
        print("Matplotlib not installed; skipping plots.")
    
    df = pd.read_csv("/Users/n_thurai/workspace/comp_6731/project/healthcare/heart.csv")
    X = df.drop(columns=['target'])
    y = df['target']

    categorical_cols = ["cp", "restecg", "slope", "ca", "thal"]
        
    X_cat = pd.get_dummies(X[categorical_cols].astype(str))
    X_num = X.drop(columns=categorical_cols)

    X = np.hstack([X_num.values, X_cat.values])
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    labels = y.values
    model_dmm_g.eval()

    from sklearn.decomposition import PCA

    # X_tensor is your FULL dataset (train + val + test)
    X_np = X_tensor.cpu().numpy()

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_np)

    plt.figure(figsize=(6,6))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap="coolwarm", alpha=0.7)
    plt.title("Raw Input Features (PCA Before Training)")
    plt.show()    

    visualize_tsne_embedding(model_dmm_g, X_tensor, labels, model_name="DMML-G")
    visualize_tsne_embedding(model_ce, X_tensor, labels, model_name="CE")

def visualize_tsne_embedding(model_ce, X_tensor, labels, model_name="CE"):
    with torch.no_grad():
        _, feats = model_ce(X_tensor, return_features=True)

    feats = feats.cpu().numpy()

    X_vis = TSNE(n_components=2, learning_rate="auto").fit_transform(feats)

    

    plt.scatter(X_vis[:,0], X_vis[:,1], c=labels, cmap="coolwarm")
    plt.title(f"t-SNE Embedding of Heart Disease Features ({model_name})")
    plt.show()


if __name__ == "__main__":
    main()
