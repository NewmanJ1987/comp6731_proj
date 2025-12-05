import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


# ============================================================
# 1. DATASET LOADING (DERMATOLOGY)
# ============================================================

class DermatologyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values if hasattr(y, 'values') else y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_dermatology(path="dermatology.csv", val_size=0.2):
    """
    Loads the dermatology dataset from Kaggle.
    Expects last column to be the class label (1–6).
    Some versions of the dataset may have missing values encoded as '?'.
    """

    df = pd.read_csv(path)

    # Replace '?' with NaN
    df = df.replace("?", np.nan).astype(float)

    # Fill missing values with column medians
    df = df.fillna(df.median())

    # The last column is the class label (1–6)
    y = df.iloc[:, -1].astype(int) - 1   # convert to 0–5
    X = df.iloc[:, :-1].values           # all other columns are features

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
# 3. LOSS FUNCTIONS
# ============================================================
def dmml_gaussian(features, logits, labels, classifier,
                  ce_weight=1.0, mm_weight=1.0, var_weight=0.1,
                  beta=1.5, sigma=1.0):
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


def eval_accuracy_loss_ce(model, loader, device):
    model.eval()
    val_losses = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            output_val = model(X)
            preds = output_val.argmax(dim=1)
            ce_loss_val = nn.functional.cross_entropy(output_val, y)
            val_losses += ce_loss_val.item() * X.size(0)
            correct += (preds == y).sum().item()
            total += len(X)
    return correct / total, val_losses/ len(loader.dataset)


def eval_accuracy_loss_dmml(model, loader, device, loss_fn):
    model.eval()
    val_losses = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            output_val = model(X)
            preds = output_val.argmax(dim=1)
            logits, feats = model(X, return_features=True)
            dmml_loss_val = loss_fn(feats, logits, y, model.classifier)
            val_losses += dmml_loss_val.item() * X.size(0)
            correct += (preds == y).sum().item()
            total += len(X)
    return correct / total, val_losses/ len(loader.dataset)

def visualize_pca_features_dermatology(df, X_tensor):
    X_np = X_tensor.cpu().numpy()

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_np)
    plt.figure(figsize=(6,6))
    cmap = plt.get_cmap("tab10")  # tab10 has 10 distinct colors
    labels = df.iloc[:, -1].astype(int) - 1
    labels_np = labels.to_numpy()  # ensure numpy array of ints
    # Scatter each class separately so the legend is correct
    for class_id in range(6):
        idx = (labels_np == class_id)
        plt.scatter(
            X_pca[idx, 0],
            X_pca[idx, 1],
            color=cmap(class_id),
            alpha=0.7,
            label=f"Class {class_id}",
            s=40
        )
    plt.title("Raw Input Features (PCA Before Training)")
    plt.legend()
    plt.show()
    return labels_np

def visualize_tsne_embedding_dermatology(model_ce, X_tensor, labels_np, model_name="CE"):
    with torch.no_grad():
        _, feats = model_ce(X_tensor, return_features=True)

    feats = feats.cpu().numpy()

    X_vis = TSNE(n_components=2, learning_rate="auto").fit_transform(feats)

    # Scatter each class separately so the legend is correct
    cmap = plt.get_cmap("tab10")  # tab10 has 10 distinct colors
    for class_id in range(6):
        idx = (labels_np == class_id)
        plt.scatter(
            X_vis[idx, 0],
            X_vis[idx, 1],
            color=cmap(class_id),
            alpha=0.7,
            label=f"Class {class_id}",
            s=40
        )
    plt.title(f"t-SNE Embedding of Heart Disease Features ({model_name})")
    plt.legend()
    plt.show()

# ============================================================
# 5. MAIN EXPERIMENT
# ============================================================

def main():
    X_train, X_val, y_train, y_val = load_dermatology("dermatology.csv")

    train_ds = DermatologyDataset(X_train, y_train)
    val_ds   = DermatologyDataset(X_val, y_val)

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
    model_ce = MLPClassifier(input_dim, hidden_dim=64, num_classes=num_classes).to(device)
    opt_ce = torch.optim.Adam(model_ce.parameters(), lr=1e-3)

    best_val_loss = float('inf')
    best_val_acc = 0
    patience = 10
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
        tl = train_epoch_dmml(model_dmm_g, train_loader, opt_dmm_g, device, loss_fn=dmml_gaussian)
        acc, loss = eval_accuracy_loss_dmml(model_dmm_g, val_loader, device, loss_fn=dmml_gaussian)

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

    # ====================================================
    # OPTIONAL: PLOTTING
    # ====================================================

        
    plt.figure(figsize=(12,5))
    plt.plot(ce_acc, label="CE Acc")
    plt.legend() 
    plt.title("Validation Accuracy")
    plt.show()


    plt.figure(figsize=(12,5))
    plt.plot(dmm_g_acc, label="DMML-G Acc")
    plt.legend()
    plt.title("Validation Accuracy")
    plt.show()




    df = pd.read_csv("dermatology.csv")

    # Replace '?' with NaN
    df = df.replace("?", np.nan).astype(float)

    # Fill missing values with column medians
    df = df.fillna(df.median())

    # The last column is the class label (1–6)
    X = df.iloc[:, :-1].values           # all other columns are features
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)


    

    # Create plots and visualize results. 
    labels_np = visualize_pca_features_dermatology(df, X_tensor)    
    visualize_tsne_embedding_dermatology(model_dmm_g, X_tensor, labels_np, model_name="DMML-G")
    visualize_tsne_embedding_dermatology(model_ce, X_tensor, labels_np, model_name="CE")




if __name__ == "__main__":
    main()
