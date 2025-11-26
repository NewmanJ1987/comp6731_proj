import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# -----------------------
# 1. Dataset & preprocessing
# -----------------------

class TitanicDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_titanic(path="train.csv", val_size=0.2, random_state=42):
    df = pd.read_csv(path)

    # Target
    y = df["Survived"].values

    # Features to use
    cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    df = df[cols]

    # Encode Sex
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    # Handle Embarked (one-hot)
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    embarked_dummies = pd.get_dummies(df["Embarked"], prefix="Embarked")
    df = pd.concat([df.drop(columns=["Embarked"]), embarked_dummies], axis=1)

    # Fill missing numeric values
    for col in ["Age", "Fare"]:
        df[col] = df[col].fillna(df[col].median())

    X = df.values

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=val_size,
        random_state=random_state,
        # stratify=y,
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    return X_train, X_val, y_train, y_val


# -----------------------
# 2. Model definition
# -----------------------

class MLPClassifier(nn.Module):
    """
    Simple MLP for Titanic.

    Returns logits and (optionally) features (penultimate layer).
    """
    def __init__(self, input_dim, hidden_dim=32, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, return_features: bool = False):
        features = self.net(x)              # [batch, hidden_dim]
        logits = self.classifier(features)  # [batch, num_classes]
        if return_features:
            return logits, features
        return logits


# -----------------------
# 3. Losses
# -----------------------

def dense_max_margin_loss_simplified(
    features: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    classifier: nn.Linear,
    ce_weight: float = 1.0,
    mm_weight: float = 1.0,
    var_weight: float = 0.1,
    margin: float = 1.0,
):
    """
    Simplified DMML:
      - Margin in *distance* space between embeddings and class centers.
      - Plus intra-class variance term.
    """
    device = features.device
    N, D = features.shape
    num_classes = logits.shape[1]

    # Cross-entropy
    ce_loss = nn.functional.cross_entropy(logits, labels)

    # Class centers = classifier weights (P1 choice)
    W = classifier.weight  # [C, D]

    # Distances to class centers
    diff = features.unsqueeze(1) - W.unsqueeze(0)  # [N, C, D]
    dist2 = (diff ** 2).sum(dim=2)                 # [N, C]

    # Positive distance
    pos = dist2[torch.arange(N, device=device), labels]

    # Negative distances (mask out the positive)
    mask = torch.zeros_like(dist2, dtype=torch.bool)
    mask[torch.arange(N, device=device), labels] = True
    neg_dist = dist2.masked_fill(mask, float("inf"))

    # Hardest negative (closest)
    neg_min, _ = neg_dist.min(dim=1)

    # Margin in distance space: we want d_neg >= d_pos + margin
    margin_violation = margin + pos - neg_min
    mm_loss = torch.relu(margin_violation).mean()

    # Intra-class variance loss
    var_loss = 0.0
    num_present_classes = 0
    for c in range(num_classes):
        mask_c = (labels == c)
        if mask_c.sum() > 1:
            feats_c = features[mask_c]
            mu_c = feats_c.mean(dim=0, keepdim=True)
            var_c = ((feats_c - mu_c) ** 2).sum(dim=1).mean()
            var_loss += var_c
            num_present_classes += 1

    if num_present_classes > 0:
        var_loss = var_loss / num_present_classes
    else:
        var_loss = torch.tensor(0.0, device=device)

    total_loss = (
        ce_weight * ce_loss
        + mm_weight * mm_loss
        + var_weight * var_loss
    )

    return total_loss, {
        "ce": ce_loss.item(),
        "mm": mm_loss.item(),
        "var": var_loss.item(),
    }


def dense_max_margin_loss_gaussian(
    features: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    classifier: nn.Linear,
    ce_weight: float = 1.0,
    mm_weight: float = 1.0,
    var_weight: float = 0.1,
    beta: float = 0.5,
    sigma: float = 1.0,
):
    """
    DMML-Gaussian:
      - Uses Gaussian similarity to class centers:
          s = exp(-||f - w_c||^2 / (2 * sigma^2))
      - Margin in *similarity* space:
          we want s_pos >= s_neg + beta
      - Same variance term as simplified version (on features).
    """
    device = features.device
    N, D = features.shape
    num_classes = logits.shape[1]

    # Cross-entropy
    ce_loss = nn.functional.cross_entropy(logits, labels)

    # Class centers
    W = classifier.weight  # [C, D]

    # Distances & Gaussian similarities
    sim = gaussian_similarity(features, sigma, W)   # [N, C]

    # Positive similarity
    s_pos = sim[torch.arange(N, device=device), labels]

    # Negative sims
    mask = torch.zeros_like(sim, dtype=torch.bool)
    mask[torch.arange(N, device=device), labels] = True
    neg_sim = sim.masked_fill(mask, float("-inf"))

    # Hardest negative (most similar)
    s_neg, _ = neg_sim.max(dim=1)

    # Margin in similarity space: want s_pos >= s_neg + beta
    violation = beta + s_neg - s_pos
    mm_loss = torch.relu(violation ** 2).mean()

    # Same variance term as before
    var_loss = 0.0
    num_present_classes = 0
    for c in range(num_classes):
        mask_c = (labels == c)
        if mask_c.sum() > 1:
            feats_c = features[mask_c]
            mu_c = feats_c.mean(dim=0, keepdim=True)
            var_c = ((feats_c - mu_c) ** 2).sum(dim=1).mean()
            var_loss += var_c
            num_present_classes += 1

    if num_present_classes > 0:
        var_loss = var_loss / num_present_classes
    else:
        var_loss = torch.tensor(0.0, device=device)

    total_loss = (
        ce_weight * ce_loss
        + mm_weight * mm_loss
        + var_weight * var_loss
    )

    return total_loss, {
        "ce": ce_loss.item(),
        "mm": mm_loss.item(),
        "var": var_loss.item(),
    }

def gaussian_similarity(features, sigma, W):
    diff = features.unsqueeze(1) - W.unsqueeze(0)  # [N, C, D]
    dist2 = (diff ** 2).sum(dim=2)                 # [N, C]

    sim = torch.exp(-dist2 / (sigma ** 2))
    return sim


# -----------------------
# 4. Training / evaluation loops
# -----------------------

def train_epoch_ce(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        total_samples += X.size(0)

    return total_loss / total_samples


def train_epoch_dmml_simplified(model, loader, optimizer, device,
                                ce_weight=1.0, mm_weight=1.0,
                                var_weight=0.1, margin=1.0):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits, feats = model(X, return_features=True)
        loss, _ = dense_max_margin_loss_simplified(
            feats, logits, y, model.classifier,
            ce_weight=ce_weight,
            mm_weight=mm_weight,
            var_weight=var_weight,
            margin=margin,
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        total_samples += X.size(0)

    return total_loss / total_samples


def train_epoch_dmml_gaussian(model, loader, optimizer, device,
                              ce_weight=1.0, mm_weight=1.0,
                              var_weight=0.1, beta=0.5, sigma=1.0):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits, feats = model(X, return_features=True)
        loss, _ = dense_max_margin_loss_gaussian(
            feats, logits, y, model.classifier,
            ce_weight=ce_weight,
            mm_weight=mm_weight,
            var_weight=var_weight,
            beta=beta,
            sigma=sigma,
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        total_samples += X.size(0)

    return total_loss / total_samples


def eval_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()
    return correct / total if total > 0 else 0.0


# -----------------------
# 5. Main experiment
# -----------------------

def main():
    # Load data
    X_train, X_val, y_train, y_val = load_titanic("train.csv")

    train_ds = TitanicDataset(X_train, y_train)
    val_ds = TitanicDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    input_dim = X_train.shape[1]
    num_classes = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ce_train_losses = []
    ce_val_accs = []

    dmml_s_train_losses = []
    dmml_s_val_accs = []

    dmml_g_train_losses = []
    dmml_g_val_accs = []


    # -------------------------
    # Model 1: Cross Entropy
    # -------------------------
    model_ce = MLPClassifier(input_dim=input_dim, hidden_dim=32, num_classes=num_classes).to(device)
    optim_ce = torch.optim.Adam(model_ce.parameters(), lr=1e-3)

    print("\n=== Training baseline (Cross Entropy) ===")
    for epoch in range(1, 21):
        train_loss = train_epoch_ce(model_ce, train_loader, optim_ce, device)
        val_acc = eval_accuracy(model_ce, val_loader, device)
        print(f"[CE]    Epoch {epoch:02d}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")
        ce_train_losses.append(train_loss)
        ce_val_accs.append(val_acc)

    # -------------------------
    # Model 2: DMML-simplified (distance)
    # -------------------------
    model_dmml_simple = MLPClassifier(input_dim=input_dim, hidden_dim=32, num_classes=num_classes).to(device)
    optim_dmml_simple = torch.optim.Adam(model_dmml_simple.parameters(), lr=1e-3)

    print("\n=== Training DMML (simplified distance version) ===")
    for epoch in range(1, 21):
        train_loss = train_epoch_dmml_simplified(
            model_dmml_simple, train_loader, optim_dmml_simple, device,
            ce_weight=1.0, mm_weight=1.0, var_weight=0.1, margin=1.0
        )
        val_acc = eval_accuracy(model_dmml_simple, val_loader, device)
        print(f"[DMML-dist] Epoch {epoch:02d}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")
        dmml_s_train_losses.append(train_loss)
        dmml_s_val_accs.append(val_acc)

    # -------------------------
    # Model 3: DMML-Gaussian (dense similarity)
    # -------------------------
    model_dmml_gauss = MLPClassifier(input_dim=input_dim, hidden_dim=32, num_classes=num_classes).to(device)
    optim_dmml_gauss = torch.optim.Adam(model_dmml_gauss.parameters(), lr=1e-3)

    print("\n=== Training DMML (Gaussian similarity version) ===")
    for epoch in range(1, 21):
        train_loss = train_epoch_dmml_gaussian(
            model_dmml_gauss, train_loader, optim_dmml_gauss, device,
            ce_weight=1.0, mm_weight=1.0, var_weight=0.1, beta=0.5, sigma=1.0
        )
        val_acc = eval_accuracy(model_dmml_gauss, val_loader, device)
        print(f"[DMML-gauss] Epoch {epoch:02d}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")
        dmml_g_train_losses.append(train_loss)
        dmml_g_val_accs.append(val_acc)
    

    epochs = range(1, 21)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, ce_train_losses, label='CE Train Loss')
    plt.plot(epochs, dmml_s_train_losses, label='DMML Simple Train Loss')
    plt.plot(epochs, dmml_g_train_losses, label='DMML Gaussian Train Loss')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
