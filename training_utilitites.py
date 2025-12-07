import torch
from torch import nn

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


def dmml_gaussian(features, logits, labels, classifier,
                  ce_weight=1.0, mm_weight=1.0, var_weight=0.1,
                  beta=1.5, sigma=1.0):
    """
    DMML using Gaussian similarities.
    """
    N, _ = features.shape
    num_classes = logits.shape[1]

    ce_loss = nn.functional.cross_entropy(logits, labels)

    W = classifier.weight  # [C, D]

    # Squared distances and Gaussian similarity
    diff = features.unsqueeze(1) - W.unsqueeze(0)
    dist2 = (diff ** 2).sum(dim=2)
    sim = torch.exp(-dist2 / (sigma**2))

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
