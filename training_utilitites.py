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