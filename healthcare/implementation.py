import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


df = pd.read_csv("/Users/n_thurai/workspace/comp_6731/project/healthcare/healthcare_dataset.csv")


class DataModel(ABC):
    @abstractmethod
    def _pre_process_data(self):
        pass

    def _split_load_data(self):
        X, Y = self._pre_process_data()
        x_trainval, x_test, y_trainval, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
        x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, random_state = 42)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)

        # Convert to pytorch tensor
        x_train_tensor = torch.tensor(x_train, dtype = torch.float32)
        x_val_tensor = torch.tensor(x_val, dtype = torch.float32)
        x_test_tensor = torch.tensor(x_test, dtype = torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype = torch.long)
        y_val_tensor = torch.tensor(y_val.values, dtype = torch.long)
        y_test_tensor = torch.tensor(y_test.values, dtype = torch.long)

        train_ds = TensorDataset(x_train_tensor, y_train_tensor)
        val_ds = TensorDataset(x_val_tensor, y_val_tensor)
        test_ds = TensorDataset(x_test_tensor, y_test_tensor)

        self.feature_count = x_train.shape[1]
        self.train_loader = DataLoader(train_ds, batch_size = 32, shuffle = True)
        self.val_loader = DataLoader(val_ds, batch_size = 64, shuffle = False)
        self.test_loader = DataLoader(test_ds, batch_size = 64, shuffle = False)


class HealthcareModel(DataModel):

    def __init__(self, source):
        self.df = pd.read_csv(source)
        self._split_load_data()

    def _pre_process_data(self):
        # Calculate length of stay before dropping date columns
        self.df['Date of Admission'] = pd.to_datetime(self.df['Date of Admission'])
        self.df['Discharge Date'] = pd.to_datetime(self.df['Discharge Date'])
        self.df['Length of Stay'] = (self.df['Discharge Date'] - self.df['Date of Admission']).dt.days
        gender_mapping = {'Male': 0, 'Female': 1}
        self.df['Gender'] = self.df['Gender'].map(gender_mapping)

        # One-hot encode
        self.df = pd.get_dummies(self.df, columns=['Blood Type'], prefix='BloodType', dtype=int)
        self.df = pd.get_dummies(self.df, columns=['Medical Condition'], prefix='med_cond', dtype=int)
        # self.df = pd.get_dummies(self.df, columns=['Insurance Provider'], prefix='insurance', dtype=int)
        self.df = pd.get_dummies(self.df, columns=['Admission Type'], prefix='admission', dtype=int)
        self.df = pd.get_dummies(self.df, columns=['Medication'], prefix='med', dtype=int)

        # Drop the columns not being used.
        self.df = self.df.drop(columns=['Name', 'Doctor', 'Hospital', 'Room Number', 'Date of Admission', 'Discharge Date', 'Insurance Provider'])
        test_results_mapping = {'Normal': 0, 'Inconclusive': 1, 'Abnormal': 2}

        self.df['Test Results'] = self.df['Test Results'].map(test_results_mapping)

        X = self.df.drop(columns=['Test Results'])
        y = self.df['Test Results']

        return X , y



class HeartDiseaseModel(DataModel):
    
    def __init__(self, source):
        self.df = pd.read_csv(source)
        self._split_load_data()

    def _pre_process_data(self):
        X = self.df.drop(columns=['target'])
        y = self.df['target']

        return X , y

class Neural_Network(nn.Module):
  def __init__(self, input_size, number_of_classes, hidden_dim = 128):
    super(Neural_Network, self).__init__()
    self.fc1 = nn.Linear(input_size, 128)
    self.af1 = nn.ReLU()
    self.fc2 = nn.Linear(128, hidden_dim)
    self.af2 = nn.ReLU()
    self.output = nn.Linear(hidden_dim, number_of_classes)

  def forward(self, x):
    x = self.af1(self.fc1(x))
    x = self.af2(self.fc2(x))
    return x, self.output(x)


def gaussian_similarity(f, w_j, sigma):
    diff = f.unsqueeze(1) - w_j.unsqueeze(0)
    sim = torch.exp(- (diff ** 2).sum(dim = 2) / sigma)
    return sim

def lmv_loss(f, labels, sigma, prototypes):
    unique_labels = labels.unique()
    class_losses = []

    for c in unique_labels:
        idx = (c == labels)
        # feature space of the current class
        feat = f[idx]

        # check if there are at least 2 samples in the same class
        if feat.shape[0] < 2:
            continue

        sim = gaussian_similarity(feat, prototypes, sigma)
        min_sim = sim.min().detach()

        # compute loss
        class_loss = ((sim - min_sim) ** 2).mean()
        class_losses.append(class_loss)

    if not class_losses:
        return torch.tensor(0.0)

    # compute the avg loss for current batch
    L_mv = torch.stack(class_losses).mean()

    return L_mv

def lmm_loss(features, labels, prototypes, beta=0.5, sigma=1.0):
    """
    features:   [B, D]
    labels:     [B]
    prototypes: [C, D]  (e.g. model.output.weight)
    """
    B = features.size(0)
    sims = gaussian_similarity(features, prototypes, sigma=sigma)  # [B, C]

    pos_sim = sims[torch.arange(B), labels]  # [B]

    mask = torch.ones_like(sims, dtype=torch.bool)
    mask[torch.arange(B), labels] = False
    neg_sims = sims.masked_fill(~mask, float('-inf'))
    hard_neg_sim, _ = neg_sims.max(dim=1)  # [B]

    margin_violation = F.relu(beta + hard_neg_sim - pos_sim)
    L_mm = (margin_violation ** 2).mean()
    return L_mm


def plot_training_validation_metrics(loss_val_list, acc_val_list, loss_train_list, acc_train_list):
    plt.subplot(1,2,1)
    plt.plot(loss_train_list, label = 'Train Loss')
    plt.plot(loss_val_list, label = 'Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch VS Loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(acc_train_list, label = 'Train Accuracy')
    plt.plot(acc_val_list, label = 'Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Epoch VS Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data_model = HeartDiseaseModel("/Users/n_thurai/workspace/comp_6731/project/healthcare/heart.csv")
    data_model.df.info()
    neural_net = Neural_Network(input_size = data_model.feature_count, number_of_classes=3, hidden_dim=32)
    optimizer = torch.optim.Adam(neural_net.parameters(), lr = 0.001)

    epochs = 1000
    sigma = 1
    best_val_loss = float('inf')
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    best_model_state = None


    loss_val_list = []
    acc_val_list = []
    loss_train_list = []
    acc_train_list = []

    for epoch in range(epochs):
        neural_net.train()
        train_losses = 0
        correct_train = 0
        total_train = 0
        

        for x_batch, y_batch in data_model.train_loader:
            optimizer.zero_grad()
            f, output_train = neural_net(x_batch)
            
            cross_entropy_loss = F.cross_entropy(output_train, y_batch)
            L_mv = lmv_loss(f, y_batch, sigma, neural_net.output.weight)
            L_mm = lmm_loss(f, y_batch, neural_net.output.weight)
            L_mvmm = 0.5 * (L_mv + L_mm)
            loss_total = cross_entropy_loss + L_mvmm
            loss_total.backward()
            optimizer.step()

            train_losses += loss_total.item() * x_batch.size(0)

            preds_train = output_train.argmax(dim = 1)
            correct_train += (preds_train == y_batch).sum().item()
            total_train += y_batch.size(0)

        train_loss = train_losses / len(data_model.train_loader.dataset)
        train_acc = correct_train / total_train
        loss_train_list.append(train_loss)
        acc_train_list.append(train_acc)


        neural_net.eval()
        val_losses = 0
        correct = 0
        total = 0

        # Validation phase
        with torch.no_grad():
            for x_batch, y_batch in data_model.val_loader:
                f_val, output_val = neural_net(x_batch)
                ce_loss_val = F.cross_entropy(output_val, y_batch)
                L_mv_val = lmv_loss(f_val, y_batch, sigma, neural_net.output.weight)
                L_mm_val = lmm_loss(f_val, y_batch, neural_net.output.weight)
                L_mvmm_val = 0.5 * (L_mv_val + L_mm_val)
                loss_total_val = ce_loss_val + L_mvmm_val

                val_losses += loss_total_val.item() * x_batch.size(0)

                preds = output_val.argmax(dim = 1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        val_loss = val_losses / len(data_model.val_loader.dataset)
        val_acc = correct / total

        loss_val_list.append(val_loss)
        acc_val_list.append(val_acc)


        print(f'Epoch {epoch + 1}/ {epochs}, Training Loss: {train_loss:.4f}, Training Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = neural_net.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'The best validation loss: {best_val_loss:.4f}, the best validation accuracy: {best_val_acc:.4f}')
                print('Early Stopping!')
                break

    plot_training_validation_metrics(loss_val_list, acc_val_list, loss_train_list, acc_train_list)

    # Load the best model
    neural_net.load_state_dict(best_model_state)
    neural_net.eval()
    test_losses = 0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for x_batch, y_batch in data_model.test_loader:
            f_test, output_test = neural_net(x_batch)
            ce_test = F.cross_entropy(output_test, y_batch)
            L_mv_test = lmv_loss(f_test, y_batch, sigma, neural_net.output.weight)
            L_mm_test = lmm_loss(f_test, y_batch, neural_net.output.weight)
            L_mvmm_test = 0.5 * (L_mv_test + L_mm_test)
            loss_total_test = ce_test + L_mvmm_test

            test_losses += loss_total_test.item() * x_batch.size(0)

            preds_test = output_test.argmax(dim = 1)
            correct_test += (preds_test == y_batch).sum().item()
            total_test += y_batch.size(0)
        test_loss = test_losses / len(data_model.test_loader.dataset)
        test_acc = correct_test / total_test
        print(f'Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}')

