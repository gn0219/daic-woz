# Multitask vs Single Task Learning for Depression Detection (Improved)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import os
import random

from src import plot_result, plot_cm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# =====================
# 1. Data Loading & Preprocessing
# =====================


def load_and_prepare_data():
    df = pd.read_csv('data/df.csv')
    smile_features = pd.read_csv('data/smile_features.csv')
    utterance_features = pd.read_csv('data/utterance_features.csv')

    smile_columns = smile_features.columns[1:].tolist()
    utterance_columns = utterance_features.columns[1:].tolist()

    missing_columns = df.columns[df.isnull().any()]
    df = df.drop(columns=missing_columns)
    utterance_columns = [col for col in utterance_columns if col not in missing_columns]
    all_columns = smile_columns + utterance_columns

    feature_df = df[['Participant_ID', 'Split', 'PHQ8_Binary', 'Gender']].copy()
    feature_df = feature_df.merge(smile_features, on='Participant_ID')
    feature_df = feature_df.merge(utterance_features, on='Participant_ID')

    train_df = feature_df[feature_df['Split'] == 'train'].copy()
    dev_df = feature_df[feature_df['Split'] == 'dev'].copy()
    test_df = feature_df[feature_df['Split'] == 'test'].copy()

    X_train = train_df[all_columns].values
    y_main_train = train_df['PHQ8_Binary'].values
    y_aux_train = train_df['Gender'].values

    X_dev = dev_df[all_columns].values
    y_main_dev = dev_df['PHQ8_Binary'].values
    y_aux_dev = dev_df['Gender'].values

    X_test = test_df[all_columns].values
    y_main_test = test_df['PHQ8_Binary'].values
    y_aux_test = test_df['Gender'].values
    test_ids = test_df['Participant_ID'].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_dev = scaler.transform(X_dev)
    X_test = scaler.transform(X_test)

    return X_train, y_main_train, y_aux_train, X_dev, y_main_dev, y_aux_dev, X_test, y_main_test, y_aux_test, test_ids

# =====================
# 2. Dataset Classes
# =====================

class MultiTaskDataset(Dataset):
    def __init__(self, X, y_main, y_aux):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_main = torch.tensor(y_main, dtype=torch.float32).unsqueeze(1)
        self.y_aux = torch.tensor(y_aux, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_main[idx], self.y_aux[idx]

class SingleTaskDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# =====================
# 3. Improved Models
# =====================

class DeepMultiTaskMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        layers = []
        for in_dim, out_dim in zip([input_dim] + hidden_dims[:-1], hidden_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.shared = nn.Sequential(*layers)

        self.head_main = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.head_aux = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        h = self.shared(x)
        out_main = torch.sigmoid(self.head_main(h))
        out_aux = torch.sigmoid(self.head_aux(h))
        return out_main, out_aux

class DeepSingleTaskMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        layers = []
        for in_dim, out_dim in zip([input_dim] + hidden_dims[:-1], hidden_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.model(x))


class DeepMultiTaskMLP_Improved(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout=0.4):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.LayerNorm(hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.se = nn.Sequential(
            nn.Linear(hidden_dims[2], hidden_dims[2]//4),
            nn.ReLU(),
            nn.Linear(hidden_dims[2]//4, hidden_dims[2]),
            nn.Sigmoid()
        )
        self.head_main = nn.Sequential(
            nn.Linear(hidden_dims[2], 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.head_aux = nn.Sequential(
            nn.Linear(hidden_dims[2], 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        h = self.shared(x)
        attention = self.se(h)
        h = h * attention
        out_main = torch.sigmoid(self.head_main(h))
        out_aux = torch.sigmoid(self.head_aux(h))
        return out_main, out_aux

class DeepSingleTaskMLP_Improved(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout=0.4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.LayerNorm(hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[2], 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.model(x))

class MultiTaskMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        layers = []
        for in_dim, out_dim in zip([input_dim]+hidden_dims[:-1], hidden_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.shared = nn.Sequential(*layers)

        self.head_main = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.head_aux = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        h = self.shared(x)
        out_main = torch.sigmoid(self.head_main(h))
        out_aux = torch.sigmoid(self.head_aux(h))
        return out_main, out_aux

class SingleTaskMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        layers = []
        for in_dim, out_dim in zip([input_dim]+hidden_dims[:-1], hidden_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.model(x))

# =====================
# 4. Training Functions with Early Stopping
# =====================

def train_mtl(model, train_loader, val_loader, epochs=50, lr=1e-3, patience=20):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = np.inf
    best_model = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_main, y_aux in train_loader:
            X_batch, y_main, y_aux = X_batch.to(device), y_main.to(device), y_aux.to(device)
        
            optimizer.zero_grad()
            out_main, out_aux = model(X_batch)
            loss_main = criterion(out_main, y_main)
            loss_aux = criterion(out_aux, y_aux)
            loss = loss_main + 0.3 * loss_aux
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_main, y_aux in val_loader:
                X_batch, y_main, y_aux = X_batch.to(device), y_main.to(device), y_aux.to(device)

                out_main, out_aux = model(X_batch)
                loss_main = criterion(out_main, y_main)
                loss_aux = criterion(out_aux, y_aux)
                val_loss += (loss_main + 0.3 * loss_aux).item()

        print(f"[MTL] Epoch {epoch+1} | Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping!")
                break

    model.load_state_dict(best_model)

def train_stl(model, train_loader, val_loader, epochs=200, lr=1e-3, patience=20):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = np.inf
    best_model = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                out = model(X_batch)
                loss = criterion(out, y_batch)
                val_loss += loss.item()

        print(f"[STL] Epoch {epoch+1} | Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping!")
                break

    model.load_state_dict(best_model)

# =====================
# 5. Evaluation Functions
# =====================

def evaluate_on_test(model, test_loader, multitask=False):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            if multitask:
                X_batch, y_batch, _ = batch
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                out, _ = model(X_batch)
            else:
                X_batch, y_batch = batch
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                out = model(X_batch)
            preds = (out.squeeze() > 0.5).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y_batch.squeeze().cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"Test Accuracy: {acc:.4f} | Test F1: {f1:.4f}")

# =====================
# 6. Main Execution
# =====================

if __name__ == "__main__":
    X_train, y_train_main, y_train_aux, X_dev, y_dev_main, y_dev_aux, X_test, y_test_main, y_test_aux, test_ids = load_and_prepare_data()

    train_dataset_mtl = MultiTaskDataset(X_train, y_train_main, y_train_aux)
    dev_dataset_mtl = MultiTaskDataset(X_dev, y_dev_main, y_dev_aux)
    test_dataset_mtl = MultiTaskDataset(X_test, y_test_main, y_test_aux)

    train_loader_mtl = DataLoader(train_dataset_mtl, batch_size=32, shuffle=True)
    dev_loader_mtl = DataLoader(dev_dataset_mtl, batch_size=32)
    test_loader_mtl = DataLoader(test_dataset_mtl, batch_size=32)

    model_mtl = DeepMultiTaskMLP_Improved(input_dim=X_train.shape[1]).to(device)
    
    # model_mtl = MultiTaskMLP(input_dim=X_train.shape[1]).to(device)

    train_mtl(model_mtl, train_loader_mtl, dev_loader_mtl)

    train_dataset_stl = SingleTaskDataset(X_train, y_train_main)
    dev_dataset_stl = SingleTaskDataset(X_dev, y_dev_main)
    test_dataset_stl = SingleTaskDataset(X_test, y_test_main)

    train_loader_stl = DataLoader(train_dataset_stl, batch_size=32, shuffle=True)
    dev_loader_stl = DataLoader(dev_dataset_stl, batch_size=32)
    test_loader_stl = DataLoader(test_dataset_stl, batch_size=32)

    model_stl = DeepSingleTaskMLP_Improved(input_dim=X_train.shape[1]).to(device)
    
    # model_stl = SingleTaskMLP(input_dim=X_train.shape[1]).to(device)

    train_stl(model_stl, train_loader_stl, dev_loader_stl)

    print("STL Model Evaluation:")
    evaluate_on_test(model_stl, test_loader_stl)

    print("MTL Model Evaluation:")
    evaluate_on_test(model_mtl, test_loader_mtl, multitask=True)
