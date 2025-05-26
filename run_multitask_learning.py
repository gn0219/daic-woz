# Multitask vs Single Task Learning for Depression Detection (Improved)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_auc_score
import os
import random
import matplotlib.pyplot as plt

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

class DeepMultiTaskMLP_Improved(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout=0.4):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
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
        out_main = self.head_main(h)
        out_aux = self.head_aux(h)
        return out_main, out_aux

class DeepSingleTaskMLP_Improved(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout=0.4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[2], 1)
        )

    def forward(self, x):
        return self.model(x)


# =====================
# 4. Training Functions with Early Stopping
# =====================

def train_mtl(model, train_loader, val_loader, epochs=1000, lr=1e-3, patience=20):
    # Calculate class weights
    pos_weight_main = torch.tensor([2.5666]).to(device)  # Adjust this weight based on your class imbalance
    pos_weight_aux = torch.tensor([1.0434]).to(device)   # Adjust this weight based on your class imbalance
    
    criterion_main = nn.BCEWithLogitsLoss(pos_weight=pos_weight_main)
    criterion_aux = nn.BCEWithLogitsLoss(pos_weight=pos_weight_aux)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = np.inf
    best_model = None
    patience_counter = 0
    
    # Store losses for plotting
    train_losses = []
    val_losses = []
    early_stop_epoch = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_main, y_aux in train_loader:
            X_batch, y_main, y_aux = X_batch.to(device), y_main.to(device), y_aux.to(device)
        
            optimizer.zero_grad()
            out_main, out_aux = model(X_batch)
            loss_main = criterion_main(out_main, y_main)
            loss_aux = criterion_aux(out_aux, y_aux)
            loss = loss_main + 0.3 * loss_aux
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        train_losses.append(total_loss / len(train_loader))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_main, y_aux in val_loader:
                X_batch, y_main, y_aux = X_batch.to(device), y_main.to(device), y_aux.to(device)

                out_main, out_aux = model(X_batch)
                loss_main = criterion_main(out_main, y_main)
                loss_aux = criterion_aux(out_aux, y_aux)
                val_loss += (loss_main + 0.3 * loss_aux).item()
        
        val_losses.append(val_loss / len(val_loader))

        print(f"[MTL] Epoch {epoch+1} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping!")
                early_stop_epoch = epoch - patience
                break

    model.load_state_dict(best_model)
    return train_losses, val_losses, early_stop_epoch

def train_stl(model, train_loader, val_loader, epochs=1000, lr=1e-3, patience=20):
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.5666]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = np.inf
    best_model = None
    patience_counter = 0
    
    # Store losses for plotting
    train_losses = []
    val_losses = []
    early_stop_epoch = None

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
        
        train_losses.append(total_loss / len(train_loader))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                out = model(X_batch)
                loss = criterion(out, y_batch)
                val_loss += loss.item()
        
        val_losses.append(val_loss / len(val_loader))

        print(f"[STL] Epoch {epoch+1} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping!")
                early_stop_epoch = epoch - patience
                break

    model.load_state_dict(best_model)
    return train_losses, val_losses, early_stop_epoch

def plot_losses(mtl_train_losses, mtl_val_losses, mtl_early_stop, stl_train_losses, stl_val_losses, stl_early_stop, save_dir='results/multitask_result'):
    plt.figure(figsize=(10, 8))
    
    # Plot MTL losses
    plt.subplot(2, 1, 1)
    plt.plot(mtl_train_losses, label='MTL Training Loss')
    plt.plot(mtl_val_losses, label='MTL Validation Loss')
    if mtl_early_stop is not None:
        plt.axvline(x=mtl_early_stop, color='r', linestyle='-', label='Early Stopping')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MTL Training and Validation Loss')
    plt.legend(loc='upper right', framealpha=0.5)
    plt.grid(True)
    
    # Plot STL losses
    plt.subplot(2, 1, 2)
    plt.plot(stl_train_losses, label='STL Training Loss')
    plt.plot(stl_val_losses, label='STL Validation Loss')
    if stl_early_stop is not None:
        plt.axvline(x=stl_early_stop, color='r', linestyle='-', label='Early Stopping')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('STL Training and Validation Loss')
    plt.legend(loc='upper right', framealpha=0.5)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_losses.png'))
    plt.close()

# =====================
# 5. Evaluation Functions
# =====================

def evaluate_on_test(model, test_loader, multitask=False, model_name="model"):
    model.eval()
    y_true, y_pred, y_pred_proba = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            if multitask:
                X_batch, y_batch, _ = batch
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                out, _ = model(X_batch)
                out = torch.sigmoid(out)  # Apply sigmoid for prediction
            else:
                X_batch, y_batch = batch
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                out = model(X_batch)
                out = torch.sigmoid(out)  # Apply sigmoid for prediction
            preds = (out.squeeze() > 0.5).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y_batch.squeeze().cpu().numpy())
            y_pred_proba.extend(out.squeeze().cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    auroc = roc_auc_score(y_true, y_pred_proba)
    cm = confusion_matrix(y_true, y_pred)
    
    # Save results
    results = {
        'accuracy': acc,
        'f1_score': f1,
        'auroc': auroc,
        'confusion_matrix': cm.tolist()
    }
    
    # Save confusion matrix plot with appropriate title
    title = f"Confusion Matrix - {model_name.upper()}"
    plot_cm(cm, class_names=['Non-depressed', 'Depressed'], 
            title=title, save_path=f'results/multitask_result/{model_name}_confusion_matrix.png')
    
    # Save results to file
    with open(f'results/multitask_result/{model_name}_results.txt', 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"AUROC: {auroc:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
    
    print(f"Test Accuracy: {acc:.4f} | Test F1: {f1:.4f} | Test AUROC: {auroc:.4f}")
    print(f"Results saved to results/multitask_result/{model_name}_results.txt")
    print(f"Confusion matrix plot saved to results/multitask_result/{model_name}_confusion_matrix.png")

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
    
    # Train MTL model and get losses
    mtl_train_losses, mtl_val_losses, mtl_early_stop = train_mtl(model_mtl, train_loader_mtl, dev_loader_mtl)

    train_dataset_stl = SingleTaskDataset(X_train, y_train_main)
    dev_dataset_stl = SingleTaskDataset(X_dev, y_dev_main)
    test_dataset_stl = SingleTaskDataset(X_test, y_test_main)

    train_loader_stl = DataLoader(train_dataset_stl, batch_size=32, shuffle=True)
    dev_loader_stl = DataLoader(dev_dataset_stl, batch_size=32)
    test_loader_stl = DataLoader(test_dataset_stl, batch_size=32)

    model_stl = DeepSingleTaskMLP_Improved(input_dim=X_train.shape[1]).to(device)
    
    # Train STL model and get losses
    stl_train_losses, stl_val_losses, stl_early_stop = train_stl(model_stl, train_loader_stl, dev_loader_stl)

    # Plot losses
    plot_losses(mtl_train_losses, mtl_val_losses, mtl_early_stop, stl_train_losses, stl_val_losses, stl_early_stop)

    print("\nMTL Model Evaluation:")
    evaluate_on_test(model_mtl, test_loader_mtl, multitask=True, model_name="mtl")

    print("STL Model Evaluation:")
    evaluate_on_test(model_stl, test_loader_stl, model_name="stl")

