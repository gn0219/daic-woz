import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import warnings
import argparse
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from src import plot_cm
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description='Train Multimodal model for depression detection')
    parser.add_argument('--save_model', action='store_true', help='Save the model')
    parser.add_argument('--hidden_size', type=int, default=32, help='Hidden size of DNN')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='L2 regularization')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
    return parser.parse_args()

def load_data(csv_path, test_size=0.2, val_size=0.2, random_state=42):
    # CSV 파일 읽기
    df = pd.read_csv(csv_path)
    print(f"Original data shape: {df.shape}")
    
    # 정보 컬럼과 특징 컬럼 분리
    info_columns = ['Participant_ID', 'Split', 'Gender', 'PHQ8_Binary', 'PHQ8_Score']
    feature_columns = [col for col in df.columns if col not in info_columns]
    
    # 발화 특징과 오디오 특징 분리 (앞 37개가 발화 특징, 나머지가 오디오 특징)
    utterance_columns = feature_columns[:37]
    audio_columns = feature_columns[37:]
    
    # NaN이 포함된 행 제거
    df = df.dropna(subset=utterance_columns + audio_columns)
    print(f"Data shape after removing NaN rows: {df.shape}")
    
    # Split 컬럼을 기준으로 데이터 분할
    train_df = df[df['Split'] == 'train'].copy()
    dev_df = df[df['Split'] == 'dev'].copy()
    test_df = df[df['Split'] == 'test'].copy()
    
    # 각 데이터셋의 클래스 분포 출력
    print("\nClass distribution in each split:")
    print("Train set:", train_df['PHQ8_Binary'].value_counts().to_dict())
    print("Dev set:", dev_df['PHQ8_Binary'].value_counts().to_dict())
    print("Test set:", test_df['PHQ8_Binary'].value_counts().to_dict())
    
    # 각 데이터셋의 특징과 레이블 추출
    X_utterance_train = train_df[utterance_columns].values
    X_audio_train = train_df[audio_columns].values
    y_train = train_df['PHQ8_Binary'].values
    
    X_utterance_val = dev_df[utterance_columns].values
    X_audio_val = dev_df[audio_columns].values
    y_val = dev_df['PHQ8_Binary'].values
    
    X_utterance_test = test_df[utterance_columns].values
    X_audio_test = test_df[audio_columns].values
    y_test = test_df['PHQ8_Binary'].values
    
    # 데이터 정규화
    utterance_scaler = StandardScaler()
    audio_scaler = StandardScaler()
    
    # Train 데이터로 스케일러 학습
    X_utterance_train = utterance_scaler.fit_transform(X_utterance_train)
    X_audio_train = audio_scaler.fit_transform(X_audio_train)
    
    # Val과 Test 데이터는 학습된 스케일러로 변환
    X_utterance_val = utterance_scaler.transform(X_utterance_val)
    X_audio_val = audio_scaler.transform(X_audio_val)
    X_utterance_test = utterance_scaler.transform(X_utterance_test)
    X_audio_test = audio_scaler.transform(X_audio_test)
    
    print(f"Training data shape - Utterance: {X_utterance_train.shape}, Audio: {X_audio_train.shape}")
    print(f"Validation data shape - Utterance: {X_utterance_val.shape}, Audio: {X_audio_val.shape}")
    print(f"Test data shape - Utterance: {X_utterance_test.shape}, Audio: {X_audio_test.shape}")
    
    return (X_utterance_train, X_audio_train, y_train), (X_utterance_val, X_audio_val, y_val), (X_utterance_test, X_audio_test, y_test)

class MultimodalDataset(Dataset):
    def __init__(self, utterance_features, audio_features, labels):
        self.utterance_features = torch.FloatTensor(utterance_features)
        self.audio_features = torch.FloatTensor(audio_features)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.utterance_features[idx], self.audio_features[idx], self.labels[idx]

class MultimodalModel(nn.Module):
    def __init__(self, utterance_input_size, audio_input_size, hidden_size, dropout=0.3):
        super(MultimodalModel, self).__init__()
        
        # Utterance modality
        self.utterance_encoder = nn.Sequential(
            nn.Linear(utterance_input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Audio modality
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        
        # Fusion layer
        fusion_input_size = hidden_size * 2
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, utterance_x, audio_x):
        # Utterance processing
        utterance_features = self.utterance_encoder(utterance_x)
        
        # Audio processing
        audio_features = self.audio_encoder(audio_x)
        
        # Combine features
        combined = torch.cat([utterance_features, audio_features], dim=1)
        
        # Final prediction
        return self.fusion_layer(combined)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, patience=30):
    train_losses = []
    val_losses = []
    train_aurocs = []
    val_aurocs = []
    train_f1s = []
    val_f1s = []
    best_val_auroc = 0.0
    best_model_state = model.state_dict().copy()
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for batch in train_loader:
            utterance_features, audio_features, labels = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            outputs = model(utterance_features, audio_features)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(outputs.cpu().detach().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Calculate training metrics
        train_preds = np.array(train_preds).flatten()
        train_labels = np.array(train_labels)
        
        # NaN 값 처리
        valid_mask = ~np.isnan(train_preds)
        if np.any(valid_mask):
            train_auroc = roc_auc_score(train_labels[valid_mask], train_preds[valid_mask])
            train_f1 = f1_score(train_labels[valid_mask], train_preds[valid_mask] > 0.5)
        else:
            train_auroc = 0.0
            train_f1 = 0.0
            
        train_aurocs.append(train_auroc)
        train_f1s.append(train_f1)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                utterance_features, audio_features, labels = [b.to(device) for b in batch]
                
                outputs = model(utterance_features, audio_features)
                loss = criterion(outputs, labels.unsqueeze(1))
                val_loss += loss.item()
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Calculate validation metrics
        val_preds = np.array(val_preds).flatten()
        val_labels = np.array(val_labels)
        
        # NaN 값 처리
        valid_mask = ~np.isnan(val_preds)
        if np.any(valid_mask):
            val_auroc = roc_auc_score(val_labels[valid_mask], val_preds[valid_mask])
            val_f1 = f1_score(val_labels[valid_mask], val_preds[valid_mask] > 0.5)
        else:
            val_auroc = 0.0
            val_f1 = 0.0
            
        val_aurocs.append(val_auroc)
        val_f1s.append(val_f1)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss:.4f}, AUROC: {train_auroc:.4f}, F1: {train_f1:.4f}')
        print(f'Validation Loss: {val_loss:.4f}, AUROC: {val_auroc:.4f}, F1: {val_f1:.4f}')
        print(f'Learning Rate: {scheduler.optimizer.param_groups[0]["lr"]:.6f}')
        
        # Early stopping check
        if val_auroc >= best_val_auroc:
            best_val_auroc = val_auroc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f'New best AUROC: {val_auroc:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping triggered after {epoch + 1} epochs')
                print(f'Best validation AUROC: {best_val_auroc:.4f}')
                early_stop_epoch = epoch - patience
                break
    
    return best_model_state, train_losses, val_losses, train_aurocs, val_aurocs, train_f1s, val_f1s, early_stop_epoch

def evaluate_model(model, test_loader, device, results_dir='results/multimodal_result'):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            utterance_features, audio_features, labels = [b.to(device) for b in batch]
            
            outputs = model(utterance_features, audio_features)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels)
    
    # Convert probabilities to binary predictions
    binary_preds = (all_preds > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, binary_preds)
    f1_macro = f1_score(all_labels, binary_preds, average='macro')
    auroc = roc_auc_score(all_labels, all_preds)
    precision = precision_score(all_labels, binary_preds)
    recall = recall_score(all_labels, binary_preds)
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, binary_preds)
    
    # Plot confusion matrix using custom plot_cm function
    plot_cm(cm, class_names=['Non-depressed', 'Depressed'], 
            title='Confusion Matrix - Multimodal Model',
            save_path=os.path.join(results_dir, 'confusion_matrix.png'))
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'auroc': auroc,
        'precision': precision,
        'recall': recall
    }

def plot_metrics(train_losses, val_losses, train_aurocs, val_aurocs, train_f1s, val_f1s, early_stop_epoch, save_dir='results/multimodal_result'):
    # Create a single figure with subplots
    plt.figure(figsize=(10, 12))
    
    # Plot losses
    plt.subplot(3, 1, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    if early_stop_epoch is not None:
        plt.axvline(x=early_stop_epoch, color='r', linestyle='-', label='Early Stopping')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='upper right', framealpha=0.5)
    plt.grid(True)
    
    # Plot AUROC
    plt.subplot(3, 1, 2)
    plt.plot(train_aurocs, label='Training AUROC')
    plt.plot(val_aurocs, label='Validation AUROC')
    if early_stop_epoch is not None:
        plt.axvline(x=early_stop_epoch, color='r', linestyle='-', label='Early Stopping')
    plt.xlabel('Epoch')
    plt.ylabel('AUROC')
    plt.title('Training and Validation AUROC')
    plt.legend(loc='upper right', framealpha=0.5)
    plt.grid(True)
    
    # Plot F1 scores
    plt.subplot(3, 1, 3)
    plt.plot(train_f1s, label='Training F1')
    plt.plot(val_f1s, label='Validation F1')
    if early_stop_epoch is not None:
        plt.axvline(x=early_stop_epoch, color='r', linestyle='-', label='Early Stopping')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Training and Validation F1 Score')
    plt.legend(loc='upper right', framealpha=0.5)
    plt.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()

def main():
    args = parse_args()
    
    # Create results directory with experiment name
    exp_name = f"multimodal_result"
    results_dir = f'results/{exp_name}'
    Path(results_dir).mkdir(exist_ok=True)
    
    # Load data
    (X_utterance_train, X_audio_train, y_train), (X_utterance_val, X_audio_val, y_val), (X_utterance_test, X_audio_test, y_test) = load_data('data/df.csv')
    
    
    print(f"Successfully processed {len(X_utterance_train)} samples")
    print(f"Utterance feature shape: {X_utterance_train.shape}")
    print(f"Audio feature shape: {X_audio_train.shape}")
    
    # Create datasets and dataloaders
    train_dataset = MultimodalDataset(X_utterance_train, X_audio_train, y_train)
    val_dataset = MultimodalDataset(X_utterance_val, X_audio_val, y_val)
    test_dataset = MultimodalDataset(X_utterance_test, X_audio_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultimodalModel(
        utterance_input_size=X_utterance_train.shape[1],
        audio_input_size=X_audio_train.shape[1],
        hidden_size=args.hidden_size,
        dropout=args.dropout
    ).to(device)
    
    # Calculate class weights for weighted loss
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print("POSITIVE WEIGHT: ", pos_weight)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))
    
    # Initialize optimizer with L2 regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Train model
    best_model_state, train_losses, val_losses, train_aurocs, val_aurocs, train_f1s, val_f1s, early_stop_epoch = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        args.epochs, device, args.patience
    )
    
    # Plot training history
    plot_metrics(train_losses, val_losses, train_aurocs, val_aurocs, train_f1s, val_f1s, early_stop_epoch=early_stop_epoch, save_dir=results_dir)
    
    # Load best model and evaluate
    model.load_state_dict(best_model_state)
    metrics = evaluate_model(model, test_loader, device, results_dir=results_dir)
    
    # Save results
    results = {
        'accuracy': metrics['accuracy'],
        'f1_macro': metrics['f1_macro'],
        'auroc': metrics['auroc'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_aurocs': train_aurocs,
        'val_aurocs': val_aurocs,
        'train_f1s': train_f1s,
        'val_f1s': val_f1s
    }
    
    # Save model
    if args.save_model:
        torch.save(model.state_dict(), os.path.join(results_dir, 'multimodal_model.pth'))
    
    # Save results to file
    with open(os.path.join(results_dir, 'results.txt'), 'w') as f:
        f.write(f'Model Configuration:\n')
        f.write(f'Hidden Size: {args.hidden_size}\n')
        f.write(f'Dropout: {args.dropout}\n')
        f.write(f'Batch Size: {args.batch_size}\n')
        f.write(f'Learning Rate: {args.lr}\n')
        f.write(f'Weight Decay: {args.weight_decay}\n')
        f.write(f'Epochs: {args.epochs}\n')
        f.write(f'Patience: {args.patience}\n\n')
        f.write(f'Results:\n')
        f.write(f'Accuracy: {metrics["accuracy"]:.4f}\n')
        f.write(f'F1 Score (Macro): {metrics["f1_macro"]:.4f}\n')
        f.write(f'AUROC: {metrics["auroc"]:.4f}\n')
        f.write(f'Precision: {metrics["precision"]:.4f}\n')
        f.write(f'Recall: {metrics["recall"]:.4f}\n')

if __name__ == '__main__':
    main() 