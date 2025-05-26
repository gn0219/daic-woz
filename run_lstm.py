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
import librosa
import warnings
import argparse
from tqdm import tqdm
warnings.filterwarnings('ignore')
# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description='Train LSTM model for depression detection')
    parser.add_argument('--use_gender', action='store_true', help='Use gender as a feature')
    parser.add_argument('--save_model', action='store_true', help='Save the model')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size of LSTM')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    return parser.parse_args()

def extract_mfcc_features(audio_path, n_mfcc=60, n_fft=2048, hop_length=512, window_size=5):
    """
    Extract MFCC features from an audio file using sliding windows
    window_size: size of window in seconds
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        if len(y) == 0:
            print(f"Warning: Empty audio file {audio_path}")
            return None
            
        # Calculate number of samples per window
        samples_per_window = int(window_size * sr)
        
        # Calculate number of windows
        n_windows = len(y) // samples_per_window
        
        if n_windows == 0:
            print(f"Warning: Audio file {audio_path} is shorter than window size")
            return None
        
        window_features = []
        
        for i in range(n_windows):
            try:
                # Extract window
                start = i * samples_per_window
                end = start + samples_per_window
                window = y[start:end]
                
                # Extract MFCC features for the window
                mfccs = librosa.feature.mfcc(
                    y=window, 
                    sr=sr, 
                    n_mfcc=n_mfcc,
                    n_fft=n_fft,
                    hop_length=hop_length
                )
                
                # Transpose to get (time, features) shape
                mfccs = mfccs.T
                
                # Calculate mean and std for this window
                window_mean = np.mean(mfccs, axis=0)
                window_std = np.std(mfccs, axis=0)
                
                # Concatenate mean and std
                window_feature = np.concatenate([window_mean, window_std])
                window_features.append(window_feature)
            except Exception as e:
                print(f"Warning: Error processing window {i} in {audio_path}: {str(e)}")
                continue
        
        if not window_features:
            print(f"Warning: No valid windows could be processed for {audio_path}")
            return None
            
        # Stack all window features
        features = np.stack(window_features)
        
        # Validate feature shape
        expected_shape = (n_windows, n_mfcc * 2)  # *2 for mean and std
        if features.shape != expected_shape:
            print(f"Warning: Unexpected feature shape for {audio_path}. Expected {expected_shape}, got {features.shape}")
            return None
            
        return features
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None

def prepare_audio_features(df, audio_dir='wav_files', save_path='data/librosa_mfcc.csv', max_windows=180):
    """
    Prepare features from audio files for all participants
    If features are already saved, load them instead of re-computing
    max_windows: maximum number of windows to keep (will pad or truncate to this length)
    """
    # Check if saved features exist
    if os.path.exists(save_path):
        print(f"Loading pre-computed MFCC features from {save_path}")
        try:
            saved_features = pd.read_csv(save_path)
            
            # Verify that all required participant IDs are present
            required_ids = set(df['Participant_ID'])
            saved_ids = set(saved_features['Participant_ID'])
            
            if required_ids.issubset(saved_ids):
                print("All required features found in saved file")
                # Get features for required participants
                features_df = saved_features[saved_features['Participant_ID'].isin(required_ids)]
                features_df = features_df.sort_values('Participant_ID')
                features = features_df.drop('Participant_ID', axis=1).values
                
                # Reshape features to (n_samples, n_windows, n_features)
                n_samples = features.shape[0]
                n_features = 120  # n_mfcc * 2 (mean + std)
                n_windows = features.shape[1] // n_features
                features = features.reshape(n_samples, n_windows, n_features)
                
                valid_indices = df[df['Participant_ID'].isin(features_df['Participant_ID'])].index.tolist()
                return features, valid_indices
            else:
                print("Some features missing in saved file, will recompute all features")
        except Exception as e:
            print(f"Error loading saved features: {str(e)}")
            print("Will recompute all features")
    
    print("Extracting MFCC features from audio files...")
    features = []
    valid_indices = []
    participant_ids = []
    
    # Create audio directory if it doesn't exist
    os.makedirs(audio_dir, exist_ok=True)
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        participant_id = row['Participant_ID']
        audio_path = os.path.join(audio_dir, f'{participant_id}_AUDIO.wav')
        
        if not os.path.exists(audio_path):
            print(f"Warning: Audio file not found for participant {participant_id}")
            continue
            
        try:
            window_features = extract_mfcc_features(audio_path)
            if window_features is not None:
                # Pad or truncate to max_windows
                if window_features.shape[0] > max_windows:
                    # Truncate
                    window_features = window_features[:max_windows]
                elif window_features.shape[0] < max_windows:
                    # Pad with zeros
                    padding = np.zeros((max_windows - window_features.shape[0], window_features.shape[1]))
                    window_features = np.vstack([window_features, padding])
                
                features.append(window_features)
                valid_indices.append(idx)
                participant_ids.append(participant_id)
            else:
                print(f"Warning: Could not extract features for participant {participant_id}")
        except Exception as e:
            print(f"Error processing participant {participant_id}: {str(e)}")
            continue
    
    if not features:
        raise ValueError("No valid features could be extracted from any audio files")
    
    # Stack features into a single array
    features = np.stack(features)
    
    # Save features to CSV
    print(f"Saving MFCC features to {save_path}")
    try:
        # Flatten the features for CSV storage
        features_flat = features.reshape(features.shape[0], -1)
        features_df = pd.DataFrame(features_flat)
        features_df['Participant_ID'] = participant_ids
        features_df = features_df[['Participant_ID'] + [col for col in features_df.columns if col != 'Participant_ID']]
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        features_df.to_csv(save_path, index=False)
    except Exception as e:
        print(f"Warning: Could not save features to CSV: {str(e)}")
    
    return features, valid_indices

class AudioDataset(Dataset):
    def __init__(self, features, labels, gender=None):
        # Features shape: (batch_size, n_windows, feature_dim)
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        if gender is not None:
            self.gender = torch.FloatTensor(gender).unsqueeze(1)
        else:
            self.gender = None
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.gender is not None:
            return self.features[idx], self.gender[idx], self.labels[idx]
        return self.features[idx], self.labels[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.5, use_gender=False):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_gender = use_gender
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        fc_input_size = hidden_size * 2  # *2 for bidirectional
        if use_gender:
            fc_input_size += 1
        
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, gender=None):
        # x shape: (batch_size, n_windows, input_size)
        lstm_out, _ = self.lstm(x)
        # Take the output of the last time step
        last_output = lstm_out[:, -1, :]
        
        if self.use_gender and gender is not None:
            last_output = torch.cat([last_output, gender], dim=1)
        
        return self.fc(last_output)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience=30):
    train_losses = []
    val_losses = []
    train_aurocs = []
    val_aurocs = []
    train_f1s = []
    val_f1s = []
    best_val_auroc = 0.0
    best_model_state = model.state_dict().copy()  # Save initial model state
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for batch in train_loader:
            if len(batch) == 3:  # with gender
                features, gender, labels = [b.to(device) for b in batch]
            else:  # without gender
                features, labels = [b.to(device) for b in batch]
                gender = None
            
            optimizer.zero_grad()
            outputs = model(features, gender)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(outputs.cpu().detach().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Calculate training metrics
        train_preds = np.array(train_preds).flatten()
        train_labels = np.array(train_labels)
        train_auroc = roc_auc_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds > 0.5)
        train_aurocs.append(train_auroc)
        train_f1s.append(train_f1)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:  # with gender
                    features, gender, labels = [b.to(device) for b in batch]
                else:  # without gender
                    features, labels = [b.to(device) for b in batch]
                    gender = None
                
                outputs = model(features, gender)
                loss = criterion(outputs, labels.unsqueeze(1))
                val_loss += loss.item()
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Calculate validation metrics
        val_preds = np.array(val_preds).flatten()
        val_labels = np.array(val_labels)
        val_auroc = roc_auc_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds > 0.5)
        val_aurocs.append(val_auroc)
        val_f1s.append(val_f1)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss:.4f}, AUROC: {train_auroc:.4f}, F1: {train_f1:.4f}')
        print(f'Validation Loss: {val_loss:.4f}, AUROC: {val_auroc:.4f}, F1: {val_f1:.4f}')
        
        # Early stopping check based on F1 score
        if val_auroc >= best_val_auroc:  # Changed from > to >= to handle initial case
            best_val_auroc = val_auroc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f'New best AUROC: {val_auroc:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping triggered after {epoch + 1} epochs')
                print(f'Best validation AUROC: {best_val_auroc:.4f}')
                break
    
    return best_model_state, train_losses, val_losses, train_aurocs, val_aurocs, train_f1s, val_f1s

def evaluate_model(model, test_loader, device, results_dir='viz/lstm_results'):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:  # with gender
                features, gender, labels = [b.to(device) for b in batch]
            else:  # without gender
                features, labels = [b.to(device) for b in batch]
                gender = None
            
            outputs = model(features, gender)
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
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    plt.close()
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'auroc': auroc,
        'precision': precision,
        'recall': recall
    }

def plot_metrics(train_losses, val_losses, train_aurocs, val_aurocs, train_f1s, val_f1s, save_dir='viz/lstm_results'):
    # Create a single figure with subplots
    plt.figure(figsize=(15, 10))
    
    # Plot losses
    plt.subplot(3, 1, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot AUROC
    plt.subplot(3, 1, 2)
    plt.plot(train_aurocs, label='Training AUROC')
    plt.plot(val_aurocs, label='Validation AUROC')
    plt.xlabel('Epoch')
    plt.ylabel('AUROC')
    plt.title('Training and Validation AUROC')
    plt.legend()
    plt.grid(True)
    
    # Plot F1 scores
    plt.subplot(3, 1, 3)
    plt.plot(train_f1s, label='Training F1')
    plt.plot(val_f1s, label='Validation F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Training and Validation F1 Score')
    plt.legend()
    plt.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()

def main():
    args = parse_args()
    
    # Create results directory with experiment name
    exp_name = f"lstm_results{'_with_gender' if args.use_gender else ''}"
    results_dir = f'viz/{exp_name}'
    Path(results_dir).mkdir(exist_ok=True)
    
    # Load data
    df = pd.read_csv('data/df.csv')
    
    # Extract or load MFCC features
    features, valid_indices = prepare_audio_features(df)
    
    # Filter dataframe to only include valid samples
    df = df.iloc[valid_indices]
    y = df['PHQ8_Binary'].values
    
    # Prepare gender feature if needed
    gender = None
    if args.use_gender:
        gender = df['Gender'].values
        # Convert gender to numeric (assuming 'M' and 'F' values)
        gender = np.array([1.0 if g == 'M' else 0.0 for g in gender])
    
    print(f"Successfully processed {len(features)} audio files")
    print(f"Feature shape: {features.shape}")
    
    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(features, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # Split gender data if using gender
    if args.use_gender:
        gender_train, gender_temp = train_test_split(gender, test_size=0.3, random_state=42, stratify=y)
        gender_val, gender_test = train_test_split(gender_temp, test_size=0.5, random_state=42, stratify=y_temp)
    else:
        gender_train = gender_val = gender_test = None
    
    # Create datasets and dataloaders
    train_dataset = AudioDataset(X_train, y_train, gender_train)
    val_dataset = AudioDataset(X_val, y_val, gender_val)
    test_dataset = AudioDataset(X_test, y_test, gender_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(
        input_size=features.shape[2],  # n_features per window
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_gender=args.use_gender
    ).to(device)
    print(f"Using device: {device}")
    
    # Calculate class weights for weighted loss
    pos_weight = (y == 0).sum() / (y == 1).sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Train model with increased epochs and early stopping
    best_model_state, train_losses, val_losses, train_aurocs, val_aurocs, train_f1s, val_f1s = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=1000, device=device, patience=30
    )
    
    # Plot training history
    plot_metrics(train_losses, val_losses, train_aurocs, val_aurocs, train_f1s, val_f1s, save_dir=results_dir)
    
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
        torch.save(model.state_dict(), os.path.join(results_dir, 'lstm_model.pth'))
    
    # Save results to file
    with open(os.path.join(results_dir, 'results.txt'), 'w') as f:
        f.write(f'Model Configuration:\n')
        f.write(f'Use Gender: {args.use_gender}\n')
        f.write(f'Hidden Size: {args.hidden_size}\n')
        f.write(f'Number of Layers: {args.num_layers}\n')
        f.write(f'Dropout: {args.dropout}\n')
        f.write(f'Batch Size: {args.batch_size}\n')
        f.write(f'Learning Rate: {args.lr}\n\n')
        f.write(f'Results:\n')
        f.write(f'Accuracy: {metrics["accuracy"]:.4f}\n')
        f.write(f'F1 Score (Macro): {metrics["f1_macro"]:.4f}\n')
        f.write(f'AUROC: {metrics["auroc"]:.4f}\n')
        f.write(f'Precision: {metrics["precision"]:.4f}\n')
        f.write(f'Recall: {metrics["recall"]:.4f}\n')

if __name__ == '__main__':
    main() 