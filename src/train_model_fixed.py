"""
Complete training script with FIXED tensor shapes
Python 3.13 & PyTorch 2.6 compatible
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

print(f"Python version: {__import__('sys').version}")
print(f"PyTorch version: {torch.__version__}")


# ============================================
# 1. DATA LOADING & PREPARATION
# ============================================

def load_stock_data(ticker: str, days: int = 252*2) -> pd.DataFrame:
    """Load historical stock data using yfinance"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    return data


def normalize_data(data: np.ndarray) -> tuple[np.ndarray, MinMaxScaler]:
    """Normalize data using MinMaxScaler (0-1 range)"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler


# ============================================
# 2. FEATURE ENGINEERING - SLIDING WINDOWS
# ============================================

def create_sequences(data: np.ndarray, seq_length: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for training
    FIXED: Return shape (N, 1, seq_length) for Conv1d
    
    Args:
        data: Normalized stock price data
        seq_length: Length of each sequence window
    
    Returns:
        X: Input sequences (N, 1, seq_length) - CORRECT FOR CONV1D
        y: Binary labels
    """
    X, y = [], []
    
    for i in range(len(data) - seq_length - 1):
        window = data[i:i + seq_length].flatten()  # Shape: (seq_length,)
        X.append(window)
        
        current_price = data[i + seq_length]
        next_price = data[i + seq_length + 1]
        label = 1 if next_price > current_price else 0
        y.append(label)
    
    X = np.array(X)  # Shape: (N, seq_length)
    X = X.reshape(X.shape[0], 1, X.shape[1])  # Shape: (N, 1, seq_length) ✅
    y = np.array(y)
    
    print(f"✓ Sequences created: X shape {X.shape}, y shape {y.shape}")
    return X, y


# ============================================
# 3. 1D CNN MODEL ARCHITECTURE
# ============================================

class StockPredictionCNN(nn.Module):
    """10-layer 1D CNN for stock price movement prediction"""
    
    def __init__(self, seq_length: int = 50, num_filters: int = 32) -> None:
        super().__init__()
        
        # Conv layers
        self.conv1 = nn.Conv1d(1, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv1d(num_filters, num_filters*2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_filters*2)
        self.dropout2 = nn.Dropout(0.2)
        
        self.conv3 = nn.Conv1d(num_filters*2, num_filters*4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(num_filters*4)
        self.dropout3 = nn.Dropout(0.2)
        
        self.conv4 = nn.Conv1d(num_filters*4, num_filters*4, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(num_filters*4)
        self.dropout4 = nn.Dropout(0.2)
        
        self.conv5 = nn.Conv1d(num_filters*4, num_filters*2, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(num_filters*2)
        self.dropout5 = nn.Dropout(0.2)
        
        self.conv6 = nn.Conv1d(num_filters*2, num_filters*2, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm1d(num_filters*2)
        self.dropout6 = nn.Dropout(0.2)
        
        self.conv7 = nn.Conv1d(num_filters*2, num_filters, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm1d(num_filters)
        self.dropout7 = nn.Dropout(0.2)
        
        self.conv8 = nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm1d(num_filters)
        self.dropout8 = nn.Dropout(0.2)
        
        self.conv9 = nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm1d(num_filters)
        self.dropout9 = nn.Dropout(0.2)
        
        self.conv10 = nn.Conv1d(num_filters, 16, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm1d(16)
        self.dropout10 = nn.Dropout(0.2)
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(16, 64)
        self.fc_dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.dropout4(x)
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.dropout5(x)
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.dropout6(x)
        x = self.relu(self.bn7(self.conv7(x)))
        x = self.dropout7(x)
        x = self.relu(self.bn8(self.conv8(x)))
        x = self.dropout8(x)
        x = self.relu(self.bn9(self.conv9(x)))
        x = self.dropout9(x)
        x = self.relu(self.bn10(self.conv10(x)))
        x = self.dropout10(x)
        
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc_dropout(x)
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        
        return x


# ============================================
# 4. TRAINING FUNCTION
# ============================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 0.001,
    device: str = 'cpu'
) -> dict:
    """Train the CNN model with early stopping"""
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    patience, patience_counter = 10, 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
        
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        val_loss /= len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return history


# ============================================
# 5. MAIN EXECUTION
# ============================================

def main() -> tuple[nn.Module, MinMaxScaler]:
    """Main pipeline"""
    
    TICKER = 'JPM'
    SEQ_LENGTH = 50
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*60}")
    print(f"Stock Prediction CNN - Training")
    print(f"Python 3.13 & PyTorch 2.6")
    print(f"{'='*60}")
    print(f"Device: {DEVICE}")
    print(f"Stock: {TICKER}\n")
    
    # Load data
    print("[1] Loading stock data...")
    raw_data = yf.download(TICKER, start="2022-01-01", end=datetime.now(), progress=False)
    prices = raw_data['Close'].values.reshape(-1, 1)
    print(f"✓ Data shape: {prices.shape}")
    
    scaled_data, scaler = normalize_data(prices)
    
    # Create sequences with CORRECT shape
    print("\n[2] Creating sequences...")
    X, y = create_sequences(scaled_data, seq_length=SEQ_LENGTH)
    
    # Verify shapes
    print(f"✓ X dtype: {X.dtype}, shape: {X.shape}")
    print(f"✓ y dtype: {y.dtype}, shape: {y.shape}")
    
    # Train/test split
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    # Convert to tensors (already have correct shape)
    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_val = torch.LongTensor(y_val)
    y_test = torch.LongTensor(y_test)
    
    print(f"✓ Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)
    
    # Train model
    print("\n[3] Building and training model...")
    model = StockPredictionCNN(seq_length=SEQ_LENGTH, num_filters=32)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✓ Model parameters: {param_count:,}")
    
    history = train_model(model, train_loader, val_loader, 
                         epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE)
    
    # Save model and scaler
    print("\n[4] Saving model and scaler...")
    torch.save(model.state_dict(), 'stock_prediction_cnn.pth')
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✓ Model saved to: stock_prediction_cnn.pth")
    print("✓ Scaler saved to: scaler.pkl")
    
    # Plot training history
    print("\n[5] Creating visualizations...")
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: training_history.png")
    
    print("\n" + "="*60)
    print("✅ Training complete!")
    print("="*60 + "\n")
    
    return model, scaler


if __name__ == "__main__":
    model, scaler = main()