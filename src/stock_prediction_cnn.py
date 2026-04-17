import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# ============================================
# 1. DATA LOADING & PREPARATION
# ============================================

def load_stock_data(ticker, days=252*2):
    """
    Load historical stock data using yfinance
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'JPM')
        days: Number of trading days to fetch
    
    Returns:
        DataFrame with closing prices
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    return data['Close'].values.reshape(-1, 1)


def normalize_data(data):
    """Normalize data using MinMaxScaler (0-1 range)"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler


# ============================================
# 2. FEATURE ENGINEERING - SLIDING WINDOWS
# ============================================

def create_sequences(data, seq_length=50):
    """
    Create sliding window sequences for training
    
    Args:
        data: Normalized stock price data
        seq_length: Length of each sequence window
    
    Returns:
        X: Input sequences (N, seq_length, 1)
        y: Binary labels (1 if next price > current, 0 otherwise)
    """
    X, y = [], []
    
    for i in range(len(data) - seq_length - 1):
        # Input: last seq_length prices
        window = data[i:i + seq_length]
        X.append(window)
        
        # Label: 1 if next price goes up, 0 if down
        current_price = data[i + seq_length]
        next_price = data[i + seq_length + 1]
        label = 1 if next_price > current_price else 0
        y.append(label)
    
    return np.array(X), np.array(y)


# ============================================
# 3. 1D CNN MODEL ARCHITECTURE
# ============================================

class StockPredictionCNN(nn.Module):
    """
    10-layer 1D CNN for stock price movement prediction
    
    Architecture:
    - Conv1d layers with ReLU activation
    - Batch normalization for stability
    - Dropout for regularization
    - Global average pooling
    - Dense layers for classification
    """
    
    def __init__(self, seq_length=50, num_filters=32):
        super(StockPredictionCNN, self).__init__()
        
        # Convolutional layers (1D)
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
        
        # Additional conv layers for depth (total 10 layers as per paper)
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
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Dense layers for classification
        self.fc1 = nn.Linear(16, 64)
        self.fc_dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Binary output (up/down)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """Forward pass through the network"""
        # Conv blocks with batch norm, ReLU, and dropout
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
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Dense layers
        x = self.relu(self.fc1(x))
        x = self.fc_dropout(x)
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        
        return x


# ============================================
# 4. TRAINING FUNCTION
# ============================================

def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, device='cpu'):
    """
    Train the CNN model
    
    Args:
        model: StockPredictionCNN instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Number of training epochs
        lr: Learning rate
        device: 'cpu' or 'cuda'
    
    Returns:
        Training history (losses and accuracies)
    """
    model.to(device)
    criterion = nn.BCELoss()  # Binary Cross Entropy for binary classification
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
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
        
        train_acc = train_correct / train_total
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        val_loss /= len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
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
# 5. EVALUATION FUNCTION
# ============================================

def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluate model on test set
    
    Returns:
        Accuracy and predictions
    """
    model.eval()
    correct = 0
    total = 0
    predictions = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            predictions.extend(predicted.cpu().numpy().flatten())
    
    accuracy = correct / total
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy, predictions


# ============================================
# 6. MAIN EXECUTION
# ============================================

def main():
    """Main pipeline for training and evaluating the model"""
    
    # Configuration
    TICKER = 'JPM'  # JPMorgan stock (mentioned in paper)
    SEQ_LENGTH = 50
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {DEVICE}")
    print(f"Stock: {TICKER}, Sequence Length: {SEQ_LENGTH}")
    
    # Step 1: Load and normalize data
    print("\n[1] Loading stock data...")
    raw_data = load_stock_data(TICKER, days=504)
    print(f"Data shape: {raw_data.shape}")
    
    scaled_data, scaler = normalize_data(raw_data)
    print(f"Data normalized to range [0, 1]")
    
    # Step 2: Create sequences
    print("\n[2] Creating sequences...")
    X, y = create_sequences(scaled_data, seq_length=SEQ_LENGTH)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Step 3: Train/test split (chronological)
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_val = torch.LongTensor(y_val)
    y_test = torch.LongTensor(y_test)
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Step 4: Initialize model
    print("\n[3] Building 10-layer CNN model...")
    model = StockPredictionCNN(seq_length=SEQ_LENGTH, num_filters=32)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Step 5: Train model
    print("\n[4] Training model...")
    history = train_model(model, train_loader, val_loader, 
                         epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE)
    
    # Step 6: Evaluate on test set
    print("\n[5] Evaluating on test set...")
    test_acc, predictions = evaluate_model(model, test_loader, device=DEVICE)
    
    # Step 7: Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history saved to 'training_history.png'")
    
    # Save model
    torch.save(model.state_dict(), 'stock_prediction_cnn.pth')
    print("Model saved to 'stock_prediction_cnn.pth'")
    
    return model, scaler


if __name__ == "__main__":
    model, scaler = main()