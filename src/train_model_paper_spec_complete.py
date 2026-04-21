"""
Stock Prediction CNN - Paper Specification Implementation (COMPLETE)
Matches paper hyperparameters with Steps 5-8:
- Step 5: Predictions with Confidence Scores
- Step 6: Backtesting Engine
- Step 7: Comprehensive Model Evaluation
- Step 8: Comprehensive Visualizations

Paper Parameters:
- Channels: 5 (OHLCV)
- Height: 256 (window size)
- Width: 9 (kernel size)
- Learning Rate: 1e-3
- Keep Prob (Dropout): 0.6
- Batch Size: 250
- Optimizer: Adam
- 8 Conv layers + 2 FC layers
- Output: 2-class (Bullish/Bearish confidence)

Python 3.13 & PyTorch 2.6 compatible
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

print(f"Python version: {__import__('sys').version}")
print(f"PyTorch version: {torch.__version__}")


# ============================================
# 1. DATA LOADING & PREPARATION
# ============================================

def load_stock_data(ticker: str, days: int = 252*3) -> pd.DataFrame:
    """Load historical stock data - extended to 3 years for 256-window size"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    return data


def extract_features(data: pd.DataFrame) -> np.ndarray:
    """Extract OHLCV features from stock data"""
    features = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
    print(f"✓ Extracted features shape: {features.shape}")
    print(f"  Channels: 5 (Open, High, Low, Close, Volume)")
    return features


def normalize_data(data: np.ndarray) -> tuple[np.ndarray, dict]:
    """Normalize features with shared OHLC scaler and independent Volume scaler"""
    scaled_data = np.zeros_like(data, dtype=np.float32)
    
    # Shared OHLC scaler
    ohlc_data = data[:, :4]
    ohlc_flat = ohlc_data.reshape(-1, 1)
    ohlc_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data[:, :4] = ohlc_scaler.fit_transform(ohlc_flat).reshape(-1, 4)
    
    # Independent Volume scaler
    volume_data = data[:, 4:5]
    volume_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data[:, 4] = volume_scaler.fit_transform(volume_data).flatten()
    
    scalers_dict = {
        'ohlc_scaler': ohlc_scaler,
        'volume_scaler': volume_scaler,
        'feature_names': ['Open', 'High', 'Low', 'Close', 'Volume']
    }
    
    print(f"✓ Data normalized: shape {scaled_data.shape}")
    return scaled_data, scalers_dict


def create_sequences(data: np.ndarray, seq_length: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding window sequences"""
    X, y = [], []
    
    for i in range(len(data) - seq_length - 1):
        window = data[i:i + seq_length]  # (256, 5)
        window = window.T  # Shape: (5, 256)
        X.append(window)
        
        current_close = data[i + seq_length, 3]
        next_close = data[i + seq_length + 1, 3]
        label = 1 if next_close > current_close else 0  # 1=Bullish, 0=Bearish
        y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"✓ Sequences created: X shape {X.shape}, y shape {y.shape}")
    return X, y


# ============================================
# 2. PAPER-SPEC CNN MODEL ARCHITECTURE
# ============================================

class PaperSpecCNN(nn.Module):
    """
    CNN matching paper specification:
    8 Conv layers + 2 FC layers with LeakyReLU and BatchNorm
    """
    
    def __init__(self):
        super().__init__()
        
        self.kernel_size = 9
        self.dropout_rate = 0.4  # 1 - keep_prob(0.6)
        self.padding = (self.kernel_size - 1) // 2
        
        # Conv1 - ReLU
        self.conv1 = nn.Conv1d(5, 128, self.kernel_size, padding=self.padding, bias=True)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(self.dropout_rate)
        
        # Conv2-8 - LeakyReLU + BatchNorm + Dropout
        self.conv2 = nn.Conv1d(128, 256, self.kernel_size, padding=self.padding, bias=True)
        self.bn2 = nn.BatchNorm1d(256)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout2 = nn.Dropout(self.dropout_rate)
        
        self.conv3 = nn.Conv1d(256, 256, self.kernel_size, padding=self.padding, bias=True)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(self.dropout_rate)
        
        self.conv4 = nn.Conv1d(256, 512, self.kernel_size, padding=self.padding, bias=True)
        self.bn4 = nn.BatchNorm1d(512)
        self.dropout4 = nn.Dropout(self.dropout_rate)
        
        self.conv5 = nn.Conv1d(512, 1024, self.kernel_size, padding=self.padding, bias=True)
        self.bn5 = nn.BatchNorm1d(1024)
        self.dropout5 = nn.Dropout(self.dropout_rate)
        
        self.conv6 = nn.Conv1d(1024, 1024, self.kernel_size, padding=self.padding, bias=True)
        self.bn6 = nn.BatchNorm1d(1024)
        self.dropout6 = nn.Dropout(self.dropout_rate)
        
        self.conv7 = nn.Conv1d(1024, 1024, self.kernel_size, padding=self.padding, bias=True)
        self.bn7 = nn.BatchNorm1d(1024)
        self.dropout7 = nn.Dropout(self.dropout_rate)
        
        self.conv8 = nn.Conv1d(1024, 1024, self.kernel_size, padding=self.padding, bias=True)
        self.bn8 = nn.BatchNorm1d(1024)
        self.dropout8 = nn.Dropout(self.dropout_rate)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # FC Layers
        self.fc1 = nn.Linear(1024, 256)
        self.fc1_dropout = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(256, 2)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lrelu(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.lrelu(x)
        x = self.dropout3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.lrelu(x)
        x = self.dropout4(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.lrelu(x)
        x = self.dropout5(x)
        
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.lrelu(x)
        x = self.dropout6(x)
        
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.lrelu(x)
        x = self.dropout7(x)
        
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.lrelu(x)
        x = self.dropout8(x)
        
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.lrelu(x)
        x = self.fc1_dropout(x)
        
        x = self.fc2(x)
        x = self.softmax(x)
        
        return x


# ============================================
# 3. TRAINING FUNCTION
# ============================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = 'cpu'
) -> dict:
    """Train the model with paper specifications"""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    patience, patience_counter = 15, 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).long()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
        
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).long()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        val_loss /= len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.6f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.6f}, Acc: {val_acc:.4f}")
        
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
# 5. PREDICTION WITH CONFIDENCE SCORES
# ============================================

def predict_with_confidence(
    model: nn.Module,
    X: np.ndarray,
    device: str = 'cpu'
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Make predictions with confidence scores
    
    Returns:
        predictions: Binary predictions (0=Bearish, 1=Bullish)
        bullish_confidence: Confidence scores for Bullish (0-100%)
        bearish_confidence: Confidence scores for Bearish (0-100%)
        probabilities: Raw 2-class probabilities
    """
    model.eval()
    
    # Ensure correct shape: (batch, 5, 256)
    if X.ndim == 2:
        X = X[:, np.newaxis, :]
    
    X_tensor = torch.FloatTensor(X).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)  # (batch, 2)
    
    probabilities = outputs.cpu().numpy()  # (batch, 2)
    
    # Extract confidence scores
    bullish_confidence = probabilities[:, 1] * 100  # Bullish = class 1
    bearish_confidence = probabilities[:, 0] * 100  # Bearish = class 0
    
    # Predictions: argmax of 2 classes
    predictions = np.argmax(probabilities, axis=1)  # 0 or 1
    
    return predictions, bullish_confidence, bearish_confidence, probabilities


# ============================================
# 6. BACKTESTING ENGINE
# ============================================

class BacktestingEngine:
    """Complete backtesting engine for trading strategy evaluation"""
    
    def __init__(self, initial_capital: float = 10000, transaction_cost: float = 0.001) -> None:
        """
        Args:
            initial_capital: Starting capital
            transaction_cost: Cost per transaction as percentage
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.results = None
    
    def run_backtest(
        self,
        prices: np.ndarray,
        predictions: np.ndarray,
        bullish_confidence: np.ndarray,
        min_confidence: float = 60.0,
        position_size: float = 0.95
    ) -> dict:
        """Run backtest on historical data"""
        
        # Ensure 1D arrays
        if prices.ndim > 1:
            prices = prices.flatten()
        if predictions.ndim > 1:
            predictions = predictions.flatten()
        if bullish_confidence.ndim > 1:
            bullish_confidence = bullish_confidence.flatten()
        
        portfolio_value = [self.initial_capital]
        trades = []
        holdings = 0.0
        cash = float(self.initial_capital)
        entry_price = None
        entry_date = None
        
        for i in range(len(predictions)):
            current_price = float(prices[i])
            prediction = int(predictions[i])
            confidence = float(bullish_confidence[i])
            
            # Only trade if confidence exceeds threshold
            if confidence < min_confidence and (100 - confidence) < min_confidence:
                portfolio_value.append(portfolio_value[-1])
                continue
            
            # Buy signal (Bullish prediction with high confidence)
            if prediction == 1 and holdings == 0 and confidence >= min_confidence:
                position_value = portfolio_value[-1] * position_size
                position_size_shares = position_value / current_price
                transaction_fee = position_value * self.transaction_cost
                
                holdings = float(position_size_shares)
                cash = portfolio_value[-1] - position_value - transaction_fee
                entry_price = current_price
                entry_date = i
            
            # Sell signal (Bearish prediction with high confidence)
            elif prediction == 0 and holdings > 0 and (100 - confidence) >= min_confidence:
                exit_value = holdings * current_price
                transaction_fee = exit_value * self.transaction_cost
                cash = cash + exit_value - transaction_fee
                
                trade_return = (current_price - entry_price) / entry_price if entry_price else 0
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': i,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'return': trade_return,
                    'shares': holdings,
                    'days_held': i - entry_date if entry_date else 0
                })
                
                holdings = 0.0
                entry_price = None
                entry_date = None
            
            # Update portfolio value
            current_portfolio_value = cash + (holdings * current_price)
            portfolio_value.append(current_portfolio_value)
        
        # Close any remaining position
        if holdings > 0:
            exit_value = holdings * prices[-1]
            cash = cash + exit_value
            portfolio_value[-1] = cash
        
        # Calculate metrics
        results_df = pd.DataFrame({'portfolio_value': portfolio_value[1:]})
        results_df['daily_return'] = results_df['portfolio_value'].pct_change()
        
        self.results = {
            'portfolio_values': portfolio_value[1:],
            'trades': pd.DataFrame(trades) if trades else pd.DataFrame(),
            'daily_returns': results_df['daily_return'].dropna().values,
            'total_trades': len(trades)
        }
        
        return self.results
    
    def get_metrics(self) -> dict:
        """Calculate comprehensive performance metrics"""
        if self.results is None:
            return None
        
        portfolio_values = self.results['portfolio_values']
        daily_returns = self.results['daily_returns']
        trades_df = self.results['trades']
        
        # Returns
        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital
        
        # Sharpe Ratio
        if len(daily_returns) > 0:
            mean_ret = float(np.mean(daily_returns))
            std_ret = float(np.std(daily_returns))
            sharpe_ratio = (mean_ret / (std_ret + 1e-8)) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Maximum Drawdown
        if len(daily_returns) > 0:
            cumulative_returns = np.cumprod(1 + daily_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = float(np.min(drawdown))
        else:
            max_drawdown = 0.0
        
        # Win Rate
        if len(trades_df) > 0:
            win_count = int((trades_df['return'] > 0).sum())
            win_rate = win_count / len(trades_df)
            avg_win = float(trades_df[trades_df['return'] > 0]['return'].mean()) if (trades_df['return'] > 0).any() else 0.0
            avg_loss = float(trades_df[trades_df['return'] <= 0]['return'].mean()) if (trades_df['return'] <= 0).any() else 0.0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
        else:
            win_rate = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            profit_factor = 0.0
        
        metrics = {
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'total_trades': self.results['total_trades'],
            'win_rate': float(win_rate),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'profit_factor': float(profit_factor),
            'final_portfolio_value': float(portfolio_values[-1]),
            'days_tested': len(portfolio_values)
        }
        
        return metrics


# ============================================
# 7. COMPREHENSIVE MODEL EVALUATION
# ============================================

def evaluate_model_comprehensive(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cpu'
) -> dict:
    """Comprehensive model evaluation with multiple metrics"""
    model.eval()
    
    all_predictions = []
    all_bullish_prob = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)  # (batch, 2)
            
            predicted = torch.argmax(outputs, dim=1).cpu().numpy()
            bullish_prob = outputs[:, 1].cpu().numpy()
            labels_np = labels.numpy()
            
            all_predictions.extend(predicted)
            all_bullish_prob.extend(bullish_prob)
            all_labels.extend(labels_np)
    
    all_predictions = np.array(all_predictions)
    all_bullish_prob = np.array(all_bullish_prob)
    all_labels = np.array(all_labels)
    
    # Classification metrics
    accuracy = np.mean(all_predictions == all_labels)
    auc_score = roc_auc_score(all_labels, all_bullish_prob)
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Per-class metrics
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*70)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"AUC-ROC Score: {auc_score:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives (Bearish->Bearish):  {tn}")
    print(f"  False Positives (Bullish->Bearish): {fp}")
    print(f"  False Negatives (Bearish->Bullish): {fn}")
    print(f"  True Positives (Bullish->Bullish):  {tp}")
    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_predictions,
                              target_names=['Bearish (0)', 'Bullish (1)']))
    print("="*70 + "\n")
    
    return {
        'accuracy': float(accuracy),
        'auc_score': float(auc_score),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'f1_score': float(f1_score),
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'bullish_probabilities': all_bullish_prob,
        'labels': all_labels,
        'roc_curve': roc_curve(all_labels, all_bullish_prob)
    }


# ============================================
# 8. COMPREHENSIVE VISUALIZATIONS
# ============================================

def plot_training_history(history: dict, filename: str = 'training_history.png') -> None:
    """Plot 4 training visualizations"""
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Training Loss
    plt.subplot(3, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2, color='steelblue')
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2, color='orange')
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('CrossEntropy Loss', fontsize=11)
    plt.title('Training and Validation Loss', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Training Accuracy
    plt.subplot(3, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy', linewidth=2, color='steelblue')
    plt.plot(history['val_acc'], label='Validation Accuracy', linewidth=2, color='orange')
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Accuracy', fontsize=11)
    plt.title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Loss + Accuracy Combined
    plt.subplot(3, 2, 3)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    line1 = ax1.plot(history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    line2 = ax2.plot(history['train_acc'], 'r-', label='Train Accuracy', linewidth=2)
    
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', color='b', fontsize=11)
    ax2.set_ylabel('Accuracy', color='r', fontsize=11)
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    ax1.set_title('Training Metrics Combined', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=10)
    
    # Plot 4: Epoch vs Validation Metrics
    plt.subplot(3, 2, 4)
    epochs_range = range(1, len(history['val_acc']) + 1)
    plt.plot(epochs_range, history['val_acc'], marker='o', linewidth=2, 
             markersize=4, color='green', label='Validation Accuracy')
    plt.fill_between(epochs_range, history['val_acc'], alpha=0.3, color='green')
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Accuracy', fontsize=11)
    plt.title('Validation Accuracy Progression', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Loss Distribution
    plt.subplot(3, 2, 5)
    plt.hist(history['train_loss'], bins=20, alpha=0.7, label='Train Loss', color='steelblue', edgecolor='black')
    plt.hist(history['val_loss'], bins=20, alpha=0.7, label='Val Loss', color='orange', edgecolor='black')
    plt.xlabel('Loss Value', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Loss Distribution', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Summary Statistics
    plt.subplot(3, 2, 6)
    plt.axis('off')
    summary_text = f"""
    TRAINING SUMMARY
    ────────────────────────────
    Total Epochs: {len(history['train_loss'])}
    
    Best Validation Accuracy: {max(history['val_acc'])*100:.2f}%
    Best Training Accuracy: {max(history['train_acc'])*100:.2f}%
    
    Final Train Loss: {history['train_loss'][-1]:.6f}
    Final Val Loss: {history['val_loss'][-1]:.6f}
    
    Loss Improvement: {(history['train_loss'][0] - history['train_loss'][-1])/history['train_loss'][0]*100:.2f}%
    Accuracy Improvement: {(history['train_acc'][-1] - history['train_acc'][0])*100:.2f}%
    
    Average Val Accuracy: {np.mean(history['val_acc'])*100:.2f}%
    Std Dev Val Accuracy: {np.std(history['val_acc'])*100:.2f}%
    """
    plt.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Training History - Paper Specification CNN', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


def plot_evaluation_metrics(eval_results: dict, filename: str = 'evaluation_metrics.png') -> None:
    """Plot 4 evaluation visualizations"""
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Confusion Matrix
    plt.subplot(2, 2, 1)
    cm = eval_results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Bearish', 'Bullish'],
                yticklabels=['Bearish', 'Bullish'],
                annot_kws={'size': 14, 'weight': 'bold'})
    plt.ylabel('True Label', fontsize=11)
    plt.xlabel('Predicted Label', fontsize=11)
    plt.title('Confusion Matrix', fontsize=12, fontweight='bold')
    
    # Plot 2: ROC Curve
    plt.subplot(2, 2, 2)
    fpr, tpr, _ = eval_results['roc_curve']
    roc_auc = eval_results['auc_score']
    plt.plot(fpr, tpr, linewidth=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})', color='steelblue')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
    plt.fill_between(fpr, tpr, alpha=0.3, color='steelblue')
    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate', fontsize=11)
    plt.title('ROC Curve', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Predicted Bullish Probability Distribution
    plt.subplot(2, 2, 3)
    bullish_probs = eval_results['bullish_probabilities']
    labels = eval_results['labels']
    
    plt.hist(bullish_probs[labels == 0], bins=50, alpha=0.7, label='Actual Bearish', 
             color='red', edgecolor='black')
    plt.hist(bullish_probs[labels == 1], bins=50, alpha=0.7, label='Actual Bullish',
             color='green', edgecolor='black')
    plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision Boundary')
    plt.xlabel('Predicted Bullish Probability', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Probability Distribution by True Label', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Classification Metrics Summary
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    metrics_text = f"""
    EVALUATION METRICS SUMMARY
    ────────────────────────────────────
    Accuracy:       {eval_results['accuracy']*100:6.2f}%
    Precision:      {eval_results['precision']*100:6.2f}%
    Recall:         {eval_results['recall']*100:6.2f}%
    Specificity:    {eval_results['specificity']*100:6.2f}%
    F1-Score:       {eval_results['f1_score']:6.4f}
    AUC-ROC:        {eval_results['auc_score']:6.4f}
    
    CONFUSION MATRIX BREAKDOWN
    ────────────────────────────────────
    True Negatives:   {cm[0,0]:6d}
    False Positives:  {cm[0,1]:6d}
    False Negatives:  {cm[1,0]:6d}
    True Positives:   {cm[1,1]:6d}
    
    Total Predictions: {np.sum(cm):6d}
    Correct:          {cm[0,0] + cm[1,1]:6d}
    Incorrect:        {cm[0,1] + cm[1,0]:6d}
    """
    
    plt.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.suptitle('Model Evaluation Metrics - Paper Specification CNN', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


def plot_backtest_results(
    backtest_engine: BacktestingEngine,
    prices: np.ndarray,
    predictions: np.ndarray,
    bullish_confidence: np.ndarray,
    filename: str = 'backtest_results.png'
) -> None:
    """Plot 4 backtesting visualizations"""
    
    if prices.ndim > 1:
        prices = prices.flatten()
    if predictions.ndim > 1:
        predictions = predictions.flatten()
    if bullish_confidence.ndim > 1:
        bullish_confidence = bullish_confidence.flatten()
    
    metrics = backtest_engine.get_metrics()
    portfolio_values = backtest_engine.results['portfolio_values']
    trades_df = backtest_engine.results['trades']
    daily_returns = backtest_engine.results['daily_returns']
    
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Portfolio Value Over Time
    plt.subplot(2, 2, 1)
    plt.plot(portfolio_values, linewidth=2, color='steelblue')
    plt.axhline(y=backtest_engine.initial_capital, color='red', linestyle='--', 
               linewidth=2, label=f'Initial Capital: ${backtest_engine.initial_capital:.0f}')
    plt.fill_between(range(len(portfolio_values)), backtest_engine.initial_capital, 
                     portfolio_values, alpha=0.3, color='steelblue')
    plt.xlabel('Time Period', fontsize=11)
    plt.ylabel('Portfolio Value ($)', fontsize=11)
    plt.title('Portfolio Value Over Time', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Price with Buy/Sell Signals
    plt.subplot(2, 2, 2)
    plt.plot(prices, label='Close Price', color='black', linewidth=1.5)
    
    buy_signals = np.where((predictions == 1) & (bullish_confidence >= 60))[0]
    sell_signals = np.where((predictions == 0) & (bullish_confidence <= 40))[0]
    
    if len(buy_signals) > 0:
        plt.scatter(buy_signals, prices[buy_signals], color='green', marker='^', 
                   s=100, label=f'Buy Signals ({len(buy_signals)})', zorder=5)
    if len(sell_signals) > 0:
        plt.scatter(sell_signals, prices[sell_signals], color='red', marker='v',
                   s=100, label=f'Sell Signals ({len(sell_signals)})', zorder=5)
    
    plt.xlabel('Time Period', fontsize=11)
    plt.ylabel('Close Price ($)', fontsize=11)
    plt.title('Price Action with Trading Signals', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Daily Returns Distribution
    plt.subplot(2, 2, 3)
    plt.hist(daily_returns, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    mean_ret = np.mean(daily_returns)
    std_ret = np.std(daily_returns)
    plt.axvline(x=mean_ret, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_ret:.4f}, Std: {std_ret:.4f}')
    plt.xlabel('Daily Return', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Daily Returns Distribution', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Backtest Metrics Table
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    metrics_text = f"""
    BACKTEST PERFORMANCE METRICS
    ────────────────────────────────────
    Initial Capital:      ${backtest_engine.initial_capital:>10,.0f}
    Final Portfolio:      ${metrics['final_portfolio_value']:>10,.0f}
    Total Return:         {metrics['total_return']*100:>10.2f}%
    
    Number of Trades:     {metrics['total_trades']:>10d}
    Win Rate:             {metrics['win_rate']*100:>10.2f}%
    Avg Win:              {metrics['avg_win']*100:>10.2f}%
    Avg Loss:             {metrics['avg_loss']*100:>10.2f}%
    Profit Factor:        {metrics['profit_factor']:>10.2f}x
    
    Sharpe Ratio:         {metrics['sharpe_ratio']:>10.4f}
    Max Drawdown:         {metrics['max_drawdown']*100:>10.2f}%
    
    Days Tested:          {metrics['days_tested']:>10d}
    Avg Trade Duration:   {trades_df['days_held'].mean() if len(trades_df) > 0 else 0:>10.0f} days
    """
    
    plt.text(0.05, 0.5, metrics_text, fontsize=11, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.suptitle('Backtest Results - Paper Specification CNN', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


def plot_confidence_analysis(
    bullish_confidence: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    filename: str = 'confidence_analysis.png'
) -> None:
    """Plot confidence score analysis"""
    
    plt.figure(figsize=(16, 10))
    
    # Plot 1: Confidence by Prediction
    plt.subplot(2, 2, 1)
    bullish_preds = bullish_confidence[predictions == 1]
    bearish_preds = 100 - bullish_confidence[predictions == 0]
    
    plt.boxplot([bullish_preds, bearish_preds], labels=['Bullish Pred', 'Bearish Pred'],
               patch_artist=True, boxprops=dict(facecolor='lightblue', alpha=0.7))
    plt.ylabel('Confidence (%)', fontsize=11)
    plt.title('Confidence Score Distribution by Prediction', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Accuracy by Confidence
    plt.subplot(2, 2, 2)
    confidence_bins = np.arange(0, 101, 10)
    accuracies = []
    bin_centers = []
    
    for i in range(len(confidence_bins)-1):
        lower = confidence_bins[i]
        upper = confidence_bins[i+1]
        mask = (np.maximum(bullish_confidence, 100-bullish_confidence) >= lower) & \
               (np.maximum(bullish_confidence, 100-bullish_confidence) < upper)
        
        if np.sum(mask) > 0:
            acc = np.mean(predictions[mask] == labels[mask])
            accuracies.append(acc)
            bin_centers.append((lower + upper) / 2)
    
    plt.plot(bin_centers, accuracies, marker='o', linewidth=2, markersize=8, color='steelblue')
    plt.xlabel('Confidence Level (%)', fontsize=11)
    plt.ylabel('Accuracy', fontsize=11)
    plt.title('Accuracy vs Confidence Threshold', fontsize=12, fontweight='bold')
    plt.ylim([0.4, 1.0])
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Confidence Distribution - Correct vs Incorrect
    plt.subplot(2, 2, 3)
    max_conf = np.maximum(bullish_confidence, 100 - bullish_confidence)
    correct = predictions == labels
    
    plt.hist(max_conf[correct], bins=50, alpha=0.7, label='Correct',
            color='green', edgecolor='black')
    plt.hist(max_conf[~correct], bins=50, alpha=0.7, label='Incorrect',
            color='red', edgecolor='black')
    plt.xlabel('Max Confidence (%)', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Confidence Distribution: Correct vs Incorrect Predictions', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Confidence Statistics
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    stats_text = f"""
    CONFIDENCE SCORE ANALYSIS
    ────────────────────────────────────
    Bullish Confidence (Predictions=1):
      Mean:     {np.mean(bullish_preds):.2f}%
      Std:      {np.std(bullish_preds):.2f}%
      Min:      {np.min(bullish_preds):.2f}%
      Max:      {np.max(bullish_preds):.2f}%
    
    Bearish Confidence (Predictions=0):
      Mean:     {np.mean(bearish_preds):.2f}%
      Std:      {np.std(bearish_preds):.2f}%
      Min:      {np.min(bearish_preds):.2f}%
      Max:      {np.max(bearish_preds):.2f}%
    
    Overall Statistics:
      Correct with 80%+ conf:  {np.sum((max_conf >= 80) & correct)} / {np.sum(max_conf >= 80)}
      Correct with 60-80% conf: {np.sum(((max_conf >= 60) & (max_conf < 80)) & correct)} / {np.sum((max_conf >= 60) & (max_conf < 80))}
      Correct with 50-60% conf: {np.sum(((max_conf >= 50) & (max_conf < 60)) & correct)} / {np.sum((max_conf >= 50) & (max_conf < 60))}
    """
    
    plt.text(0.05, 0.5, stats_text, fontsize=10, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))
    
    plt.suptitle('Confidence Score Analysis', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


# ============================================
# 9. MAIN EXECUTION
# ============================================

def main() -> tuple[nn.Module, dict]:
    """Complete pipeline with all steps"""
    
    TICKER = 'JPM'
    SEQ_LENGTH = 256
    BATCH_SIZE = 250
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    NUM_FEATURES = 5
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*70}")
    print(f"Stock Prediction CNN - COMPLETE IMPLEMENTATION")
    print(f"Paper Specification with Steps 5-8")
    print(f"{'='*70}")
    print(f"Device: {DEVICE}, Stock: {TICKER}\n")
    
    # ============ LOAD DATA ============
    print("[1] Loading stock data...")
    raw_data = yf.download(TICKER, start="2021-01-01", end=datetime.now(), progress=False)
    print(f"✓ Raw data shape: {raw_data.shape}")
    
    # ============ EXTRACT FEATURES ============
    print("\n[2] Extracting OHLCV features...")
    features = extract_features(raw_data)
    
    # ============ NORMALIZE ============
    print("\n[3] Normalizing features...")
    scaled_data, scalers_dict = normalize_data(features)
    
    # ============ CREATE SEQUENCES ============
    print("\n[4] Creating sequences (height=256)...")
    X, y = create_sequences(scaled_data, seq_length=SEQ_LENGTH)
    
    # Train/test split
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    # Convert to tensors
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
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)
    
    # ============ BUILD & TRAIN ============
    print("\n[5] Building and training model...")
    model = PaperSpecCNN()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✓ Model parameters: {param_count:,}")
    
    history = train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE)
    
    # ============ STEP 5: PREDICTIONS WITH CONFIDENCE ============
    print("\n[6] Making predictions with confidence scores...")
    test_predictions, test_bullish_conf, test_bearish_conf, test_probs = predict_with_confidence(
        model, X_test.numpy(), device=DEVICE
    )
    
    print(f"✓ Predictions generated: {len(test_predictions)}")
    print(f"  Bullish confidence - Mean: {np.mean(test_bullish_conf):.2f}%, Std: {np.std(test_bullish_conf):.2f}%")
    print(f"  Bearish confidence - Mean: {np.mean(test_bearish_conf):.2f}%, Std: {np.std(test_bearish_conf):.2f}%")
    
    # ============ STEP 6: BACKTESTING ============
    print("\n[7] Running backtest analysis...")
    test_prices = features[train_size + val_size + SEQ_LENGTH + 1:]
    if test_prices.ndim > 1:
        test_prices = test_prices[:, 3].flatten()  # Close price
    else:
        test_prices = test_prices.flatten()
    
    backtest_engine = BacktestingEngine(initial_capital=10000, transaction_cost=0.001)
    backtest_results = backtest_engine.run_backtest(
        test_prices, test_predictions, test_bullish_conf,
        min_confidence=60.0, position_size=0.95
    )
    
    metrics = backtest_engine.get_metrics()
    print("\n✓ Backtest Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:.<30} {value:.4f}")
        else:
            print(f"  {key:.<30} {value}")
    
    # ============ STEP 7: COMPREHENSIVE EVALUATION ============
    print("\n[8] Comprehensive model evaluation...")
    eval_results = evaluate_model_comprehensive(model, test_loader, device=DEVICE)
    
    # ============ STEP 8: VISUALIZATIONS ============
    print("\n[9] Creating visualizations...")
    
    # Visualization 1: Training History
    plot_training_history(history, 'paper_spec_training_history.png')
    
    # Visualization 2: Evaluation Metrics
    plot_evaluation_metrics(eval_results, 'paper_spec_evaluation_metrics.png')
    
    # Visualization 3: Backtest Results
    plot_backtest_results(backtest_engine, test_prices, test_predictions, test_bullish_conf,
                         'paper_spec_backtest_results.png')
    
    # Visualization 4: Confidence Analysis
    plot_confidence_analysis(test_bullish_conf, test_predictions, 
                            y_test.numpy(), 'paper_spec_confidence_analysis.png')
    
    # ============ SAVE MODEL ============
    print("\n[10] Saving model and scalers...")
    torch.save(model.state_dict(), 'stock_prediction_paper_spec_complete.pth')
    with open('scalers_paper_spec_complete.pkl', 'wb') as f:
        pickle.dump(scalers_dict, f)
    print("✓ Model saved to: stock_prediction_paper_spec_complete.pth")
    print("✓ Scalers saved to: scalers_paper_spec_complete.pkl")
    
    # ============ FINAL SUMMARY ============
    print("\n" + "="*70)
    print("COMPLETE TRAINING SUMMARY")
    print("="*70)
    print(f"\n✓ STEP 1-4: Data Preparation")
    print(f"  - Loaded {len(features)} trading days")
    print(f"  - Created {len(X)} sequences (height=256, channels=5)")
    print(f"  - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    print(f"\n✓ STEP 5: Predictions with Confidence")
    print(f"  - Generated predictions for {len(test_predictions)} test samples")
    print(f"  - Bullish avg confidence: {np.mean(test_bullish_conf):.2f}%")
    print(f"  - High confidence predictions (>70%): {np.sum(np.maximum(test_bullish_conf, 100-test_bullish_conf) > 70)}")
    
    print(f"\n✓ STEP 6: Backtesting Engine")
    print(f"  - Total trades executed: {metrics['total_trades']}")
    print(f"  - Win rate: {metrics['win_rate']*100:.2f}%")
    print(f"  - Total return: {metrics['total_return']*100:.2f}%")
    print(f"  - Sharpe ratio: {metrics['sharpe_ratio']:.4f}")
    
    print(f"\n✓ STEP 7: Comprehensive Evaluation")
    print(f"  - Accuracy: {eval_results['accuracy']*100:.2f}%")
    print(f"  - Precision: {eval_results['precision']*100:.2f}%")
    print(f"  - Recall: {eval_results['recall']*100:.2f}%")
    print(f"  - F1-Score: {eval_results['f1_score']:.4f}")
    print(f"  - AUC-ROC: {eval_results['auc_score']:.4f}")
    
    print(f"\n✓ STEP 8: Comprehensive Visualizations")
    print(f"  - paper_spec_training_history.png (6 subplots)")
    print(f"  - paper_spec_evaluation_metrics.png (4 subplots)")
    print(f"  - paper_spec_backtest_results.png (4 subplots)")
    print(f"  - paper_spec_confidence_analysis.png (4 subplots)")
    
    print("\n" + "="*70)
    print("✅ COMPLETE PIPELINE FINISHED!")
    print("="*70 + "\n")
    
    return model, scalers_dict


if __name__ == "__main__":
    model, scalers_dict = main()