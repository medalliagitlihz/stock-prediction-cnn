"""
Complete training script with MULTIPLE FEATURES (Open, Close, High, Low, Volume)
with backtesting, evaluation, and comprehensive visualizations
MODIFIED: Shared scaler for OHLC, independent scaler for Volume
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
import seaborn as sns
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


def extract_features(data: pd.DataFrame) -> np.ndarray:
    """
    Extract OHLCV features from stock data
    
    Args:
        data: DataFrame with columns [Open, High, Low, Close, Volume]
    
    Returns:
        features: Array of shape (N, 5) with [Open, High, Low, Close, Volume]
    """
    features = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
    print(f"✓ Extracted features shape: {features.shape}")
    print(f"  Features: Open, High, Low, Close, Volume")
    return features


def normalize_data(data: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Normalize features with different scaling strategies:
    - Shared scaler for OHLC (Open, High, Low, Close) - indices 0-3
    - Independent scaler for Volume - index 4
    
    Args:
        data: Array of shape (N, 5) with features [Open, High, Low, Close, Volume]
    
    Returns:
        scaled_data: Normalized data (N, 5)
        scalers_dict: Dict with 'ohlc_scaler' and 'volume_scaler'
    """
    scaled_data = np.zeros_like(data, dtype=np.float32)
    
    # ============ OHLC: Shared Scaler (indices 0-3) ============
    # Combine all OHLC values for a single scaler
    ohlc_data = data[:, :4]  # Shape: (N, 4)
    ohlc_scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Reshape to (N*4, 1) for fitting, then transform all OHLC values
    ohlc_flat = ohlc_data.reshape(-1, 1)  # Shape: (N*4, 1)
    ohlc_scaled = ohlc_scaler.fit_transform(ohlc_flat)  # Shape: (N*4, 1)
    scaled_data[:, :4] = ohlc_scaled.reshape(-1, 4)  # Shape: (N, 4)
    
    print(f"✓ OHLC scaled with shared scaler")
    print(f"  OHLC range: [{ohlc_scaler.data_min_[0]:.2f}, {ohlc_scaler.data_max_[0]:.2f}]")
    
    # ============ Volume: Independent Scaler (index 4) ============
    volume_data = data[:, 4:5]  # Shape: (N, 1)
    volume_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data[:, 4] = volume_scaler.fit_transform(volume_data).flatten()
    
    print(f"✓ Volume scaled with independent scaler")
    print(f"  Volume range: [{volume_scaler.data_min_[0]:.0f}, {volume_scaler.data_max_[0]:.0f}]")
    
    # Store scalers in dictionary
    scalers_dict = {
        'ohlc_scaler': ohlc_scaler,
        'volume_scaler': volume_scaler,
        'feature_names': ['Open', 'High', 'Low', 'Close', 'Volume']
    }
    
    print(f"✓ Data normalized: shape {scaled_data.shape}")
    print(f"  OHLC (Open, High, Low, Close) use shared scaler")
    print(f"  Volume uses independent scaler")
    
    return scaled_data, scalers_dict


# ============================================
# 2. FEATURE ENGINEERING - SLIDING WINDOWS
# ============================================

def create_sequences(data: np.ndarray, seq_length: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for training
    
    Args:
        data: Normalized stock data of shape (N, 5) with features
        seq_length: Length of each sequence window
    
    Returns:
        X: Input sequences (N, 5, seq_length) - CORRECT FOR CONV1D
        y: Binary labels (Close price movement)
    """
    X, y = [], []
    
    for i in range(len(data) - seq_length - 1):
        window = data[i:i + seq_length]  # (seq_length, 5)
        window = window.T  # Shape: (5, seq_length)
        X.append(window)
        
        # Use Close price (index 3) for label
        current_close = data[i + seq_length, 3]
        next_close = data[i + seq_length + 1, 3]
        label = 1 if next_close > current_close else 0
        y.append(label)
    
    X = np.array(X)  # Shape: (N, 5, seq_length)
    y = np.array(y)
    
    print(f"✓ Sequences created: X shape {X.shape}, y shape {y.shape}")
    print(f"  X format: (batch, features=5, sequence_length={seq_length})")
    return X, y


# ============================================
# 3. 1D CNN MODEL ARCHITECTURE (MULTI-FEATURE)
# ============================================

class StockPredictionCNN(nn.Module):
    """
    10-layer 1D CNN for stock price movement prediction
    UPDATED: Accepts 5 input features (Open, High, Low, Close, Volume)
    """
    
    def __init__(self, seq_length: int = 50, num_filters: int = 32, num_features: int = 5) -> None:
        super().__init__()
        
        # Conv layers - First layer accepts 5 channels (OHLCV)
        self.conv1 = nn.Conv1d(num_features, num_filters, kernel_size=3, padding=1)
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
        """Forward pass - x shape: (batch, 5, seq_length)"""
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
# 5. PREDICTION WITH CONFIDENCE SCORES
# ============================================

def predict_with_confidence(
    model: nn.Module,
    X: np.ndarray,
    device: str = 'cpu'
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Make predictions with confidence scores
    
    Args:
        model: Trained model
        X: Input data (batch, 5, seq_length)
        device: Device to use
    
    Returns:
        predictions: Binary predictions (0 or 1)
        confidences: Confidence scores (0-100%)
        probabilities: Raw probabilities from sigmoid
    """
    model.eval()
    
    # Ensure correct shape: (batch, 5, seq_length)
    if X.ndim == 2:
        X = X[:, np.newaxis, :]
    
    X_tensor = torch.FloatTensor(X).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)
    
    probabilities = outputs.cpu().numpy()
    if probabilities.ndim > 1:
        probabilities = probabilities.flatten()
    
    predictions = (probabilities > 0.5).astype(int)
    confidences = np.abs(probabilities - 0.5) * 2 * 100
    
    return predictions, confidences, probabilities


# ============================================
# 6. BACKTESTING ENGINE
# ============================================

class BacktestingEngine:
    """Backtesting engine for evaluating trading strategy"""
    
    def __init__(self, initial_capital: float = 10000, transaction_cost: float = 0.001) -> None:
        """
        Args:
            initial_capital: Starting capital for trading
            transaction_cost: Transaction cost as percentage
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.results = None
    
    def run_backtest(
        self,
        prices: np.ndarray,
        predictions: np.ndarray,
        confidences: np.ndarray,
        min_confidence: float = 50.0,
        position_size: float = 0.95
    ) -> dict:
        """Run backtest on historical data"""
        
        # Ensure prices is 1D
        if prices.ndim > 1:
            prices = prices.flatten()
        if predictions.ndim > 1:
            predictions = predictions.flatten()
        if confidences.ndim > 1:
            confidences = confidences.flatten()
        
        portfolio_value = [self.initial_capital]
        trades = []
        holdings = 0.0
        cash = float(self.initial_capital)
        entry_price = None
        entry_date = None
        
        for i in range(len(predictions)):
            current_price = float(prices[i])
            prediction = int(predictions[i])
            confidence = float(confidences[i])
            
            # Only trade if confidence exceeds threshold
            if confidence < min_confidence:
                portfolio_value.append(portfolio_value[-1])
                continue
            
            # Buy signal
            if prediction == 1 and holdings == 0 and confidence >= min_confidence:
                position_value = portfolio_value[-1] * position_size
                position_size_shares = position_value / current_price
                transaction_fee = position_value * self.transaction_cost
                
                holdings = float(position_size_shares)
                cash = portfolio_value[-1] - position_value - transaction_fee
                entry_price = current_price
                entry_date = i
            
            # Sell signal
            elif prediction == 0 and holdings > 0:
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
                    'shares': holdings
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
    
    def get_metrics(self) -> dict | None:
        """Calculate performance metrics"""
        if self.results is None:
            return None
        
        portfolio_values = self.results['portfolio_values']
        daily_returns = self.results['daily_returns']
        trades_df = self.results['trades']
        
        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital
        annual_return = total_return
        
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
            'annual_return': float(annual_return),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'total_trades': self.results['total_trades'],
            'win_rate': float(win_rate),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': float(profit_factor),
            'final_portfolio_value': float(portfolio_values[-1])
        }
        
        return metrics


# ============================================
# 7. EVALUATION & VISUALIZATION FUNCTIONS
# ============================================

def evaluate_model_comprehensive(
    model: nn.Module,
    test_loader: DataLoader,
    test_data: torch.Tensor,
    test_labels: torch.Tensor,
    device: str = 'cpu'
) -> dict:
    """Comprehensive model evaluation with multiple metrics"""
    model.eval()
    
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            predicted = (outputs > 0.5).float().cpu().numpy().flatten()
            probs = outputs.cpu().numpy().flatten()
            labels_np = labels.numpy().flatten()
            
            all_predictions.extend(predicted)
            all_probabilities.extend(probs)
            all_labels.extend(labels_np)
    
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = np.mean(all_predictions == all_labels)
    auc_score = roc_auc_score(all_labels, all_probabilities)
    cm = confusion_matrix(all_labels, all_predictions)
    
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*60)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"AUC-ROC Score: {auc_score:.4f}")
    print("\nConfusion Matrix:")
    print(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, 
                               target_names=['DOWN', 'UP']))
    print("="*60 + "\n")
    
    return {
        'accuracy': float(accuracy),
        'auc_score': float(auc_score),
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'labels': all_labels
    }


def plot_training_history(history: dict, filename: str = 'training_history.png') -> None:
    """Plot training history"""
    plt.figure(figsize=(14, 5))
    
    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train', linewidth=2)
    plt.plot(history['val_loss'], label='Val', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True, alpha=0.3)
    
    # Accuracy plot
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train', linewidth=2)
    plt.plot(history['val_acc'], label='Val', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.grid(True, alpha=0.3)
    
    # Both on same plot
    plt.subplot(1, 3, 3)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    line1 = ax1.plot(history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    line2 = ax2.plot(history['train_acc'], 'r-', label='Train Acc', linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='b')
    ax2.set_ylabel('Accuracy', color='r')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    ax1.set_title('Training Metrics Combined')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


def plot_backtest_results(
    backtest_engine: BacktestingEngine,
    prices: np.ndarray,
    predictions: np.ndarray,
    title: str = "Backtest Results"
) -> None:
    """Plot backtest results and metrics"""
    if prices.ndim > 1:
        prices = prices.flatten()
    
    metrics = backtest_engine.get_metrics()
    portfolio_values = backtest_engine.results['portfolio_values']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Portfolio value over time
    ax = axes[0, 0]
    ax.plot(portfolio_values, linewidth=2, color='blue')
    ax.axhline(y=backtest_engine.initial_capital, color='red', linestyle='--', 
               label=f'Initial: ${backtest_engine.initial_capital:.0f}')
    ax.fill_between(range(len(portfolio_values)), backtest_engine.initial_capital, 
                     portfolio_values, alpha=0.3)
    ax.set_xlabel('Time Period')
    ax.set_ylabel('Portfolio Value ($)')
    ax.set_title('Portfolio Value Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Price and predictions
    ax = axes[0, 1]
    ax.plot(prices, label='Price', color='black', linewidth=2)
    buy_signals = np.where(predictions == 1)[0]
    sell_signals = np.where(predictions == 0)[0]
    ax.scatter(buy_signals, prices[buy_signals], color='green', marker='^', 
               s=100, label=f'Buy Signals ({len(buy_signals)})', zorder=5)
    ax.scatter(sell_signals, prices[sell_signals], color='red', marker='v', 
               s=100, label=f'Sell Signals ({len(sell_signals)})', zorder=5)
    ax.set_xlabel('Time Period')
    ax.set_ylabel('Price ($)')
    ax.set_title('Price with Buy/Sell Signals')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Metrics table
    ax = axes[1, 0]
    ax.axis('tight')
    ax.axis('off')
    metrics_data = [
        ['Metric', 'Value'],
        ['Total Return', f"{metrics['total_return']*100:.2f}%"],
        ['Sharpe Ratio', f"{metrics['sharpe_ratio']:.4f}"],
        ['Max Drawdown', f"{metrics['max_drawdown']*100:.2f}%"],
        ['Total Trades', f"{metrics['total_trades']}"],
        ['Win Rate', f"{metrics['win_rate']*100:.2f}%"],
        ['Profit Factor', f"{metrics['profit_factor']:.4f}"],
        ['Final Value', f"${metrics['final_portfolio_value']:.2f}"]
    ]
    table = ax.table(cellText=metrics_data, cellLoc='center', loc='center',
                    colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Backtest Metrics', fontsize=12, fontweight='bold')
    
    # Daily returns histogram
    ax = axes[1, 1]
    daily_returns = backtest_engine.results['daily_returns']
    ax.hist(daily_returns, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    mean_ret = float(np.mean(daily_returns))
    ax.axvline(x=mean_ret, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_ret:.4f}')
    ax.set_xlabel('Daily Return')
    ax.set_ylabel('Frequency')
    ax.set_title('Daily Returns Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_roc_curve(eval_results: dict, filename: str = 'roc_curve.png') -> None:
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(eval_results['labels'], eval_results['probabilities'])
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2.5, label=f"AUC = {eval_results['auc_score']:.4f}", 
             color='steelblue')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


def plot_confusion_matrix(eval_results: dict, filename: str = 'confusion_matrix.png') -> None:
    """Plot confusion matrix"""
    cm = eval_results['confusion_matrix']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['DOWN', 'UP'],
                yticklabels=['DOWN', 'UP'],
                cbar_kws={'label': 'Count'})
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


def plot_scaler_info(scalers_dict: dict, filename: str = 'scaler_info.png') -> None:
    """Plot information about scalers"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # OHLC Scaler info
    ax = axes[0]
    ax.axis('tight')
    ax.axis('off')
    
    ohlc_scaler = scalers_dict['ohlc_scaler']
    ohlc_data = [
        ['Feature', 'Min', 'Max', 'Range'],
        ['OHLC', f"{ohlc_scaler.data_min_[0]:.2f}", 
         f"{ohlc_scaler.data_max_[0]:.2f}",
         f"{ohlc_scaler.data_max_[0] - ohlc_scaler.data_min_[0]:.2f}"]
    ]
    table1 = ax.table(cellText=ohlc_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table1.auto_set_font_size(False)
    table1.set_fontsize(11)
    table1.scale(1, 2.5)
    
    # Header styling
    for i in range(4):
        table1[(0, i)].set_facecolor('#40466e')
        table1[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('OHLC Scaler (Shared)\nOpen, High, Low, Close', 
                fontsize=12, fontweight='bold', pad=20)
    
    # Volume Scaler info
    ax = axes[1]
    ax.axis('tight')
    ax.axis('off')
    
    volume_scaler = scalers_dict['volume_scaler']
    volume_data = [
        ['Feature', 'Min', 'Max', 'Range'],
        ['Volume', f"{volume_scaler.data_min_[0]:.0f}", 
         f"{volume_scaler.data_max_[0]:.0f}",
         f"{volume_scaler.data_max_[0] - volume_scaler.data_min_[0]:.0f}"]
    ]
    table2 = ax.table(cellText=volume_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table2.auto_set_font_size(False)
    table2.set_fontsize(11)
    table2.scale(1, 2.5)
    
    # Header styling
    for i in range(4):
        table2[(0, i)].set_facecolor('#40466e')
        table2[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Volume Scaler (Independent)\nVolume Only', 
                fontsize=12, fontweight='bold', pad=20)
    
    plt.suptitle('Feature Scaling Information', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


# ============================================
# 8. MAIN EXECUTION
# ============================================

def main() -> tuple[nn.Module, dict]:
    """Main pipeline with all steps"""
    
    TICKER = 'JPM'
    SEQ_LENGTH = 50
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    NUM_FEATURES = 5
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*60}")
    print(f"Stock Prediction CNN - Enhanced Multi-Feature Training")
    print(f"Features: Open, High, Low, Close, Volume")
    print(f"Scaling: Shared OHLC + Independent Volume")
    print(f"Python 3.13 & PyTorch 2.6")
    print(f"{'='*60}")
    print(f"Device: {DEVICE}")
    print(f"Stock: {TICKER}\n")
    
    # ============ STEP 1: LOAD DATA ============
    print("[1] Loading stock data...")
    raw_data = yf.download(TICKER, start="2022-01-01", end=datetime.now(), progress=False)
    print(f"✓ Raw data shape: {raw_data.shape}")
    
    # ============ STEP 2: EXTRACT FEATURES ============
    print("\n[2] Extracting features...")
    features = extract_features(raw_data)
    
    # ============ STEP 3: NORMALIZE FEATURES (MODIFIED) ============
    print("\n[3] Normalizing features (Shared OHLC + Independent Volume)...")
    scaled_data, scalers_dict = normalize_data(features)
    
    # ============ STEP 4: CREATE SEQUENCES ============
    print("\n[4] Creating sequences...")
    X, y = create_sequences(scaled_data, seq_length=SEQ_LENGTH)
    
    print(f"✓ X dtype: {X.dtype}, shape: {X.shape}")
    print(f"✓ y dtype: {y.dtype}, shape: {y.shape}")
    
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
    
    # ============ STEP 5: BUILD & TRAIN MODEL ============
    print("\n[5] Building and training model...")
    model = StockPredictionCNN(seq_length=SEQ_LENGTH, num_filters=32, num_features=NUM_FEATURES)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✓ Model parameters: {param_count:,}")
    print(f"✓ Input channels: {NUM_FEATURES} (Open, High, Low, Close, Volume)")
    
    history = train_model(model, train_loader, val_loader, 
                         epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE)
    
    # ============ STEP 6: MAKE PREDICTIONS & BACKTEST ============
    print("\n[6] Making predictions and running backtest...")
    test_predictions, test_confidences, _ = predict_with_confidence(
        model, X_test.numpy(), device=DEVICE
    )
    
    # Get actual prices for backtesting (use Close price - index 3)
    test_prices = features[train_size + val_size + SEQ_LENGTH + 1:]
    if test_prices.ndim > 1:
        test_prices = test_prices[:, 3].flatten()  # Extract Close price
    else:
        test_prices = test_prices.flatten()
    
    backtest_engine = BacktestingEngine(initial_capital=10000, transaction_cost=0.001)
    backtest_results = backtest_engine.run_backtest(
        test_prices, test_predictions, test_confidences,
        min_confidence=60.0, position_size=0.95
    )
    
    metrics = backtest_engine.get_metrics()
    print("\n✓ Backtest Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:.<30} {value:.4f}")
        else:
            print(f"  {key:.<30} {value}")
    
    # ============ STEP 7: EVALUATE MODEL ============
    print("\n[7] Evaluating model...")
    eval_results = evaluate_model_comprehensive(model, test_loader, X_test, y_test, device=DEVICE)
    
    # ============ STEP 8: SAVE & VISUALIZE ============
    print("\n[8] Saving model and creating visualizations...")
    torch.save(model.state_dict(), 'stock_prediction_cnn.pth')
    with open('scalers.pkl', 'wb') as f:
        pickle.dump(scalers_dict, f)
    print("✓ Model saved to: stock_prediction_cnn.pth")
    print("✓ Scalers saved to: scalers.pkl")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_training_history(history, 'training_history.png')
    plot_backtest_results(backtest_engine, test_prices, test_predictions, 
                         'Backtest Results - Multi-Feature Model').savefig('backtest_results.png', 
                                                                           dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: backtest_results.png")
    
    plot_roc_curve(eval_results, 'roc_curve.png')
    plot_confusion_matrix(eval_results, 'confusion_matrix.png')
    plot_scaler_info(scalers_dict, 'scaler_info.png')
    
    # Summary report
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Model: 10-Layer 1D CNN (Multi-Feature)")
    print(f"Input Features: Open, High, Low, Close, Volume (5 channels)")
    print(f"Parameters: {param_count:,}")
    print(f"\nFeature Scaling:")
    print(f"  OHLC (Open, High, Low, Close): Shared scaler")
    print(f"    Range: [{scalers_dict['ohlc_scaler'].data_min_[0]:.2f}, {scalers_dict['ohlc_scaler'].data_max_[0]:.2f}]")
    print(f"  Volume: Independent scaler")
    print(f"    Range: [{scalers_dict['volume_scaler'].data_min_[0]:.0f}, {scalers_dict['volume_scaler'].data_max_[0]:.0f}]")
    print(f"\nTraining Results:")
    print(f"  Best Validation Accuracy: {max(history['val_acc'])*100:.2f}%")
    print(f"  Test Accuracy: {eval_results['accuracy']*100:.2f}%")
    print(f"  AUC-ROC Score: {eval_results['auc_score']:.4f}")
    print(f"\nBacktest Results:")
    print(f"  Total Return: {metrics['total_return']*100:.2f}%")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"  Win Rate: {metrics['win_rate']*100:.2f}%")
    print(f"  Total Trades: {metrics['total_trades']}")
    print(f"\nGenerated Files:")
    print(f"  ✓ stock_prediction_cnn.pth (model)")
    print(f"  ✓ scalers.pkl (OHLC + Volume scalers)")
    print(f"  ✓ training_history.png")
    print(f"  ✓ backtest_results.png")
    print(f"  ✓ roc_curve.png")
    print(f"  ✓ confusion_matrix.png")
    print(f"  ✓ scaler_info.png")
    print("="*60 + "\n")
    
    return model, scalers_dict


if __name__ == "__main__":
    model, scalers_dict = main()