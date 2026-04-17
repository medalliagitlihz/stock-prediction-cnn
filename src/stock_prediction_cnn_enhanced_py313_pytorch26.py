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
import warnings
import pickle
warnings.filterwarnings('ignore')

print(f"Python version: {__import__('sys').version}")
print(f"PyTorch version: {torch.__version__}")

# ============================================
# 1. DATA LOADING & PREPARATION
# ============================================

def load_stock_data(ticker: str, days: int = 252*2) -> pd.DataFrame:
    """
    Load historical stock data using yfinance
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'JPM')
        days: Number of trading days to fetch
    
    Returns:
        DataFrame with OHLCV data
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    return data


def normalize_data(data: np.ndarray) -> tuple[np.ndarray, MinMaxScaler]:
    """Normalize data using MinMaxScaler (0-1 range)
    
    Python 3.13 compatible type hints with tuple[T, U] syntax
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler


# ============================================
# 2. FEATURE ENGINEERING - SLIDING WINDOWS
# ============================================

def create_sequences(data: np.ndarray, seq_length: int = 50) -> tuple[np.ndarray, np.ndarray]:
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
        window = data[i:i + seq_length]
        X.append(window)
        
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
    Updated for PyTorch 2.6 compatibility
    """
    
    def __init__(self, seq_length: int = 50, num_filters: int = 32) -> None:
        super().__init__()  # PyTorch 2.6: Use super() without args
        
        # Conv layers with BatchNorm and Dropout
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
        """Forward pass through the network"""
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
    """Train the CNN model with early stopping
    
    PyTorch 2.6: Updated to use modern APIs
    """
    model.to(device)
    criterion = nn.BCELoss()
    # PyTorch 2.6: Use weight_decay directly, no lambda_l2 needed
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
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
        
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # PyTorch 2.6: Use torch.no_grad() context manager
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
# 5. PREDICTION WITH CONFIDENCE SCORES
# ============================================

def predict_with_confidence(
    model: nn.Module,
    X: np.ndarray,
    device: str = 'cpu'
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Make predictions with confidence scores
    
    PyTorch 2.6 compatible return type hints
    
    Returns:
        predictions: Binary predictions (0 or 1)
        confidences: Confidence scores (0-100%)
        probabilities: Raw probabilities from sigmoid
    """
    model.eval()
    
    X_tensor = torch.FloatTensor(X).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)
    
    probabilities = outputs.cpu().numpy()
    # Handle PyTorch 2.6 tensor shape changes
    if probabilities.ndim > 1:
        probabilities = probabilities.flatten()
    
    predictions = (probabilities > 0.5).astype(int)
    confidences = np.abs(probabilities - 0.5) * 2 * 100
    
    return predictions, confidences, probabilities


# ============================================
# 6. BACKTESTING ENGINE
# ============================================

class BacktestingEngine:
    """
    Backtesting engine for evaluating trading strategy
    Python 3.13 compatible with type hints
    """
    
    def __init__(self, initial_capital: float = 10000, transaction_cost: float = 0.001) -> None:
        """
        Args:
            initial_capital: Starting capital for trading
            transaction_cost: Transaction cost as percentage (e.g., 0.001 = 0.1%)
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
        """
        Run backtest on historical data
        """
        # Initialize tracking variables
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
                
                # Record trade
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
        results_df = pd.DataFrame({
            'portfolio_value': portfolio_value[1:]
        })
        results_df['daily_return'] = results_df['portfolio_value'].pct_change()
        
        # Store results
        self.results = {
            'portfolio_values': portfolio_value[1:],
            'trades': pd.DataFrame(trades) if trades else pd.DataFrame(),
            'daily_returns': results_df['daily_return'].dropna().values,
            'total_trades': len(trades)
        }
        
        return self.results
    
    def get_metrics(self) -> dict | None:
        """Calculate performance metrics
        
        Python 3.13: Use union syntax (dict | None) instead of Optional[dict]
        """
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
# 7. PREDICTION UTILITY CLASS
# ============================================

class StockPredictor:
    """
    Complete prediction utility for making real-time predictions
    PyTorch 2.6 and Python 3.13 compatible
    """
    
    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        seq_length: int = 50,
        device: str = 'cpu'
    ) -> None:
        """
        Load pretrained model and scaler
        """
        self.device = device
        self.seq_length = seq_length
        
        # Load model
        self.model = StockPredictionCNN(seq_length=seq_length, num_filters=32)
        # PyTorch 2.6: Use map_location for device placement
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        print(f"✓ Model loaded from {model_path}")
        print(f"✓ Scaler loaded from {scaler_path}")
    
    def predict_next_movement(
        self,
        ticker: str,
        confidence_threshold: float = 50.0
    ) -> dict:
        """
        Predict next day's price movement for a stock
        """
        # Fetch recent price data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)
        
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        except Exception as e:
            return {'error': f'Failed to fetch data: {str(e)}'}
        
        prices = data['Close'].values.reshape(-1, 1)
        scaled_prices = self.scaler.transform(prices)
        
        # Use the last seq_length prices for prediction
        if len(scaled_prices) < self.seq_length:
            return {
                'error': f'Insufficient data. Need {self.seq_length} prices, got {len(scaled_prices)}'
            }
        
        recent_sequence = scaled_prices[-self.seq_length:].reshape(1, self.seq_length, 1)
        
        # Get prediction
        prediction, confidence, probability = predict_with_confidence(
            self.model, recent_sequence, device=self.device
        )
        
        current_price = float(prices[-1][0])
        prediction_text = "UP ⬆️" if prediction[0] == 1 else "DOWN ⬇️"
        confidence_val = float(confidence[0])
        
        result = {
            'ticker': ticker,
            'current_price': current_price,
            'prediction': prediction_text,
            'probability': float(probability[0]),
            'confidence': confidence_val,
            'meets_threshold': confidence_val >= confidence_threshold,
            'timestamp': datetime.now().isoformat(),
            'recommendation': 'BUY' if prediction[0] == 1 and confidence_val >= confidence_threshold else (
                'SELL' if prediction[0] == 0 and confidence_val >= confidence_threshold else 'HOLD'
            )
        }
        
        return result
    
    def batch_predict(
        self,
        tickers: list[str],
        confidence_threshold: float = 50.0
    ) -> pd.DataFrame:
        """
        Predict for multiple stocks
        
        Python 3.13: Use list[str] instead of List[str]
        """
        results = []
        
        for ticker in tickers:
            try:
                prediction = self.predict_next_movement(ticker, confidence_threshold)
                results.append(prediction)
            except Exception as e:
                print(f"⚠ Error predicting {ticker}: {str(e)}")
        
        return pd.DataFrame(results)
    
    def predict_with_historical_context(
        self,
        ticker: str,
        days_back: int = 20
    ) -> dict:
        """
        Predict with historical accuracy metrics
        """
        # Fetch data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)
        
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        except Exception as e:
            return {'error': f'Failed to fetch data: {str(e)}'}
        
        prices = data['Close'].values.reshape(-1, 1)
        scaled_prices = self.scaler.transform(prices)
        
        # Get predictions for historical data
        if len(scaled_prices) < self.seq_length + days_back:
            return {'error': 'Insufficient historical data'}
        
        historical_predictions = []
        for i in range(len(scaled_prices) - self.seq_length - 1):
            if i >= len(scaled_prices) - self.seq_length - days_back:
                window = scaled_prices[i:i + self.seq_length].reshape(1, self.seq_length, 1)
                pred, conf, prob = predict_with_confidence(self.model, window, device=self.device)
                
                actual_movement = 1 if prices[i + self.seq_length + 1][0] > prices[i + self.seq_length][0] else 0
                historical_predictions.append({
                    'prediction': pred[0],
                    'actual': actual_movement,
                    'confidence': conf[0],
                    'correct': pred[0] == actual_movement
                })
        
        # Calculate historical accuracy
        if historical_predictions:
            hist_df = pd.DataFrame(historical_predictions)
            recent_accuracy = hist_df['correct'].sum() / len(hist_df)
            high_conf_mask = hist_df['confidence'] >= 60
            high_conf_accuracy = hist_df[high_conf_mask]['correct'].mean() if high_conf_mask.any() else 0.0
        else:
            recent_accuracy = 0.0
            high_conf_accuracy = 0.0
        
        # Get current prediction
        current_pred = self.predict_next_movement(ticker)
        current_pred['recent_accuracy'] = float(recent_accuracy)
        current_pred['high_confidence_accuracy'] = float(high_conf_accuracy)
        
        return current_pred


# ============================================
# 8. EVALUATION FUNCTIONS
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


def plot_backtest_results(
    backtest_engine: BacktestingEngine,
    prices: np.ndarray,
    predictions: np.ndarray,
    title: str = "Backtest Results"
) -> plt.Figure:
    """Plot backtest results and metrics"""
    metrics = backtest_engine.get_metrics()
    portfolio_values = backtest_engine.results['portfolio_values']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Portfolio value over time
    ax = axes[0, 0]
    ax.plot(portfolio_values, linewidth=2, color='blue')
    ax.axhline(y=backtest_engine.initial_capital, color='red', linestyle='--', label='Initial Capital')
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
    ax.scatter(buy_signals, prices[buy_signals], color='green', marker='^', s=100, label='Buy Signal')
    ax.scatter(sell_signals, prices[sell_signals], color='red', marker='v', s=100, label='Sell Signal')
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
    ]
    table = ax.table(cellText=metrics_data, cellLoc='center', loc='center',
                    colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax.set_title('Backtest Metrics')
    
    # Daily returns histogram
    ax = axes[1, 1]
    daily_returns = backtest_engine.results['daily_returns']
    ax.hist(daily_returns, bins=50, edgecolor='black', alpha=0.7, color='blue')
    mean_ret = float(np.mean(daily_returns))
    ax.axvline(x=mean_ret, color='red', linestyle='--', label=f'Mean: {mean_ret:.4f}')
    ax.set_xlabel('Daily Return')
    ax.set_ylabel('Frequency')
    ax.set_title('Daily Returns Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================
# 9. MAIN EXECUTION
# ============================================

def main() -> tuple[nn.Module, MinMaxScaler]:
    """Main pipeline - Python 3.13 and PyTorch 2.6 compatible"""
    
    # Configuration
    TICKER = 'JPM'
    SEQ_LENGTH = 50
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*60}")
    print(f"Stock Prediction CNN - Python 3.13 & PyTorch 2.6")
    print(f"{'='*60}")
    print(f"Device: {DEVICE}")
    print(f"Stock: {TICKER}, Sequence Length: {SEQ_LENGTH}\n")
    
    # Load data
    print("[1] Loading stock data...")
    raw_data = yf.download(TICKER, start="2022-01-01", end=datetime.now(), progress=False)
    prices = raw_data['Close'].values.reshape(-1, 1)
    print(f"✓ Data shape: {prices.shape}")
    
    scaled_data, scaler = normalize_data(prices)
    
    # Create sequences
    print("\n[2] Creating sequences...")
    X, y = create_sequences(scaled_data, seq_length=SEQ_LENGTH)
    print(f"✓ X shape: {X.shape}, y shape: {y.shape}")
    
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
    
    # DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)
    
    # Train model
    print("\n[3] Building and training model...")
    model = StockPredictionCNN(seq_length=SEQ_LENGTH, num_filters=32)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✓ Model parameters: {param_count:,}")
    
    history = train_model(model, train_loader, val_loader, 
                         epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE)
    
    # Comprehensive evaluation
    print("\n[4] Evaluating model...")
    eval_results = evaluate_model_comprehensive(model, test_loader, X_test, y_test, device=DEVICE)
    
    # Backtesting
    print("\n[5] Running backtest analysis...")
    test_predictions, test_confidences, _ = predict_with_confidence(
        model, X_test.numpy(), device=DEVICE
    )
    
    # Get actual prices for backtesting
    test_prices = prices[train_size + val_size + SEQ_LENGTH + 1:]
    
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
    
    # Save model and scaler
    print("\n[6] Saving model and scaler...")
    torch.save(model.state_dict(), 'stock_prediction_cnn.pth')
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✓ Model saved to: stock_prediction_cnn.pth")
    print("✓ Scaler saved to: scaler.pkl")
    
    # Create visualizations
    print("\n[7] Creating visualizations...")
    
    # Training history
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
    print("  ✓ Saved: training_history.png")
    
    # Backtest results
    backtest_fig = plot_backtest_results(backtest_engine, test_prices, test_predictions)
    backtest_fig.savefig('backtest_results.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: backtest_results.png")
    
    # ROC curve
    fpr, tpr, _ = roc_curve(eval_results['labels'], eval_results['probabilities'])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {eval_results['auc_score']:.4f}")
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: roc_curve.png")
    
    print("\n" + "="*60)
    print("✅ Training complete!")
    print("="*60 + "\n")
    
    return model, scaler


if __name__ == "__main__":
    model, scaler = main()