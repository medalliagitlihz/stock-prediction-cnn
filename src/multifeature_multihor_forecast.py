"""
Multi-Horizon Forecasting Implementation
Predict stock prices at different future time horizons: T+1, T+5, T+30
Python 3.13 & PyTorch 2.6 compatible
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')


# ============================================
# CONCEPT: Multi-Horizon Forecasting
# ============================================

"""
CURRENT IMPLEMENTATION (T+1 forecast):
Window: [Day 1, Day 2, ..., Day 50]
Label:  Day 51 > Day 50?  (1-day forecast)
        └─ predict next day

PROPOSED: T+5 Forecast
Window: [Day 1, Day 2, ..., Day 50]
Label:  Day 55 > Day 50?  (5-day forecast)
        └─ predict 5 days ahead

PROPOSED: T+30 Forecast
Window: [Day 1, Day 2, ..., Day 50]
Label:  Day 80 > Day 50?  (30-day forecast)
        └─ predict 30 days ahead

KEY CHANGE:
    current_close = data[i + seq_length, 3]
    next_close = data[i + seq_length + 1, 3]      # T+1
    
    BECOMES:
    current_close = data[i + seq_length, 3]
    future_close = data[i + seq_length + horizon, 3]  # T+horizon
"""


# ============================================
# 1. MULTI-HORIZON DATA PREPARATION
# ============================================

def load_stock_data(ticker: str, days: int = 252*2) -> pd.DataFrame:
    """Load historical stock data"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    return data


def extract_features(data: pd.DataFrame) -> np.ndarray:
    """Extract OHLCV features"""
    features = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
    print(f"✓ Extracted features shape: {features.shape}")
    print(f"  Features: Open, High, Low, Close, Volume")
    return features


def normalize_data(data: np.ndarray) -> tuple[np.ndarray, dict]:
    """Normalize OHLC (shared) and Volume (independent)"""
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


def create_sequences_multihorizon(
    data: np.ndarray,
    seq_length: int = 50,
    horizon: int = 1,
    step: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create sequences with multiple forecasting horizons
    
    Args:
        data: Normalized stock data (N, 5)
        seq_length: Historical window length (e.g., 50 days)
        horizon: How many days ahead to forecast (1, 5, 30, etc.)
        step: Sliding window step size (1 = no skip, >1 = data augmentation)
    
    Returns:
        X: Input sequences (M, 5, seq_length)
        y: Binary labels for horizon prediction
    
    PARAMETERS TO CHANGE FOR DIFFERENT HORIZONS:
    
    ┌─────────────────────────────────────────────────────┐
    │ Horizon  │ horizon │ min_data_needed │ step │ Note   │
    ├──────────┼─────────┼─────────────────┼──────┼────────┤
    │ T+1      │ 1       │ seq_len + 2     │ 1    │ Default│
    │ T+5      │ 5       │ seq_len + 6     │ 1    │ Week   │
    │ T+30     │ 30      │ seq_len + 31    │ 1    │ Month  │
    │ Augment  │ 1       │ seq_len + 2     │ 5    │ Every 5│
    └─────────────────────────────────────────────────────┘
    
    The key formula:
        label = 1 if data[i + seq_length + horizon, 3] > data[i + seq_length, 3]
    """
    
    X, y = [], []
    
    # Must have enough data for: seq_length + horizon + 1
    max_idx = len(data) - seq_length - horizon
    
    print(f"Creating sequences with horizon={horizon}, step={step}")
    print(f"  Sequence length: {seq_length}")
    print(f"  Forecast horizon: {horizon} days")
    print(f"  Data points available: {len(data)}")
    print(f"  Required minimum: {seq_length + horizon + 1}")
    print(f"  Creating sequences from index 0 to {max_idx}")
    
    for i in range(0, max_idx, step):  # step controls sliding window
        # Get historical window
        window = data[i:i + seq_length]  # (seq_length, 5)
        window = window.T  # (5, seq_length)
        X.append(window)
        
        # Compare current close with future close (horizon ahead)
        current_close = data[i + seq_length, 3]      # Day seq_length
        future_close = data[i + seq_length + horizon, 3]  # Day seq_length + horizon
        
        label = 1 if future_close > current_close else 0
        y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"✓ Sequences created: X shape {X.shape}, y shape {y.shape}")
    print(f"  Total sequences: {len(X)}")
    print(f"  Data augmentation factor: {1/step if step > 1 else 1}x")
    
    return X, y


# ============================================
# 2. MULTI-HORIZON DATASET ANALYSIS
# ============================================

def analyze_horizon_feasibility(data: np.ndarray, horizons: list[int], seq_length: int = 50):
    """Analyze how much data is needed for each horizon"""
    print(f"\n{'='*70}")
    print(f"MULTI-HORIZON FEASIBILITY ANALYSIS")
    print(f"{'='*70}")
    print(f"Total data points available: {len(data)}")
    print(f"Sequence length: {seq_length}\n")
    
    for horizon in horizons:
        min_required = seq_length + horizon + 1
        sequences_possible = len(data) - seq_length - horizon
        
        feasible = "✅" if sequences_possible > 0 else "❌"
        
        print(f"{feasible} Horizon T+{horizon}:")
        print(f"   Minimum required: {min_required} days")
        print(f"   Maximum sequences: {max(0, sequences_possible)}")
        
        if sequences_possible > 0:
            print(f"   Data coverage: {(sequences_possible / len(data)) * 100:.1f}%")
        else:
            print(f"   ⚠️  NOT ENOUGH DATA! Need {min_required - len(data)} more days")
        print()


# ============================================
# 3. MODIFIED CNN FOR MULTI-HORIZON
# ============================================

class MultiHorizonCNN(nn.Module):
    """
    Multi-horizon forecasting CNN
    Single model can predict multiple horizons with different outputs
    """
    
    def __init__(
        self,
        seq_length: int = 50,
        num_filters: int = 32,
        num_features: int = 5,
        num_horizons: int = 1
    ) -> None:
        super().__init__()
        
        # Shared convolutional backbone (same for all horizons)
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
        
        # Shared fully connected layers
        self.fc1 = nn.Linear(16, 64)
        self.fc_dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        
        # HORIZON-SPECIFIC OUTPUT HEADS
        # Each horizon has its own output layer
        self.num_horizons = num_horizons
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
            for _ in range(num_horizons)
        ])
        
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass
        Returns: List of predictions, one per horizon
        """
        # Shared backbone
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
        
        # Multiple outputs for different horizons
        outputs = [head(x) for head in self.output_heads]
        return outputs


# ============================================
# 4. MULTI-HORIZON TRAINING
# ============================================

def train_multihorizon_model(
    model: MultiHorizonCNN,
    train_loaders: list[DataLoader],
    val_loaders: list[DataLoader],
    horizons: list[int],
    epochs: int = 50,
    lr: float = 0.001,
    device: str = 'cpu'
) -> dict:
    """
    Train model on multiple horizons simultaneously
    Each horizon has its own loss that contributes to total loss
    """
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': {h: [] for h in horizons},
        'val_acc': {h: [] for h in horizons}
    }
    
    best_val_acc = 0.0
    patience, patience_counter = 10, 0
    
    for epoch in range(epochs):
        # ========== TRAINING ==========
        model.train()
        train_loss_total = 0.0
        train_accs = {h: 0.0 for h in horizons}
        train_counts = {h: 0 for h in horizons}
        
        # Train on all horizons
        for horizon_idx, (train_loader, horizon) in enumerate(zip(train_loaders, horizons)):
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs[horizon_idx], labels)
                loss.backward()
                optimizer.step()
                
                train_loss_total += loss.item()
                
                predicted = (outputs[horizon_idx] > 0.5).float()
                correct = (predicted == labels).sum().item()
                train_accs[horizon] += correct
                train_counts[horizon] += labels.size(0)
        
        train_loss_avg = train_loss_total / sum(len(dl) for dl in train_loaders)
        
        # Normalize accuracies
        for h in horizons:
            if train_counts[h] > 0:
                train_accs[h] /= train_counts[h]
        
        # ========== VALIDATION ==========
        model.eval()
        val_loss_total = 0.0
        val_accs = {h: 0.0 for h in horizons}
        val_counts = {h: 0 for h in horizons}
        
        with torch.no_grad():
            for horizon_idx, (val_loader, horizon) in enumerate(zip(val_loaders, horizons)):
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device).float().unsqueeze(1)
                    outputs = model(inputs)
                    loss = criterion(outputs[horizon_idx], labels)
                    
                    val_loss_total += loss.item()
                    
                    predicted = (outputs[horizon_idx] > 0.5).float()
                    correct = (predicted == labels).sum().item()
                    val_accs[horizon] += correct
                    val_counts[horizon] += labels.size(0)
        
        val_loss_avg = val_loss_total / sum(len(dl) for dl in val_loaders)
        
        # Normalize accuracies
        for h in horizons:
            if val_counts[h] > 0:
                val_accs[h] /= val_counts[h]
        
        # Record history
        history['train_loss'].append(train_loss_avg)
        history['val_loss'].append(val_loss_avg)
        for h in horizons:
            history['train_acc'][h].append(train_accs[h])
            history['val_acc'][h].append(val_accs[h])
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            acc_str = " | ".join([f"T+{h}: {val_accs[h]:.4f}" for h in horizons])
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {val_loss_avg:.4f} | {acc_str}")
        
        # Early stopping
        avg_val_acc = np.mean(list(val_accs.values()))
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return history


# ============================================
# 5. COMPARATIVE ANALYSIS
# ============================================

def create_comparison_guide() -> str:
    """Create parameter change guide for different horizons"""
    
    guide = """
╔════════════════════════════════════════════════════════════════════════════╗
║           PARAMETER CHANGES FOR MULTI-HORIZON FORECASTING                  ║
╚════════════════════════════════════════════════════════════════════════════╝

┌─ KEY PARAMETERS TO MODIFY ─────────────────────────────────────────────────┐

1. HORIZON PARAMETER (Most Important)
   ├─ Location: create_sequences_multihorizon(horizon=X)
   ├─ T+1:  horizon = 1
   ├─ T+5:  horizon = 5
   ├─ T+30: horizon = 30
   └─ Effect: Changes which future price we compare against

2. SEQUENCE LENGTH (Sometimes)
   ├─ Current: seq_length = 50 (good for T+1, T+5)
   ├─ For T+30: Consider seq_length = 100 (longer lookback)
   ├─ Reason: Longer trends need longer history
   └─ Formula: total_data_needed = seq_length + horizon + 1

3. DATA AUGMENTATION STEP (For more training data)
   ├─ Location: create_sequences_multihorizon(step=X)
   ├─ step = 1: No augmentation (use every day)
   ├─ step = 5: Use every 5th day (5x more sequences)
   ├─ Example with step=5:
   │  └─ Window 0: [Day 0-49]
   │  └─ Window 1: [Day 5-54]  (skip 1-4)
   │  └─ Window 2: [Day 10-59] (skip 6-9)
   └─ Effect: More training data, but overlapping windows

4. BATCH SIZE (Usually unchanged)
   ├─ Current: 32 is standard
   ├─ For T+30 with less data: might increase to 64
   └─ Trade-off: Larger batch = more stable, slower training

5. NUMBER OF EPOCHS (Usually unchanged)
   ├─ Current: 50 epochs
   ├─ With data augmentation (step>1): might reduce to 30
   ├─ With less data: might increase to 100
   └─ Early stopping handles this automatically

6. LEARNING RATE (Usually unchanged)
   ├─ Current: 0.001 is standard
   ├─ Could slightly reduce for T+30: 0.0005
   └─ Reason: Smaller gradients with sparser future signals

7. MODEL ARCHITECTURE (Usually unchanged)
   ├─ Current: 10 conv layers work well
   ├─ For T+30: Could increase to 12 layers
   ├─ Reason: Longer temporal patterns need deeper network
   └─ Effect: More parameters, slower training

┌─ EXAMPLE CONFIGURATIONS ────────────────────────────────────────────────────┐

CONFIGURATION 1: T+1 (Next Day)
├─ horizon = 1
├─ seq_length = 50
├─ step = 1 (no augmentation)
├─ Min data: 52 days
├─ Expected sequences: N - 51
├─ Use case: Day trading
└─ Data: ~2 months needed

CONFIGURATION 2: T+5 (Weekly)
├─ horizon = 5
├─ seq_length = 50
├─ step = 2 (2x data augmentation)
├─ Min data: 56 days
├─ Expected sequences: (N - 55) / 2
├─ Use case: Swing trading
└─ Data: ~3 months needed

CONFIGURATION 3: T+30 (Monthly)
├─ horizon = 30
├─ seq_length = 100 (longer lookback!)
├─ step = 5 (5x data augmentation!)
├─ Min data: 131 days
├─ Expected sequences: (N - 130) / 5
├─ Use case: Position trading
└─ Data: ~6 months recommended, 1 year+ better

CONFIGURATION 4: T+60 (Bi-Monthly)
├─ horizon = 60
├─ seq_length = 150 (very long lookback)
├─ step = 10 (heavy augmentation)
├─ Min data: 211 days
├─ Expected sequences: (N - 210) / 10
├─ Use case: Long-term investing
└─ Data: 2 years recommended

┌─ CALCULATION FORMULAS ──────────────────────────────────────────────────────┐

Minimum data points needed:
    min_data = seq_length + horizon + 1

Maximum sequences you can create:
    max_sequences = floor((total_data - seq_length - horizon) / step)

Training data size:
    training_sequences = max_sequences * train_ratio
    training_sequences ≈ 0.7 * max_sequences

Data coverage (how much of history used):
    coverage = (max_sequences * step) / total_data

Example: 1000 days of data, seq=50, horizon=30, step=1
    min_data = 50 + 30 + 1 = 81 ✓ (have 1000)
    max_sequences = floor((1000 - 50 - 30) / 1) = 920
    training_sequences ≈ 0.7 * 920 = 644
    coverage = (920 * 1) / 1000 = 92%

Example: 500 days, seq=100, horizon=30, step=5
    min_data = 100 + 30 + 1 = 131 ✓ (have 500)
    max_sequences = floor((500 - 100 - 30) / 5) = 74
    training_sequences ≈ 0.7 * 74 = 51 (might be low!)
    coverage = (74 * 5) / 500 = 74%

┌─ IMPACT ANALYSIS ───────────────────────────────────────────────────────────┐

As horizon increases:
├─ ✓ Signal becomes weaker (harder to predict far future)
├─ ✓ Need more data (fewer unique windows possible)
├─ ✓ May need longer historical context (larger seq_length)
├─ ✓ Might need data augmentation (step > 1)
├─ ✓ Model may struggle to converge
└─ ✓ Accuracy typically decreases with horizon

Recommended step sizes:
├─ T+1 to T+5:   step = 1 or 2 (5-10 overlapping days)
├─ T+10 to T+20: step = 2-5 (10-25% overlap)
├─ T+30 to T+60: step = 5-10 (heavy augmentation)
└─ Note: More overlap = more correlated training samples

═══════════════════════════════════════════════════════════════════════════════
"""
    
    return guide


# ============================================
# 6. PRACTICAL EXAMPLE
# ============================================

def main_multihor izon_example():
    """Demonstrate multi-horizon forecasting"""
    
    print(create_comparison_guide())
    
    TICKER = 'JPM'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ============ LOAD DATA ============
    print("\n[1] Loading stock data...")
    raw_data = yf.download(TICKER, start="2020-01-01", end=datetime.now(), progress=False)
    features = extract_features(raw_data)
    scaled_data, scalers_dict = normalize_data(features)
    
    # ============ ANALYZE FEASIBILITY ============
    horizons = [1, 5, 30]
    analyze_horizon_feasibility(scaled_data, horizons, seq_length=50)
    
    # ============ CREATE SEQUENCES FOR EACH HORIZON ============
    sequences_by_horizon = {}
    for horizon in horizons:
        print(f"\n[2.{horizons.index(horizon)+1}] Creating sequences for T+{horizon}...")
        if horizon == 1:
            X, y = create_sequences_multihorizon(scaled_data, seq_length=50, horizon=1, step=1)
        elif horizon == 5:
            X, y = create_sequences_multihorizon(scaled_data, seq_length=50, horizon=5, step=2)
        else:  # T+30
            X, y = create_sequences_multihorizon(scaled_data, seq_length=100, horizon=30, step=5)
        
        sequences_by_horizon[horizon] = (X, y)
    
    # ============ COMPARE DATA AVAILABILITY ============
    print("\n" + "="*70)
    print("TRAINING DATA AVAILABILITY COMPARISON")
    print("="*70)
    for horizon in horizons:
        X, y = sequences_by_horizon[horizon]
        print(f"\nHorizon T+{horizon}:")
        print(f"  Total sequences: {len(X)}")
        print(f"  Label distribution: {(y==1).sum()} UP, {(y==0).sum()} DOWN")
        print(f"  Balance: {(y==1).sum()/len(y)*100:.1f}% UP")
    
    # ============ PREPARE DATALOADERS ============
    train_loaders = []
    val_loaders = []
    
    for horizon in horizons:
        X, y = sequences_by_horizon[horizon]
        
        train_size = int(len(X) * 0.7)
        val_size = int(len(X) * 0.15)
        
        X_train = torch.FloatTensor(X[:train_size])
        y_train = torch.LongTensor(y[:train_size])
        X_val = torch.FloatTensor(X[train_size:train_size+val_size])
        y_val = torch.LongTensor(y[train_size:train_size+val_size])
        
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)
        
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
    
    # ============ BUILD MULTI-HORIZON MODEL ============
    print("\n[3] Building multi-horizon model...")
    model = MultiHorizonCNN(
        seq_length=50,
        num_filters=32,
        num_features=5,
        num_horizons=len(horizons)
    )
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✓ Model parameters: {param_count:,}")
    print(f"✓ Number of horizons: {len(horizons)}")
    print(f"✓ Output heads: {len(model.output_heads)}")
    
    # ============ TRAIN ============
    print("\n[4] Training multi-horizon model...")
    history = train_multihorizon_model(
        model, train_loaders, val_loaders, horizons,
        epochs=50, lr=0.001, device=DEVICE
    )
    
    # ============ VISUALIZE RESULTS ============
    print("\n[5] Visualizing results...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Multi-Horizon Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy per horizon
    ax = axes[0, 1]
    for horizon in horizons:
        ax.plot(history['val_acc'][horizon], label=f'T+{horizon}', linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Accuracy by Horizon')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Data available per horizon
    ax = axes[1, 0]
    horizon_names = [f'T+{h}' for h in horizons]
    data_counts = [len(sequences_by_horizon[h][0]) for h in horizons]
    ax.bar(horizon_names, data_counts, color=['green', 'orange', 'red'], alpha=0.7)
    ax.set_ylabel('Number of Sequences')
    ax.set_title('Data Available per Horizon')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(data_counts):
        ax.text(i, v + 10, str(v), ha='center', fontweight='bold')
    
    # Horizon vs Accuracy trade-off
    ax = axes[1, 1]
    final_accs = [history['val_acc'][h][-1] for h in horizons]
    ax.plot(horizons, final_accs, marker='o', markersize=10, linewidth=2, color='blue')
    ax.set_xlabel('Forecast Horizon (days)')
    ax.set_ylabel('Final Validation Accuracy')
    ax.set_title('Accuracy Degradation vs Horizon')
    ax.grid(True, alpha=0.3)
    for i, (h, acc) in enumerate(zip(horizons, final_accs)):
        ax.text(h, acc + 0.01, f'{acc:.3f}', ha='center')
    
    plt.suptitle('Multi-Horizon Forecasting Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('multihor izon_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: multihor izon_analysis.png")
    
    # ============ SUMMARY ============
    print("\n" + "="*70)
    print("MULTI-HORIZON TRAINING SUMMARY")
    print("="*70)
    print(f"\nTraining Results:")
    for horizon in horizons:
        final_acc = history['val_acc'][horizon][-1]
        print(f"  T+{horizon:2d}: Accuracy = {final_acc*100:6.2f}%")
    
    print(f"\nData Efficiency:")
    for horizon in horizons:
        X, y = sequences_by_horizon[horizon]
        print(f"  T+{horizon:2d}: {len(X):5d} sequences available")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main_multihorizon_example()