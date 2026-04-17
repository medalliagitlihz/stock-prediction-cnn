# Stock Price Movement Prediction using CNN

A PyTorch implementation of a 10-layer 1D Convolutional Neural Network for predicting stock price movements. Based on the arXiv paper 2512.21804.

## Features

✅ **Deep 1D CNN Architecture** - 10 convolutional layers with batch normalization and dropout
✅ **Confidence Scores** - Get prediction confidence (0-100%) for each prediction
✅ **Backtesting Engine** - Evaluate trading strategies with transaction costs
✅ **Prediction Utility** - Easy-to-use interface for making predictions
✅ **Historical Context** - View accuracy on recent data before trading
✅ **Batch Predictions** - Predict for multiple stocks at once
✅ **Comprehensive Metrics** - Sharpe ratio, max drawdown, win rate, profit factor

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd stock-prediction-cnn

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- numpy
- pandas
- scikit-learn
- yfinance
- matplotlib
- seaborn

## Quick Start

### 1. Train the Model

```bash
python stock_prediction_cnn_enhanced.py
```

This will:
- Download historical stock data
- Preprocess and normalize the data
- Train the 10-layer CNN model
- Evaluate on test set
- Run backtesting analysis
- Save visualizations

### 2. Make Predictions

```python
from stock_prediction_cnn_enhanced import StockPredictor

# Load pretrained model
predictor = StockPredictor('stock_prediction_cnn.pth', 'scaler.pkl')

# Single prediction
result = predictor.predict_next_movement('AAPL', confidence_threshold=60.0)
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2f}%")
print(f"Recommendation: {result['recommendation']}")
```

### 3. Batch Predictions

```python
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
results = predictor.batch_predict(tickers, confidence_threshold=50.0)
print(results)
```

### 4. Run Backtesting

```python
from stock_prediction_cnn_enhanced import BacktestingEngine
import pickle

# Load model and scaler
model.load_state_dict(torch.load('stock_prediction_cnn.pth'))
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Run backtest
backtest_engine = BacktestingEngine(initial_capital=10000, transaction_cost=0.001)
results = backtest_engine.run_backtest(prices, predictions, confidences, 
                                       min_confidence=60.0)

# Get metrics
metrics = backtest_engine.get_metrics()
print(f"Total Return: {metrics['total_return']*100:.2f}%")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
```

## Command Line Interface

```bash
# Single prediction
python predictor.py --ticker JPM --confidence 60

# Batch prediction
python predictor.py --batch --tickers "AAPL,MSFT,GOOGL" --confidence 70

# Prediction with historical context
python predictor.py --ticker JPM --context

# Custom model path
python predictor.py --ticker JPM --model custom_model.pth --scaler custom_scaler.pkl
```

## Model Architecture

The model consists of:
- **10 Convolutional Layers** with 1D filters (kernel_size=3)
- **Batch Normalization** after each conv layer for training stability
- **Dropout** (0.2-0.3) to prevent overfitting
- **Global Average Pooling** to reduce dimensionality
- **3 Dense Layers** for binary classification
- **Sigmoid Activation** for output (probability 0-1)

```
Input (seq_length, 1)
  ↓
Conv1d(1, 32) + BatchNorm + ReLU + Dropout
  ↓
Conv1d(32, 64) + BatchNorm + ReLU + Dropout
  ↓
... (6 more conv blocks)
  ↓
GlobalAvgPool
  ↓
Dense(16 → 64) + ReLU + Dropout
  ↓
Dense(64 → 32) + ReLU
  ↓
Dense(32 → 1) + Sigmoid
  ↓
Output: Prediction (0 or 1) + Confidence
```

## Features Explained

### Confidence Scores
- Range: 0-100%
- Calculated as: `|probability - 0.5| * 2 * 100`
- 50% = completely uncertain
- 100% = completely confident

### Backtesting Metrics
- **Total Return**: Percentage gain/loss
- **Sharpe Ratio**: Risk-adjusted returns (higher is better)
- **Max Drawdown**: Maximum peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gains to losses

### Trading Recommendations
- **BUY**: Prediction is UP with high confidence
- **SELL**: Prediction is DOWN with high confidence
- **HOLD**: Prediction confidence below threshold

## Data Requirements

- Minimum 100 trading days of historical data
- Uses closing prices for prediction
- Normalizes data to 0-1 range using MinMaxScaler
- Creates 50-day sequences for each prediction

## Example Usage Patterns

### Pattern 1: Conservative Trading (High Confidence)
```python
predictor = StockPredictor('stock_prediction_cnn.pth', 'scaler.pkl')
prediction = predictor.predict_next_movement('JPM', confidence_threshold=75.0)
# Only trades when very confident
```

### Pattern 2: Aggressive Trading (Lower Confidence)
```python
prediction = predictor.predict_next_movement('JPM', confidence_threshold=50.0)
# More frequent trades but higher risk
```

### Pattern 3: Risk Assessment
```python
prediction = predictor.predict_with_historical_context('JPM', days_back=20)
# Check model accuracy before trading
```

## Performance Metrics (JPMorgan Example)

Based on paper results:
- **Accuracy**: ~91%
- **Precision**: High on up/down predictions
- **Recall**: Good identification of trends

Backtesting results depend on:
- Confidence thresholds
- Transaction costs
- Position sizing
- Market conditions

## Configuration Parameters

```python
# Model parameters
SEQ_LENGTH = 50          # Historical days to consider
NUM_FILTERS = 32         # Initial conv filters

# Training parameters
EPOCHS = 50              # Training epochs
LEARNING_RATE = 0.001    # Adam optimizer learning rate
BATCH_SIZE = 32          # Batch size for training

# Backtesting parameters
INITIAL_CAPITAL = 10000  # Starting capital
TRANSACTION_COST = 0.001 # 0.1% transaction cost
MIN_CONFIDENCE = 60.0    # Minimum confidence to trade
POSITION_SIZE = 0.95     # Fraction of capital per trade
```

## Output Files

After training, the following files are generated:
- `stock_prediction_cnn.pth` - Trained model weights
- `scaler.pkl` - Data scaler for preprocessing
- `training_history.png` - Training curves
- `backtest_results.png` - Backtest analysis charts
- `roc_curve.png` - ROC curve for model evaluation
- `predictions.csv` - Batch prediction results (if applicable)

## Troubleshooting

**Q: Model not predicting correctly?**
A: Ensure you have sufficient historical data (100+ trading days) and confidence threshold is appropriate.

**Q: CUDA out of memory?**
A: Reduce batch size or use CPU by setting `DEVICE = 'cpu'`

**Q: Poor backtest results?**
A: Try adjusting confidence thresholds and position sizes. Historical accuracy varies by stock.

## Limitations

- Predictions are probabilistic, not guaranteed
- Historical performance doesn't guarantee future results
- Market conditions can change model effectiveness
- Stock-specific tuning may be needed
- Transaction costs significantly impact returns

## References

- Paper: arXiv 2512.21804 - "S&P 500 Stock's Movement Prediction using CNN"
- CNN for Time Series: [Link to research]
- PyTorch Documentation: [pytorch.org](https://pytorch.org)

## License

This project is for educational purposes. Use at your own risk.

## Contributing

Contributions are welcome! Please submit pull requests or issues on GitHub.

## Disclaimer

This is an educational project. Stock market predictions are inherently uncertain. 
**Never trade with real money based solely on this model's predictions.**
Always do your own research and consult with financial advisors.