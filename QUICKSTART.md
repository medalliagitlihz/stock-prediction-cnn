# Quick Start Guide

## 5-Minute Setup

### Step 1: Install
```bash
pip install -r requirements.txt
```

### Step 2: Train Model (First Time Only)
```bash
python stock_prediction_cnn_enhanced.py
```

This generates:
- `stock_prediction_cnn.pth` (model)
- `scaler.pkl` (data normalizer)
- Visualization charts

### Step 3: Make Predictions

**Option A: Python**
```python
from stock_prediction_cnn_enhanced import StockPredictor

predictor = StockPredictor('stock_prediction_cnn.pth', 'scaler.pkl')
result = predictor.predict_next_movement('AAPL')

print(f"Stock: {result['ticker']}")
print(f"Price: ${result['current_price']:.2f}")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2f}%")
print(f"Action: {result['recommendation']}")
```

**Option B: Command Line**
```bash
python predictor.py --ticker AAPL --confidence 60
python predictor.py --batch --tickers "AAPL,MSFT,GOOGL"
```

## Common Tasks

### Predict Multiple Stocks
```python
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM']
results = predictor.batch_predict(tickers)
results.to_csv('predictions.csv')
```

### Check Model Accuracy on Recent Data
```python
prediction = predictor.predict_with_historical_context('JPM', days_back=20)
print(f"Recent Accuracy: {prediction['recent_accuracy']*100:.2f}%")
```

### Backtest a Strategy
```python
from stock_prediction_cnn_enhanced import BacktestingEngine

backtest = BacktestingEngine(initial_capital=10000)
results = backtest.run_backtest(prices, predictions, confidences, 
                                min_confidence=70.0)
metrics = backtest.get_metrics()

print(f"Return: {metrics['total_return']*100:.2f}%")
print(f"Sharpe: {metrics['sharpe_ratio']:.4f}")
```

### Try Different Confidence Thresholds
```python
for threshold in [50, 60, 70, 80, 90]:
    backtest = BacktestingEngine(initial_capital=10000)
    results = backtest.run_backtest(prices, predictions, confidences, 
                                    min_confidence=threshold)
    metrics = backtest.get_metrics()
    print(f"Threshold {threshold}%: {metrics['total_return']*100:.2f}% return")
```

## Understanding Outputs

### Prediction Result
```
{
  'ticker': 'AAPL',
  'current_price': 150.25,
  'prediction': 'UP ⬆️',          # Direction: UP or DOWN
  'probability': 0.7234,          # Raw probability (0-1)
  'confidence': 44.68,            # Confidence in prediction (0-100%)
  'meets_threshold': True,        # Whether confidence >= threshold
  'recommendation': 'BUY',        # Action: BUY, SELL, or HOLD
}
```

### Backtest Metrics
```
total_return: 0.2342              # 23.42% total return
sharpe_ratio: 1.2345              # Risk-adjusted returns (>1.0 is good)
max_drawdown: -0.1523              # Worst peak-to-trough (-15.23%)
total_trades: 12                   # Number of trades executed
win_rate: 0.6667                   # 66.67% of trades were profitable
profit_factor: 2.1234              # Gains to losses ratio (>1.0 is good)
```

## Tips & Tricks

1. **Confidence Thresholds**
   - High threshold (80+): Fewer trades, higher win rate
   - Low threshold (50-60): More trades, lower win rate
   - Sweet spot: Usually 60-75%

2. **Position Sizing**
   - Conservative: 50% of capital per trade
   - Moderate: 75% of capital per trade
   - Aggressive: 95% of capital per trade

3. **Stock Selection**
   - Model works best on liquid stocks (high trading volume)
   - Newer stocks may not have enough historical data
   - Well-established companies (JPM, AAPL) generally perform better

4. **Model Retraining**
   - Retrain monthly for best results
   - More data = better predictions
   - Stock-specific models can outperform general models

## Examples

See `example_usage.py` for 5 complete examples:
1. Single stock prediction
2. Batch predictions
3. Historical context
4. Backtesting
5. Confidence threshold analysis

Run with:
```bash
python example_usage.py
```

## Frequently Asked Questions

**Q: What if confidence is very low?**
A: Skip the trade. Wait for a higher confidence signal.

**Q: Can I use this for real trading?**
A: Use for paper trading first. Never risk real money without thoroughly testing.

**Q: How often should I retrain?**
A: Monthly retraining is recommended as market conditions change.

**Q: Which stocks work best?**
A: Large-cap, liquid stocks (JPM, AAPL, MSFT, GOOGL) generally perform better.

**Q: What's a good Sharpe ratio?**
A: >1.0 is acceptable, >2.0 is very good, >3.0 is excellent.

## Next Steps

1. Run training: `python stock_prediction_cnn_enhanced.py`
2. Make predictions: `python predictor.py --ticker AAPL`
3. Backtest strategy: See `example_usage.py`
4. Deploy to production (optional)

Enjoy! 🚀