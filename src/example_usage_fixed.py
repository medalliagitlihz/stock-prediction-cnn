"""
Example usage of the Stock Prediction CNN with all features
FIXED: Correct tensor shapes for Conv1d (batch, channels, length)
"""

import torch
import numpy as np
from stock_prediction_cnn_enhanced_py313_pytorch26 import (
    StockPredictor, StockPredictionCNN, BacktestingEngine, 
    load_stock_data, normalize_data, create_sequences,
    predict_with_confidence
)
import yfinance as yf
from datetime import datetime, timedelta
import pickle


def example_1_single_prediction():
    """Example 1: Make a single stock prediction"""
    print("\n" + "="*60)
    print("EXAMPLE 1: SINGLE STOCK PREDICTION")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        predictor = StockPredictor('stock_prediction_cnn.pth', 'scaler.pkl', device=device)
        
        # Predict for Apple
        prediction = predictor.predict_next_movement('AAPL', confidence_threshold=60.0)
        
        print(f"\nStock: {prediction['ticker']}")
        print(f"Current Price: ${prediction['current_price']:.2f}")
        print(f"Prediction: {prediction['prediction']}")
        print(f"Confidence: {prediction['confidence']:.2f}%")
        print(f"Recommendation: {prediction['recommendation']}")
    except FileNotFoundError:
        print("⚠ Model files not found. Please train the model first:")
        print("  python stock_prediction_cnn_enhanced_py313_pytorch26.py")


def example_2_batch_predictions():
    """Example 2: Predict for multiple stocks"""
    print("\n" + "="*60)
    print("EXAMPLE 2: BATCH PREDICTIONS FOR MULTIPLE STOCKS")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        predictor = StockPredictor('stock_prediction_cnn.pth', 'scaler.pkl', device=device)
        
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        results = predictor.batch_predict(tickers, confidence_threshold=50.0)
        
        print("\nBatch Prediction Results:")
        print(results[['ticker', 'current_price', 'prediction', 'confidence', 'recommendation']].to_string())
    except FileNotFoundError:
        print("⚠ Model files not found. Please train the model first.")


def example_3_historical_context():
    """Example 3: Prediction with historical accuracy"""
    print("\n" + "="*60)
    print("EXAMPLE 3: PREDICTION WITH HISTORICAL CONTEXT")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        predictor = StockPredictor('stock_prediction_cnn.pth', 'scaler.pkl', device=device)
        
        prediction = predictor.predict_with_historical_context('JPM', days_back=20)
        
        print(f"\nStock: {prediction['ticker']}")
        print(f"Current Prediction: {prediction['prediction']} (Confidence: {prediction['confidence']:.2f}%)")
        print(f"Recent Accuracy (20 days): {prediction.get('recent_accuracy', 0)*100:.2f}%")
        print(f"High Confidence Accuracy: {prediction.get('high_confidence_accuracy', 0)*100:.2f}%")
    except FileNotFoundError:
        print("⚠ Model files not found. Please train the model first.")


def example_4_backtesting():
    """Example 4: Full backtesting analysis"""
    print("\n" + "="*60)
    print("EXAMPLE 4: BACKTESTING WITH CONFIDENCE SCORES")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Load model and data
        model = StockPredictionCNN(seq_length=50, num_filters=32)
        model.load_state_dict(torch.load('stock_prediction_cnn.pth', map_location=device, weights_only=True))
        model.to(device)
        
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Load test data
        print("\nLoading data...")
        raw_data = yf.download('JPM', start="2023-01-01", end=datetime.now(), progress=False)
        prices = raw_data['Close'].values.reshape(-1, 1)
        
        scaled_data, _ = normalize_data(prices)
        X, y = create_sequences(scaled_data, seq_length=50)
        
        # FIXED: Ensure X has correct shape (N, 1, seq_length)
        if X.ndim == 2:
            X = X[:, np.newaxis, :]
        print(f"✓ Data shape: {X.shape}")
        
        # Get predictions for last 100 samples
        test_data = X[-100:]
        predictions, confidences, _ = predict_with_confidence(model, test_data, device=device)
        
        # FIXED: Flatten prices for backtesting
        test_prices = prices[-(len(test_data)+1):-1].flatten()
        
        # Run backtest
        backtest_engine = BacktestingEngine(initial_capital=10000, transaction_cost=0.001)
        results = backtest_engine.run_backtest(
            test_prices, predictions, confidences,
            min_confidence=60.0, position_size=0.95
        )
        
        metrics = backtest_engine.get_metrics()
        print("\n✓ Backtest Results:")
        print(f"  Initial Capital: $10,000")
        print(f"  Final Portfolio Value: ${metrics['final_portfolio_value']:.2f}")
        print(f"  Total Return: {metrics['total_return']*100:.2f}%")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        print(f"  Total Trades: {metrics['total_trades']}")
        print(f"  Win Rate: {metrics['win_rate']*100:.2f}%")
        print(f"  Profit Factor: {metrics['profit_factor']:.4f}")
    except FileNotFoundError:
        print("⚠ Model files not found. Please train the model first.")


def example_5_confidence_based_trading():
    """Example 5: Trading strategy based on confidence thresholds"""
    print("\n" + "="*60)
    print("EXAMPLE 5: CONFIDENCE-BASED TRADING STRATEGY")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        model = StockPredictionCNN(seq_length=50, num_filters=32)
        model.load_state_dict(torch.load('stock_prediction_cnn.pth', map_location=device, weights_only=True))
        model.to(device)
        
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Load data
        raw_data = yf.download('JPM', start="2023-01-01", end=datetime.now(), progress=False)
        prices = raw_data['Close'].values.reshape(-1, 1)
        
        scaled_data, _ = normalize_data(prices)
        X, y = create_sequences(scaled_data, seq_length=50)
        
        # FIXED: Ensure X has correct shape
        if X.ndim == 2:
            X = X[:, np.newaxis, :]
        
        # Get predictions
        test_data = X[-100:]
        predictions, confidences, _ = predict_with_confidence(model, test_data, device=device)
        
        # FIXED: Flatten prices
        test_prices = prices[-(len(test_data)+1):-1].flatten()
        
        # Test different confidence thresholds
        print("\nTesting different confidence thresholds:")
        print("-" * 60)
        
        confidence_thresholds = [50, 60, 70, 80]
        results_summary = []
        
        for threshold in confidence_thresholds:
            backtest_engine = BacktestingEngine(initial_capital=10000, transaction_cost=0.001)
            results = backtest_engine.run_backtest(
                test_prices, predictions, confidences,
                min_confidence=threshold, position_size=0.95
            )
            
            metrics = backtest_engine.get_metrics()
            results_summary.append({
                'threshold': threshold,
                'trades': metrics['total_trades'],
                'win_rate': metrics['win_rate'],
                'return': metrics['total_return'],
                'sharpe': metrics['sharpe_ratio']
            })
            
            print(f"\nConfidence Threshold: {threshold}%")
            print(f"  Trades: {metrics['total_trades']} | "
                  f"Win Rate: {metrics['win_rate']*100:.1f}% | "
                  f"Return: {metrics['total_return']*100:.2f}% | "
                  f"Sharpe: {metrics['sharpe_ratio']:.4f}")
        
        # Find best threshold
        best_result = max(results_summary, key=lambda x: x['return'])
        print(f"\n✓ Best threshold: {best_result['threshold']}% (Return: {best_result['return']*100:.2f}%)")
        
    except FileNotFoundError:
        print("⚠ Model files not found. Please train the model first.")


def run_all_examples():
    """Run all examples"""
    examples = [
        ("Example 1", example_1_single_prediction),
        ("Example 2", example_2_batch_predictions),
        ("Example 3", example_3_historical_context),
        ("Example 4", example_4_backtesting),
        ("Example 5", example_5_confidence_based_trading),
    ]
    
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"❌ {name} error: {type(e).__name__}: {str(e)[:100]}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Stock Prediction CNN - Usage Examples")
    print("Python 3.13 & PyTorch 2.6")
    print("="*60)
    
    run_all_examples()
    
    print("\n" + "="*60)
    print("✅ Examples completed!")
    print("="*60)