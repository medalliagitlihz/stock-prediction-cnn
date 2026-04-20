"""
Example usage of the Stock Prediction CNN with MULTI-FEATURE data
Uses Open, High, Low, Close, Volume as inputs
Python 3.13 & PyTorch 2.6 compatible
"""

import torch
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import pickle
import pandas as pd


# Import from training module
from train_model_multifeature import (
    StockPredictionCNN, load_stock_data, extract_features,
    normalize_data, create_sequences
)


class StockPredictorMultiFeature:
    """
    Prediction utility for multi-feature stock predictions
    Uses Open, High, Low, Close, Volume as inputs
    """
    
    def __init__(
        self,
        model_path: str,
        scalers_path: str,
        seq_length: int = 50,
        device: str = 'cpu'
    ) -> None:
        """Load pretrained model and scalers"""
        self.device = device
        self.seq_length = seq_length
        
        # Load model
        self.model = StockPredictionCNN(seq_length=seq_length, num_filters=32, num_features=5)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        
        # Load scalers (list of 5 scalers)
        with open(scalers_path, 'rb') as f:
            self.scalers = pickle.load(f)
        
        print(f"✓ Model loaded from {model_path}")
        print(f"✓ Scalers loaded from {scalers_path}")
    
    def predict_next_movement(
        self,
        ticker: str,
        confidence_threshold: float = 50.0
    ) -> dict:
        """
        Predict next day's price movement for a stock
        Uses OHLCV features
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)
        
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        except Exception as e:
            return {'error': f'Failed to fetch data: {str(e)}'}
        
        # Extract and normalize features
        features = extract_features(data)  # Shape: (N, 5)
        
        # Normalize using saved scalers
        scaled_features = np.zeros_like(features, dtype=np.float32)
        for i, scaler in enumerate(self.scalers):
            scaled_features[:, i] = scaler.transform(features[:, i:i+1]).flatten()
        
        # Check sufficient data
        if len(scaled_features) < self.seq_length:
            return {
                'error': f'Insufficient data. Need {self.seq_length} samples, got {len(scaled_features)}'
            }
        
        # Create sequence - last seq_length rows
        recent_window = scaled_features[-self.seq_length:]  # Shape: (50, 5)
        recent_window = recent_window.T  # Shape: (5, 50) - (features, sequence)
        recent_sequence = recent_window.reshape(1, 5, self.seq_length)  # Shape: (1, 5, 50)
        
        # Get prediction
        X_tensor = torch.FloatTensor(recent_sequence).to(self.device)
        
        with torch.no_grad():
            output = self.model(X_tensor)
        
        probability = float(output.cpu().numpy()[0][0])
        prediction = 1 if probability > 0.5 else 0
        confidence = abs(probability - 0.5) * 2 * 100
        
        # Get current close price
        current_close = float(features[-1, 3])  # Close is at index 3
        
        prediction_text = "UP ⬆️" if prediction == 1 else "DOWN ⬇️"
        
        result = {
            'ticker': ticker,
            'current_price': current_close,
            'prediction': prediction_text,
            'probability': probability,
            'confidence': float(confidence),
            'meets_threshold': float(confidence) >= confidence_threshold,
            'timestamp': datetime.now().isoformat(),
            'recommendation': 'BUY' if prediction == 1 and confidence >= confidence_threshold else (
                'SELL' if prediction == 0 and confidence >= confidence_threshold else 'HOLD'
            )
        }
        
        return result
    
    def batch_predict(
        self,
        tickers: list[str],
        confidence_threshold: float = 50.0
    ) -> pd.DataFrame:
        """Predict for multiple stocks"""
        results = []
        
        for ticker in tickers:
            try:
                prediction = self.predict_next_movement(ticker, confidence_threshold)
                results.append(prediction)
            except Exception as e:
                print(f"⚠ Error predicting {ticker}: {str(e)}")
        
        return pd.DataFrame(results)


# ============================================
# EXAMPLES
# ============================================

def example_1_single_prediction():
    """Example 1: Make a single stock prediction"""
    print("\n" + "="*60)
    print("EXAMPLE 1: SINGLE STOCK PREDICTION (Multi-Feature)")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        predictor = StockPredictorMultiFeature(
            'stock_prediction_cnn.pth',
            'scalers.pkl',
            device=device
        )
        
        prediction = predictor.predict_next_movement('AAPL', confidence_threshold=60.0)
        
        print(f"\nStock: {prediction['ticker']}")
        print(f"Current Price: ${prediction['current_price']:.2f}")
        print(f"Prediction: {prediction['prediction']}")
        print(f"Probability: {prediction['probability']:.4f}")
        print(f"Confidence: {prediction['confidence']:.2f}%")
        print(f"Recommendation: {prediction['recommendation']}")
    except FileNotFoundError:
        print("⚠ Model files not found. Please train the model first:")
        print("  python train_model_multifeature.py")


def example_2_batch_predictions():
    """Example 2: Predict for multiple stocks"""
    print("\n" + "="*60)
    print("EXAMPLE 2: BATCH PREDICTIONS (Multi-Feature)")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        predictor = StockPredictorMultiFeature(
            'stock_prediction_cnn.pth',
            'scalers.pkl',
            device=device
        )
        
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        results = predictor.batch_predict(tickers, confidence_threshold=50.0)
        
        print("\nBatch Prediction Results:")
        print(results[['ticker', 'current_price', 'prediction', 'confidence', 'recommendation']].to_string())
    except FileNotFoundError:
        print("⚠ Model files not found. Please train the model first.")


def example_3_feature_analysis():
    """Example 3: Analyze individual features"""
    print("\n" + "="*60)
    print("EXAMPLE 3: FEATURE ANALYSIS")
    print("="*60)
    
    try:
        ticker = 'JPM'
        print(f"\nAnalyzing features for {ticker}...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)
        
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        print(f"\n✓ Data shape: {data.shape}")
        print(f"\nFeature Statistics:")
        print(f"  Open  - Mean: ${data['Open'].mean():.2f}, Range: ${data['Open'].min():.2f} - ${data['Open'].max():.2f}")
        print(f"  High  - Mean: ${data['High'].mean():.2f}, Range: ${data['High'].min():.2f} - ${data['High'].max():.2f}")
        print(f"  Low   - Mean: ${data['Low'].mean():.2f}, Range: ${data['Low'].min():.2f} - ${data['Low'].max():.2f}")
        print(f"  Close - Mean: ${data['Close'].mean():.2f}, Range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
        print(f"  Vol   - Mean: {data['Volume'].mean():.0f}, Range: {data['Volume'].min():.0f} - {data['Volume'].max():.0f}")
        
        # Show last 5 days
        print(f"\nLast 5 days of data:")
        print(data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(5).to_string())
        
    except Exception as e:
        print(f"⚠ Error: {str(e)}")


def example_4_model_comparison():
    """Example 4: Compare predictions with actual movement"""
    print("\n" + "="*60)
    print("EXAMPLE 4: MODEL ACCURACY ON RECENT DATA")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        predictor = StockPredictorMultiFeature(
            'stock_prediction_cnn.pth',
            'scalers.pkl',
            device=device
        )
        
        # Get data for last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)
        data = yf.download('JPM', start=start_date, end=end_date, progress=False)
        
        # Extract features
        from train_model_multifeature import extract_features, normalize_data
        features = extract_features(data)
        scaled_features, _ = normalize_data(features)
        
        # Test on last 30 days
        test_days = 30
        predictions = []
        actuals = []
        
        for i in range(len(scaled_features) - predictor.seq_length - test_days, 
                       len(scaled_features) - predictor.seq_length - 1):
            window = scaled_features[i:i + predictor.seq_length].T
            window = torch.FloatTensor(window.reshape(1, 5, predictor.seq_length)).to(device)
            
            with torch.no_grad():
                output = predictor.model(window)
            
            pred = 1 if output > 0.5 else 0
            predictions.append(pred)
            
            # Actual movement (close price)
            actual = 1 if features[i + predictor.seq_length + 1, 3] > features[i + predictor.seq_length, 3] else 0
            actuals.append(actual)
        
        # Calculate accuracy
        accuracy = sum([p == a for p, a in zip(predictions, actuals)]) / len(predictions)
        
        print(f"\n✓ Tested on JPM (last {test_days} days)")
        print(f"  Accuracy: {accuracy*100:.2f}%")
        print(f"  Correct predictions: {sum([p == a for p, a in zip(predictions, actuals)])}/{len(predictions)}")
        
    except FileNotFoundError:
        print("⚠ Model files not found. Please train the model first.")


def example_5_feature_importance():
    """Example 5: Visualize feature importance"""
    print("\n" + "="*60)
    print("EXAMPLE 5: FEATURE SCALING VERIFICATION")
    print("="*60)
    
    try:
        with open('scalers.pkl', 'rb') as f:
            scalers = pickle.load(f)
        
        feature_names = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        print(f"\n✓ Loaded {len(scalers)} scalers")
        print("\nFeature scaling information:")
        for i, (name, scaler) in enumerate(zip(feature_names, scalers)):
            print(f"  {i+1}. {name:8} - Min: {scaler.data_min_[0]:.2f}, Max: {scaler.data_max_[0]:.2f}")
        
    except FileNotFoundError:
        print("⚠ Scalers not found. Please train the model first.")


def run_all_examples():
    """Run all examples"""
    examples = [
        ("Example 1", example_1_single_prediction),
        ("Example 2", example_2_batch_predictions),
        ("Example 3", example_3_feature_analysis),
        ("Example 4", example_4_model_comparison),
        ("Example 5", example_5_feature_importance),
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
    print("Stock Prediction CNN - Multi-Feature Examples")
    print("Features: Open, High, Low, Close, Volume")
    print("Python 3.13 & PyTorch 2.6")
    print("="*60)
    
    run_all_examples()
    
    print("\n" + "="*60)
    print("✅ Examples completed!")
    print("="*60)