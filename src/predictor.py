"""
Stock Prediction Utility
Used for making real-time predictions with trained models
"""

import torch
import pickle
import argparse
from stock_prediction_cnn_enhanced import StockPredictor, StockPredictionCNN
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Stock Price Movement Prediction')
    parser.add_argument('--ticker', type=str, default='JPM', help='Stock ticker symbol')
    parser.add_argument('--model', type=str, default='stock_prediction_cnn.pth', help='Path to model weights')
    parser.add_argument('--scaler', type=str, default='scaler.pkl', help='Path to scaler')
    parser.add_argument('--batch', action='store_true', help='Batch predict multiple stocks')
    parser.add_argument('--tickers', type=str, help='Comma-separated list of tickers for batch prediction')
    parser.add_argument('--confidence', type=float, default=50.0, help='Minimum confidence threshold')
    parser.add_argument('--context', action='store_true', help='Show historical accuracy context')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Initialize predictor
    print("Loading model and scaler...")
    predictor = StockPredictor(args.model, args.scaler, seq_length=50, device=device)
    print("✓ Model loaded\n")
    
    if args.batch and args.tickers:
        # Batch prediction
        tickers = args.tickers.split(',')
        print(f"Predicting for {len(tickers)} stocks...\n")
        
        results = predictor.batch_predict(tickers, confidence_threshold=args.confidence)
        print(results.to_string())
        
        # Save to CSV
        results.to_csv('predictions.csv', index=False)
        print(f"\n✓ Results saved to predictions.csv")
    
    else:
        # Single prediction
        print(f"Predicting for {args.ticker}...\n")
        
        if args.context:
            prediction = predictor.predict_with_historical_context(args.ticker, days_back=20)
        else:
            prediction = predictor.predict_next_movement(args.ticker, args.confidence)
        
        # Display results
        print("="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        for key, value in prediction.items():
            if isinstance(value, float):
                if key == 'probability':
                    print(f"{key:.<30} {value:.4f}")
                elif key == 'confidence':
                    print(f"{key:.<30} {value:.2f}%")
                elif key == 'recent_accuracy' or key == 'high_confidence_accuracy':
                    print(f"{key:.<30} {value:.2f}%")
                else:
                    print(f"{key:.<30} {value:.4f}")
            else:
                print(f"{key:.<30} {value}")
        print("="*60)


if __name__ == "__main__":
    main()