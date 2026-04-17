def main() -> tuple[nn.Module, MinMaxScaler]:
    """Main pipeline - Python 3.13 and PyTorch 2.6 compatible"""
    
    # ... [previous code] ...
    
    # Backtesting
    print("\n[5] Running backtest analysis...")
    test_predictions, test_confidences, _ = predict_with_confidence(
        model, X_test.numpy(), device=DEVICE
    )
    
    # FIXED: Get and flatten prices for backtesting
    test_prices = prices[train_size + val_size + SEQ_LENGTH + 1:]
    if test_prices.ndim > 1:
        test_prices = test_prices.flatten()  # Convert (N, 1) -> (N,)
    
    backtest_engine = BacktestingEngine(initial_capital=10000, transaction_cost=0.001)
    backtest_results = backtest_engine.run_backtest(
        test_prices, test_predictions, test_confidences,
        min_confidence=60.0, position_size=0.95
    )
    
    # ... [rest of code] ...