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
    FIXED: Handle 2D price array correctly
    """
    # FIXED: Flatten prices if it's 2D (N, 1) -> (N,)
    if prices.ndim > 1:
        prices = prices.flatten()
    
    # Initialize tracking variables
    portfolio_value = [self.initial_capital]
    trades = []
    holdings = 0.0
    cash = float(self.initial_capital)
    entry_price = None
    entry_date = None
    
    for i in range(len(predictions)):
        # FIXED: Now prices[i] is a scalar
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