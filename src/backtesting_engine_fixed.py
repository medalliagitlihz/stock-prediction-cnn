class BacktestingEngine:
    """
    Backtesting engine for evaluating trading strategy
    Python 3.13 compatible with type hints
    FIXED: Handle array dimensions correctly
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
        
        Args:
            prices: 1D array of prices (flatten if needed)
            predictions: 1D array of binary predictions
            confidences: 1D array of confidence scores
            min_confidence: Minimum confidence threshold
            position_size: Fraction of capital to use per trade
        """
        # FIXED: Ensure prices is 1D
        if prices.ndim > 1:
            prices = prices.flatten()
        if predictions.ndim > 1:
            predictions = predictions.flatten()
        if confidences.ndim > 1:
            confidences = confidences.flatten()
        
        print(f"Debug: prices shape={prices.shape}, predictions shape={predictions.shape}, confidences shape={confidences.shape}")
        
        # Initialize tracking variables
        portfolio_value = [self.initial_capital]
        trades = []
        holdings = 0.0
        cash = float(self.initial_capital)
        entry_price = None
        entry_date = None
        
        for i in range(len(predictions)):
            # FIXED: Extract scalar from 1D array
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