def plot_backtest_results(
    backtest_engine: BacktestingEngine,
    prices: np.ndarray,
    predictions: np.ndarray,
    title: str = "Backtest Results"
) -> plt.Figure:
    """Plot backtest results and metrics"""
    # FIXED: Flatten prices if 2D
    if prices.ndim > 1:
        prices = prices.flatten()
    
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