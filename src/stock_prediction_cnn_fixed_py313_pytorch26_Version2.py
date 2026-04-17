# ... [Keep all previous code until create_sequences function] ...

def create_sequences(data: np.ndarray, seq_length: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for training
    FIXED: Return shape (N, 1, seq_length) for Conv1d compatibility
    
    Args:
        data: Normalized stock price data
        seq_length: Length of each sequence window
    
    Returns:
        X: Input sequences (N, 1, seq_length) - FIXED SHAPE
        y: Binary labels (1 if next price > current, 0 otherwise)
    """
    X, y = [], []
    
    for i in range(len(data) - seq_length - 1):
        # FIXED: Keep only 1D array (price sequence)
        window = data[i:i + seq_length].flatten()  # Shape: (seq_length,)
        X.append(window)
        
        current_price = data[i + seq_length]
        next_price = data[i + seq_length + 1]
        label = 1 if next_price > current_price else 0
        y.append(label)
    
    X = np.array(X)  # Shape: (N, seq_length)
    X = X.reshape(X.shape[0], 1, X.shape[1])  # Shape: (N, 1, seq_length) - CORRECT FOR CONV1D
    y = np.array(y)
    
    return X, y


# ... [Keep train_model, predict_with_confidence, BacktestingEngine, StockPredictor classes] ...

def predict_with_confidence(
    model: nn.Module,
    X: np.ndarray,
    device: str = 'cpu'
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Make predictions with confidence scores
    FIXED: Handle correct tensor shape (batch, 1, seq_length)
    
    Returns:
        predictions: Binary predictions (0 or 1)
        confidences: Confidence scores (0-100%)
        probabilities: Raw probabilities from sigmoid
    """
    model.eval()
    
    # Ensure correct shape: (batch, 1, seq_length)
    if X.ndim == 2:
        X = X[:, np.newaxis, :]  # Add channel dimension if missing
    
    X_tensor = torch.FloatTensor(X).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)
    
    probabilities = outputs.cpu().numpy()
    # Handle PyTorch 2.6 tensor shape changes
    if probabilities.ndim > 1:
        probabilities = probabilities.flatten()
    
    predictions = (probabilities > 0.5).astype(int)
    confidences = np.abs(probabilities - 0.5) * 2 * 100
    
    return predictions, confidences, probabilities


# ... [Keep all other code the same until StockPredictor class] ...

class StockPredictor:
    """
    Complete prediction utility for making real-time predictions
    PyTorch 2.6 and Python 3.13 compatible
    FIXED: Correct tensor shapes for Conv1d
    """
    
    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        seq_length: int = 50,
        device: str = 'cpu'
    ) -> None:
        """
        Load pretrained model and scaler
        """
        self.device = device
        self.seq_length = seq_length
        
        # Load model
        self.model = StockPredictionCNN(seq_length=seq_length, num_filters=32)
        # PyTorch 2.6: Use map_location for device placement
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        print(f"✓ Model loaded from {model_path}")
        print(f"✓ Scaler loaded from {scaler_path}")
    
    def predict_next_movement(
        self,
        ticker: str,
        confidence_threshold: float = 50.0
    ) -> dict:
        """
        Predict next day's price movement for a stock
        FIXED: Correct tensor shape
        """
        # Fetch recent price data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)
        
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        except Exception as e:
            return {'error': f'Failed to fetch data: {str(e)}'}
        
        prices = data['Close'].values.reshape(-1, 1)
        scaled_prices = self.scaler.transform(prices)
        
        # Use the last seq_length prices for prediction
        if len(scaled_prices) < self.seq_length:
            return {
                'error': f'Insufficient data. Need {self.seq_length} prices, got {len(scaled_prices)}'
            }
        
        # FIXED: Shape should be (1, 1, seq_length) for Conv1d
        recent_sequence = scaled_prices[-self.seq_length:].flatten()  # Shape: (seq_length,)
        recent_sequence = recent_sequence.reshape(1, 1, -1)  # Shape: (1, 1, seq_length)
        
        # Get prediction
        prediction, confidence, probability = predict_with_confidence(
            self.model, recent_sequence, device=self.device
        )
        
        current_price = float(prices[-1][0])
        prediction_text = "UP ⬆️" if prediction[0] == 1 else "DOWN ⬇️"
        confidence_val = float(confidence[0])
        
        result = {
            'ticker': ticker,
            'current_price': current_price,
            'prediction': prediction_text,
            'probability': float(probability[0]),
            'confidence': confidence_val,
            'meets_threshold': confidence_val >= confidence_threshold,
            'timestamp': datetime.now().isoformat(),
            'recommendation': 'BUY' if prediction[0] == 1 and confidence_val >= confidence_threshold else (
                'SELL' if prediction[0] == 0 and confidence_val >= confidence_threshold else 'HOLD'
            )
        }
        
        return result
    
    def predict_with_historical_context(
        self,
        ticker: str,
        days_back: int = 20
    ) -> dict:
        """
        Predict with historical accuracy metrics
        FIXED: Correct tensor shape
        """
        # Fetch data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)
        
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        except Exception as e:
            return {'error': f'Failed to fetch data: {str(e)}'}
        
        prices = data['Close'].values.reshape(-1, 1)
        scaled_prices = self.scaler.transform(prices)
        
        # Get predictions for historical data
        if len(scaled_prices) < self.seq_length + days_back:
            return {'error': 'Insufficient historical data'}
        
        historical_predictions = []
        for i in range(len(scaled_prices) - self.seq_length - 1):
            if i >= len(scaled_prices) - self.seq_length - days_back:
                # FIXED: Shape should be (1, 1, seq_length)
                window = scaled_prices[i:i + self.seq_length].flatten()  # Shape: (seq_length,)
                window = window.reshape(1, 1, -1)  # Shape: (1, 1, seq_length)
                
                pred, conf, prob = predict_with_confidence(self.model, window, device=self.device)
                
                actual_movement = 1 if prices[i + self.seq_length + 1][0] > prices[i + self.seq_length][0] else 0
                historical_predictions.append({
                    'prediction': pred[0],
                    'actual': actual_movement,
                    'confidence': conf[0],
                    'correct': pred[0] == actual_movement
                })
        
        # Calculate historical accuracy
        if historical_predictions:
            hist_df = pd.DataFrame(historical_predictions)
            recent_accuracy = hist_df['correct'].sum() / len(hist_df)
            high_conf_mask = hist_df['confidence'] >= 60
            high_conf_accuracy = hist_df[high_conf_mask]['correct'].mean() if high_conf_mask.any() else 0.0
        else:
            recent_accuracy = 0.0
            high_conf_accuracy = 0.0
        
        # Get current prediction
        current_pred = self.predict_next_movement(ticker)
        current_pred['recent_accuracy'] = float(recent_accuracy)
        current_pred['high_confidence_accuracy'] = float(high_conf_accuracy)
        
        return current_pred


# ... [Rest of code remains the same] ...