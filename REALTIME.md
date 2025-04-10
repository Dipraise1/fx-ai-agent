# Real-Time Trading Platform Documentation

This document provides detailed information about the real-time capabilities of the trading agent platform, including WebSocket implementation, error handling, and real-time model training.

## WebSocket Implementation

The trading agent uses WebSocket connections to receive real-time market data from Finnhub. The implementation includes several key features to ensure reliable and continuous operation:

### Authentication

- **API Key**: Required for establishing the WebSocket connection
- **Secret Key**: Used in the `X-Finnhub-Secret` header for authenticated HTTP requests
- **Connection URL**: `wss://ws.finnhub.io?token={your_api_key}`

### Message Acknowledgment

To prevent timeouts on the WebSocket connection, the system implements an acknowledgment mechanism:

```python
def on_message(self, ws, message):
    """Handle incoming WebSocket messages."""
    try:
        # Parse the message
        msg = json.loads(message)
        
        # Send acknowledgment to prevent timeouts
        if ws and ws.sock and ws.sock.connected:
            ws.send(json.dumps({"type": "ping"}))
            self.logger.debug("Sent ping acknowledgment")
        
        # Process the message based on type
        if "type" in msg:
            # Process trade data
            if msg["type"] == "trade":
                # Process trade data
            # Handle other message types
            elif msg["type"] == "ping":
                self.logger.debug("Received ping from server")
            elif msg["type"] == "error":
                self.logger.error(f"Received error: {msg}")
    except Exception as e:
        self.logger.error(f"Error processing WebSocket message: {e}")
```

### Error Handling

The WebSocket implementation includes comprehensive error handling:

- **Connection Errors**: Logged and trigger reconnection attempt
- **Message Processing Errors**: Caught and logged to prevent connection termination
- **Subscription Errors**: Retried with exponential backoff
- **Send Errors**: Handled to prevent application crashes when connection is lost

### Reconnection Logic

If the WebSocket connection is closed unexpectedly, the system will automatically attempt to reconnect:

```python
def on_close(self, ws, close_status_code, close_msg):
    """Handle WebSocket connection closure."""
    self.logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")
    self.is_connected = False
    
    # Schedule reconnection attempt
    if not self.stopping:
        self.logger.info(f"Attempting to reconnect in {self.reconnect_delay} seconds...")
        threading.Timer(self.reconnect_delay, self._setup_real_time_connection).start()
```

The reconnection logic includes:
- **Delay**: Initial 30-second delay before reconnecting
- **Exponential Backoff**: Increasing delay for subsequent reconnection attempts
- **Maximum Retries**: Configurable maximum number of reconnection attempts

### Keep-Alive Mechanism

In addition to message acknowledgments, the system implements a keep-alive mechanism:

- **Ping Interval**: Sends a ping message every 15 seconds
- **Pong Timeout**: Monitors response time and triggers reconnection if pong is not received

## Real-Time Subscriptions

### Symbol Mapping

The system maps common symbol formats to the format required by the data provider:

```python
def _map_symbol_to_provider(self, symbol):
    """Map a generic symbol to the provider-specific format."""
    # Example mapping for stocks
    if symbol.startswith('STOCK:'):
        return symbol.replace('STOCK:', '')
    # Example mapping for forex
    elif symbol.startswith('FOREX:'):
        # Convert FOREX:USD/JPY to OANDA:USD_JPY
        return f"OANDA:{symbol.replace('FOREX:', '').replace('/', '_')}"
    # No mapping needed
    return symbol
```

### Subscription Management

The system manages subscriptions to multiple symbols:

```python
def subscribe_real_time(self, symbol):
    """Subscribe to real-time updates for a symbol."""
    if not self.api_key or self.api_key == "demo":
        self.logger.warning("Real-time updates not available in demo mode")
        return False
    
    provider_symbol = self._map_symbol_to_provider(symbol)
    
    if self.ws and self.is_connected:
        try:
            self.ws.send(json.dumps({'type': 'subscribe', 'symbol': provider_symbol}))
            self.real_time_subscriptions.add(symbol)
            self.logger.info(f"Subscribed to real-time updates for {symbol}")
            return True
        except Exception as e:
            self.logger.error(f"Error subscribing to {symbol}: {e}")
            return False
    else:
        self.logger.warning(f"WebSocket not connected, can't subscribe to {symbol}")
        self.pending_subscriptions.add(symbol)
        self._setup_real_time_connection()
        return False
```

## Real-Time Model Training

The platform includes a real-time model trainer that continuously updates machine learning models with the latest market data.

### Training Cycle

The real-time model trainer operates on a configurable cycle:

1. **Data Collection**: Fetches the latest market data
2. **Feature Engineering**: Calculates technical indicators and creates feature sets
3. **Model Training**: Retrains prediction models with the new data
4. **Model Evaluation**: Evaluates model performance
5. **Model Deployment**: Updates the live model if performance improves

### Configuration

The real-time model training can be configured through environment variables:

```
# Model training settings
MODEL_TRAINING_INTERVAL=3600  # Seconds between model training cycles (1 hour)
FEATURE_WINDOW=20             # Number of historical periods for feature calculation
PREDICTION_HORIZON=5          # Number of periods ahead to predict
MIN_TRAINING_SAMPLES=100      # Minimum samples required for training
```

### Handling Missing Data

The real-time model trainer includes mechanisms for handling missing data:

- **Interpolation**: Fills small gaps in time series data
- **Imputation**: Fills missing values with appropriate statistics
- **Outlier Handling**: Detects and manages outliers to prevent model distortion

## Running the Real-Time Platform

### Environment Setup

Before running the real-time platform, ensure your environment variables are properly configured in the `.env` file:

```
FINNHUB_API_KEY=your_api_key_here
FINNHUB_SECRET=your_secret_here
REAL_TIME_MODE=true
UPDATE_INTERVAL=60
MODEL_TRAINING_INTERVAL=3600
LOG_LEVEL=INFO
```

### Starting the Platform

To start the full real-time trading platform:

```bash
./start_realtime_trading.sh
```

This script:
1. Loads environment variables from `.env`
2. Starts the real-time trading agent (WebSocket connections and signal generation)
3. Starts the real-time model trainer
4. Redirects output to log files

### Demo Mode

For testing without API credentials, use demo mode:

```bash
./start_realtime_trading.sh --demo
```

In demo mode:
- Mock data is used instead of real market data
- WebSocket connections are simulated
- Model training uses historical data snippets

### Monitoring

Monitor the real-time operations through log files:

```bash
# View trading agent logs
tail -f logs/trading_agent.log

# View model trainer logs
tail -f logs/model_trainer.log
```

### Stopping the Platform

To stop all real-time components:

```bash
./stop_realtime_trading.sh
```

## Troubleshooting

### Common Issues

1. **WebSocket Connection Errors**
   - Check API key and internet connection
   - Verify Finnhub account status and subscription tier
   - Check for rate limiting (free tier has limitations)

2. **403 Errors**
   - Verify that your API key has access to the requested data endpoints
   - Check that your secret key is correctly formatted in the `.env` file

3. **Acknowledgment Timeouts**
   - May indicate network latency issues
   - Check for firewall or proxy interference

4. **Model Training Errors**
   - Ensure sufficient historical data is available
   - Check for feature calculation errors
   - Verify minimum sample requirements are met

### Debug Mode

For more detailed logging, set `LOG_LEVEL=DEBUG` in your `.env` file:

```
LOG_LEVEL=DEBUG
```

This will provide detailed information about WebSocket connections, message handling, and model training processes.

## Integration with Trading Strategies

Real-time data and model predictions can be integrated with trading strategies:

```python
def update_strategy(self, symbol, price, timestamp):
    """Update trading strategy with real-time data."""
    # Update price history
    self.price_history[symbol].append((timestamp, price))
    
    # Get latest model prediction
    prediction = self.model_manager.get_prediction(symbol)
    
    # Update strategy state
    signal = self.strategy_manager.update(symbol, price, prediction)
    
    # Execute trade if signal generated
    if signal:
        self.execute_trade(symbol, signal)
```

Each strategy can subscribe to real-time updates and receive notifications when new data or predictions are available. 