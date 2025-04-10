# Trading Agent

A machine learning-powered trading agent that analyzes market data, generates trading signals, and performs real-time market analysis with secure API authentication and reliable WebSocket connections.

## Features

- **Price Prediction Models**: Machine learning models for predicting price movements.
- **Sentiment Analysis**: NLP-based sentiment analysis of financial news and social media.
- **Market Analysis**: Technical and fundamental analysis indicators.
- **Trading Strategies**: Customizable trading strategies for different markets.
- **Real-time Monitoring**: Real-time price and news monitoring with WebSocket connections.
- **Real-time Model Training**: Continuous retraining of prediction models with latest data.
- **Secure Authentication**: API key and secret-based authentication with Finnhub.
- **WebSocket Acknowledgments**: Automatic acknowledgment of received messages to prevent timeouts.
- **Reconnection Logic**: Automatic reconnection for WebSocket connections to ensure data continuity.
- **Risk Management**: Position sizing and risk controls.

## Project Structure

```
.
├── data/                 # Market data storage
├── logs/                 # Application logs
├── models_data/          # Trained model storage
├── results/              # Analysis results
├── src/                  # Source code
│   ├── models/           # Machine learning models
│   ├── strategies/       # Trading strategies
│   ├── utils/            # Utility functions
│   ├── agent.py          # Main trading agent class
│   ├── data_fetcher.py   # Market data fetcher
│   └── trade_setup.py    # Trade setup generator
├── test_model.py         # Unit tests for models
├── test_price_prediction.py # Integrated tests for price prediction
├── main.py               # Main application entry point
├── run_trading_agent.py  # Script to run trading agent
├── run_trading_agent_realtime.py # Script to run real-time trading agent
├── realtime_model_trainer.py # Script for real-time model training
├── start_realtime_trading.sh # Script to start full real-time platform
├── stop_realtime_trading.sh # Script to stop real-time platform
├── REALTIME.md           # Documentation for real-time features
└── requirements.txt      # Project dependencies
```

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/tradingagent.git
cd tradingagent
```

2. Create and activate a virtual environment:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Set up environment variables:
```
cp .env.example .env
```
Edit `.env` with your API credentials, including your Finnhub API key and secret.

## Real-Time Features

### WebSocket Implementation
The trading agent uses WebSocket connections to receive real-time market data:
- **Automatic Acknowledgments**: Sends a ping message after each received message to prevent timeouts
- **Keep-Alive Pings**: Maintains connection with 15-second ping intervals
- **Secure Authentication**: Uses API key in WebSocket connection and X-Finnhub-Secret header for HTTP requests
- **Reconnection Logic**: Automatically attempts to reconnect after 30 seconds if the connection drops
- **Error Handling**: Comprehensive error handling for all WebSocket operations

### Real-Time Model Training
The agent continuously retrains its prediction models with the latest market data:
- **Periodic Training**: Models are retrained at configurable intervals (default: 1 hour)
- **Fresh Data**: Always uses latest market data for training
- **Multiple Timeframes**: Trains models for various timeframes (15-min, 1-hour, Daily)
- **Multiple Instruments**: Supports multiple trading instruments simultaneously

## Usage

### Running the Trading Agent

```bash
# Basic usage (one-time analysis)
python main.py

# Real-time mode with continuous updates
python main.py --realtime

# Continuous polling mode
python main.py --continuous

# Demo mode (with mock data, no API key needed)
python main.py --demo
```

### Running the Real-Time Platform

For full real-time capabilities including WebSocket connections and model training:

```bash
# Start the full real-time trading platform with your API credentials
./start_realtime_trading.sh

# Start in demo mode (with mock data)
./start_realtime_trading.sh --demo

# Stop all real-time components
./stop_realtime_trading.sh
```

### Monitoring Real-Time Operations

Monitor the real-time operation logs:

```bash
# View trading agent logs
tail -f logs/trading_agent.log

# View model trainer logs
tail -f logs/model_trainer.log
```

For more details on real-time capabilities, see [REALTIME.md](REALTIME.md).

### Running Tests

```bash
# Run model unit tests
python -m test_model

# Run integrated price prediction test
python -m test_price_prediction
```

## Machine Learning Models

### Price Prediction Model

The price prediction model uses various technical indicators and price patterns to forecast future price movements. It supports multiple algorithms:

- Random Forest Regressor
- Gradient Boosting Regressor
- Linear Regression

Features used include:

- Price returns
- Volatility measures
- Moving averages
- Price/volume ratios
- OHLC relationships

The models automatically retrain with the latest market data in real-time mode.

### Sentiment Analysis

Sentiment analysis processes financial news and social media content to gauge market sentiment, which can be used as an additional signal for trading decisions.

## Trading Strategies

The project includes several trading strategies:

1. **GBPUSDScalpingStrategy**: Short-term trading strategy for GBP/USD
2. **USDJPYEventStrategy**: Event-driven strategy for USD/JPY
3. **VIXVolatilityStrategy**: Volatility-based strategy using VIX
4. **CryptoStrategy**: Strategy for cryptocurrency markets

## Configuration

Configuration is managed through environment variables in the `.env` file:

```
# Finnhub API credentials
FINNHUB_API_KEY=your_api_key_here
FINNHUB_SECRET=your_secret_here

# Real-time settings
REAL_TIME_MODE=true
UPDATE_INTERVAL=60  # Seconds between price updates

# Model training settings
MODEL_TRAINING_INTERVAL=3600  # Seconds between model training cycles (1 hour)
```

### API Keys

The system supports various financial data APIs:
- Finnhub (with API key and secret for authentication)
- Alpha Vantage
- CoinAPI (for cryptocurrencies)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Do not use it to make financial decisions. Always consult with a financial advisor before trading.
# fx-ai-agent
