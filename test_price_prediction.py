import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import sys
from dotenv import load_dotenv

# Import NLTK setup utility for fallbacks
from src.utils.nltk_setup import setup_fallbacks

# After setting up fallbacks, import the model
from src.models.price_prediction import PricePredictionModel

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_price_prediction")

def generate_test_data(n_samples=500):
    """Generate synthetic price data for testing"""
    np.random.seed(42)
    
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='h')
    
    # Generate price data with some realistic patterns
    base_price = 1.2000  # Starting price
    price = base_price
    
    prices = []
    for i in range(n_samples):
        # Add some randomness and trend
        daily_cycle = 0.0002 * np.sin(i/24 * 2 * np.pi)
        weekly_cycle = 0.0005 * np.sin(i/168 * 2 * np.pi)
        trend = 0.00001 * i
        noise = np.random.normal(0, 0.0003)
        
        price = price + daily_cycle + weekly_cycle + trend + noise
        prices.append(price)
    
    # Create price data with typical OHLC structure
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p + np.random.uniform(0.0001, 0.0015) for p in prices],
        'low': [p - np.random.uniform(0.0001, 0.0015) for p in prices],
        'close': [p + np.random.normal(0, 0.0002) for p in prices],
        'volume': np.random.randint(100, 1000, size=n_samples)
    })
    
    return df

def test_price_prediction_model():
    """Test the price prediction model with synthetic data"""
    # Generate test data
    logger.info("Generating test data")
    df = generate_test_data(1000)
    
    # Create model instance
    logger.info("Creating model")
    model = PricePredictionModel(instrument='TEST', timeframe='1h', model_type='random_forest')
    
    # Split into training and testing data
    train_size = int(len(df) * 0.8)
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]
    
    # Train the model
    logger.info("Training model")
    mse = model.train(train_data, forecast_periods=1, test_size=0.2)
    logger.info(f"Training completed with MSE: {mse}")
    
    # Get feature importance
    feature_importance = model.get_feature_importance()
    if feature_importance:
        logger.info("Top 5 important features:")
        for name, importance in feature_importance[:5]:
            logger.info(f"{name}: {importance:.4f}")
    
    # Test prediction on the test data
    logger.info("Testing predictions")
    
    # Make predictions on each point in the test set
    # Make sure to use a window that's large enough for feature creation
    min_window_size = 50  # Ensure at least 50 data points for feature creation
    
    actual_prices = []
    predicted_prices = []
    
    # Start with a large enough window and increment by 1
    for i in range(len(test_data) - 5):
        # Create a window with enough history (include the last min_window_size points from training)
        # plus i points from test data
        window_start = max(0, train_size - min_window_size)
        window_end = train_size + i + 1
        data_slice = df.iloc[window_start:window_end]
        
        # Check if we have enough data
        if len(data_slice) < min_window_size:
            logger.warning(f"Skipping prediction at index {i}: insufficient data ({len(data_slice)} < {min_window_size})")
            continue
        
        # Make prediction
        prediction = model.predict(data_slice)
        
        # Get the actual next price
        if i + 1 < len(test_data):
            actual_next_price = test_data.iloc[i+1]['close']
            predicted_next_price = data_slice.iloc[-1]['close'] * (1 + prediction)
            
            actual_prices.append(actual_next_price)
            predicted_prices.append(predicted_next_price)
    
    # Calculate prediction accuracy
    if actual_prices and predicted_prices:
        prediction_error = np.mean(np.abs(np.array(actual_prices) - np.array(predicted_prices)))
        logger.info(f"Mean Absolute Error: {prediction_error:.6f}")
        
        # Plot actual vs predicted prices
        plt.figure(figsize=(12, 6))
        plt.plot(actual_prices, label='Actual')
        plt.plot(predicted_prices, label='Predicted')
        plt.title('Price Prediction Test')
        plt.xlabel('Time Step')
        plt.ylabel('Price')
        plt.legend()
        
        # Make sure results directory exists
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/price_prediction_test.png')
        logger.info("Plot saved to results/price_prediction_test.png")
    else:
        logger.warning("No valid predictions made, cannot calculate error or generate plot")
    
    # Test forecast
    logger.info("Testing forecast")
    forecast = model.get_forecast(df.iloc[-100:], n_future=5, return_values=True)
    logger.info(f"5-period forecast: {forecast}")
    
    return model, prediction_error if actual_prices else 0.0

if __name__ == "__main__":
    # Setup NLTK fallbacks first
    logger.info("Setting up NLTK fallbacks")
    setup_fallbacks()
    
    # Load .env file if exists
    load_dotenv()
    
    # Make sure results directory exists
    os.makedirs('results', exist_ok=True)
    
    # Run test
    try:
        model, error = test_price_prediction_model()
        logger.info(f"Test completed successfully with error: {error}")
    except Exception as e:
        logger.error(f"Error running test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    sys.exit(0) 