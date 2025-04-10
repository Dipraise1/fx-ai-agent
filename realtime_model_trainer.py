import os
import time
import argparse
import datetime
import threading
import logging
import pandas as pd
from dotenv import load_dotenv

# Import NLTK setup utility for fallbacks
from src.utils.nltk_setup import setup_fallbacks
setup_fallbacks()

# Now import the rest of the modules
from src.data_fetcher import DataFetcher
from src.models.price_prediction import PricePredictionModel
from src.utils.logger import Logger

class RealtimeModelTrainer:
    """Continuously trains and updates models with real-time market data"""
    
    def __init__(self, data_fetcher, training_interval=3600, demo_mode=False):
        """
        Initialize the real-time model trainer
        
        Args:
            data_fetcher: DataFetcher instance for retrieving market data
            training_interval: Interval in seconds between model training cycles
            demo_mode: Whether running in demo mode without real API credentials
        """
        self.data_fetcher = data_fetcher
        self.training_interval = training_interval
        self.demo_mode = demo_mode
        self.running = False
        self.training_thread = None
        self.models = {}
        self.logger = logging.getLogger('trading_agent.model_trainer')
        
        # Define the instruments and timeframes to train models for
        self.instruments = ['GBPUSD', 'USDJPY', 'EURUSD', 'ETHUSD', 'SOLUSD']
        self.timeframes = ['15', '60', 'D']  # 15min, 1hour, Daily
        self.model_types = ['random_forest', 'gbm']
        
        # Create and initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all prediction models"""
        for instrument in self.instruments:
            self.models[instrument] = {}
            for timeframe in self.timeframes:
                self.models[instrument][timeframe] = PricePredictionModel(
                    instrument=instrument,
                    timeframe=timeframe,
                    model_type='random_forest'  # Default model type
                )
        
        self.logger.info(f"Initialized {len(self.instruments) * len(self.timeframes)} prediction models")
        
    def start(self):
        """Start the real-time model training process"""
        if self.running:
            self.logger.warning("Real-time model training is already running")
            return
            
        self.running = True
        self.logger.info("Starting real-time model training")
        
        # Run initial training for all models
        self._train_all_models()
        
        # Start the training thread
        self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.training_thread.start()
        
    def stop(self):
        """Stop the real-time model training process"""
        if not self.running:
            return
            
        self.running = False
        
        # Wait for thread to stop
        if self.training_thread:
            self.training_thread.join(timeout=5)
            
        self.logger.info("Real-time model training stopped")
        
    def _training_loop(self):
        """Background thread for periodically training models"""
        while self.running:
            try:
                # Wait for next training cycle
                time.sleep(self.training_interval)
                
                # Train all models
                self._train_all_models()
                
            except Exception as e:
                self.logger.error(f"Error in training loop: {e}")
                time.sleep(60)  # Wait and retry
                
    def _train_all_models(self):
        """Train all prediction models with the latest data"""
        self.logger.info("Starting model training cycle")
        start_time = time.time()
        
        trained_count = 0
        error_count = 0
        
        # Determine the number of candles to fetch for each timeframe
        candle_counts = {
            '15': 1000,    # 15-minute candles (about 10 days)
            '60': 1000,    # 1-hour candles (about 40 days)
            'D': 365       # Daily candles (1 year)
        }
        
        for instrument in self.instruments:
            for timeframe in self.timeframes:
                try:
                    self.logger.info(f"Fetching data for {instrument} {timeframe}")
                    
                    # Fetch market data
                    df = self.data_fetcher.get_forex_data(
                        symbol=instrument,
                        resolution=timeframe,
                        count=candle_counts.get(timeframe, 100),
                        use_cache=False  # Force fresh data for training
                    )
                    
                    if df.empty:
                        self.logger.warning(f"No data available for {instrument} {timeframe}")
                        error_count += 1
                        continue
                    
                    # Log data shape for debugging
                    self.logger.info(f"Received {len(df)} data points for {instrument} {timeframe}")
                    
                    # Add technical indicators
                    df = self.data_fetcher.add_technical_indicators(df)
                    
                    # Check if indicators were successfully added
                    required_indicators = ['rsi', 'sma_20', 'momentum']
                    missing_indicators = [ind for ind in required_indicators if ind not in df.columns]
                    if missing_indicators:
                        self.logger.warning(f"Missing indicators for {instrument} {timeframe}: {missing_indicators}")
                    
                    # Train the model
                    model = self.models[instrument][timeframe]
                    
                    # Check if we have sufficient data for training
                    if len(df) < 50:  # Minimum data points needed
                        self.logger.warning(f"Insufficient data for training {instrument} {timeframe}: {len(df)} rows")
                        error_count += 1
                        continue
                    
                    self.logger.info(f"Training model for {instrument} {timeframe} with {len(df)} data points")
                    mse = model.train(df)
                    
                    self.logger.info(f"Trained {instrument} {timeframe} model, MSE: {mse:.6f}")
                    trained_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Error training {instrument} {timeframe} model: {str(e)}")
                    # Log the traceback for debugging
                    import traceback
                    self.logger.error(traceback.format_exc())
                    error_count += 1
        
        duration = time.time() - start_time
        self.logger.info(f"Training cycle completed in {duration:.1f} seconds. "
                        f"Trained: {trained_count}, Errors: {error_count}")
                    
    def get_forecast(self, instrument, timeframe='15'):
        """
        Get a forecast for a specific instrument and timeframe
        
        Args:
            instrument: Trading instrument (e.g., 'GBPUSD')
            timeframe: Timeframe for the forecast (e.g., '15', '60', 'D')
            
        Returns:
            Dictionary with forecast information
        """
        if instrument not in self.models or timeframe not in self.models[instrument]:
            return {"error": f"No model available for {instrument} {timeframe}"}
            
        model = self.models[instrument][timeframe]
        
        if not model.is_trained:
            return {"error": "Model not trained yet"}
            
        try:
            # Fetch the latest data
            df = self.data_fetcher.get_forex_data(
                symbol=instrument,
                resolution=timeframe,
                count=50,  # Need enough data for feature creation
                use_cache=False  # Get latest data
            )
            
            if df.empty:
                return {"error": "No recent data available"}
                
            # Add technical indicators
            df = self.data_fetcher.add_technical_indicators(df)
            
            # Get forecast
            forecast = model.get_forecast(df, n_future=5, return_values=True)
            return forecast
            
        except Exception as e:
            self.logger.error(f"Error getting forecast for {instrument} {timeframe}: {e}")
            return {"error": str(e)}

def run_realtime_model_trainer(demo_mode=False):
    """Run the real-time model trainer"""
    print("\n" + "="*80)
    print(" "*25 + "REAL-TIME MODEL TRAINER")
    print("="*80 + "\n")
    
    # Load environment variables
    load_dotenv()
    
    # Initialize logger
    logger = Logger()
    logger.info("Starting real-time model trainer")
    
    # Get API credentials
    api_key = "demo" if demo_mode else os.getenv("FINNHUB_API_KEY")
    api_secret = None if demo_mode else os.getenv("FINNHUB_SECRET")
    
    if not api_key and not demo_mode:
        logger.error("Finnhub API key not found. Run with --demo flag or set FINNHUB_API_KEY in .env")
        return
    
    if not api_secret and not demo_mode:
        logger.warning("Finnhub secret not found. Set FINNHUB_SECRET in .env for proper authentication")
    
    # Get settings from environment
    use_cache = os.getenv("USE_CACHE", "true").lower() == "true"
    cache_expiry = int(os.getenv("CACHE_EXPIRY", "3600"))
    training_interval = int(os.getenv("MODEL_TRAINING_INTERVAL", "3600"))  # Default 1 hour
    
    # Initialize data fetcher
    data_fetcher = DataFetcher(
        api_key=api_key, 
        api_secret=api_secret,
        use_cache=use_cache,
        cache_expiry=cache_expiry
    )
    
    # Initialize and start real-time model trainer
    trainer = RealtimeModelTrainer(
        data_fetcher=data_fetcher,
        training_interval=training_interval,
        demo_mode=demo_mode
    )
    
    try:
        # Start the trainer
        trainer.start()
        
        print("\n" + "="*80)
        print(" "*25 + "MODEL TRAINING ACTIVE")
        print("="*80)
        print(f"Training interval: {training_interval} seconds")
        print("Press Ctrl+C to stop\n")
        
        # Keep the main thread running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        # Handle clean shutdown
        print("\nStopping real-time model training...")
        trainer.stop()
        print("Model training stopped. Exiting...")

if __name__ == "__main__":
    # Setup command line arguments
    parser = argparse.ArgumentParser(description="Run real-time model trainer")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode without API credentials")
    args = parser.parse_args()
    
    run_realtime_model_trainer(demo_mode=args.demo) 