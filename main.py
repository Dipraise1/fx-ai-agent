import time
import os
import argparse
from dotenv import load_dotenv

# Import NLTK setup utility for fallbacks (do this first)
from src.utils.nltk_setup import setup_fallbacks

# Set up NLTK fallbacks before importing other modules
setup_fallbacks()

# Now import the rest of the modules
from src.agent import TradingAgent
from src.data_fetcher import DataFetcher
from src.trade_setup import TradeSetup
from src.utils.logger import Logger

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Trading Agent")
    parser.add_argument("--realtime", action="store_true", help="Run in real-time mode")
    parser.add_argument("--continuous", action="store_true", help="Run in continuous mode")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode without API credentials")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Initialize logger
    logger = Logger()
    logger.info("Starting Trading Agent")
    
    # Use environment variables or command line args for mode selection
    real_time_mode = args.realtime or os.getenv("REAL_TIME_MODE", "false").lower() == "true"
    continuous_mode = args.continuous or os.getenv("CONTINUOUS_MODE", "false").lower() == "true"
    
    # Get API credentials
    api_key = "demo" if args.demo else os.getenv("FINNHUB_API_KEY")
    api_secret = None if args.demo else os.getenv("FINNHUB_SECRET")
    
    # Check credentials
    if not api_key and not args.demo:
        logger.error("Finnhub API key not found. Run with --demo flag or set FINNHUB_API_KEY in .env")
        return
    
    if not api_secret and not args.demo:
        logger.warning("Finnhub secret not found. Set FINNHUB_SECRET in .env for proper authentication")
    
    # Get other settings from environment
    use_cache = os.getenv("USE_CACHE", "true").lower() == "true"
    cache_expiry = int(os.getenv("CACHE_EXPIRY", "3600"))
    update_interval = int(os.getenv("UPDATE_INTERVAL", "60"))
    polling_interval = int(os.getenv("POLLING_INTERVAL_MINUTES", "15"))
    
    # Initialize data fetcher
    data_fetcher = DataFetcher(
        api_key=api_key,
        api_secret=api_secret,
        use_cache=use_cache,
        cache_expiry=cache_expiry
    )
    
    # Initialize trading agent with real-time mode if specified
    agent = TradingAgent(
        data_fetcher=data_fetcher,
        real_time_mode=real_time_mode,
        update_interval=update_interval
    )
    
    # Run trading agent once
    agent.analyze_markets()
    
    # Handle different modes
    if real_time_mode:
        # Keep the main thread running for real-time mode
        print("\n" + "="*80)
        print(" "*25 + "REAL-TIME MONITORING ACTIVE")
        print("="*80 + "\n")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            # Handle clean shutdown
            logger.info("Stopping real-time monitoring...")
            agent.stop_real_time_monitoring()
            print("\nReal-time monitoring stopped. Exiting...")
    elif continuous_mode:
        # Run in continuous polling mode
        run_continuous(agent, polling_interval)

def run_continuous(agent, interval_minutes=15):
    """Run the trading agent continuously at specified intervals"""
    print(f"\nRunning in continuous mode with {interval_minutes} minute intervals")
    print("Press Ctrl+C to stop\n")
    try:
        while True:
            agent.analyze_markets()
            print(f"\nWaiting for {interval_minutes} minutes until next analysis...")
            time.sleep(interval_minutes * 60)
    except KeyboardInterrupt:
        print("\nContinuous mode stopped. Exiting...")

if __name__ == "__main__":
    main() 