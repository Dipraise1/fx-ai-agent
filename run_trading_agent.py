import os
import time
from dotenv import load_dotenv

# Import NLTK setup utility for fallbacks (do this first)
from src.utils.nltk_setup import setup_fallbacks

# Set up NLTK fallbacks before importing other modules
setup_fallbacks()

# Now import the rest of the modules
from src.agent import TradingAgent
from src.data_fetcher import DataFetcher
from src.utils.logger import Logger

def run_agent():
    """Run the complete trading agent with all strategies"""
    print("\n" + "="*80)
    print(" "*30 + "TRADING AGENT")
    print("="*80 + "\n")
    
    # Load environment variables
    load_dotenv()
    
    # Initialize logger
    logger = Logger()
    logger.info("Starting Trading Agent")
    
    # Use API key from environment variables
    api_key = os.getenv("FINNHUB_API_KEY", "demo")
    print(f"Using API key: {'demo' if api_key == 'demo' else '****' + api_key[-4:]}")
    
    # Initialize data fetcher
    data_fetcher = DataFetcher(api_key=api_key)
    
    # Initialize trading agent
    agent = TradingAgent(data_fetcher)
    
    # Check if continuous mode is enabled
    continuous_mode = os.getenv("CONTINUOUS_MODE", "false").lower() == "true"
    
    if continuous_mode:
        # Get polling interval from environment variables
        polling_interval = int(os.getenv("POLLING_INTERVAL_MINUTES", "15"))
        print(f"Running in continuous mode with {polling_interval} minute intervals")
        
        # Run trading agent continuously
        while True:
            # Run analysis
            agent.analyze_markets()
            
            # Wait for next interval
            print(f"\nWaiting for {polling_interval} minutes until next analysis...")
            time.sleep(polling_interval * 60)
    else:
        # Run trading agent once
        print("Running in single execution mode")
        agent.analyze_markets()
        
    print("\nTrading agent execution complete!\n")

if __name__ == "__main__":
    run_agent() 