#!/usr/bin/env python3
"""
Trading Agent Setup Script

This script sets up the trading agent by:
1. Installing required packages
2. Creating necessary directories
3. Setting up configuration
"""

import os
import sys
import importlib
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    required_version = (3, 7)
    current_version = sys.version_info
    
    if current_version < required_version:
        print(f"❌ Python {required_version[0]}.{required_version[1]} or higher is required")
        print(f"   Current version: {current_version[0]}.{current_version[1]}")
        return False
    
    print(f"✅ Python version: {current_version[0]}.{current_version[1]}.{current_version[2]}")
    return True

def check_required_packages():
    """Check if required packages are installed"""
    required_packages = [
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'nltk',
        'python-dotenv',
        'joblib'
    ]
    
    missing_packages = []
    
    print("\nChecking required packages:")
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is missing")
            missing_packages.append(package)
    
    if missing_packages:
        print("\n⚠️ Some required packages are missing. Install them with:")
        print(f"pip install {' '.join(missing_packages)}")
        
        # Ask to install
        answer = input("\nDo you want to install missing packages now? [y/N]: ")
        if answer.lower() == 'y':
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])
                print("✅ Packages installed successfully")
            except Exception as e:
                print(f"❌ Error installing packages: {e}")
                return False
        else:
            print("⚠️ Please install the missing packages manually")
            return False
    
    return True

def check_nltk_setup():
    """Test the NLTK fallbacks setup"""
    print("\nTesting NLTK fallbacks:")
    
    try:
        # Import our setup module
        from src.utils.nltk_setup import setup_fallbacks
        
        # Set up fallbacks
        setup_fallbacks()
        
        # Test stopwords
        import nltk
        stopwords = nltk.corpus.stopwords.words('english')
        print(f"✅ NLTK stopwords are working (found {len(stopwords)} words)")
        
        # Test tokenization
        tokens = nltk.word_tokenize("This is a test sentence.")
        print(f"✅ NLTK tokenization is working (got {len(tokens)} tokens)")
        
        # Test sentiment analysis if used
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
            sentiment = analyzer.polarity_scores("This is excellent!")
            print(f"✅ NLTK sentiment analysis is working (positive score: {sentiment['pos']:.2f})")
        except:
            print("⚠️ NLTK sentiment analysis is not set up")
        
        return True
    except Exception as e:
        print(f"❌ Error testing NLTK setup: {e}")
        print("⚠️ NLTK fallbacks may not be working correctly")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        'data',
        'logs',
        'models_data',
        'results',
        'nltk_data'
    ]
    
    print("\nCreating directories:")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created {directory}/")
    
    return True

def setup_config():
    """Set up configuration"""
    env_example = '.env.example'
    env_file = '.env'
    
    print("\nSetting up configuration:")
    
    if not os.path.exists(env_example):
        print(f"❌ {env_example} not found")
        return False
    
    if os.path.exists(env_file):
        print(f"⚠️ {env_file} already exists")
        answer = input("Do you want to overwrite it? [y/N]: ")
        if answer.lower() != 'y':
            print(f"✅ Keeping existing {env_file}")
            return True
    
    # Copy example to actual .env file
    with open(env_example, 'r') as example_file:
        content = example_file.read()
    
    with open(env_file, 'w') as env_file_obj:
        env_file_obj.write(content)
    
    print(f"✅ Created {env_file}")
    print("⚠️ Remember to edit .env with your API keys and settings")
    
    return True

def main():
    """Run setup process"""
    print("="*80)
    print(" "*25 + "TRADING AGENT SETUP")
    print("="*80)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check required packages
    if not check_required_packages():
        print("\n⚠️ Continuing setup, but some features may not work")
    
    # Create directories
    create_directories()
    
    # Check NLTK setup
    check_nltk_setup()
    
    # Set up configuration
    setup_config()
    
    print("\n" + "="*80)
    print("🎉 Setup completed!")
    print("="*80)
    print("\nYou can now run the trading agent with:")
    print("python main.py")
    print("\nOr run in real-time mode with:")
    print("python run_trading_agent_realtime.py")

if __name__ == "__main__":
    main() 