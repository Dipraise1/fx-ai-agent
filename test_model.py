import unittest
import pandas as pd
import numpy as np
import os
import shutil
import logging
import sys
from datetime import datetime, timedelta

# Import NLTK setup utility for fallbacks (import this first)
from src.utils.nltk_setup import setup_fallbacks

# Set up NLTK fallbacks before importing any other modules
setup_fallbacks()

# Now import the model
from src.models.price_prediction import PricePredictionModel

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_models")

class TestPricePredictionModel(unittest.TestCase):
    """Test cases for the PricePredictionModel class."""

    @classmethod
    def setUpClass(cls):
        """Set up test data that can be used for all tests."""
        # Create test data
        cls.test_data = cls.generate_test_data(300)
        
        # Create test directory for models
        os.makedirs('models_data', exist_ok=True)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests have completed."""
        # Remove any test model files
        test_files = [f for f in os.listdir('models_data') if f.startswith('TEST_')]
        for file in test_files:
            try:
                os.remove(os.path.join('models_data', file))
            except Exception as e:
                logger.error(f"Error removing test file {file}: {e}")
    
    @staticmethod
    def generate_test_data(n_samples=300):
        """Generate synthetic price data for testing."""
        np.random.seed(42)
        
        dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
        
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

    def test_init(self):
        """Test model initialization."""
        model = PricePredictionModel(instrument='TEST_INIT', timeframe='1h', model_type='random_forest')
        
        self.assertEqual(model.instrument, 'TEST_INIT')
        self.assertEqual(model.timeframe, '1h')
        self.assertEqual(model.model_type, 'random_forest')
        self.assertFalse(model.is_trained)
        self.assertIsNone(model.model)
    
    def test_feature_creation(self):
        """Test feature creation from price data."""
        model = PricePredictionModel(instrument='TEST_FEATURES', timeframe='1h')
        
        # Create features
        df_features = model._create_features(self.test_data.iloc[:50])
        
        # Check if features were created and NaN values were dropped
        self.assertGreater(len(df_features), 0)
        self.assertGreater(len(model.features), 0)
        self.assertFalse(df_features.isnull().any().any())
        
        # Check if all expected features exist
        expected_features = [
            'return_1', 'return_5', 'return_10', 
            'volatility_5', 'volatility_10',
            'sma_5', 'sma_10', 'sma_20',
            'sma_ratio_5_10', 'sma_ratio_5_20', 'sma_ratio_10_20',
            'close_sma_5_ratio', 'close_sma_10_ratio', 'close_sma_20_ratio',
            'volume_sma_5', 'volume_ratio',
            'hl_ratio', 'co_ratio'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, df_features.columns)
    
    def test_train(self):
        """Test model training."""
        model = PricePredictionModel(instrument='TEST_TRAIN', timeframe='1h', model_type='random_forest')
        
        # Train the model
        mse = model.train(self.test_data, forecast_periods=1, test_size=0.2)
        
        # Check training results
        self.assertTrue(model.is_trained)
        self.assertIsNotNone(model.model)
        self.assertGreater(len(model.features), 0)
        self.assertIsInstance(mse, float)
        
        # Check if model files were created
        self.assertTrue(os.path.exists(model.model_file))
        self.assertTrue(os.path.exists(model.scaler_X_file))
        self.assertTrue(os.path.exists(model.scaler_y_file))
    
    def test_predict(self):
        """Test price prediction."""
        model = PricePredictionModel(instrument='TEST_PREDICT', timeframe='1h', model_type='random_forest')
        
        # Train the model
        model.train(self.test_data, forecast_periods=1, test_size=0.2)
        
        # Test single-period prediction
        prediction = model.predict(self.test_data.iloc[-50:])
        self.assertIsInstance(prediction, float)
        
        # Test multi-period prediction
        predictions = model.predict(self.test_data.iloc[-50:], n_future=5)
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 5)
    
    def test_get_forecast(self):
        """Test forecast generation."""
        model = PricePredictionModel(instrument='TEST_FORECAST', timeframe='1h', model_type='random_forest')
        
        # Train the model
        model.train(self.test_data, forecast_periods=1, test_size=0.2)
        
        # Test forecast
        forecast = model.get_forecast(self.test_data.iloc[-50:], n_future=3)
        
        # Check forecast structure
        self.assertIsInstance(forecast, dict)
        expected_keys = [
            'instrument', 'timeframe', 'current_price', 'forecast_direction',
            'avg_predicted_change', 'max_predicted_price', 'min_predicted_price',
            'confidence', 'timestamp'
        ]
        for key in expected_keys:
            self.assertIn(key, forecast)
        
        # Test with return_values=True
        forecast_with_values = model.get_forecast(self.test_data.iloc[-50:], n_future=3, return_values=True)
        self.assertIn('forecasted_prices', forecast_with_values)
        self.assertIn('predicted_changes', forecast_with_values)
    
    def test_calculate_confidence(self):
        """Test confidence calculation."""
        model = PricePredictionModel(instrument='TEST_CONFIDENCE', timeframe='1h')
        
        # Test with all positive predictions
        confidence1 = model._calculate_confidence([0.001, 0.002, 0.003])
        self.assertIsInstance(confidence1, int)
        self.assertGreaterEqual(confidence1, 0)
        self.assertLessEqual(confidence1, 100)
        
        # Test with mixed predictions
        confidence2 = model._calculate_confidence([0.001, -0.002, 0.003])
        self.assertIsInstance(confidence2, int)
        self.assertGreaterEqual(confidence2, 0)
        self.assertLessEqual(confidence2, 100)
        
        # All positive should have higher confidence than mixed
        self.assertGreater(confidence1, confidence2)
    
    def test_feature_importance(self):
        """Test feature importance calculation."""
        model = PricePredictionModel(instrument='TEST_IMPORTANCE', timeframe='1h', model_type='random_forest')
        
        # Before training
        importance_before = model.get_feature_importance()
        self.assertIsNone(importance_before)
        
        # Train the model
        model.train(self.test_data, forecast_periods=1, test_size=0.2)
        
        # After training
        importance_after = model.get_feature_importance()
        self.assertIsNotNone(importance_after)
        self.assertIsInstance(importance_after, list)
        self.assertGreater(len(importance_after), 0)
        
        # Check structure of feature importance
        first_item = importance_after[0]
        self.assertIsInstance(first_item, tuple)
        self.assertEqual(len(first_item), 2)
        self.assertIsInstance(first_item[0], str)
        self.assertIsInstance(first_item[1], float)
    
    def test_different_model_types(self):
        """Test different model types."""
        try:
            train_data = self.test_data.iloc[:200]  # Use subset for faster testing
            
            # Test random forest
            rf_model = PricePredictionModel(instrument='TEST_RF', timeframe='1h', model_type='random_forest')
            rf_model.train(train_data)
            self.assertEqual(rf_model.model.__class__.__name__, 'RandomForestRegressor')
            
            # Test gradient boosting
            gbm_model = PricePredictionModel(instrument='TEST_GBM', timeframe='1h', model_type='gbm')
            gbm_model.train(train_data)
            self.assertEqual(gbm_model.model.__class__.__name__, 'GradientBoostingRegressor')
            
            # Test linear regression
            lr_model = PricePredictionModel(instrument='TEST_LR', timeframe='1h', model_type='linear')
            lr_model.train(train_data)
            self.assertEqual(lr_model.model.__class__.__name__, 'LinearRegression')
            
            # Test invalid model type
            with self.assertRaises(ValueError):
                invalid_model = PricePredictionModel(instrument='TEST_INVALID', timeframe='1h', model_type='invalid')
                invalid_model.train(train_data)
        except Exception as e:
            logger.error(f"Error in test_different_model_types: {e}")
            raise

if __name__ == '__main__':
    unittest.main() 