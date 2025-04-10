import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import datetime
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

class PricePredictionModel:
    """Machine learning model for financial price prediction.
    
    This model creates features from price data, trains a machine learning model,
    and provides price predictions for future periods.
    """
    
    def __init__(self, instrument: str, timeframe: str = '1h', model_type: str = 'random_forest'):
        """
        Initialize the price prediction model
        
        Args:
            instrument: Trading instrument symbol (e.g., 'GBPUSD')
            timeframe: Timeframe for analysis (e.g., '1h', '4h', '1d')
            model_type: Type of model to use ('random_forest', 'gbm', 'linear')
        """
        self.instrument = instrument
        self.timeframe = timeframe
        self.model_type = model_type
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_trained = False
        self.features: List[str] = []
        
        # Create models directory if it doesn't exist
        os.makedirs('models_data', exist_ok=True)
        
        # Model file path
        self.model_file = f"models_data/{instrument}_{timeframe}_{model_type}_model.joblib"
        self.scaler_X_file = f"models_data/{instrument}_{timeframe}_scaler_X.joblib"
        self.scaler_y_file = f"models_data/{instrument}_{timeframe}_scaler_y.joblib"
        
        # Logger
        self.logger = logging.getLogger('trading_agent.price_prediction')
        
        # Try to load pre-trained model
        self._load_model()
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from price data
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added features
        """
        # Check for empty DataFrame
        if df.empty:
            self.logger.warning("Empty DataFrame provided to _create_features")
            # Return empty DataFrame with expected columns
            empty_df = pd.DataFrame(columns=['close'] + self.features)
            return empty_df
            
        # Check for minimum required data
        if len(df) < 20:  # Need at least 20 rows for all features
            self.logger.warning(f"Insufficient data for feature creation: {len(df)} rows provided, 20 required")
            # Return empty DataFrame with expected columns
            empty_df = pd.DataFrame(columns=['close'] + self.features)
            return empty_df
            
        # Create a copy to avoid modifying the original
        df_features = df.copy()
        
        # Handle missing required columns
        required_columns = ['close']
        if 'volume' in df_features.columns:
            required_columns.append('volume')
            
        for col in required_columns:
            if col not in df_features.columns:
                self.logger.error(f"Required column '{col}' not found in DataFrame")
                # Return empty DataFrame with expected columns
                empty_df = pd.DataFrame(columns=['close'] + self.features)
                return empty_df
        
        try:
            # Price-based features
            df_features['return_1'] = df_features['close'].pct_change(1)
            df_features['return_5'] = df_features['close'].pct_change(5)
            df_features['return_10'] = df_features['close'].pct_change(10)
            
            # Volatility features
            df_features['volatility_5'] = df_features['return_1'].rolling(window=5).std()
            df_features['volatility_10'] = df_features['return_1'].rolling(window=10).std()
            
            # Moving averages
            df_features['sma_5'] = df_features['close'].rolling(window=5).mean()
            df_features['sma_10'] = df_features['close'].rolling(window=10).mean()
            df_features['sma_20'] = df_features['close'].rolling(window=20).mean()
            
            # Moving average ratios
            df_features['sma_ratio_5_10'] = df_features['sma_5'] / df_features['sma_10']
            df_features['sma_ratio_5_20'] = df_features['sma_5'] / df_features['sma_20']
            df_features['sma_ratio_10_20'] = df_features['sma_10'] / df_features['sma_20']
            
            # Price relative to moving averages
            df_features['close_sma_5_ratio'] = df_features['close'] / df_features['sma_5']
            df_features['close_sma_10_ratio'] = df_features['close'] / df_features['sma_10']
            df_features['close_sma_20_ratio'] = df_features['close'] / df_features['sma_20']
            
            # Volume features
            if 'volume' in df_features.columns:
                df_features['volume_sma_5'] = df_features['volume'].rolling(window=5).mean()
                df_features['volume_ratio'] = df_features['volume'] / df_features['volume_sma_5']
            
            # OHLC relationships
            if all(col in df_features.columns for col in ['open', 'high', 'low']):
                df_features['hl_ratio'] = (df_features['high'] - df_features['low']) / df_features['close']
                df_features['co_ratio'] = (df_features['close'] - df_features['open']) / df_features['open']
            
            # Drop rows with NaN values
            df_features = df_features.dropna()
            
            # If we have no features defined yet, store them now
            if not self.features:
                self.features = [col for col in df_features.columns 
                                if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
            
            return df_features
            
        except Exception as e:
            self.logger.error(f"Error creating features: {e}")
            # Return empty DataFrame with expected columns
            empty_df = pd.DataFrame(columns=['close'] + self.features)
            return empty_df
    
    def train(self, df: pd.DataFrame, forecast_periods: int = 1, test_size: float = 0.2) -> float:
        """
        Train the price prediction model
        
        Args:
            df: DataFrame with OHLCV data
            forecast_periods: Number of periods to forecast
            test_size: Proportion of data to use for testing
            
        Returns:
            Mean squared error on test data
        """
        # Check for empty DataFrame
        if df.empty:
            self.logger.error("Empty DataFrame provided for training")
            return 0.0
            
        # Create features
        df_features = self._create_features(df)
        
        # Check if feature creation succeeded
        if df_features.empty:
            self.logger.error("Feature creation failed, cannot train model")
            return 0.0
        
        # Create target variable (future price change)
        df_features['target'] = df_features['close'].shift(-forecast_periods) / df_features['close'] - 1
        
        # Drop rows with NaN in target
        df_features = df_features.dropna()
        
        # Ensure we have enough data to train
        if len(df_features) < 30:
            self.logger.error(f"Insufficient data for training after preprocessing: {len(df_features)} rows")
            return 0.0
        
        # Split into train and test sets
        split_idx = int(len(df_features) * (1 - test_size))
        train_data = df_features.iloc[:split_idx]
        test_data = df_features.iloc[split_idx:]
        
        # Prepare feature matrix and target vector
        X_train = train_data[self.features].values
        y_train = train_data['target'].values
        X_test = test_data[self.features].values
        y_test = test_data['target'].values
        
        # Scale features and target
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        X_test_scaled = self.scaler_X.transform(X_test)
        y_test_scaled = self.scaler_y.transform(y_test.reshape(-1, 1)).ravel()
        
        # Initialize the model based on model_type
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.model_type == 'gbm':
            self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif self.model_type == 'linear':
            self.model = LinearRegression()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Train the model
        self.model.fit(X_train_scaled, y_train_scaled)
        
        # Make predictions on the test set
        y_pred_scaled = self.model.predict(X_test_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.logger.info(f"Model trained: {self.instrument} {self.timeframe} {self.model_type}")
        self.logger.info(f"Test MSE: {mse:.6f}, R2: {r2:.4f}")
        
        # Save the model and scalers
        self._save_model()
        
        self.is_trained = True
        
        return mse
    
    def predict(self, df: pd.DataFrame, n_future: int = 1) -> Union[float, List[float]]:
        """
        Make price predictions
        
        Args:
            df: DataFrame with OHLCV data
            n_future: Number of future periods to predict
            
        Returns:
            Predicted price changes for the next n_future periods
        """
        if not self.is_trained:
            self.logger.warning("Model not trained. Please train the model first.")
            return 0.0 if n_future == 1 else [0.0] * n_future
        
        # Check for empty DataFrame
        if df.empty:
            self.logger.warning("Empty DataFrame provided for prediction")
            return 0.0 if n_future == 1 else [0.0] * n_future
        
        # Create features
        df_features = self._create_features(df)
        
        # Check if feature creation succeeded and we have at least one row
        if df_features.empty:
            self.logger.warning("Feature creation failed, cannot make prediction")
            return 0.0 if n_future == 1 else [0.0] * n_future
        
        try:
            # Get the latest data point
            latest_data = df_features.iloc[-1:]
            
            # Prepare feature matrix
            X = latest_data[self.features].values
            
            # Check for empty feature matrix
            if X.shape[0] == 0 or X.shape[1] != len(self.features):
                self.logger.warning(f"Invalid feature matrix shape: {X.shape}, expected (1, {len(self.features)})")
                return 0.0 if n_future == 1 else [0.0] * n_future
                
            # Scale features
            X_scaled = self.scaler_X.transform(X)
            
            # Make prediction
            prediction_scaled = self.model.predict(X_scaled)
            
            # Inverse transform prediction
            prediction = self.scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1)).ravel()
            
            # If predicting multiple periods
            if n_future > 1:
                predictions = [prediction[0]]
                current_price = latest_data['close'].values[0]
                
                for i in range(1, n_future):
                    # Update price based on prediction
                    next_price = current_price * (1 + predictions[-1])
                    
                    # Create synthetic next data point
                    next_data = latest_data.copy()
                    next_data['close'] = next_price
                    
                    # Update features for the synthetic data point
                    next_features = self._create_features(next_data)
                    
                    if next_features.empty:
                        self.logger.warning("Failed to create features for next prediction step")
                        predictions.extend([predictions[-1]] * (n_future - i))
                        break
                        
                    # Make prediction for the next period
                    next_X = next_features[self.features].values
                    
                    # Check for valid feature matrix
                    if next_X.shape[0] == 0 or next_X.shape[1] != len(self.features):
                        self.logger.warning("Invalid feature matrix for multi-step prediction")
                        predictions.extend([predictions[-1]] * (n_future - i))
                        break
                        
                    next_X_scaled = self.scaler_X.transform(next_X)
                    next_prediction_scaled = self.model.predict(next_X_scaled)
                    next_prediction = self.scaler_y.inverse_transform(
                        next_prediction_scaled.reshape(-1, 1)
                    ).ravel()[0]
                    
                    predictions.append(next_prediction)
                    current_price = next_price
                    
                return predictions
            
            return prediction[0]
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {e}")
            return 0.0 if n_future == 1 else [0.0] * n_future
    
    def get_forecast(self, df: pd.DataFrame, n_future: int = 5, return_values: bool = False) -> Dict[str, Any]:
        """
        Get price forecast for the future periods
        
        Args:
            df: DataFrame with OHLCV data
            n_future: Number of future periods to forecast
            return_values: If True, return actual forecasted prices
            
        Returns:
            Dictionary with forecast information
        """
        if not self.is_trained:
            self.logger.warning("Model not trained. Please train the model first.")
            return {"error": "Model not trained"}
        
        # Check for empty DataFrame
        if df.empty:
            self.logger.warning("Empty DataFrame provided for forecast")
            return {"error": "Empty data provided"}
        
        try:
            # Get the current price
            current_price = df['close'].iloc[-1]
            
            # Make predictions
            predicted_changes = self.predict(df, n_future=n_future)
            if not isinstance(predicted_changes, list):
                predicted_changes = [predicted_changes]
            
            # Calculate forecasted prices
            forecasted_prices = [current_price]
            for change in predicted_changes:
                next_price = forecasted_prices[-1] * (1 + change)
                forecasted_prices.append(next_price)
            
            # Remove the current price
            forecasted_prices = forecasted_prices[1:]
            
            # Calculate forecast statistics
            avg_change = sum(predicted_changes) / len(predicted_changes)
            max_price = max(forecasted_prices)
            min_price = min(forecasted_prices)
            
            # Determine forecast direction
            if avg_change > 0.0005:  # More than 5 pips for FX
                forecast_direction = "bullish"
            elif avg_change < -0.0005:
                forecast_direction = "bearish"
            else:
                forecast_direction = "neutral"
                
            # Create forecast dictionary
            forecast = {
                "instrument": self.instrument,
                "timeframe": self.timeframe,
                "current_price": current_price,
                "forecast_direction": forecast_direction,
                "avg_predicted_change": avg_change,
                "max_predicted_price": max_price,
                "min_predicted_price": min_price,
                "confidence": self._calculate_confidence(predicted_changes),
                "timestamp": datetime.datetime.now().isoformat(),
            }
            
            if return_values:
                forecast["forecasted_prices"] = forecasted_prices
                forecast["predicted_changes"] = predicted_changes
            
            return forecast
            
        except Exception as e:
            self.logger.error(f"Error generating forecast: {e}")
            return {"error": f"Forecast generation failed: {str(e)}"}
    
    def _calculate_confidence(self, predictions: List[float]) -> int:
        """
        Calculate confidence level based on prediction consistency
        
        Args:
            predictions: List of predicted changes
            
        Returns:
            Confidence level (0-100)
        """
        # Check for empty predictions
        if not predictions:
            return 0
            
        try:
            # Determine if all predictions are in the same direction
            all_positive = all(p > 0 for p in predictions)
            all_negative = all(p < 0 for p in predictions)
            
            # Calculate absolute average prediction
            abs_avg = sum(abs(p) for p in predictions) / len(predictions)
            
            # Scale the absolute average to get a base confidence
            base_confidence = min(60, abs_avg * 10000)  # Scale for typical FX movements
            
            # Add consistency bonus
            if all_positive or all_negative:
                consistency_bonus = 15
            else:
                consistency_bonus = 0
                
            # Calculate variance in predictions
            variance = np.var(predictions)
            
            # Scale variance to get a consistency penalty
            variance_penalty = min(30, variance * 100000)
            
            # Final confidence
            confidence = base_confidence + consistency_bonus - variance_penalty
            
            # Ensure confidence is in the range [0, 100]
            return max(0, min(100, int(confidence)))
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0
    
    def _save_model(self) -> None:
        """Save model and scalers to disk"""
        try:
            if self.model is not None:
                joblib.dump(self.model, self.model_file)
                joblib.dump(self.scaler_X, self.scaler_X_file)
                joblib.dump(self.scaler_y, self.scaler_y_file)
                self.logger.info(f"Model saved to {self.model_file}")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            
    def _load_model(self) -> bool:
        """
        Load model and scalers from disk if they exist
        
        Returns:
            True if model was loaded successfully, False otherwise
        """
        try:
            if os.path.exists(self.model_file):
                self.model = joblib.load(self.model_file)
                self.scaler_X = joblib.load(self.scaler_X_file)
                self.scaler_y = joblib.load(self.scaler_y_file)
                self.is_trained = True
                self.logger.info(f"Model loaded from {self.model_file}")
                return True
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
        return False
        
    def get_feature_importance(self) -> Optional[List[Tuple[str, float]]]:
        """
        Get feature importance if the model supports it
        
        Returns:
            List of (feature, importance) tuples sorted by importance, or None if not supported
        """
        if not self.is_trained:
            self.logger.warning("Model not trained. Please train the model first.")
            return None
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                feature_importance = list(zip(self.features, importances))
                return sorted(feature_importance, key=lambda x: x[1], reverse=True)
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            
        return None 