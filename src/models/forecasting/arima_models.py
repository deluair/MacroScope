"""
ARIMA and Auto-ARIMA models for time series forecasting
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ARIMAForecaster:
    """ARIMA-based forecasting models"""
    
    def __init__(self, auto_arima_enabled: bool = True):
        self.auto_arima_enabled = auto_arima_enabled
        self.fitted_models: Dict[str, Any] = {}
        self.model_params: Dict[str, Tuple[int, int, int]] = {}
    
    def check_stationarity(self, series: pd.Series) -> bool:
        """Check if series is stationary using Augmented Dickey-Fuller test"""
        try:
            result = adfuller(series.dropna())
            p_value = result[1]
            is_stationary = p_value < 0.05
            
            logger.info(f"Stationarity test - p-value: {p_value:.4f}, Stationary: {is_stationary}")
            return is_stationary
        except Exception as e:
            logger.error(f"Error in stationarity test: {e}")
            return False
    
    def difference_series(self, series: pd.Series, max_diff: int = 2) -> Tuple[pd.Series, int]:
        """Difference series to achieve stationarity"""
        diff_order = 0
        current_series = series.copy()
        
        for i in range(max_diff):
            if self.check_stationarity(current_series):
                break
            current_series = current_series.diff().dropna()
            diff_order += 1
            
        return current_series, diff_order
    
    def find_best_arima_params(self, series: pd.Series) -> Tuple[int, int, int]:
        """Find best ARIMA parameters using auto_arima or grid search"""
        if self.auto_arima_enabled:
            try:
                model = auto_arima(
                    series,
                    start_p=0, start_q=0,
                    max_p=5, max_q=5,
                    seasonal=False,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action='ignore'
                )
                return model.order
            except Exception as e:
                logger.warning(f"Auto ARIMA failed: {e}, using manual selection")
        
        # Manual parameter selection
        best_aic = np.inf
        best_params = (1, 1, 1)
        
        for p in range(3):
            for d in range(2):
                for q in range(3):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted_model = model.fit()
                        
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_params = (p, d, q)
                    except:
                        continue
        
        return best_params
    
    def fit(self, data: pd.DataFrame, target_column: str, 
            external_vars: Optional[List[str]] = None) -> bool:
        """Fit ARIMA model to data"""
        try:
            series = data[target_column].dropna()
            
            if len(series) < 10:
                logger.error(f"Insufficient data for {target_column}: {len(series)} observations")
                return False
            
            # Find best parameters
            params = self.find_best_arima_params(series)
            self.model_params[target_column] = params
            
            # Fit model
            if external_vars and len(external_vars) > 0:
                # ARIMAX model with external variables
                exog_data = data[external_vars].dropna()
                # Align series and exogenous data
                aligned_data = pd.concat([series, exog_data], axis=1).dropna()
                series_aligned = aligned_data[target_column]
                exog_aligned = aligned_data[external_vars]
                
                model = ARIMA(series_aligned, exog=exog_aligned, order=params)
            else:
                model = ARIMA(series, order=params)
            
            fitted_model = model.fit()
            self.fitted_models[target_column] = fitted_model
            
            logger.info(f"ARIMA{params} fitted for {target_column} - AIC: {fitted_model.aic:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error fitting ARIMA model for {target_column}: {e}")
            return False
    
    def forecast(self, target_column: str, steps: int = 30,
                exog_future: Optional[np.ndarray] = None,
                confidence_intervals: List[float] = [0.8, 0.9, 0.95]) -> Dict[str, Any]:
        """Generate forecasts with confidence intervals"""
        if target_column not in self.fitted_models:
            raise ValueError(f"Model not fitted for {target_column}")
        
        try:
            model = self.fitted_models[target_column]
            
            # Generate forecast
            if exog_future is not None:
                forecast_result = model.forecast(steps=steps, exog=exog_future)
                conf_int = model.get_forecast(steps=steps, exog=exog_future).conf_int()
            else:
                forecast_result = model.forecast(steps=steps)
                conf_int = model.get_forecast(steps=steps).conf_int()
            
            # Extract confidence intervals
            forecast_data = {
                'forecast': forecast_result,
                'confidence_intervals': {}
            }
            
            # Calculate different confidence levels
            for alpha in confidence_intervals:
                forecast_obj = model.get_forecast(steps=steps, exog=exog_future)
                ci = forecast_obj.conf_int(alpha=1-alpha)
                forecast_data['confidence_intervals'][alpha] = {
                    'lower': ci.iloc[:, 0].values,
                    'upper': ci.iloc[:, 1].values
                }
            
            # Model diagnostics
            forecast_data['model_diagnostics'] = {
                'aic': model.aic,
                'bic': model.bic,
                'params': self.model_params[target_column],
                'residuals_stats': {
                    'mean': np.mean(model.resid),
                    'std': np.std(model.resid),
                    'skewness': float(model.resid.skew()),
                    'kurtosis': float(model.resid.kurtosis())
                }
            }
            
            return forecast_data
            
        except Exception as e:
            logger.error(f"Error generating forecast for {target_column}: {e}")
            raise
    
    def evaluate_forecast(self, actual: pd.Series, forecast: np.ndarray) -> Dict[str, float]:
        """Evaluate forecast accuracy"""
        # Align actual and forecast data
        min_length = min(len(actual), len(forecast))
        actual_aligned = actual.iloc[-min_length:].values
        forecast_aligned = forecast[:min_length]
        
        # Calculate metrics
        mae = np.mean(np.abs(actual_aligned - forecast_aligned))
        rmse = np.sqrt(np.mean((actual_aligned - forecast_aligned) ** 2))
        mape = np.mean(np.abs((actual_aligned - forecast_aligned) / actual_aligned)) * 100
        
        # R-squared
        ss_res = np.sum((actual_aligned - forecast_aligned) ** 2)
        ss_tot = np.sum((actual_aligned - np.mean(actual_aligned)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2_score': r2
        }
    
    def update_model(self, new_data: pd.DataFrame, target_column: str,
                    refit: bool = False) -> bool:
        """Update model with new data"""
        if target_column not in self.fitted_models:
            logger.error(f"Model not fitted for {target_column}")
            return False
        
        try:
            if refit:
                # Refit the entire model with new data
                return self.fit(new_data, target_column)
            else:
                # Update existing model (if supported by statsmodels)
                # For now, we'll refit since online updating is complex
                return self.fit(new_data, target_column)
            
        except Exception as e:
            logger.error(f"Error updating model for {target_column}: {e}")
            return False
    
    def get_model_summary(self, target_column: str) -> str:
        """Get model summary statistics"""
        if target_column not in self.fitted_models:
            return f"Model not fitted for {target_column}"
        
        model = self.fitted_models[target_column]
        return str(model.summary())
    
    def save_model(self, target_column: str, filepath: str) -> bool:
        """Save fitted model to file"""
        if target_column not in self.fitted_models:
            logger.error(f"Model not fitted for {target_column}")
            return False
        
        try:
            model = self.fitted_models[target_column]
            model.save(filepath)
            logger.info(f"Model saved for {target_column} to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving model for {target_column}: {e}")
            return False
    
    def load_model(self, target_column: str, filepath: str) -> bool:
        """Load fitted model from file"""
        try:
            from statsmodels.tsa.arima.model import ARIMAResults
            model = ARIMAResults.load(filepath)
            self.fitted_models[target_column] = model
            logger.info(f"Model loaded for {target_column} from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model for {target_column}: {e}")
            return False

class MultiVariateARIMA:
    """Multivariate ARIMA for multiple time series"""
    
    def __init__(self):
        self.forecasters: Dict[str, ARIMAForecaster] = {}
        self.variables: List[str] = []
    
    def fit(self, data: pd.DataFrame, target_variables: List[str],
           cross_effects: bool = True) -> bool:
        """Fit ARIMA models for multiple variables"""
        self.variables = target_variables
        success_count = 0
        
        for var in target_variables:
            forecaster = ARIMAForecaster()
            
            # Include other variables as external regressors if cross_effects is True
            if cross_effects:
                external_vars = [v for v in target_variables if v != var]
                success = forecaster.fit(data, var, external_vars[:3])  # Limit to 3 external vars
            else:
                success = forecaster.fit(data, var)
            
            if success:
                self.forecasters[var] = forecaster
                success_count += 1
        
        logger.info(f"Successfully fitted {success_count}/{len(target_variables)} ARIMA models")
        return success_count > 0
    
    def forecast_all(self, steps: int = 30,
                    confidence_intervals: List[float] = [0.8, 0.9, 0.95]) -> Dict[str, Dict[str, Any]]:
        """Generate forecasts for all variables"""
        forecasts = {}
        
        for var, forecaster in self.forecasters.items():
            try:
                forecast_data = forecaster.forecast(var, steps, confidence_intervals=confidence_intervals)
                forecasts[var] = forecast_data
            except Exception as e:
                logger.error(f"Error forecasting {var}: {e}")
                continue
        
        return forecasts
    
    def get_cross_correlations(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate cross-correlations between variables"""
        subset_data = data[self.variables].dropna()
        return subset_data.corr() 