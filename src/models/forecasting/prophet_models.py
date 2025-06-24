"""
Advanced Prophet-based forecasting models for economic indicators
"""
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class EconomicProphetForecaster:
    """Advanced Prophet forecasting for economic indicators"""
    
    def __init__(self, indicator_configs: Optional[Dict] = None):
        self.models = {}
        self.forecasts = {}
        self.indicator_configs = indicator_configs or self._default_configs()
        
    def _default_configs(self) -> Dict:
        """Default Prophet configurations for different economic indicators"""
        return {
            'gdp_growth': {
                'seasonality_mode': 'additive',
                'yearly_seasonality': True,
                'weekly_seasonality': False,
                'daily_seasonality': False,
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10.0
            },
            'unemployment_rate': {
                'seasonality_mode': 'additive',
                'yearly_seasonality': True,
                'weekly_seasonality': False,
                'daily_seasonality': False,
                'changepoint_prior_scale': 0.01,
                'seasonality_prior_scale': 10.0
            },
            'cpi_inflation': {
                'seasonality_mode': 'multiplicative',
                'yearly_seasonality': True,
                'weekly_seasonality': False,
                'daily_seasonality': False,
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 15.0
            },
            'sp500_close': {
                'seasonality_mode': 'multiplicative',
                'yearly_seasonality': True,
                'weekly_seasonality': True,
                'daily_seasonality': False,
                'changepoint_prior_scale': 0.1,
                'seasonality_prior_scale': 20.0
            },
            'federal_funds_rate': {
                'seasonality_mode': 'additive',
                'yearly_seasonality': True,
                'weekly_seasonality': False,
                'daily_seasonality': False,
                'changepoint_prior_scale': 0.01,
                'seasonality_prior_scale': 5.0
            }
        }
    
    def prepare_data(self, df: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
        """Prepare data for Prophet (requires 'ds' and 'y' columns)"""
        if date_col not in df.columns or target_col not in df.columns:
            raise ValueError(f"Columns {date_col} or {target_col} not found in dataframe")
        
        # Create Prophet-compatible dataframe
        prophet_df = df[[date_col, target_col]].copy()
        prophet_df = prophet_df.rename(columns={date_col: 'ds', target_col: 'y'})
        
        # Ensure datetime and numeric types
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        prophet_df['y'] = pd.to_numeric(prophet_df['y'], errors='coerce')
        
        # Remove missing values
        prophet_df = prophet_df.dropna()
        
        # Sort by date
        prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
        
        logger.info(f"Prepared {len(prophet_df)} data points for {target_col}")
        return prophet_df
    
    def train_model(self, data: pd.DataFrame, indicator: str) -> Prophet:
        """Train Prophet model for specific economic indicator"""
        
        # Get configuration for this indicator
        config = self.indicator_configs.get(indicator, self.indicator_configs['gdp_growth'])
        
        # Initialize Prophet model
        model = Prophet(**config)
        
        # Add custom seasonalities for economic indicators
        if indicator in ['unemployment_rate', 'gdp_growth']:
            # Business cycle seasonality (every ~7-10 years)
            model.add_seasonality(name='business_cycle', period=365.25*8, fourier_order=3)
        
        if indicator in ['sp500_close', 'nasdaq_close']:
            # Market seasonality effects
            model.add_seasonality(name='quarterly_earnings', period=365.25/4, fourier_order=2)
        
        # Fit the model
        try:
            model.fit(data)
            self.models[indicator] = model
            logger.info(f"Successfully trained Prophet model for {indicator}")
            return model
        except Exception as e:
            logger.error(f"Error training Prophet model for {indicator}: {e}")
            raise
    
    def generate_forecast(self, indicator: str, periods: int = 365, 
                         include_history: bool = True) -> pd.DataFrame:
        """Generate forecast for economic indicator"""
        
        if indicator not in self.models:
            raise ValueError(f"Model for {indicator} not trained yet")
        
        model = self.models[indicator]
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods, include_history=include_history)
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Store forecast
        self.forecasts[indicator] = forecast
        
        logger.info(f"Generated {periods}-day forecast for {indicator}")
        return forecast
    
    def get_forecast_summary(self, indicator: str) -> Dict:
        """Get summary statistics for forecast"""
        
        if indicator not in self.forecasts:
            raise ValueError(f"Forecast for {indicator} not generated yet")
        
        forecast = self.forecasts[indicator]
        
        # Get future predictions only
        future_forecast = forecast[forecast['ds'] > forecast['ds'].iloc[-365]]
        
        if len(future_forecast) == 0:
            return {}
        
        summary = {
            'indicator': indicator,
            'forecast_start': future_forecast['ds'].min(),
            'forecast_end': future_forecast['ds'].max(),
            'predicted_mean': future_forecast['yhat'].mean(),
            'predicted_trend': future_forecast['trend'].iloc[-1] - future_forecast['trend'].iloc[0],
            'confidence_interval_width': (future_forecast['yhat_upper'] - future_forecast['yhat_lower']).mean(),
            'next_30_days': {
                'mean': future_forecast['yhat'].head(30).mean(),
                'min': future_forecast['yhat_lower'].head(30).min(),
                'max': future_forecast['yhat_upper'].head(30).max(),
                'trend': 'increasing' if future_forecast['yhat'].head(30).iloc[-1] > future_forecast['yhat'].head(30).iloc[0] else 'decreasing'
            },
            'next_90_days': {
                'mean': future_forecast['yhat'].head(90).mean(),
                'min': future_forecast['yhat_lower'].head(90).min(),
                'max': future_forecast['yhat_upper'].head(90).max(),
                'trend': 'increasing' if future_forecast['yhat'].head(90).iloc[-1] > future_forecast['yhat'].head(90).iloc[0] else 'decreasing'
            }
        }
        
        return summary
    
    def batch_forecast(self, data_dict: Dict[str, pd.DataFrame], 
                      date_col: str = 'date', periods: int = 365) -> Dict[str, pd.DataFrame]:
        """Generate forecasts for multiple indicators"""
        
        results = {}
        
        for indicator, df in data_dict.items():
            try:
                # Find numeric columns for forecasting
                numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
                
                for col in numeric_cols:
                    forecast_key = f"{indicator}_{col}"
                    
                    # Prepare data
                    prophet_data = self.prepare_data(df, date_col, col)
                    
                    if len(prophet_data) < 30:  # Need minimum data points
                        logger.warning(f"Insufficient data for {forecast_key}, skipping")
                        continue
                    
                    # Train model
                    model = self.train_model(prophet_data, forecast_key)
                    
                    # Generate forecast
                    forecast = self.generate_forecast(forecast_key, periods)
                    
                    results[forecast_key] = {
                        'forecast': forecast,
                        'summary': self.get_forecast_summary(forecast_key),
                        'model': model
                    }
                    
            except Exception as e:
                logger.error(f"Error forecasting {indicator}: {e}")
                continue
        
        logger.info(f"Generated forecasts for {len(results)} indicators")
        return results
    
    def get_investment_signals(self, forecasts: Dict) -> Dict[str, Dict]:
        """Generate investment signals based on forecasts"""
        
        signals = {}
        
        for indicator, forecast_data in forecasts.items():
            if 'summary' not in forecast_data:
                continue
                
            summary = forecast_data['summary']
            
            # Investment signal logic
            signal = {
                'indicator': indicator,
                'signal': 'NEUTRAL',
                'confidence': 0.5,
                'reasoning': '',
                'time_horizon': '90_days'
            }
            
            # Economic indicators signals
            if 'gdp_growth' in indicator:
                if summary['next_90_days']['trend'] == 'increasing' and summary['next_90_days']['mean'] > 2.0:
                    signal['signal'] = 'BULLISH'
                    signal['confidence'] = 0.75
                    signal['reasoning'] = 'Strong GDP growth expected, positive for equities'
                elif summary['next_90_days']['mean'] < 1.0:
                    signal['signal'] = 'BEARISH'
                    signal['confidence'] = 0.7
                    signal['reasoning'] = 'Weak GDP growth expected, consider defensive positions'
            
            elif 'unemployment_rate' in indicator:
                if summary['next_90_days']['trend'] == 'decreasing' and summary['next_90_days']['mean'] < 4.5:
                    signal['signal'] = 'BULLISH'
                    signal['confidence'] = 0.7
                    signal['reasoning'] = 'Falling unemployment, strong labor market'
                elif summary['next_90_days']['trend'] == 'increasing':
                    signal['signal'] = 'BEARISH'
                    signal['confidence'] = 0.65
                    signal['reasoning'] = 'Rising unemployment, economic weakness'
            
            elif 'cpi_inflation' in indicator:
                if summary['next_90_days']['mean'] > 4.0:
                    signal['signal'] = 'BEARISH'
                    signal['confidence'] = 0.8
                    signal['reasoning'] = 'High inflation expected, pressure on bonds and growth stocks'
                elif 1.5 < summary['next_90_days']['mean'] < 3.0:
                    signal['signal'] = 'BULLISH'
                    signal['confidence'] = 0.6
                    signal['reasoning'] = 'Moderate inflation, favorable for equities'
            
            elif any(x in indicator for x in ['sp500', 'nasdaq', 'dow_jones']):
                trend_strength = abs(summary['predicted_trend']) / summary['predicted_mean']
                if summary['next_90_days']['trend'] == 'increasing' and trend_strength > 0.05:
                    signal['signal'] = 'BULLISH'
                    signal['confidence'] = 0.7
                    signal['reasoning'] = 'Strong upward momentum in equity markets'
                elif summary['next_90_days']['trend'] == 'decreasing' and trend_strength > 0.05:
                    signal['signal'] = 'BEARISH'
                    signal['confidence'] = 0.7
                    signal['reasoning'] = 'Downward pressure in equity markets'
            
            signals[indicator] = signal
        
        return signals

def main():
    """Example usage of the Prophet forecasting system"""
    
    # This would typically be called from the dashboard
    logger.info("Prophet forecasting system initialized")
    
    # Example: Create sample data
    dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
    sample_data = pd.DataFrame({
        'date': dates,
        'gdp_growth': 2.0 + 0.5 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + np.random.normal(0, 0.3, len(dates))
    })
    
    # Initialize forecaster
    forecaster = EconomicProphetForecaster()
    
    # Prepare and forecast
    prophet_data = forecaster.prepare_data(sample_data, 'date', 'gdp_growth')
    model = forecaster.train_model(prophet_data, 'gdp_growth')
    forecast = forecaster.generate_forecast('gdp_growth', periods=90)
    summary = forecaster.get_forecast_summary('gdp_growth')
    
    logger.info(f"Forecast summary: {summary}")

if __name__ == "__main__":
    main() 