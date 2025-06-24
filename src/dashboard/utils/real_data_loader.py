import pandas as pd
import streamlit as st
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional
import yfinance as yf
from fredapi import Fred
from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FRED API
fred = Fred(api_key=settings.FRED_API_KEY)

class RealDataLoader:
    """Loads real economic and financial data from various APIs"""
    
    def __init__(self):
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        self.start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')  # 5 years of data
    
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def load_fred_data(_self, series_ids: Dict[str, str]) -> pd.DataFrame:
        """Load data from FRED API"""
        data = {}
        for name, series_id in series_ids.items():
            try:
                series = fred.get_series(series_id, observation_start=_self.start_date, observation_end=_self.end_date)
                if not series.empty:
                    data[name] = series
                    logger.info(f"Successfully loaded FRED series: {name} ({series_id})")
                else:
                    logger.warning(f"No data found for FRED series: {name} ({series_id})")
            except Exception as e:
                logger.error(f"Error loading FRED series {name} ({series_id}): {e}")
                continue
        
        if data:
            df = pd.DataFrame(data)
            df.index.name = 'date'
            df = df.reset_index()
            return df
        return pd.DataFrame()
    
    @st.cache_data(ttl=3600)
    def load_yahoo_finance_data(_self, tickers: Dict[str, str]) -> pd.DataFrame:
        """Load data from Yahoo Finance"""
        data = {}
        for name, ticker in tickers.items():
            try:
                yf_ticker = yf.Ticker(ticker)
                hist = yf_ticker.history(start=_self.start_date, end=_self.end_date)
                if not hist.empty:
                    data[f"{name}_close"] = hist['Close']
                    data[f"{name}_volume"] = hist['Volume']
                    logger.info(f"Successfully loaded Yahoo Finance data: {name} ({ticker})")
                else:
                    logger.warning(f"No data found for ticker: {name} ({ticker})")
            except Exception as e:
                logger.error(f"Error loading Yahoo Finance data for {name} ({ticker}): {e}")
                continue
        
        if data:
            df = pd.DataFrame(data)
            df.index.name = 'date'
            df = df.reset_index()
            return df
        return pd.DataFrame()
    
    @st.cache_data(ttl=3600)
    def load_bls_data(_self, series_ids: Dict[str, str]) -> pd.DataFrame:
        """Load data from BLS API"""
        try:
            headers = {'Content-type': 'application/json'}
            data_payload = {
                "seriesid": list(series_ids.values()),
                "startyear": str(datetime.now().year - 5),
                "endyear": str(datetime.now().year),
                "registrationkey": settings.BLS_API_KEY
            }
            
            response = requests.post(
                'https://api.bls.gov/publicAPI/v2/timeseries/data/',
                json=data_payload,
                headers=headers
            )
            
            if response.status_code == 200:
                json_data = response.json()
                if json_data.get('status') == 'REQUEST_SUCCEEDED':
                    results = json_data.get('Results', {}).get('series', [])
                    
                    all_data = {}
                    for series in results:
                        series_id = series['seriesID']
                        # Find the name for this series_id
                        name = next((k for k, v in series_ids.items() if v == series_id), series_id)
                        
                        df_data = []
                        for item in series['data']:
                            try:
                                # Handle different period formats
                                period = item['period']
                                if period.startswith('M'):
                                    month = period[1:]
                                    date_str = f"{item['year']}-{month.zfill(2)}-01"
                                elif period.startswith('Q'):
                                    quarter = int(period[1:])
                                    month = (quarter - 1) * 3 + 1
                                    date_str = f"{item['year']}-{month:02d}-01"
                                else:
                                    date_str = f"{item['year']}-01-01"
                                
                                df_data.append({
                                    'date': pd.to_datetime(date_str),
                                    name: float(item['value']) if item['value'] != '.' else None
                                })
                            except (ValueError, KeyError) as e:
                                logger.warning(f"Error parsing BLS data point: {e}")
                                continue
                        
                        if df_data:
                            series_df = pd.DataFrame(df_data)
                            series_df = series_df.set_index('date').sort_index()
                            all_data[name] = series_df[name]
                    
                    if all_data:
                        df = pd.DataFrame(all_data)
                        df.index.name = 'date'
                        df = df.reset_index()
                        logger.info(f"Successfully loaded BLS data for {len(all_data)} series")
                        return df
            
            logger.error(f"BLS API request failed: {response.status_code}")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error loading BLS data: {e}")
            return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_all_real_data() -> Dict[str, pd.DataFrame]:
    """Load all real economic and financial data"""
    loader = RealDataLoader()
    datasets = {}
    
    # Primary Economic Indicators from FRED
    fred_primary = {
        'gdp': 'GDP',
        'gdp_growth': 'A191RL1Q225SBEA',  # Real GDP, Percent Change from Year Ago
        'unemployment_rate': 'UNRATE',  # Unemployment Rate
        'cpi_inflation': 'CPIAUCSL',  # Consumer Price Index
        'federal_funds_rate': 'FEDFUNDS',
        'consumer_sentiment': 'UMCSENT',
        'industrial_production': 'INDPRO',
        'retail_sales': 'RSAFS',
        'housing_starts': 'HOUST'
    }
    
    primary_data = loader.load_fred_data(fred_primary)
    if not primary_data.empty:
        datasets['primary_indicators'] = primary_data
    
    # Financial Markets from Yahoo Finance
    yahoo_tickers = {
        'sp500': '^GSPC',
        'nasdaq': '^IXIC',
        'dow_jones': '^DJI',
        'vix': '^VIX',
        'treasury_10y': '^TNX',
        'dollar_index': 'DX-Y.NYB'
    }
    
    financial_data = loader.load_yahoo_finance_data(yahoo_tickers)
    if not financial_data.empty:
        datasets['financial_markets'] = financial_data
    
    # Regional Economic Data from FRED
    fred_regional = {
        'california_unemployment': 'CAUR',
        'texas_unemployment': 'TXUR',
        'new_york_unemployment': 'NYUR',
        'florida_unemployment': 'FLUR'
    }
    
    regional_data = loader.load_fred_data(fred_regional)
    if not regional_data.empty:
        datasets['regional_economic'] = regional_data
    
    # Labor Statistics from BLS
    bls_series = {
        'nonfarm_payrolls': 'CES0000000001',
        'average_hourly_earnings': 'CES0500000003',
        'labor_force_participation': 'LNS11300000'
    }
    
    bls_data = loader.load_bls_data(bls_series)
    if not bls_data.empty:
        datasets['labor_statistics'] = bls_data
    
    # International Trade from FRED
    fred_trade = {
        'trade_balance': 'BOPGSTB',
        'exports': 'EXPGS',
        'imports': 'IMPGS',
        'exchange_rate_eur': 'DEXUSEU',
        'exchange_rate_cny': 'DEXCHUS'
    }
    
    trade_data = loader.load_fred_data(fred_trade)
    if not trade_data.empty:
        datasets['international_trade'] = trade_data
    
    # Alternative Indicators from FRED
    fred_alternative = {
        'yield_curve_spread': 'T10Y2Y',
        'credit_spread': 'BAA10Y',
        'money_supply': 'M2SL',
        'bank_credit': 'TOTBKCR'
    }
    
    alternative_data = loader.load_fred_data(fred_alternative)
    if not alternative_data.empty:
        datasets['alternative_indicators'] = alternative_data
    
    logger.info(f"Successfully loaded {len(datasets)} real datasets")
    return datasets

if __name__ == '__main__':
    # For testing purposes
    all_data = load_all_real_data()
    if all_data:
        for name, df in all_data.items():
            print(f"\n--- {name} ---")
            print(df.head())
            print(df.info())