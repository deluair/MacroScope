"""
Comprehensive synthetic data generator for MacroScope economic intelligence platform
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

from config.settings import settings
from src.data_generation.economic_relationships import EconomicRelationships, EconomicParameters
from src.data_generation.shock_integration import ShockIntegration

logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    """Main synthetic data generator for MacroScope platform"""
    
    def __init__(self, start_date: str = None, end_date: str = None, frequency: str = "D"):
        self.start_date = start_date or settings.START_DATE
        self.end_date = end_date or settings.END_DATE
        self.frequency = frequency
        
        # Initialize economic relationships and shock integration
        self.econ_relationships = EconomicRelationships()
        self.shock_integration = ShockIntegration()
        
        # Create date range
        self.dates = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq=self.frequency
        )
        
        self.n_periods = len(self.dates)
        logger.info(f"Initialized data generator for {self.n_periods} periods from {self.start_date} to {self.end_date}")
    
    def generate_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Generate all synthetic datasets"""
        logger.info("Starting comprehensive data generation...")
        
        datasets = {}
        
        # Generate core datasets
        datasets["primary_indicators"] = self.generate_primary_indicators()
        datasets["regional_economic"] = self.generate_regional_economic()
        datasets["financial_markets"] = self.generate_financial_markets()
        datasets["international_trade"] = self.generate_international_trade()
        datasets["alternative_indicators"] = self.generate_alternative_indicators()
        
        # Save all datasets
        self.save_datasets(datasets)
        
        return datasets
    
    def generate_primary_indicators(self) -> pd.DataFrame:
        """Generate primary economic indicators dataset"""
        logger.info("Generating primary economic indicators...")
        
        # Initialize base series
        data = pd.DataFrame(index=self.dates)
        data['date'] = self.dates
        
        # GDP Growth (quarterly, seasonally adjusted)
        gdp_trend = 2.5  # Trend growth rate
        gdp_cycle = 0.5 * np.sin(2 * np.pi * np.arange(self.n_periods) / 252) # Annual business cycle
        gdp_noise = np.random.normal(0, 0.3, self.n_periods)
        productivity_shocks = self.shock_integration.generate_technological_shocks(
            self.dates, ["productivity"]
        )["productivity"]
        
        data['gdp_growth'] = gdp_trend + gdp_cycle + gdp_noise + productivity_shocks
        data['gdp_level'] = 100 * np.cumprod(1 + data['gdp_growth'] / 100)
        
        # Unemployment Rate (using Okun's Law)
        data['unemployment_rate'] = self.econ_relationships.okuns_law(
            data['gdp_growth'].values,
            trend_growth=gdp_trend
        )
        
        # Inflation (CPI and Core CPI using Phillips Curve)
        inflation_expectations = np.full(self.n_periods, 2.0)  # Fed target
        supply_shocks = self.shock_integration.generate_external_shocks(
            self.dates, ["oil_price"]
        )["oil_price"] * 0.1  # Oil price impact on inflation
        
        data['cpi_inflation'] = self.econ_relationships.phillips_curve(
            data['unemployment_rate'].values,
            inflation_expectations,
            supply_shocks
        )
        
        # Core CPI (less volatile)
        data['core_cpi_inflation'] = data['cpi_inflation'] * 0.8 + np.random.normal(0, 0.1, self.n_periods)
        
        # Federal Funds Rate (using Taylor Rule)
        output_gap = data['gdp_growth'] - gdp_trend
        data['fed_funds_rate'] = self.econ_relationships.taylor_rule(
            data['cpi_inflation'].values,
            output_gap.values
        )
        
        # Yield Curve
        yield_curves = self.econ_relationships.yield_curve(data['fed_funds_rate'].values)
        for maturity, rates in yield_curves.items():
            data[f'yield_{maturity.lower()}'] = rates
        
        # Employment metrics
        data['payroll_changes'] = np.random.normal(200, 100, self.n_periods)  # Thousands
        data['jolts_openings'] = 7000 + 500 * np.sin(2 * np.pi * np.arange(self.n_periods) / 252) + np.random.normal(0, 200, self.n_periods)
        
        # Exchange rates (USD perspective)
        eur_inflation = np.random.normal(1.8, 0.5, self.n_periods)  # ECB target
        gbp_inflation = np.random.normal(2.0, 0.6, self.n_periods)  # BoE target
        jpy_inflation = np.random.normal(0.5, 0.3, self.n_periods)  # BoJ target
        cny_inflation = np.random.normal(2.5, 0.4, self.n_periods)  # PBOC target
        
        # Initialize exchange rates
        data.loc[data.index[0], 'usd_eur'] = 0.85
        data.loc[data.index[0], 'usd_gbp'] = 0.75
        data.loc[data.index[0], 'usd_jpy'] = 110.0
        data.loc[data.index[0], 'usd_cny'] = 6.5
        
        # Apply PPP relationships
        for i in range(1, self.n_periods):
            data.loc[data.index[i], 'usd_eur'] = self.econ_relationships.purchasing_power_parity(
                np.array([data['cpi_inflation'].iloc[i]]),
                np.array([eur_inflation[i]]),
                np.array([data['usd_eur'].iloc[i-1]])
            )[0]
            
            data.loc[data.index[i], 'usd_gbp'] = self.econ_relationships.purchasing_power_parity(
                np.array([data['cpi_inflation'].iloc[i]]),
                np.array([gbp_inflation[i]]),
                np.array([data['usd_gbp'].iloc[i-1]])
            )[0]
            
            data.loc[data.index[i], 'usd_jpy'] = self.econ_relationships.purchasing_power_parity(
                np.array([data['cpi_inflation'].iloc[i]]),
                np.array([jpy_inflation[i]]),
                np.array([data['usd_jpy'].iloc[i-1]])
            )[0]
            
            data.loc[data.index[i], 'usd_cny'] = self.econ_relationships.purchasing_power_parity(
                np.array([data['cpi_inflation'].iloc[i]]),
                np.array([cny_inflation[i]]),
                np.array([data['usd_cny'].iloc[i-1]])
            )[0]
        
        # Apply economic shocks
        data = self.shock_integration.apply_historical_shocks(data, self.dates)
        
        return data
    
    def generate_regional_economic(self) -> pd.DataFrame:
        """Generate regional economic dataset"""
        logger.info("Generating regional economic data...")
        
        # Major US states and metro areas
        regions = [
            "California", "Texas", "New York", "Florida", "Illinois",
            "Pennsylvania", "Ohio", "Georgia", "North Carolina", "Michigan",
            "NYC Metro", "LA Metro", "Chicago Metro", "Dallas Metro", "Houston Metro"
        ]
        
        regional_data = []
        
        for region in regions:
            region_df = pd.DataFrame(index=self.dates)
            region_df['date'] = self.dates
            region_df['region'] = region
            
            # Regional GDP (correlated with national but with regional variation)
            national_gdp_growth = 2.5
            regional_factor = np.random.normal(1.0, 0.2)  # Regional multiplier
            regional_cycle = 0.3 * np.sin(2 * np.pi * np.arange(self.n_periods) / 252 + np.random.uniform(0, 2*np.pi))
            
            region_df['gdp_growth'] = (national_gdp_growth * regional_factor + 
                                     regional_cycle + 
                                     np.random.normal(0, 0.4, self.n_periods))
            
            # Regional unemployment
            national_unemployment = 5.0
            region_df['unemployment_rate'] = (national_unemployment + 
                                            np.random.normal(0, 1.0) + 
                                            0.5 * np.sin(2 * np.pi * np.arange(self.n_periods) / 252))
            
            # Housing indicators
            region_df['housing_starts'] = np.maximum(
                1000 + 200 * np.sin(2 * np.pi * np.arange(self.n_periods) / 252) + 
                np.random.normal(0, 100, self.n_periods),
                100
            )
            
            region_df['home_prices'] = 100 * np.cumprod(
                1 + np.random.normal(0.05, 0.02, self.n_periods)
            )
            
            # Industry-specific employment
            if "CA" in region or "California" in region:
                # Tech-heavy
                region_df['tech_employment'] = 100 + 10 * np.cumsum(np.random.normal(0.1, 0.5, self.n_periods))
            elif "TX" in region or "Texas" in region:
                # Energy-heavy
                region_df['energy_employment'] = 100 + 5 * np.cumsum(np.random.normal(0.05, 0.8, self.n_periods))
            else:
                # Manufacturing
                region_df['manufacturing_employment'] = 100 + 3 * np.cumsum(np.random.normal(0, 0.3, self.n_periods))
            
            regional_data.append(region_df)
        
        return pd.concat(regional_data, ignore_index=True)
    
    def generate_financial_markets(self) -> pd.DataFrame:
        """Generate financial markets dataset"""
        logger.info("Generating financial markets data...")
        
        data = pd.DataFrame(index=self.dates)
        data['date'] = self.dates
        
        # Equity indices
        # S&P 500
        base_return = 0.08 / 252  # 8% annual return
        volatility_shocks = self.shock_integration.generate_financial_shocks(
            self.dates, ["volatility"]
        )["volatility"]
        
        sp500_returns = np.random.normal(base_return, 0.015, self.n_periods) + volatility_shocks * 0.01
        data['sp500'] = 4000 * np.cumprod(1 + sp500_returns)
        
        # NASDAQ (more volatile)
        nasdaq_returns = sp500_returns * 1.2 + np.random.normal(0, 0.005, self.n_periods)
        data['nasdaq'] = 12000 * np.cumprod(1 + nasdaq_returns)
        
        # Russell 2000 (small caps)
        russell_returns = sp500_returns * 0.9 + np.random.normal(0, 0.008, self.n_periods)
        data['russell_2000'] = 2000 * np.cumprod(1 + russell_returns)
        
        # VIX (volatility index)
        data['vix'] = 20 + 15 * np.abs(volatility_shocks) + np.random.normal(0, 2, self.n_periods)
        data['vix'] = np.clip(data['vix'], 10, 80)
        
        # Bond market indicators
        data['treasury_10y'] = data.index.to_series().apply(lambda x: 
            self.generate_primary_indicators()['yield_10y'].iloc[0] if x == data.index[0] else np.nan
        ).ffill()
        
        # Corporate spreads
        credit_shocks = self.shock_integration.generate_financial_shocks(
            self.dates, ["credit_spread"]
        )["credit_spread"]
        
        data['ig_credit_spread'] = 1.5 + credit_shocks + np.random.normal(0, 0.1, self.n_periods)
        data['hy_credit_spread'] = 4.0 + 2 * credit_shocks + np.random.normal(0, 0.3, self.n_periods)
        
        # Commodity prices
        oil_shocks = self.shock_integration.generate_external_shocks(
            self.dates, ["oil_price"]
        )["oil_price"]
        
        data['oil_price'] = 70 * np.cumprod(1 + oil_shocks * 0.01 + np.random.normal(0, 0.02, self.n_periods))
        data['gold_price'] = 1800 * np.cumprod(1 + np.random.normal(0, 0.015, self.n_periods))
        
        # Currency and international
        data['dxy'] = 100 + 10 * np.cumsum(np.random.normal(0, 0.01, self.n_periods))
        
        return data
    
    def generate_international_trade(self) -> pd.DataFrame:
        """Generate international trade dataset"""
        logger.info("Generating international trade data...")
        
        # Major trading partners
        partners = ["China", "Canada", "Mexico", "Germany", "Japan", "UK", "South Korea"]
        
        trade_data = []
        
        for partner in partners:
            partner_df = pd.DataFrame(index=self.dates)
            partner_df['date'] = self.dates
            partner_df['partner'] = partner
            
            # Base trade volumes (in billions USD)
            if partner == "China":
                base_imports = 400
                base_exports = 150
            elif partner == "Canada":
                base_imports = 300
                base_exports = 280
            elif partner == "Mexico":
                base_imports = 350
                base_exports = 250
            else:
                base_imports = np.random.uniform(50, 200)
                base_exports = np.random.uniform(40, 180)
            
            # Trade flows with economic relationships
            domestic_gdp = 100 + 2.5 * np.arange(self.n_periods) / 252  # Simplified GDP
            foreign_gdp = 100 + np.random.normal(2.0, 1.0) * np.arange(self.n_periods) / 252
            exchange_rate = 1.0 + np.cumsum(np.random.normal(0, 0.001, self.n_periods))
            
            trade_results = self.econ_relationships.trade_balance(
                domestic_gdp, foreign_gdp, exchange_rate
            )
            
            partner_df['imports'] = base_imports * (trade_results['imports'] / 100)
            partner_df['exports'] = base_exports * (trade_results['exports'] / 100)
            partner_df['trade_balance'] = partner_df['exports'] - partner_df['imports']
            
            # Tariff effects (simulate trade war periods)
            if partner == "China":
                # China trade war effects 2018-2020
                tariff_mask = (self.dates >= '2018-01-01') & (self.dates <= '2020-12-31')
                partner_df.loc[tariff_mask, 'imports'] *= 0.85
                partner_df.loc[tariff_mask, 'exports'] *= 0.90
            
            # Container throughput
            partner_df['container_throughput'] = (partner_df['imports'] + partner_df['exports']) * 0.1
            
            trade_data.append(partner_df)
        
        return pd.concat(trade_data, ignore_index=True)
    
    def generate_alternative_indicators(self) -> pd.DataFrame:
        """Generate alternative data sources"""
        logger.info("Generating alternative indicators...")
        
        data = pd.DataFrame(index=self.dates)
        data['date'] = self.dates
        
        # Google Trends economic search indices
        search_terms = ['unemployment', 'inflation', 'recession', 'jobs', 'mortgage rates']
        
        for term in search_terms:
            # Base search volume
            base_volume = np.random.uniform(30, 70)
            
            # Economic stress increases search volume
            if term in ['unemployment', 'recession']:
                stress_factor = 1 + np.random.exponential(0.1, self.n_periods)
            else:
                stress_factor = 1 + np.random.normal(0, 0.1, self.n_periods)
            
            # Seasonal patterns
            seasonal = 1 + 0.1 * np.sin(2 * np.pi * np.arange(self.n_periods) / 252)
            
            data[f'google_trends_{term}'] = base_volume * stress_factor * seasonal
        
        # Satellite-derived economic activity
        # Nighttime lights as proxy for economic activity
        data['nighttime_lights'] = 100 + 2 * np.cumsum(np.random.normal(0.01, 0.02, self.n_periods))
        
        # Shipping activity from satellite data
        data['shipping_density'] = 100 + 10 * np.sin(2 * np.pi * np.arange(self.n_periods) / 252) + np.random.normal(0, 5, self.n_periods)
        
        # Social media sentiment scores
        data['twitter_economic_sentiment'] = np.random.beta(2, 2, self.n_periods) * 100  # 0-100 scale
        data['news_sentiment'] = np.random.normal(50, 15, self.n_periods)  # Neutral = 50
        
        # Corporate earnings call sentiment
        quarterly_mask = (self.dates.month % 3 == 1) & (self.dates.day <= 7)  # Earnings season
        data['earnings_sentiment'] = np.where(
            quarterly_mask,
            np.random.normal(55, 10, self.n_periods),  # Slightly positive during earnings
            np.random.normal(50, 5, self.n_periods)    # Neutral otherwise
        )
        
        # Economic surprise indices
        data['economic_surprise_index'] = np.random.normal(0, 20, self.n_periods)  # Standardized
        
        # High-frequency GDP nowcasting proxies
        data['railroad_traffic'] = 100 + 5 * np.sin(2 * np.pi * np.arange(self.n_periods) / 252) + np.random.normal(0, 3, self.n_periods)
        data['truck_traffic'] = 100 + 3 * np.sin(2 * np.pi * np.arange(self.n_periods) / 252) + np.random.normal(0, 2, self.n_periods)
        data['electricity_consumption'] = 100 + 2 * np.cumsum(np.random.normal(0.005, 0.01, self.n_periods))
        
        return data
    
    def save_datasets(self, datasets: Dict[str, pd.DataFrame]) -> None:
        """Save all generated datasets to CSV files"""
        logger.info("Saving datasets to files...")
        
        for name, df in datasets.items():
            filepath = settings.SYNTHETIC_DATA_DIR / f"{name}.csv"
            df.to_csv(filepath, index=False)
            logger.info(f"Saved {name} dataset with {len(df)} rows to {filepath}")
    
    def generate_scenario_data(self, scenario_name: str, 
                             parameter_adjustments: Dict[str, float]) -> Dict[str, pd.DataFrame]:
        """Generate data for specific economic scenarios"""
        logger.info(f"Generating scenario data: {scenario_name}")
        
        # Adjust economic parameters based on scenario
        adjusted_params = EconomicParameters()
        
        for param, adjustment in parameter_adjustments.items():
            if hasattr(adjusted_params, param):
                current_value = getattr(adjusted_params, param)
                setattr(adjusted_params, param, current_value * adjustment)
        
        # Temporarily replace economic relationships
        original_relationships = self.econ_relationships
        self.econ_relationships = EconomicRelationships(adjusted_params)
        
        try:
            # Generate scenario datasets
            scenario_datasets = self.generate_all_datasets()
            
            # Add scenario metadata
            for name, df in scenario_datasets.items():
                df['scenario'] = scenario_name
                
            return scenario_datasets
            
        finally:
            # Restore original relationships
            self.econ_relationships = original_relationships

def main():
    """Main function to generate all synthetic data"""
    generator = SyntheticDataGenerator()
    
    # Generate baseline datasets
    datasets = generator.generate_all_datasets()
    
    # Generate scenario datasets
    scenarios = {
        "recession": {
            "phillips_alpha": 0.8,  # Steeper Phillips curve
            "taylor_rule_alpha": 2.0,  # More aggressive Fed response
            "okun_alpha": -0.7  # Stronger GDP-unemployment relationship
        },
        "high_inflation": {
            "phillips_beta": 0.6,  # Higher inflation persistence
            "taylor_rule_pi_target": 3.0,  # Higher inflation target
        },
        "productivity_boom": {
            "phillips_alpha": 0.3,  # Flatter Phillips curve
            "okun_alpha": -0.3,  # Weaker GDP-unemployment relationship
        }
    }
    
    for scenario_name, adjustments in scenarios.items():
        scenario_data = generator.generate_scenario_data(scenario_name, adjustments)
        
        # Save scenario data
        scenario_dir = settings.SYNTHETIC_DATA_DIR / scenario_name
        scenario_dir.mkdir(exist_ok=True)
        
        for name, df in scenario_data.items():
            filepath = scenario_dir / f"{name}.csv"
            df.to_csv(filepath, index=False)
    
    logger.info("Synthetic data generation completed successfully!")

if __name__ == "__main__":
    main() 