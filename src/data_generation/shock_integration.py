"""
Shock integration system for realistic economic disruptions and events
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ShockEvent:
    """Economic shock event definition"""
    name: str
    start_date: datetime
    duration_days: int
    intensity: float  # Multiplier for shock magnitude
    affected_variables: List[str]
    shock_type: str  # 'systematic', 'external', 'financial', 'technological'
    description: str

class ShockIntegration:
    """Integrates various economic shocks into synthetic data"""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Define historical shock events for calibration
        self.historical_shocks = self._define_historical_shocks()
        
    def _define_historical_shocks(self) -> List[ShockEvent]:
        """Define historical shock events for realistic calibration"""
        shocks = [
            ShockEvent(
                name="COVID-19 Pandemic",
                start_date=datetime(2020, 3, 15),
                duration_days=365,
                intensity=3.0,
                affected_variables=["gdp", "unemployment", "inflation", "interest_rate"],
                shock_type="external",
                description="Global pandemic causing economic disruption"
            ),
            ShockEvent(
                name="Supply Chain Crisis",
                start_date=datetime(2021, 8, 1),
                duration_days=180,
                intensity=1.5,
                affected_variables=["inflation", "trade_balance", "industrial_production"],
                shock_type="external",
                description="Global supply chain disruptions"
            ),
            ShockEvent(
                name="Tech Sector Correction",
                start_date=datetime(2022, 1, 1),
                duration_days=120,
                intensity=1.2,
                affected_variables=["stock_prices", "investment", "business_confidence"],
                shock_type="financial",
                description="Technology sector valuation correction"
            ),
            ShockEvent(
                name="Energy Price Spike",
                start_date=datetime(2022, 2, 24),
                duration_days=240,
                intensity=2.0,
                affected_variables=["inflation", "oil_price", "exchange_rate"],
                shock_type="external",
                description="Geopolitical tensions affecting energy markets"
            ),
            ShockEvent(
                name="Banking Sector Stress",
                start_date=datetime(2023, 3, 10),
                duration_days=90,
                intensity=1.8,
                affected_variables=["credit_spreads", "financial_conditions", "stock_prices"],
                shock_type="financial",
                description="Regional banking sector stress events"
            )
        ]
        return shocks
    
    def generate_systematic_shocks(self, dates: pd.DatetimeIndex, 
                                 variables: List[str]) -> Dict[str, np.ndarray]:
        """
        Generate systematic shocks (monetary policy, fiscal policy)
        
        Args:
            dates: Date index
            variables: List of economic variables
            
        Returns:
            Dictionary of shock series for each variable
        """
        shocks = {}
        n_periods = len(dates)
        
        for var in variables:
            if var == "monetary_policy":
                # Federal Reserve policy changes
                shocks[var] = self._generate_policy_shocks(dates, policy_type="monetary")
            elif var == "fiscal_policy":
                # Government fiscal policy changes
                shocks[var] = self._generate_policy_shocks(dates, policy_type="fiscal")
            else:
                # Default systematic shocks
                shocks[var] = np.random.normal(0, 0.1, n_periods)
                
        return shocks
    
    def generate_external_shocks(self, dates: pd.DatetimeIndex,
                               variables: List[str]) -> Dict[str, np.ndarray]:
        """
        Generate external shocks (oil prices, geopolitical events, natural disasters)
        
        Args:
            dates: Date index
            variables: List of economic variables
            
        Returns:
            Dictionary of shock series for each variable
        """
        shocks = {}
        n_periods = len(dates)
        
        for var in variables:
            if var == "oil_price":
                shocks[var] = self._generate_commodity_shocks(dates, commodity="oil")
            elif var == "geopolitical":
                shocks[var] = self._generate_geopolitical_shocks(dates)
            elif var == "natural_disaster":
                shocks[var] = self._generate_disaster_shocks(dates)
            else:
                # Default external shocks with fat tails
                shocks[var] = np.random.standard_t(df=3, size=n_periods) * 0.15
                
        return shocks
    
    def generate_financial_shocks(self, dates: pd.DatetimeIndex,
                                variables: List[str]) -> Dict[str, np.ndarray]:
        """
        Generate financial shocks (banking crises, credit events, market volatility)
        
        Args:
            dates: Date index
            variables: List of economic variables
            
        Returns:
            Dictionary of shock series for each variable
        """
        shocks = {}
        n_periods = len(dates)
        
        for var in variables:
            if var == "credit_spread":
                shocks[var] = self._generate_credit_shocks(dates)
            elif var == "volatility":
                shocks[var] = self._generate_volatility_shocks(dates)
            elif var == "liquidity":
                shocks[var] = self._generate_liquidity_shocks(dates)
            else:
                # Default financial shocks with clustering
                shocks[var] = self._generate_garch_shocks(n_periods)
                
        return shocks
    
    def generate_technological_shocks(self, dates: pd.DatetimeIndex,
                                    variables: List[str]) -> Dict[str, np.ndarray]:
        """
        Generate technological shocks (productivity improvements, automation)
        
        Args:
            dates: Date index
            variables: List of economic variables
            
        Returns:
            Dictionary of shock series for each variable
        """
        shocks = {}
        n_periods = len(dates)
        
        for var in variables:
            if var == "productivity":
                shocks[var] = self._generate_productivity_shocks(dates)
            elif var == "automation":
                shocks[var] = self._generate_automation_shocks(dates)
            else:
                # Default technology shocks with trend
                trend = np.linspace(0, 0.1, n_periods)
                shocks[var] = trend + np.random.normal(0, 0.05, n_periods)
                
        return shocks
    
    def _generate_policy_shocks(self, dates: pd.DatetimeIndex, 
                              policy_type: str) -> np.ndarray:
        """Generate policy shock series"""
        n_periods = len(dates)
        shocks = np.zeros(n_periods)
        
        # Major policy announcement dates
        if policy_type == "monetary":
            # Fed meeting dates (approximately every 6 weeks)
            fed_meetings = [i for i in range(0, n_periods, 42)]
            for meeting in fed_meetings:
                if meeting < n_periods:
                    # 20% chance of significant policy change
                    if np.random.random() < 0.2:
                        shocks[meeting:meeting+5] = np.random.normal(0, 0.3, 5)
        
        elif policy_type == "fiscal":
            # Major fiscal announcements (quarterly)
            fiscal_announcements = [i for i in range(0, n_periods, 90)]
            for announcement in fiscal_announcements:
                if announcement < n_periods:
                    # 15% chance of significant fiscal policy change
                    if np.random.random() < 0.15:
                        duration = min(30, n_periods - announcement)
                        shocks[announcement:announcement+duration] = np.random.normal(0, 0.25, duration)
        
        return shocks
    
    def _generate_commodity_shocks(self, dates: pd.DatetimeIndex, 
                                 commodity: str) -> np.ndarray:
        """Generate commodity price shocks"""
        n_periods = len(dates)
        
        # Base volatility for commodity
        base_vol = 0.3 if commodity == "oil" else 0.2
        
        # Generate shocks with occasional spikes
        shocks = np.random.normal(0, base_vol, n_periods)
        
        # Add occasional price spikes (supply disruptions)
        spike_probability = 0.01  # 1% chance per period
        spike_indices = np.random.random(n_periods) < spike_probability
        shocks[spike_indices] *= 3  # Triple the shock magnitude
        
        return shocks
    
    def _generate_geopolitical_shocks(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """Generate geopolitical tension shocks"""
        n_periods = len(dates)
        
        # Base geopolitical tension (usually low)
        base_tension = np.random.exponential(0.1, n_periods)
        
        # Occasional geopolitical events
        event_probability = 0.005  # 0.5% chance per period
        event_indices = np.random.random(n_periods) < event_probability
        
        # Events have persistence
        for i, is_event in enumerate(event_indices):
            if is_event:
                duration = np.random.randint(5, 30)  # 5-30 day events
                end_idx = min(i + duration, n_periods)
                event_magnitude = np.random.exponential(1.0)
                base_tension[i:end_idx] += event_magnitude
        
        return base_tension
    
    def _generate_disaster_shocks(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """Generate natural disaster shocks"""
        n_periods = len(dates)
        shocks = np.zeros(n_periods)
        
        # Seasonal pattern for disasters
        for i, date in enumerate(dates):
            # Hurricane season, wildfire season, etc.
            seasonal_factor = 1 + 0.5 * np.sin(2 * np.pi * date.dayofyear / 365)
            disaster_prob = 0.002 * seasonal_factor  # Base 0.2% chance
            
            if np.random.random() < disaster_prob:
                magnitude = np.random.exponential(0.5)
                duration = np.random.randint(3, 15)
                end_idx = min(i + duration, n_periods)
                shocks[i:end_idx] = magnitude
        
        return shocks
    
    def _generate_credit_shocks(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """Generate credit market shocks"""
        n_periods = len(dates)
        
        # Credit shocks with regime switching
        regime_prob = 0.01  # 1% chance of regime change
        current_regime = "normal"  # normal, stress
        
        shocks = np.zeros(n_periods)
        
        for i in range(n_periods):
            # Regime switching
            if np.random.random() < regime_prob:
                current_regime = "stress" if current_regime == "normal" else "normal"
            
            # Generate shocks based on regime
            if current_regime == "normal":
                shocks[i] = np.random.normal(0, 0.1)
            else:
                shocks[i] = np.random.normal(0.2, 0.3)  # Higher mean and volatility in stress
        
        return shocks
    
    def _generate_volatility_shocks(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """Generate financial volatility shocks"""
        n_periods = len(dates)
        
        # GARCH-like volatility clustering
        volatility = np.zeros(n_periods)
        volatility[0] = 0.2  # Initial volatility
        
        alpha = 0.1  # ARCH coefficient
        beta = 0.85  # GARCH coefficient
        omega = 0.01  # Constant term
        
        for i in range(1, n_periods):
            # GARCH(1,1) process
            volatility[i] = np.sqrt(omega + alpha * volatility[i-1]**2 + beta * volatility[i-1])
            
        # Generate shocks using time-varying volatility
        shocks = np.random.normal(0, 1, n_periods) * volatility
        
        return shocks
    
    def _generate_liquidity_shocks(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """Generate liquidity shocks"""
        n_periods = len(dates)
        
        # Base liquidity conditions
        base_liquidity = np.random.normal(0, 0.1, n_periods)
        
        # Occasional liquidity crises
        crisis_prob = 0.005  # 0.5% chance per period
        for i in range(n_periods):
            if np.random.random() < crisis_prob:
                # Liquidity crisis lasting 1-2 weeks
                duration = np.random.randint(5, 14)
                end_idx = min(i + duration, n_periods)
                crisis_magnitude = np.random.exponential(1.5)
                base_liquidity[i:end_idx] -= crisis_magnitude
        
        return base_liquidity
    
    def _generate_garch_shocks(self, n_periods: int) -> np.ndarray:
        """Generate GARCH-type shocks with volatility clustering"""
        shocks = np.zeros(n_periods)
        volatility = 0.1
        
        alpha = 0.1
        beta = 0.8
        omega = 0.01
        
        for i in range(n_periods):
            shocks[i] = np.random.normal(0, volatility)
            # Update volatility
            volatility = np.sqrt(omega + alpha * shocks[i]**2 + beta * volatility**2)
        
        return shocks
    
    def _generate_productivity_shocks(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """Generate productivity shocks"""
        n_periods = len(dates)
        
        # Long-term productivity trend
        trend = np.linspace(0, 0.2, n_periods)  # 20% productivity growth over period
        
        # Short-term productivity fluctuations
        cyclical = 0.05 * np.sin(2 * np.pi * np.arange(n_periods) / 252)  # Annual cycle
        
        # Random productivity shocks
        random_shocks = np.random.normal(0, 0.02, n_periods)
        
        # Occasional productivity breakthroughs
        breakthrough_prob = 0.001
        breakthroughs = np.random.random(n_periods) < breakthrough_prob
        breakthrough_effects = np.where(breakthroughs, np.random.exponential(0.1), 0)
        
        return trend + cyclical + random_shocks + breakthrough_effects
    
    def _generate_automation_shocks(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """Generate automation impact shocks"""
        n_periods = len(dates)
        
        # Gradual automation trend
        base_trend = np.linspace(0, 0.1, n_periods)
        
        # Sudden automation adoption (e.g., AI breakthroughs)
        adoption_events = np.random.random(n_periods) < 0.002  # 0.2% chance
        
        shocks = base_trend.copy()
        for i, is_event in enumerate(adoption_events):
            if is_event:
                # Automation shock with persistence
                shock_magnitude = np.random.exponential(0.05)
                decay = np.exp(-np.arange(n_periods - i) / 50)  # 50-day half-life
                shocks[i:] += shock_magnitude * decay
        
        return shocks
    
    def apply_historical_shocks(self, data: pd.DataFrame, 
                              dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Apply historically-calibrated shocks to synthetic data
        
        Args:
            data: Synthetic economic data
            dates: Date index
            
        Returns:
            Data with historical shocks applied
        """
        shocked_data = data.copy()
        
        for shock in self.historical_shocks:
            # Find date range for shock
            shock_start = max(shock.start_date, dates[0])
            shock_end = shock.start_date + timedelta(days=shock.duration_days)
            shock_end = min(shock_end, dates[-1])
            
            # Get indices for shock period
            mask = (dates >= shock_start) & (dates <= shock_end)
            shock_indices = np.where(mask)[0]
            
            if len(shock_indices) > 0:
                logger.info(f"Applying {shock.name} shock from {shock_start} to {shock_end}")
                
                # Apply shock to affected variables
                for var in shock.affected_variables:
                    if var in shocked_data.columns:
                        shock_profile = self._create_shock_profile(
                            len(shock_indices), shock.intensity, shock.shock_type
                        )
                        
                        if shock.shock_type == "external" and var in ["gdp", "unemployment"]:
                            # Negative supply shock
                            shocked_data.loc[mask, var] *= (1 - shock_profile * 0.1)
                        elif shock.shock_type == "financial" and var in ["stock_prices", "credit_spreads"]:
                            # Financial stress
                            if var == "stock_prices":
                                shocked_data.loc[mask, var] *= (1 - shock_profile * 0.2)
                            else:
                                shocked_data.loc[mask, var] *= (1 + shock_profile * 0.5)
                        else:
                            # General additive shock
                            shocked_data.loc[mask, var] += shock_profile * np.std(data[var])
        
        return shocked_data
    
    def _create_shock_profile(self, n_periods: int, intensity: float, 
                            shock_type: str) -> np.ndarray:
        """Create shock intensity profile over time"""
        if shock_type == "external":
            # Sharp initial impact, gradual recovery
            t = np.linspace(0, 1, n_periods)
            profile = intensity * np.exp(-3 * t)
        elif shock_type == "financial":
            # Gradual buildup, sharp peak, slow recovery
            t = np.linspace(0, 1, n_periods)
            profile = intensity * np.exp(-((t - 0.3) / 0.2)**2)
        elif shock_type == "systematic":
            # Persistent policy effects
            profile = intensity * np.ones(n_periods) * np.exp(-t / 2)
        else:
            # Default: symmetric shock
            t = np.linspace(0, 1, n_periods)
            profile = intensity * np.exp(-((t - 0.5) / 0.3)**2)
        
        return profile 