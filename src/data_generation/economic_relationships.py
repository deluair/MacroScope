"""
Economic relationships and theoretical foundations for synthetic data generation
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class EconomicParameters:
    """Economic model parameters"""
    phillips_alpha: float = 0.5  # Phillips curve slope
    phillips_beta: float = 0.3   # Inflation expectations coefficient
    taylor_rule_r_star: float = 2.0  # Natural interest rate
    taylor_rule_pi_target: float = 2.0  # Inflation target
    taylor_rule_alpha: float = 1.5  # Inflation response
    taylor_rule_beta: float = 0.5   # Output gap response
    okun_alpha: float = -0.5  # Okun's law coefficient
    ppp_adjustment: float = 0.1  # PPP adjustment speed

class EconomicRelationships:
    """Implements realistic economic relationships for synthetic data generation"""
    
    def __init__(self, parameters: Optional[EconomicParameters] = None):
        self.params = parameters or EconomicParameters()
        
    def phillips_curve(self, unemployment: np.ndarray, 
                      inflation_expectations: np.ndarray,
                      supply_shock: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Phillips Curve: π_t = α * (u_n - u_t) + β * π_e_t + ε_t
        
        Args:
            unemployment: Current unemployment rate
            inflation_expectations: Expected inflation
            supply_shock: Supply shock (optional)
        
        Returns:
            Inflation rate
        """
        natural_unemployment = 5.0  # Natural rate of unemployment
        
        # Core Phillips curve relationship
        inflation = (self.params.phillips_alpha * (natural_unemployment - unemployment) + 
                    self.params.phillips_beta * inflation_expectations)
        
        # Add supply shocks if provided
        if supply_shock is not None:
            inflation += supply_shock
            
        return inflation
    
    def taylor_rule(self, inflation: np.ndarray, 
                   output_gap: np.ndarray,
                   lagged_rate: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Taylor Rule: i_t = r* + π_target + α(π_t - π_target) + β * y_gap_t
        
        Args:
            inflation: Current inflation rate
            output_gap: Output gap (actual - potential GDP)
            lagged_rate: Previous period interest rate for smoothing
        
        Returns:
            Federal funds rate
        """
        # Taylor rule formula
        rate = (self.params.taylor_rule_r_star + 
                self.params.taylor_rule_pi_target +
                self.params.taylor_rule_alpha * (inflation - self.params.taylor_rule_pi_target) +
                self.params.taylor_rule_beta * output_gap)
        
        # Interest rate smoothing
        if lagged_rate is not None:
            smoothing_param = 0.8
            rate = smoothing_param * lagged_rate + (1 - smoothing_param) * rate
            
        # Zero lower bound constraint
        rate = np.maximum(rate, 0.0)
        
        return rate
    
    def okuns_law(self, gdp_growth: np.ndarray, 
                  trend_growth: float = 2.5,
                  lagged_unemployment: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Okun's Law: Δu_t = α * (y_t - y_trend)
        
        Args:
            gdp_growth: Real GDP growth rate
            trend_growth: Trend GDP growth rate
            lagged_unemployment: Previous period unemployment
        
        Returns:
            Unemployment rate
        """
        # Change in unemployment based on GDP growth
        unemployment_change = self.params.okun_alpha * (gdp_growth - trend_growth)
        
        if lagged_unemployment is not None:
            unemployment = lagged_unemployment + unemployment_change
        else:
            # Start with natural rate if no lagged values
            unemployment = 5.0 + unemployment_change
            
        # Ensure unemployment stays within reasonable bounds
        unemployment = np.clip(unemployment, 2.0, 15.0)
        
        return unemployment
    
    def purchasing_power_parity(self, domestic_inflation: np.ndarray,
                               foreign_inflation: np.ndarray,
                               lagged_exchange_rate: np.ndarray) -> np.ndarray:
        """
        Purchasing Power Parity: Δe_t = θ * (π_domestic - π_foreign)
        
        Args:
            domestic_inflation: Domestic inflation rate
            foreign_inflation: Foreign inflation rate  
            lagged_exchange_rate: Previous period exchange rate
        
        Returns:
            Exchange rate
        """
        # PPP-implied exchange rate change
        exchange_rate_change = self.params.ppp_adjustment * (domestic_inflation - foreign_inflation)
        
        # Update exchange rate
        exchange_rate = lagged_exchange_rate * (1 + exchange_rate_change / 100)
        
        return exchange_rate
    
    def money_demand(self, gdp: np.ndarray, 
                    interest_rate: np.ndarray,
                    income_elasticity: float = 1.0,
                    interest_elasticity: float = -0.5) -> np.ndarray:
        """
        Money demand function: M/P = Y^α * i^β
        
        Args:
            gdp: Real GDP
            interest_rate: Nominal interest rate
            income_elasticity: Income elasticity of money demand
            interest_elasticity: Interest elasticity of money demand
        
        Returns:
            Real money demand
        """
        # Log-linear money demand
        log_money_demand = (income_elasticity * np.log(gdp) + 
                           interest_elasticity * np.log(interest_rate + 0.01))  # Add small constant to avoid log(0)
        
        return np.exp(log_money_demand)
    
    def consumption_function(self, income: np.ndarray,
                           wealth: np.ndarray,
                           interest_rate: np.ndarray,
                           mpc: float = 0.8,
                           wealth_effect: float = 0.05,
                           interest_elasticity: float = -0.2) -> np.ndarray:
        """
        Consumption function: C = mpc * Y + wealth_effect * W - interest_elasticity * i
        
        Args:
            income: Disposable income
            wealth: Household wealth
            interest_rate: Real interest rate
            mpc: Marginal propensity to consume
            wealth_effect: Wealth effect coefficient
            interest_elasticity: Interest rate elasticity
        
        Returns:
            Consumption
        """
        consumption = (mpc * income + 
                      wealth_effect * wealth + 
                      interest_elasticity * interest_rate)
        
        return np.maximum(consumption, 0.1 * income)  # Minimum consumption level
    
    def investment_function(self, gdp: np.ndarray,
                           interest_rate: np.ndarray,
                           business_confidence: np.ndarray,
                           accelerator: float = 0.3,
                           cost_of_capital: float = -1.5,
                           confidence_effect: float = 0.1) -> np.ndarray:
        """
        Investment function based on accelerator model and cost of capital
        
        Args:
            gdp: Real GDP
            interest_rate: Real interest rate
            business_confidence: Business confidence index
            accelerator: Accelerator coefficient
            cost_of_capital: Interest rate sensitivity
            confidence_effect: Business confidence effect
        
        Returns:
            Investment
        """
        # Base investment from accelerator model
        gdp_growth = np.diff(gdp, prepend=gdp[0])
        base_investment = accelerator * gdp_growth
        
        # Adjust for cost of capital and confidence
        investment = (base_investment + 
                     cost_of_capital * interest_rate +
                     confidence_effect * business_confidence)
        
        return np.maximum(investment, 0.05 * gdp)  # Minimum investment level
    
    def aggregate_supply(self, potential_gdp: np.ndarray,
                        productivity_shock: np.ndarray,
                        oil_price_shock: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Aggregate supply function with productivity and supply shocks
        
        Args:
            potential_gdp: Potential GDP
            productivity_shock: Total factor productivity shock
            oil_price_shock: Oil price shock (optional)
        
        Returns:
            Actual GDP supply
        """
        # Base supply from potential GDP and productivity
        gdp_supply = potential_gdp * (1 + productivity_shock)
        
        # Oil price shock effects (negative supply shock)
        if oil_price_shock is not None:
            oil_effect = -0.1 * oil_price_shock  # 10% oil price increase reduces GDP by 1%
            gdp_supply *= (1 + oil_effect)
            
        return gdp_supply
    
    def yield_curve(self, short_rate: np.ndarray,
                   term_premium: float = 1.0,
                   expectations_hypothesis: bool = True) -> Dict[str, np.ndarray]:
        """
        Generate yield curve based on expectations hypothesis and term premium
        
        Args:
            short_rate: Short-term interest rate (3-month)
            term_premium: Term premium for long-term rates
            expectations_hypothesis: Whether to use expectations hypothesis
        
        Returns:
            Dictionary of rates for different maturities
        """
        rates = {}
        maturities = ['3M', '6M', '1Y', '2Y', '5Y', '10Y', '30Y']
        
        for i, maturity in enumerate(maturities):
            if i == 0:
                rates[maturity] = short_rate
            else:
                # Term structure based on expectations and risk premium
                maturity_years = [0.25, 0.5, 1, 2, 5, 10, 30][i]
                if expectations_hypothesis:
                    # Simple expectations hypothesis with term premium
                    rates[maturity] = short_rate + term_premium * np.sqrt(maturity_years)
                else:
                    # More complex term structure
                    rates[maturity] = short_rate + term_premium * maturity_years * 0.5
                    
        return rates
    
    def trade_balance(self, domestic_gdp: np.ndarray,
                     foreign_gdp: np.ndarray,
                     exchange_rate: np.ndarray,
                     import_elasticity: float = 1.5,
                     export_elasticity: float = 1.2) -> Dict[str, np.ndarray]:
        """
        Trade balance function with income and price elasticities
        
        Args:
            domestic_gdp: Domestic GDP
            foreign_gdp: Foreign GDP 
            exchange_rate: Real exchange rate
            import_elasticity: Import income elasticity
            export_elasticity: Export price elasticity
        
        Returns:
            Dictionary with imports, exports, and trade balance
        """
        # Import function (depends on domestic income and exchange rate)
        imports = import_elasticity * domestic_gdp / exchange_rate
        
        # Export function (depends on foreign income and exchange rate)
        exports = export_elasticity * foreign_gdp * exchange_rate
        
        # Trade balance
        trade_balance = exports - imports
        
        return {
            'imports': imports,
            'exports': exports,
            'trade_balance': trade_balance
        } 