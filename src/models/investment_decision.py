"""
Investment Decision Framework for MacroScope
Transforms economic forecasts into actionable investment recommendations
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class AssetClass(Enum):
    EQUITIES = "equities"
    BONDS = "bonds"
    COMMODITIES = "commodities"
    CASH = "cash"
    REAL_ESTATE = "real_estate"
    ALTERNATIVES = "alternatives"

class InvestmentSignal(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

@dataclass
class InvestmentRecommendation:
    asset_class: AssetClass
    signal: InvestmentSignal
    confidence: float  # 0-1
    target_allocation: float  # percentage
    time_horizon: str  # "30d", "90d", "1y"
    reasoning: str
    risk_level: str  # "Low", "Medium", "High"
    expected_return: Optional[float] = None
    max_drawdown: Optional[float] = None

class InvestmentDecisionEngine:
    """Advanced investment decision engine based on economic forecasts"""
    
    def __init__(self):
        self.current_allocations = self._default_allocations()
        self.risk_tolerance = "Medium"  # Low, Medium, High
        self.investment_horizon = "Medium"  # Short (< 1yr), Medium (1-3yr), Long (3yr+)
        
    def _default_allocations(self) -> Dict[AssetClass, float]:
        """Default balanced portfolio allocation"""
        return {
            AssetClass.EQUITIES: 60.0,
            AssetClass.BONDS: 30.0,
            AssetClass.COMMODITIES: 5.0,
            AssetClass.CASH: 5.0,
            AssetClass.REAL_ESTATE: 0.0,
            AssetClass.ALTERNATIVES: 0.0
        }
    
    def analyze_economic_environment(self, forecasts: Dict, signals: Dict) -> Dict[str, str]:
        """Analyze overall economic environment from forecasts"""
        
        environment = {
            'growth_outlook': 'neutral',
            'inflation_outlook': 'neutral',
            'monetary_policy': 'neutral',
            'market_sentiment': 'neutral',
            'overall_regime': 'neutral'
        }
        
        # Analyze GDP growth outlook
        gdp_signals = [s for k, s in signals.items() if 'gdp' in k.lower()]
        if gdp_signals:
            bullish_count = sum(1 for s in gdp_signals if s['signal'] == 'BULLISH')
            if bullish_count > len(gdp_signals) / 2:
                environment['growth_outlook'] = 'expansionary'
            elif any(s['signal'] == 'BEARISH' for s in gdp_signals):
                environment['growth_outlook'] = 'contractionary'
        
        # Analyze inflation outlook
        inflation_signals = [s for k, s in signals.items() if 'inflation' in k.lower() or 'cpi' in k.lower()]
        if inflation_signals:
            for signal in inflation_signals:
                if 'high inflation' in signal.get('reasoning', '').lower():
                    environment['inflation_outlook'] = 'rising'
                elif 'moderate inflation' in signal.get('reasoning', '').lower():
                    environment['inflation_outlook'] = 'stable'
        
        # Analyze monetary policy environment
        fed_signals = [s for k, s in signals.items() if 'fed' in k.lower() or 'rate' in k.lower()]
        if fed_signals:
            for signal in fed_signals:
                if any(word in signal.get('reasoning', '').lower() for word in ['rising', 'tightening']):
                    environment['monetary_policy'] = 'tightening'
                elif any(word in signal.get('reasoning', '').lower() for word in ['cutting', 'easing']):
                    environment['monetary_policy'] = 'easing'
        
        # Analyze market sentiment
        market_signals = [s for k, s in signals.items() if any(x in k.lower() for x in ['sp500', 'nasdaq', 'vix'])]
        if market_signals:
            bullish_market = sum(1 for s in market_signals if s['signal'] == 'BULLISH')
            bearish_market = sum(1 for s in market_signals if s['signal'] == 'BEARISH')
            
            if bullish_market > bearish_market:
                environment['market_sentiment'] = 'bullish'
            elif bearish_market > bullish_market:
                environment['market_sentiment'] = 'bearish'
        
        # Determine overall regime
        if environment['growth_outlook'] == 'expansionary' and environment['inflation_outlook'] == 'stable':
            environment['overall_regime'] = 'goldilocks'
        elif environment['growth_outlook'] == 'contractionary':
            environment['overall_regime'] = 'recession'
        elif environment['inflation_outlook'] == 'rising':
            environment['overall_regime'] = 'stagflation'
        elif environment['monetary_policy'] == 'easing':
            environment['overall_regime'] = 'recovery'
        
        return environment
    
    def generate_asset_class_recommendations(self, environment: Dict[str, str], 
                                           signals: Dict) -> List[InvestmentRecommendation]:
        """Generate recommendations for each asset class"""
        
        recommendations = []
        
        # EQUITIES RECOMMENDATION
        equity_rec = self._analyze_equities(environment, signals)
        recommendations.append(equity_rec)
        
        # BONDS RECOMMENDATION
        bonds_rec = self._analyze_bonds(environment, signals)
        recommendations.append(bonds_rec)
        
        # COMMODITIES RECOMMENDATION
        commodities_rec = self._analyze_commodities(environment, signals)
        recommendations.append(commodities_rec)
        
        # CASH RECOMMENDATION
        cash_rec = self._analyze_cash(environment, signals)
        recommendations.append(cash_rec)
        
        return recommendations
    
    def _analyze_equities(self, environment: Dict, signals: Dict) -> InvestmentRecommendation:
        """Analyze equities allocation recommendation"""
        
        base_allocation = 60.0
        signal = InvestmentSignal.HOLD
        confidence = 0.5
        reasoning = "Neutral equity outlook"
        risk_level = "Medium"
        
        # Regime-based analysis
        if environment['overall_regime'] == 'goldilocks':
            signal = InvestmentSignal.BUY
            base_allocation = 70.0
            confidence = 0.8
            reasoning = "Goldilocks environment favorable for equities - growth with stable inflation"
            
        elif environment['overall_regime'] == 'recovery':
            signal = InvestmentSignal.BUY
            base_allocation = 65.0
            confidence = 0.7
            reasoning = "Recovery phase with monetary easing supports equity valuations"
            
        elif environment['overall_regime'] == 'recession':
            signal = InvestmentSignal.SELL
            base_allocation = 40.0
            confidence = 0.75
            reasoning = "Recessionary environment - reduce equity exposure"
            risk_level = "High"
            
        elif environment['overall_regime'] == 'stagflation':
            signal = InvestmentSignal.SELL
            base_allocation = 45.0
            confidence = 0.7
            reasoning = "Stagflation pressures equity margins and valuations"
            risk_level = "High"
        
        # Market sentiment overlay
        market_signals = [s for k, s in signals.items() if any(x in k.lower() for x in ['sp500', 'nasdaq'])]
        if market_signals:
            bullish_count = sum(1 for s in market_signals if s['signal'] == 'BULLISH')
            if bullish_count > len(market_signals) / 2:
                base_allocation += 5.0
                confidence = min(confidence + 0.1, 1.0)
            elif any(s['signal'] == 'BEARISH' for s in market_signals):
                base_allocation -= 5.0
                confidence = min(confidence + 0.1, 1.0)
        
        # Risk adjustment
        if self.risk_tolerance == "Low":
            base_allocation *= 0.8
        elif self.risk_tolerance == "High":
            base_allocation *= 1.2
        
        base_allocation = max(20.0, min(base_allocation, 80.0))  # Bounds
        
        return InvestmentRecommendation(
            asset_class=AssetClass.EQUITIES,
            signal=signal,
            confidence=confidence,
            target_allocation=base_allocation,
            time_horizon="90d",
            reasoning=reasoning,
            risk_level=risk_level,
            expected_return=8.0 if signal in [InvestmentSignal.BUY, InvestmentSignal.STRONG_BUY] else 5.0,
            max_drawdown=-15.0 if risk_level == "High" else -10.0
        )
    
    def _analyze_bonds(self, environment: Dict, signals: Dict) -> InvestmentRecommendation:
        """Analyze bonds allocation recommendation"""
        
        base_allocation = 30.0
        signal = InvestmentSignal.HOLD
        confidence = 0.5
        reasoning = "Neutral bond outlook"
        risk_level = "Low"
        
        # Interest rate environment analysis
        if environment['monetary_policy'] == 'tightening':
            signal = InvestmentSignal.SELL
            base_allocation = 20.0
            confidence = 0.75
            reasoning = "Rising rates environment pressures bond prices"
            risk_level = "Medium"
            
        elif environment['monetary_policy'] == 'easing':
            signal = InvestmentSignal.BUY
            base_allocation = 40.0
            confidence = 0.8
            reasoning = "Easing monetary policy supports bond prices"
            
        elif environment['inflation_outlook'] == 'rising':
            signal = InvestmentSignal.SELL
            base_allocation = 25.0
            confidence = 0.7
            reasoning = "Rising inflation erodes real bond returns"
            risk_level = "Medium"
        
        # Recession hedge
        if environment['overall_regime'] == 'recession':
            signal = InvestmentSignal.BUY
            base_allocation = 45.0
            confidence = 0.8
            reasoning = "Bonds provide defensive characteristics during recession"
        
        # Risk adjustment
        if self.risk_tolerance == "Low":
            base_allocation *= 1.2
        elif self.risk_tolerance == "High":
            base_allocation *= 0.8
        
        base_allocation = max(10.0, min(base_allocation, 60.0))  # Bounds
        
        return InvestmentRecommendation(
            asset_class=AssetClass.BONDS,
            signal=signal,
            confidence=confidence,
            target_allocation=base_allocation,
            time_horizon="90d",
            reasoning=reasoning,
            risk_level=risk_level,
            expected_return=4.0 if signal in [InvestmentSignal.BUY, InvestmentSignal.STRONG_BUY] else 2.5,
            max_drawdown=-5.0
        )
    
    def _analyze_commodities(self, environment: Dict, signals: Dict) -> InvestmentRecommendation:
        """Analyze commodities allocation recommendation"""
        
        base_allocation = 5.0
        signal = InvestmentSignal.HOLD
        confidence = 0.5
        reasoning = "Neutral commodities outlook"
        risk_level = "High"
        
        # Inflation hedge analysis
        if environment['inflation_outlook'] == 'rising':
            signal = InvestmentSignal.BUY
            base_allocation = 15.0
            confidence = 0.75
            reasoning = "Commodities provide inflation hedge"
            
        elif environment['overall_regime'] == 'stagflation':
            signal = InvestmentSignal.STRONG_BUY
            base_allocation = 20.0
            confidence = 0.8
            reasoning = "Stagflation environment highly favorable for commodities"
        
        # Growth environment
        elif environment['growth_outlook'] == 'expansionary':
            signal = InvestmentSignal.BUY
            base_allocation = 10.0
            confidence = 0.6
            reasoning = "Strong growth drives commodity demand"
        
        # Risk adjustment
        if self.risk_tolerance == "Low":
            base_allocation *= 0.5
        elif self.risk_tolerance == "High":
            base_allocation *= 1.5
        
        base_allocation = max(0.0, min(base_allocation, 25.0))  # Bounds
        
        return InvestmentRecommendation(
            asset_class=AssetClass.COMMODITIES,
            signal=signal,
            confidence=confidence,
            target_allocation=base_allocation,
            time_horizon="90d",
            reasoning=reasoning,
            risk_level=risk_level,
            expected_return=12.0 if signal in [InvestmentSignal.BUY, InvestmentSignal.STRONG_BUY] else 6.0,
            max_drawdown=-25.0
        )
    
    def _analyze_cash(self, environment: Dict, signals: Dict) -> InvestmentRecommendation:
        """Analyze cash allocation recommendation"""
        
        base_allocation = 5.0
        signal = InvestmentSignal.HOLD
        confidence = 0.5
        reasoning = "Standard cash allocation for liquidity"
        risk_level = "Low"
        
        # High uncertainty environments
        if environment['overall_regime'] == 'recession':
            signal = InvestmentSignal.BUY
            base_allocation = 15.0
            confidence = 0.8
            reasoning = "Increased cash allocation for defensive positioning and opportunities"
            
        elif environment['monetary_policy'] == 'tightening':
            signal = InvestmentSignal.BUY
            base_allocation = 10.0
            confidence = 0.7
            reasoning = "Rising rates make cash more attractive"
        
        # High growth, low volatility
        elif environment['overall_regime'] == 'goldilocks':
            base_allocation = 3.0
            reasoning = "Minimize cash drag in favorable growth environment"
        
        base_allocation = max(3.0, min(base_allocation, 20.0))  # Bounds
        
        return InvestmentRecommendation(
            asset_class=AssetClass.CASH,
            signal=signal,
            confidence=confidence,
            target_allocation=base_allocation,
            time_horizon="30d",
            reasoning=reasoning,
            risk_level=risk_level,
            expected_return=environment.get('fed_rate', 4.0) if signal == InvestmentSignal.BUY else 3.0,
            max_drawdown=0.0
        )
    
    def generate_portfolio_recommendation(self, forecasts: Dict, signals: Dict) -> Dict:
        """Generate comprehensive portfolio recommendation"""
        
        # Analyze economic environment
        environment = self.analyze_economic_environment(forecasts, signals)
        
        # Generate asset class recommendations
        recommendations = self.generate_asset_class_recommendations(environment, signals)
        
        # Normalize allocations to 100%
        total_allocation = sum(rec.target_allocation for rec in recommendations)
        for rec in recommendations:
            rec.target_allocation = (rec.target_allocation / total_allocation) * 100
        
        # Calculate portfolio metrics
        expected_return = sum(rec.target_allocation * rec.expected_return / 100 
                            for rec in recommendations if rec.expected_return)
        
        max_drawdown = max(rec.max_drawdown for rec in recommendations if rec.max_drawdown)
        
        # Generate rebalancing actions
        rebalancing_actions = self._generate_rebalancing_actions(recommendations)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'economic_environment': environment,
            'recommendations': recommendations,
            'portfolio_metrics': {
                'expected_annual_return': expected_return,
                'estimated_max_drawdown': max_drawdown,
                'risk_level': self._calculate_portfolio_risk(recommendations),
                'diversification_score': self._calculate_diversification(recommendations)
            },
            'rebalancing_actions': rebalancing_actions,
            'key_insights': self._generate_key_insights(environment, recommendations)
        }
    
    def _generate_rebalancing_actions(self, recommendations: List[InvestmentRecommendation]) -> List[Dict]:
        """Generate specific rebalancing actions"""
        actions = []
        
        for rec in recommendations:
            current_alloc = self.current_allocations.get(rec.asset_class, 0.0)
            target_alloc = rec.target_allocation
            difference = target_alloc - current_alloc
            
            if abs(difference) > 2.0:  # Only suggest changes > 2%
                action = "INCREASE" if difference > 0 else "DECREASE"
                actions.append({
                    'asset_class': rec.asset_class.value,
                    'action': action,
                    'current_allocation': current_alloc,
                    'target_allocation': target_alloc,
                    'change_percent': difference,
                    'priority': 'HIGH' if abs(difference) > 10 else 'MEDIUM' if abs(difference) > 5 else 'LOW',
                    'reasoning': rec.reasoning
                })
        
        return sorted(actions, key=lambda x: abs(x['change_percent']), reverse=True)
    
    def _calculate_portfolio_risk(self, recommendations: List[InvestmentRecommendation]) -> str:
        """Calculate overall portfolio risk level"""
        risk_scores = {'Low': 1, 'Medium': 2, 'High': 3}
        
        weighted_risk = sum(
            rec.target_allocation * risk_scores[rec.risk_level] / 100
            for rec in recommendations
        )
        
        if weighted_risk < 1.5:
            return 'Low'
        elif weighted_risk < 2.5:
            return 'Medium'
        else:
            return 'High'
    
    def _calculate_diversification(self, recommendations: List[InvestmentRecommendation]) -> float:
        """Calculate diversification score (0-100)"""
        allocations = [rec.target_allocation for rec in recommendations]
        
        # Higher diversification when allocations are more evenly distributed
        max_allocation = max(allocations)
        
        if max_allocation > 80:
            return 20.0
        elif max_allocation > 70:
            return 40.0
        elif max_allocation > 60:
            return 60.0
        elif max_allocation > 50:
            return 80.0
        else:
            return 100.0
    
    def _generate_key_insights(self, environment: Dict, recommendations: List[InvestmentRecommendation]) -> List[str]:
        """Generate key investment insights"""
        insights = []
        
        # Environment-based insights
        if environment['overall_regime'] == 'goldilocks':
            insights.append("🌟 Goldilocks environment detected - favorable for growth assets")
        elif environment['overall_regime'] == 'recession':
            insights.append("⚠️ Recessionary environment - focus on defensive positioning")
        elif environment['overall_regime'] == 'stagflation':
            insights.append("📈 Stagflation risks - consider inflation hedges")
        
        # Asset class insights
        equity_rec = next(r for r in recommendations if r.asset_class == AssetClass.EQUITIES)
        if equity_rec.signal in [InvestmentSignal.BUY, InvestmentSignal.STRONG_BUY]:
            insights.append(f"📊 Bullish equity outlook - {equity_rec.reasoning}")
        elif equity_rec.signal in [InvestmentSignal.SELL, InvestmentSignal.STRONG_SELL]:
            insights.append(f"📉 Bearish equity outlook - {equity_rec.reasoning}")
        
        # Risk insights
        high_risk_assets = [r for r in recommendations if r.risk_level == 'High' and r.target_allocation > 10]
        if high_risk_assets:
            insights.append(f"⚡ Higher risk positioning due to {', '.join(r.asset_class.value for r in high_risk_assets)}")
        
        return insights

def main():
    """Example usage of the Investment Decision Engine"""
    
    # Example forecast signals
    example_signals = {
        'gdp_growth': {'signal': 'BULLISH', 'confidence': 0.7, 'reasoning': 'Strong growth expected'},
        'unemployment_rate': {'signal': 'NEUTRAL', 'confidence': 0.5, 'reasoning': 'Stable employment'},
        'cpi_inflation': {'signal': 'BEARISH', 'confidence': 0.8, 'reasoning': 'High inflation expected'},
        'sp500_close': {'signal': 'BULLISH', 'confidence': 0.6, 'reasoning': 'Market momentum'}
    }
    
    # Initialize engine
    engine = InvestmentDecisionEngine()
    
    # Generate recommendations
    portfolio_rec = engine.generate_portfolio_recommendation({}, example_signals)
    
    logger.info("Generated portfolio recommendation")
    for rec in portfolio_rec['recommendations']:
        logger.info(f"{rec.asset_class.value}: {rec.target_allocation:.1f}% ({rec.signal.value})")

if __name__ == "__main__":
    main() 