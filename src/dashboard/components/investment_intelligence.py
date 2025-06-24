"""
Investment Intelligence Dashboard Component
Combines economic forecasting with actionable investment recommendations
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from src.models.forecasting.prophet_models import EconomicProphetForecaster
    from src.models.investment_decision import InvestmentDecisionEngine, AssetClass, InvestmentSignal
except ImportError as e:
    st.error(f"Import error: {e}. Please ensure all forecasting modules are available.")
    EconomicProphetForecaster = None
    InvestmentDecisionEngine = None

def render_investment_intelligence(datasets: Dict[str, pd.DataFrame], use_real_data: bool = True):
    """Render the investment intelligence dashboard"""
    
    st.title("🎯 Investment Intelligence")
    st.markdown("**Transform economic forecasts into actionable investment decisions**")
    
    if not EconomicProphetForecaster or not InvestmentDecisionEngine:
        st.error("⚠️ Forecasting modules not available. Please install Prophet and other dependencies.")
        return
    
    # Create tabs for different analysis views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Economic Forecasts", 
        "💼 Portfolio Recommendations", 
        "📊 Asset Allocation", 
        "🔍 Market Intelligence"
    ])
    
    with tab1:
        render_economic_forecasts(datasets)
    
    with tab2:
        render_portfolio_recommendations(datasets)
    
    with tab3:
        render_asset_allocation(datasets)
    
    with tab4:
        render_market_intelligence(datasets)

def render_economic_forecasts(datasets: Dict[str, pd.DataFrame]):
    """Render economic forecasting section"""
    
    st.header("📈 Economic Forecasting Engine")
    
    # Forecast configuration
    col1, col2, col3 = st.columns(3)
    with col1:
        forecast_horizon = st.selectbox(
            "Forecast Horizon",
            [30, 60, 90, 180, 365],
            index=2,
            help="Number of days to forecast"
        )
    
    with col2:
        confidence_level = st.selectbox(
            "Confidence Level",
            [80, 90, 95],
            index=1,
            help="Confidence interval for forecasts"
        )
    
    with col3:
        key_indicators = st.multiselect(
            "Key Indicators",
            ["GDP Growth", "Unemployment", "CPI Inflation", "Fed Funds Rate", "S&P 500"],
            default=["GDP Growth", "CPI Inflation", "S&P 500"],
            help="Select indicators to forecast"
        )
    
    if st.button("🚀 Generate Forecasts", key="generate_forecasts"):
        with st.spinner("Generating economic forecasts..."):
            forecasts_data = generate_forecasts(datasets, forecast_horizon, key_indicators)
            
            if forecasts_data:
                st.success("✅ Forecasts generated successfully!")
                
                # Display forecast charts
                for indicator, forecast_info in forecasts_data.items():
                    if 'forecast' in forecast_info:
                        st.subheader(f"📊 {indicator.replace('_', ' ').title()} Forecast")
                        
                        fig = create_forecast_chart(
                            forecast_info['forecast'], 
                            indicator, 
                            confidence_level
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Summary metrics
                        if 'summary' in forecast_info:
                            summary = forecast_info['summary']
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric(
                                    "30-Day Trend",
                                    summary.get('next_30_days', {}).get('trend', 'N/A').title(),
                                    delta=f"{summary.get('next_30_days', {}).get('mean', 0):.2f}"
                                )
                            
                            with col2:
                                st.metric(
                                    "90-Day Avg",
                                    f"{summary.get('next_90_days', {}).get('mean', 0):.2f}",
                                    delta=f"{summary.get('predicted_trend', 0):.2f}"
                                )
                            
                            with col3:
                                confidence_width = summary.get('confidence_interval_width', 0)
                                st.metric(
                                    "Uncertainty",
                                    f"±{confidence_width:.2f}",
                                    delta="Low" if confidence_width < 1 else "High"
                                )
                            
                            with col4:
                                trend_direction = "📈" if summary.get('predicted_trend', 0) > 0 else "📉"
                                st.metric(
                                    "Overall Trend",
                                    trend_direction,
                                    delta=summary.get('next_90_days', {}).get('trend', 'neutral').title()
                                )
            else:
                st.warning("⚠️ Unable to generate forecasts. Please check data availability.")

def render_portfolio_recommendations(datasets: Dict[str, pd.DataFrame]):
    """Render portfolio recommendations section"""
    
    st.header("💼 AI-Powered Portfolio Recommendations")
    
    # Investment profile configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_tolerance = st.selectbox(
            "Risk Tolerance",
            ["Low", "Medium", "High"],
            index=1,
            help="Your risk tolerance level"
        )
    
    with col2:
        investment_horizon = st.selectbox(
            "Investment Horizon",
            ["Short (< 1 year)", "Medium (1-3 years)", "Long (3+ years)"],
            index=1,
            help="Your investment time horizon"
        )
    
    with col3:
        portfolio_size = st.number_input(
            "Portfolio Size ($)",
            min_value=10000,
            max_value=10000000,
            value=100000,
            step=10000,
            help="Your total investment portfolio size"
        )
    
    if st.button("🎯 Generate Investment Recommendations", key="generate_investment"):
        with st.spinner("Analyzing economic environment and generating recommendations..."):
            
            # Generate forecasts first
            forecasts_data = generate_forecasts(datasets, 90, ["GDP Growth", "Unemployment", "CPI Inflation", "S&P 500"])
            
            if forecasts_data:
                # Generate investment signals
                signals = extract_investment_signals(forecasts_data)
                
                # Initialize investment engine
                engine = InvestmentDecisionEngine()
                engine.risk_tolerance = risk_tolerance
                engine.investment_horizon = investment_horizon.split()[0]
                
                # Generate recommendations
                portfolio_rec = engine.generate_portfolio_recommendation(forecasts_data, signals)
                
                # Display results
                st.success("✅ Investment recommendations generated!")
                
                # Economic Environment Analysis
                st.subheader("🌍 Economic Environment Analysis")
                environment = portfolio_rec['economic_environment']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    regime_color = get_regime_color(environment['overall_regime'])
                    st.markdown(f"**Overall Regime:** <span style='color: {regime_color}'>{environment['overall_regime'].title()}</span>", unsafe_allow_html=True)
                
                with col2:
                    growth_color = get_sentiment_color(environment['growth_outlook'])
                    st.markdown(f"**Growth Outlook:** <span style='color: {growth_color}'>{environment['growth_outlook'].title()}</span>", unsafe_allow_html=True)
                
                with col3:
                    inflation_color = get_sentiment_color(environment['inflation_outlook'])
                    st.markdown(f"**Inflation Outlook:** <span style='color: {inflation_color}'>{environment['inflation_outlook'].title()}</span>", unsafe_allow_html=True)
                
                with col4:
                    policy_color = get_sentiment_color(environment['monetary_policy'])
                    st.markdown(f"**Monetary Policy:** <span style='color: {policy_color}'>{environment['monetary_policy'].title()}</span>", unsafe_allow_html=True)
                
                # Portfolio Metrics
                st.subheader("📊 Portfolio Metrics")
                metrics = portfolio_rec['portfolio_metrics']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "Expected Return",
                        f"{metrics['expected_annual_return']:.1f}%",
                        delta="Annual"
                    )
                
                with col2:
                    st.metric(
                        "Risk Level",
                        metrics['risk_level'],
                        delta=f"Max DD: {metrics['estimated_max_drawdown']:.1f}%"
                    )
                
                with col3:
                    st.metric(
                        "Diversification",
                        f"{metrics['diversification_score']:.0f}/100",
                        delta="Score"
                    )
                
                with col4:
                    expected_value = portfolio_size * metrics['expected_annual_return'] / 100
                    st.metric(
                        "Expected Gain",
                        f"${expected_value:,.0f}",
                        delta="Annual"
                    )
                
                # Asset Class Recommendations
                st.subheader("💎 Asset Class Recommendations")
                
                recommendations = portfolio_rec['recommendations']
                
                # Create recommendation table
                rec_data = []
                for rec in recommendations:
                    rec_data.append({
                        'Asset Class': rec.asset_class.value.title(),
                        'Signal': get_signal_emoji(rec.signal) + " " + rec.signal.value,
                        'Target %': f"{rec.target_allocation:.1f}%",
                        'Confidence': f"{rec.confidence*100:.0f}%",
                        'Expected Return': f"{rec.expected_return:.1f}%" if rec.expected_return else "N/A",
                        'Risk Level': rec.risk_level,
                        'Reasoning': rec.reasoning
                    })
                
                rec_df = pd.DataFrame(rec_data)
                st.dataframe(rec_df, use_container_width=True)
                
                # Key Insights
                st.subheader("🔍 Key Investment Insights")
                insights = portfolio_rec['key_insights']
                
                for insight in insights:
                    st.markdown(f"• {insight}")
                
                # Rebalancing Actions
                if portfolio_rec['rebalancing_actions']:
                    st.subheader("⚖️ Rebalancing Actions")
                    
                    for action in portfolio_rec['rebalancing_actions']:
                        priority_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[action['priority']]
                        
                        st.markdown(f"""
                        {priority_color} **{action['action']} {action['asset_class'].title()}**: 
                        {action['current_allocation']:.1f}% → {action['target_allocation']:.1f}% 
                        ({action['change_percent']:+.1f}%)
                        
                        *{action['reasoning']}*
                        """)

def render_asset_allocation(datasets: Dict[str, pd.DataFrame]):
    """Render asset allocation visualization"""
    
    st.header("📊 Dynamic Asset Allocation")
    
    # Generate sample recommendations for visualization
    with st.spinner("Generating dynamic asset allocation..."):
        
        # Create sample allocation data over time
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='M')
        
        # Simulate different market regimes
        allocations_data = []
        
        for i, date in enumerate(dates):
            # Simulate market regime changes
            if i < 12:  # 2020 - Crisis
                equities, bonds, commodities, cash = 45, 35, 5, 15
            elif i < 24:  # 2021 - Recovery
                equities, bonds, commodities, cash = 70, 20, 5, 5
            elif i < 36:  # 2022 - Inflation
                equities, bonds, commodities, cash = 55, 25, 15, 5
            elif i < 48:  # 2023 - Normalization
                equities, bonds, commodities, cash = 65, 25, 5, 5
            else:  # 2024 - Current
                equities, bonds, commodities, cash = 60, 30, 5, 5
            
            allocations_data.append({
                'Date': date,
                'Equities': equities,
                'Bonds': bonds,
                'Commodities': commodities,
                'Cash': cash
            })
        
        allocation_df = pd.DataFrame(allocations_data)
        
        # Create stacked area chart
        fig = go.Figure()
        
        asset_classes = ['Equities', 'Bonds', 'Commodities', 'Cash']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, asset in enumerate(asset_classes):
            fig.add_trace(go.Scatter(
                x=allocation_df['Date'],
                y=allocation_df[asset],
                mode='lines',
                stackgroup='one',
                name=asset,
                fill='tonexty' if i > 0 else 'tozeroy',
                line=dict(color=colors[i])
            ))
        
        fig.update_layout(
            title="Dynamic Asset Allocation Over Time",
            xaxis_title="Date",
            yaxis_title="Allocation (%)",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Current allocation pie chart
        col1, col2 = st.columns(2)
        
        with col1:
            current_allocation = allocation_df.iloc[-1]
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=asset_classes,
                values=[current_allocation[asset] for asset in asset_classes],
                hole=0.3,
                marker_colors=colors
            )])
            
            fig_pie.update_layout(
                title="Current Recommended Allocation",
                height=400
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("📈 Performance Attribution")
            
            # Sample performance data
            performance_data = {
                'Asset Class': asset_classes,
                'YTD Return': ['12.5%', '2.1%', '8.3%', '4.2%'],
                'Contribution': ['7.5%', '0.6%', '0.4%', '0.2%'],
                'Weight': ['60%', '30%', '5%', '5%']
            }
            
            perf_df = pd.DataFrame(performance_data)
            st.dataframe(perf_df, use_container_width=True)
            
            st.metric(
                "Total Portfolio Return",
                "8.7%",
                delta="2.3% vs Benchmark"
            )

def render_market_intelligence(datasets: Dict[str, pd.DataFrame]):
    """Render market intelligence and signals"""
    
    st.header("🔍 Market Intelligence Dashboard")
    
    # Market regime analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Market Regime Analysis")
        
        # Sample regime data
        regime_probs = {
            'Bull Market': 0.65,
            'Bear Market': 0.15,
            'Sideways': 0.20
        }
        
        fig_regime = go.Figure(data=[go.Bar(
            x=list(regime_probs.keys()),
            y=list(regime_probs.values()),
            marker_color=['green', 'red', 'orange']
        )])
        
        fig_regime.update_layout(
            title="Market Regime Probabilities",
            yaxis_title="Probability",
            height=300
        )
        
        st.plotly_chart(fig_regime, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Investment Signals")
        
        # Sample signals
        signals_data = {
            'Indicator': ['GDP Growth', 'Unemployment', 'Inflation', 'Fed Policy', 'Market Momentum'],
            'Signal': ['🟢 Bullish', '🟡 Neutral', '🔴 Bearish', '🟡 Neutral', '🟢 Bullish'],
            'Confidence': ['85%', '60%', '90%', '70%', '75%']
        }
        
        signals_df = pd.DataFrame(signals_data)
        st.dataframe(signals_df, use_container_width=True)
    
    # Economic surprise index
    st.subheader("📈 Economic Surprise Index")
    
    # Generate sample surprise data
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    surprise_data = np.random.normal(0, 20, len(dates))
    surprise_df = pd.DataFrame({
        'Date': dates,
        'Surprise_Index': surprise_data.cumsum()
    })
    
    fig_surprise = px.line(
        surprise_df, 
        x='Date', 
        y='Surprise_Index',
        title="Economic Data Surprise Index"
    )
    
    fig_surprise.add_hline(y=0, line_dash="dash", line_color="red")
    fig_surprise.update_layout(height=400)
    
    st.plotly_chart(fig_surprise, use_container_width=True)
    
    # Key market insights
    st.subheader("💡 Key Market Insights")
    
    insights = [
        "🔥 **Strong Employment Data**: Labor market showing robust growth, supporting consumer spending",
        "⚠️ **Inflation Concerns**: CPI trending above Fed target, potential for policy tightening",
        "📈 **Equity Momentum**: Technical indicators suggest continued upward momentum in major indices",
        "🏠 **Housing Cooling**: Rising mortgage rates beginning to impact housing market activity",
        "💰 **Dollar Strength**: USD showing strength against major currencies, impacting exports"
    ]
    
    for insight in insights:
        st.markdown(insight)

def generate_forecasts(datasets: Dict[str, pd.DataFrame], horizon: int, indicators: List[str]) -> Dict:
    """Generate forecasts for selected indicators"""
    
    if not datasets or 'primary_indicators' not in datasets:
        return {}
    
    try:
        forecaster = EconomicProphetForecaster()
        forecasts = {}
        
        primary_data = datasets['primary_indicators']
        
        # Map indicator names to data columns
        indicator_mapping = {
            "GDP Growth": "gdp_growth",
            "Unemployment": "unemployment_rate", 
            "CPI Inflation": "cpi_inflation",
            "Fed Funds Rate": "federal_funds_rate",
            "S&P 500": "sp500"
        }
        
        for indicator in indicators:
            if indicator in indicator_mapping:
                col_name = indicator_mapping[indicator]
                
                # Check if column exists and has data
                if col_name in primary_data.columns and not primary_data[col_name].isna().all():
                    
                    # Prepare data for Prophet
                    forecast_data = primary_data[['date', col_name]].dropna()
                    
                    if len(forecast_data) > 30:  # Minimum data points needed
                        try:
                            prophet_data = forecaster.prepare_data(forecast_data, 'date', col_name)
                            model = forecaster.train_model(prophet_data, col_name)
                            forecast = forecaster.generate_forecast(col_name, periods=horizon)
                            summary = forecaster.get_forecast_summary(col_name)
                            
                            forecasts[col_name] = {
                                'forecast': forecast,
                                'summary': summary,
                                'model': model
                            }
                        except Exception as e:
                            st.warning(f"Could not generate forecast for {indicator}: {str(e)}")
        
        return forecasts
        
    except Exception as e:
        st.error(f"Error generating forecasts: {str(e)}")
        return {}

def extract_investment_signals(forecasts_data: Dict) -> Dict:
    """Extract investment signals from forecast data"""
    
    if not forecasts_data:
        return {}
    
    try:
        forecaster = EconomicProphetForecaster()
        signals = forecaster.get_investment_signals(forecasts_data)
        return signals
    except Exception as e:
        st.warning(f"Could not extract investment signals: {str(e)}")
        return {}

def create_forecast_chart(forecast_df: pd.DataFrame, indicator: str, confidence: int) -> go.Figure:
    """Create forecast visualization chart"""
    
    fig = go.Figure()
    
    # Historical data
    historical_mask = forecast_df['ds'] <= forecast_df['ds'].iloc[-365] if len(forecast_df) > 365 else forecast_df['ds'] <= forecast_df['ds'].max()
    future_mask = ~historical_mask
    
    # Add historical line
    fig.add_trace(go.Scatter(
        x=forecast_df[historical_mask]['ds'],
        y=forecast_df[historical_mask]['yhat'],
        mode='lines',
        name='Historical',
        line=dict(color='blue')
    ))
    
    # Add forecast line
    fig.add_trace(go.Scatter(
        x=forecast_df[future_mask]['ds'],
        y=forecast_df[future_mask]['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='red', dash='dash')
    ))
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=forecast_df[future_mask]['ds'],
        y=forecast_df[future_mask]['yhat_upper'],
        fill=None,
        mode='lines',
        line_color='rgba(0,0,0,0)',
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_df[future_mask]['ds'],
        y=forecast_df[future_mask]['yhat_lower'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,0,0,0)',
        name=f'{confidence}% Confidence',
        fillcolor='rgba(255,0,0,0.2)'
    ))
    
    fig.update_layout(
        title=f"{indicator.replace('_', ' ').title()} Forecast",
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode='x unified',
        height=400
    )
    
    return fig

def get_regime_color(regime: str) -> str:
    """Get color for economic regime"""
    colors = {
        'goldilocks': '#28a745',
        'recovery': '#17a2b8', 
        'recession': '#dc3545',
        'stagflation': '#fd7e14',
        'neutral': '#6c757d'
    }
    return colors.get(regime, '#6c757d')

def get_sentiment_color(sentiment: str) -> str:
    """Get color for sentiment indicators"""
    colors = {
        'bullish': '#28a745',
        'expansionary': '#28a745',
        'easing': '#28a745',
        'stable': '#ffc107',
        'neutral': '#6c757d',
        'bearish': '#dc3545',
        'contractionary': '#dc3545',
        'tightening': '#dc3545',
        'rising': '#fd7e14'
    }
    return colors.get(sentiment, '#6c757d')

def get_signal_emoji(signal) -> str:
    """Get emoji for investment signal"""
    if hasattr(signal, 'value'):
        signal_val = signal.value
    else:
        signal_val = str(signal)
    
    emojis = {
        'STRONG_BUY': '🚀',
        'BUY': '📈',
        'HOLD': '➡️',
        'SELL': '📉',
        'STRONG_SELL': '💥'
    }
    return emojis.get(signal_val, '➡️') 