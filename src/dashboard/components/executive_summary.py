import streamlit as st
import pandas as pd
import numpy as np
from src.dashboard.utils.plotting import plot_time_series, plot_gauge

def safe_get_value(data, column, default=0.0):
    """
    Safely extract a value from data with fallback.
    """
    try:
        if column in data.index and pd.notna(data[column]):
            return float(data[column])
        return default
    except (KeyError, ValueError, TypeError):
        return default

def calculate_change_safely(current, previous):
    """
    Safely calculate change between two values.
    """
    try:
        if pd.notna(current) and pd.notna(previous) and previous != 0:
            return current - previous
        return 0.0
    except (TypeError, ValueError):
        return 0.0

def calculate_percentage_change(current, previous):
    """
    Safely calculate percentage change.
    """
    try:
        if pd.notna(current) and pd.notna(previous) and previous != 0:
            return ((current - previous) / previous) * 100
        return 0.0
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0

def display_executive_summary(data: dict[str, pd.DataFrame]):
    """
    Renders the Executive Summary page with professional styling and robust error handling.

    Args:
        data: A dictionary of DataFrames, including 'primary_indicators'.
    """
    # Enhanced data validation
    primary_indicators = data.get("primary_indicators")
    if primary_indicators is None or primary_indicators.empty:
        st.error("⚠️ **Data Unavailable** - Primary economic indicators could not be loaded.")
        st.info("Please check your data connection or switch to synthetic data mode.")
        
        # Show available datasets for debugging
        if data:
            available_datasets = list(data.keys())
            st.info(f"Available datasets: {', '.join(available_datasets)}")
        return

    # Robust data extraction with error handling
    try:
        # Ensure we have at least one row of data
        if len(primary_indicators) == 0:
            st.error("No data rows available in primary indicators.")
            return
            
        # Get the most recent data points with fallbacks
        latest_data = primary_indicators.iloc[-1]
        previous_data = primary_indicators.iloc[-2] if len(primary_indicators) > 1 else latest_data
        
        # Validate that we have the required columns
        required_columns = ['gdp_growth', 'unemployment_rate', 'cpi_inflation']
        missing_columns = [col for col in required_columns if col not in primary_indicators.columns]
        
        if missing_columns:
            st.warning(f"⚠️ Missing columns: {', '.join(missing_columns)}. Using available data.")
            
    except Exception as e:
        st.error(f"Error processing primary indicators data: {e}")
        return

    # Professional header with context
    st.markdown("""
    <div class="metric-card">
        <h3>📈 Current Economic Snapshot</h3>
        <p>Real-time analysis of key macroeconomic indicators and their recent performance trends.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Enhanced key indicators section with robust error handling
    st.markdown("### 🎯 Key Performance Indicators")
    
    # Create metrics with better styling and error handling
    col1, col2, col3, col4 = st.columns(4)
    
    # Safely extract values with fallbacks
    gdp_growth = safe_get_value(latest_data, 'gdp_growth', 0.0)
    prev_gdp_growth = safe_get_value(previous_data, 'gdp_growth', gdp_growth)
    unemployment = safe_get_value(latest_data, 'unemployment_rate', 0.0)
    prev_unemployment = safe_get_value(previous_data, 'unemployment_rate', unemployment)
    inflation = safe_get_value(latest_data, 'cpi_inflation', 0.0)
    prev_inflation = safe_get_value(previous_data, 'cpi_inflation', inflation)
    
    with col1:
        change = calculate_change_safely(gdp_growth, prev_gdp_growth)
        try:
            st.metric(
                label="📊 GDP Growth",
                value=f"{gdp_growth:.2f}%" if not np.isnan(gdp_growth) else "N/A",
                delta=f"{change:+.2f}%" if not np.isnan(change) and change != 0 else None,
                help="Quarterly GDP growth rate"
            )
        except Exception:
            st.metric("📊 GDP Growth", "N/A", help="Data unavailable")

    with col2:
        change = calculate_change_safely(unemployment, prev_unemployment)
        try:
            st.metric(
                label="👥 Unemployment",
                value=f"{unemployment:.1f}%" if not np.isnan(unemployment) else "N/A",
                delta=f"{change:+.1f}%" if not np.isnan(change) and change != 0 else None,
                delta_color="inverse",
                help="Current unemployment rate"
            )
        except Exception:
            st.metric("👥 Unemployment", "N/A", help="Data unavailable")

    with col3:
        change = calculate_change_safely(inflation, prev_inflation)
        try:
            st.metric(
                label="💰 CPI Inflation",
                value=f"{inflation:.1f}%" if not np.isnan(inflation) else "N/A",
                delta=f"{change:+.1f}%" if not np.isnan(change) and change != 0 else None,
                help="Consumer Price Index inflation rate"
            )
        except Exception:
            st.metric("💰 CPI Inflation", "N/A", help="Data unavailable")
    
    with col4:
        # Calculate economic health score with robust error handling
        try:
            if not any(np.isnan([gdp_growth, unemployment, inflation])):
                # Improved health score calculation
                health_score = (
                    50 +  # Base score
                    (gdp_growth * 8) +  # GDP contribution (positive is good)
                    (-unemployment * 3) +  # Unemployment contribution (lower is better)
                    (-abs(inflation - 2.0) * 4)  # Inflation target around 2%
                )
                health_score = max(0, min(100, health_score))
                
                if health_score >= 75:
                    health_status = "🟢 Excellent"
                    health_color = "#28a745"
                elif health_score >= 60:
                    health_status = "🟡 Good"
                    health_color = "#ffc107"
                elif health_score >= 40:
                    health_status = "🟠 Fair"
                    health_color = "#fd7e14"
                else:
                    health_status = "🔴 Poor"
                    health_color = "#dc3545"
                    
                st.metric(
                    label="🏥 Economic Health",
                    value=f"{health_score:.0f}/100",
                    help="Composite economic health indicator based on GDP, unemployment, and inflation"
                )
                st.markdown(f'<p style="color: {health_color}; font-weight: bold; margin-top: -10px;">{health_status}</p>', 
                           unsafe_allow_html=True)
            else:
                st.metric("🏥 Economic Health", "N/A", help="Insufficient data for calculation")
        except Exception as e:
            st.metric("🏥 Economic Health", "Error", help=f"Calculation error: {e}")

    st.markdown("---")

    # Enhanced gauge charts section with error handling
    st.markdown("### 📊 Performance Gauges")
    col1, col2, col3 = st.columns(3)

    with col1:
        try:
            if not np.isnan(gdp_growth):
                fig = plot_gauge(gdp_growth, "GDP Growth (%)", prev_gdp_growth)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            else:
                st.info("📊 GDP Growth data unavailable")
        except Exception as e:
            st.error(f"Error displaying GDP gauge: {e}")

    with col2:
        try:
            if not np.isnan(unemployment):
                fig = plot_gauge(unemployment, "Unemployment Rate (%)", prev_unemployment)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            else:
                st.info("👥 Unemployment data unavailable")
        except Exception as e:
            st.error(f"Error displaying unemployment gauge: {e}")

    with col3:
        try:
            if not np.isnan(inflation):
                fig = plot_gauge(inflation, "CPI Inflation (%)", prev_inflation)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            else:
                st.info("💰 Inflation data unavailable")
        except Exception as e:
            st.error(f"Error displaying inflation gauge: {e}")

    st.markdown("---")

    # Enhanced trends section with robust error handling
    st.markdown("### 📈 Historical Trends & Analysis")
    
    # Add trend analysis with error handling
    col1, col2 = st.columns([3, 1])
    
    with col1:
        try:
            # Check which columns are available
            available_columns = [col for col in ['gdp_growth', 'unemployment_rate', 'cpi_inflation'] 
                               if col in primary_indicators.columns]
            
            if available_columns:
                fig = plot_time_series(
                    primary_indicators,
                    y_columns=available_columns,
                    title="Primary Economic Indicators - Historical Performance",
                    y_axis_title="Percentage (%)"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ No suitable columns found for time series plot")
                st.info(f"Available columns: {list(primary_indicators.columns)}")
        except Exception as e:
            st.error(f"Error creating time series plot: {e}")
            # Fallback: show raw data table
            st.subheader("📊 Recent Data (Last 10 Records)")
            st.dataframe(primary_indicators.tail(10))
    
    with col2:
        try:
            # Safe trend analysis
            gdp_trend = "Stable"
            employment_trend = "Stable"
            inflation_trend = "Stable"
            
            if not np.isnan(gdp_growth) and not np.isnan(prev_gdp_growth):
                if gdp_growth > prev_gdp_growth + 0.1:
                    gdp_trend = "📈 Expanding"
                elif gdp_growth < prev_gdp_growth - 0.1:
                    gdp_trend = "📉 Contracting"
                else:
                    gdp_trend = "➡️ Stable"
            
            if not np.isnan(unemployment) and not np.isnan(prev_unemployment):
                if unemployment < prev_unemployment - 0.1:
                    employment_trend = "📈 Improving"
                elif unemployment > prev_unemployment + 0.1:
                    employment_trend = "📉 Deteriorating"
                else:
                    employment_trend = "➡️ Stable"
            
            if not np.isnan(inflation) and not np.isnan(prev_inflation):
                if inflation > prev_inflation + 0.1:
                    inflation_trend = "📈 Rising"
                elif inflation < prev_inflation - 0.1:
                    inflation_trend = "📉 Falling"
                else:
                    inflation_trend = "➡️ Stable"
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>📋 Trend Analysis</h4>
                <div style="font-size: 0.9em; line-height: 1.8;">
                    <p><strong>GDP Growth:</strong><br>{gdp_trend}</p>
                    <p><strong>Employment:</strong><br>{employment_trend}</p>
                    <p><strong>Inflation:</strong><br>{inflation_trend}</p>
                </div>
                <hr style="margin: 10px 0;">
                <small><em>Trends based on recent period comparison</em></small>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error in trend analysis: {e}")

    # Enhanced market snapshot section with comprehensive error handling
    st.markdown("### 💹 Financial Markets Overview")
    
    financial_markets = data.get("financial_markets")
    if financial_markets is not None and not financial_markets.empty:
        try:
            latest_market_data = financial_markets.iloc[-1]
            previous_market_data = financial_markets.iloc[-2] if len(financial_markets) > 1 else latest_market_data
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Enhanced S&P500 detection with more robust error handling
                sp500_col = None
                sp500_candidates = ["sp500_price", "close", "adj_close", "S&P500", "value", "price", "SP500"]
                
                for candidate in sp500_candidates:
                    if candidate in latest_market_data.index and pd.notna(latest_market_data[candidate]):
                        sp500_col = candidate
                        break
                
                if sp500_col is not None:
                    try:
                        current_sp500 = safe_get_value(latest_market_data, sp500_col, 0)
                        prev_sp500 = safe_get_value(previous_market_data, sp500_col, current_sp500)
                        
                        if current_sp500 > 0 and prev_sp500 > 0:
                            change = current_sp500 - prev_sp500
                            change_pct = calculate_percentage_change(current_sp500, prev_sp500)
                            
                            st.metric(
                                label="📈 S&P 500 Index",
                                value=f"{current_sp500:,.0f}",
                                delta=f"{change:+.0f} ({change_pct:+.1f}%)" if change != 0 else None,
                                help="Standard & Poor's 500 stock market index"
                            )
                        else:
                            st.metric("📈 S&P 500 Index", "N/A", help="Invalid data values")
                    except Exception as e:
                        st.metric("📈 S&P 500 Index", "Error", help=f"Calculation error: {e}")
                else:
                    st.metric("📈 S&P 500 Index", "N/A", help="Data column not found")
                    # Debug info
                    with st.expander("🔍 Debug: Available Market Columns"):
                        st.write(list(latest_market_data.index))
            
            with col2:
                # Enhanced Treasury yield detection
                treasury_candidates = ["10y_treasury_yield", "treasury_yield", "10y_yield", "yield_10y", "DGS10"]
                treasury_col = None
                
                for candidate in treasury_candidates:
                    if candidate in latest_market_data.index and pd.notna(latest_market_data[candidate]):
                        treasury_col = candidate
                        break
                
                if treasury_col is not None:
                    try:
                        current_yield = safe_get_value(latest_market_data, treasury_col, 0)
                        prev_yield = safe_get_value(previous_market_data, treasury_col, current_yield)
                        
                        if current_yield >= 0 and prev_yield >= 0:
                            change = calculate_change_safely(current_yield, prev_yield)
                            
                            st.metric(
                                label="🏛️ 10Y Treasury",
                                value=f"{current_yield:.2f}%",
                                delta=f"{change:+.2f}%" if change != 0 else None,
                                help="10-Year U.S. Treasury Bond Yield"
                            )
                        else:
                            st.metric("🏛️ 10Y Treasury", "N/A", help="Invalid yield values")
                    except Exception as e:
                        st.metric("🏛️ 10Y Treasury", "Error", help=f"Calculation error: {e}")
                else:
                    st.metric("🏛️ 10Y Treasury", "N/A", help="Treasury data not found")
            
            with col3:
                # Enhanced volatility detection with multiple fallbacks
                vix_candidates = ["vix", "volatility", "fear_index", "VIX", "market_volatility"]
                vix_col = None
                
                for candidate in vix_candidates:
                    if candidate in latest_market_data.index and pd.notna(latest_market_data[candidate]):
                        vix_col = candidate
                        break
                
                if vix_col is not None:
                    try:
                        current_vix = safe_get_value(latest_market_data, vix_col, 0)
                        prev_vix = safe_get_value(previous_market_data, vix_col, current_vix)
                        
                        if current_vix >= 0:
                            change = calculate_change_safely(current_vix, prev_vix)
                            
                            st.metric(
                                label="⚡ Market Volatility",
                                value=f"{current_vix:.1f}",
                                delta=f"{change:+.1f}" if change != 0 else None,
                                help="Market volatility index (VIX or similar)"
                            )
                        else:
                            st.metric("⚡ Market Volatility", "N/A", help="Invalid volatility values")
                    except Exception as e:
                        st.metric("⚡ Market Volatility", "Error", help=f"Calculation error: {e}")
                else:
                    # Calculate a simple volatility proxy from S&P 500 if available
                    try:
                        if sp500_col is not None and len(financial_markets) >= 5:
                            recent_prices = financial_markets[sp500_col].dropna().tail(5)
                            if len(recent_prices) >= 3:
                                volatility = recent_prices.pct_change().std() * 100
                                if not np.isnan(volatility) and volatility >= 0:
                                    st.metric(
                                        label="⚡ Price Volatility",
                                        value=f"{volatility:.1f}%",
                                        help="5-day price volatility estimate"
                                    )
                                else:
                                    st.metric("⚡ Volatility", "N/A", help="Cannot calculate volatility")
                            else:
                                st.metric("⚡ Volatility", "N/A", help="Insufficient price data")
                        else:
                            st.metric("⚡ Volatility", "N/A", help="No price data available")
                    except Exception as e:
                        st.metric("⚡ Volatility", "Error", help=f"Volatility calculation error: {e}")
        
        except Exception as e:
            st.error(f"Error processing financial markets data: {e}")
            # Show available data for debugging
            with st.expander("🔍 Debug: Financial Markets Data"):
                st.write("Available columns:", list(financial_markets.columns))
                st.write("Latest data shape:", financial_markets.shape)
                st.dataframe(financial_markets.tail(3))
    else:
        st.markdown("""
        <div class="metric-card">
            <h4>⚠️ Market Data Unavailable</h4>
            <p>Financial market indicators are currently not accessible. Please check your data connection or data source configuration.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show what datasets are available
        if data:
            available_datasets = [k for k, v in data.items() if v is not None and not v.empty]
            if available_datasets:
                st.info(f"📊 Available datasets: {', '.join(available_datasets)}")
    
    # Enhanced economic summary with data quality insights
    st.markdown("---")
    
    # Calculate overall data completeness
    try:
        total_indicators = 0
        available_indicators = 0
        
        # Count primary indicators
        if primary_indicators is not None and not primary_indicators.empty:
            key_columns = ['gdp_growth', 'unemployment_rate', 'cpi_inflation']
            for col in key_columns:
                total_indicators += 1
                if col in primary_indicators.columns and not primary_indicators[col].isna().all():
                    available_indicators += 1
        
        # Count market indicators using the same candidate logic as the display
        if financial_markets is not None and not financial_markets.empty:
            # Check S&P 500 indicator
            total_indicators += 1
            sp500_candidates = ["sp500_price", "close", "adj_close", "S&P500", "value", "price", "SP500"]
            for candidate in sp500_candidates:
                if candidate in financial_markets.columns and not financial_markets[candidate].isna().all():
                    available_indicators += 1
                    break
            
            # Check Treasury yield indicator
            total_indicators += 1
            treasury_candidates = ["10y_treasury_yield", "treasury_yield", "10y_yield", "yield_10y", "DGS10"]
            for candidate in treasury_candidates:
                if candidate in financial_markets.columns and not financial_markets[candidate].isna().all():
                    available_indicators += 1
                    break
            
            # Check VIX indicator
            total_indicators += 1
            vix_candidates = ["vix", "volatility", "fear_index", "VIX", "market_volatility"]
            for candidate in vix_candidates:
                if candidate in financial_markets.columns and not financial_markets[candidate].isna().all():
                    available_indicators += 1
                    break
        
        data_completeness = (available_indicators / total_indicators * 100) if total_indicators > 0 else 0
        
        # Determine data quality status
        if data_completeness >= 80:
            quality_status = "🟢 Excellent"
            quality_color = "#28a745"
        elif data_completeness >= 60:
            quality_status = "🟡 Good"
            quality_color = "#ffc107"
        elif data_completeness >= 40:
            quality_status = "🟠 Fair"
            quality_color = "#fd7e14"
        else:
            quality_status = "🔴 Poor"
            quality_color = "#dc3545"
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>🎯 Executive Summary</h4>
            <p>This dashboard provides a comprehensive view of current economic conditions through key indicators. 
            Monitor these metrics regularly to stay informed about economic trends and potential market shifts.</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 Executive Summary</h4>
            <p>This dashboard provides a comprehensive view of current economic conditions through key indicators. 
            Monitor these metrics regularly to stay informed about economic trends and potential market shifts.</p>
            <small><em>Data refreshed from authoritative economic sources.</em></small>
        </div>
        """, unsafe_allow_html=True)
