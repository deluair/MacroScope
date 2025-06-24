import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from src.dashboard.utils.plotting import plot_time_series, plot_gauge

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "synthetic"

def load_scenario_data(scenario_name: str) -> dict[str, pd.DataFrame]:
    """
    Loads all datasets for a specific scenario.
    """
    scenario_path = DATA_DIR / scenario_name
    datasets = {}
    if not scenario_path.is_dir():
        return {}

    for filepath in scenario_path.glob("*.csv"):
        dataset_name = filepath.stem
        df = pd.read_csv(filepath)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        datasets[dataset_name] = df
    return datasets

def calculate_risk_metrics(baseline_df: pd.DataFrame, scenario_df: pd.DataFrame, metric: str) -> dict:
    """
    Calculate comprehensive risk metrics comparing baseline and scenario data.
    """
    if metric not in baseline_df.columns or metric not in scenario_df.columns:
        return {}
    
    baseline_values = baseline_df[metric].dropna()
    scenario_values = scenario_df[metric].dropna()
    
    if len(baseline_values) == 0 or len(scenario_values) == 0:
        return {}
    
    # Calculate various risk metrics
    baseline_mean = baseline_values.mean()
    scenario_mean = scenario_values.mean()
    
    baseline_std = baseline_values.std()
    scenario_std = scenario_values.std()
    
    # Impact metrics
    absolute_change = scenario_mean - baseline_mean
    percentage_change = ((scenario_mean - baseline_mean) / baseline_mean * 100) if baseline_mean != 0 else 0
    
    # Volatility metrics
    volatility_change = scenario_std - baseline_std
    volatility_change_pct = ((scenario_std - baseline_std) / baseline_std * 100) if baseline_std != 0 else 0
    
    # Risk assessment
    risk_level = "Low"
    if abs(percentage_change) > 20:
        risk_level = "High"
    elif abs(percentage_change) > 10:
        risk_level = "Medium"
    
    return {
        'baseline_mean': baseline_mean,
        'scenario_mean': scenario_mean,
        'absolute_change': absolute_change,
        'percentage_change': percentage_change,
        'baseline_volatility': baseline_std,
        'scenario_volatility': scenario_std,
        'volatility_change': volatility_change,
        'volatility_change_pct': volatility_change_pct,
        'risk_level': risk_level
    }

def display_risk_assessment(baseline_data: dict[str, pd.DataFrame]):
    """
    Renders an enhanced Risk Assessment and Scenario Analysis page.
    """
    if not baseline_data:
        st.error("⚠️ **No Baseline Data Available** - Cannot perform risk assessment.")
        st.info("Please ensure baseline data is loaded properly.")
        return
    
    # Professional header
    st.markdown("""
    <div class="metric-card">
        <h3>⚠️ Risk Assessment & Scenario Analysis</h3>
        <p>Comprehensive analysis of economic scenarios and their potential impacts on key indicators.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Check for available scenarios
    scenarios = [d.name for d in DATA_DIR.iterdir() if d.is_dir()] if DATA_DIR.exists() else []
    
    if not scenarios:
        st.warning("⚠️ **No Scenario Data Available**")
        st.info("""
        Scenario analysis requires synthetic data in the `/data/synthetic/` directory.
        
        **Available Analysis:**
        - Baseline data exploration
        - Historical volatility analysis
        - Data quality assessment
        """)
        
        # Fallback: Analyze baseline data for risk patterns
        st.markdown("---")
        st.markdown("### 📊 Baseline Risk Analysis")
        
        if "primary_indicators" in baseline_data:
            df = baseline_data["primary_indicators"]
            numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
            
            if numeric_cols:
                selected_metric = st.selectbox(
                    "Select indicator for risk analysis:",
                    numeric_cols,
                    help="Choose an economic indicator to analyze for risk patterns"
                )
                
                if selected_metric:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Volatility analysis
                        values = df[selected_metric].dropna()
                        if len(values) > 1:
                            volatility = values.std()
                            mean_val = values.mean()
                            cv = (volatility / mean_val * 100) if mean_val != 0 else 0
                            
                            st.markdown("""
                            <div class="metric-card">
                                <h4>📈 Volatility Analysis</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Standard Deviation", f"{volatility:.2f}")
                                st.metric("Mean Value", f"{mean_val:.2f}")
                            with col_b:
                                st.metric("Coefficient of Variation", f"{cv:.1f}%")
                                risk_level = "High" if cv > 20 else "Medium" if cv > 10 else "Low"
                                st.metric("Risk Level", risk_level)
                    
                    with col2:
                        # Historical trend
                        if 'date' in df.columns:
                            fig = plot_time_series(
                                df,
                                y_columns=[selected_metric],
                                title=f"{selected_metric.replace('_', ' ').title()} - Historical Trend",
                                y_axis_title="Value"
                            )
                            st.plotly_chart(fig, use_container_width=True)
        return
    
    # Scenario selection with enhanced UI
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_scenario = st.selectbox(
            "🎯 Select Economic Scenario:", 
            options=scenarios,
            help="Choose a scenario to analyze its impact on economic indicators"
        )
    
    with col2:
        st.metric("Available Scenarios", len(scenarios))
        if baseline_data:
            baseline_indicators = len([col for df in baseline_data.values() for col in df.columns if df[col].dtype in ['int64', 'float64']])
            st.metric("Baseline Indicators", baseline_indicators)

    if selected_scenario:
        scenario_data = load_scenario_data(selected_scenario)

        if not scenario_data:
            st.error(f"⚠️ **Scenario Loading Error** - Could not load data for: {selected_scenario}")
            st.info("Please check if the scenario data files exist and are properly formatted.")
            return

        st.markdown("---")
        st.markdown(f"### 🎯 Scenario Impact Analysis: {selected_scenario.replace('_', ' ').title()}")
        
        # Find common datasets and metrics
        common_datasets = set(baseline_data.keys()) & set(scenario_data.keys())
        
        if not common_datasets:
            st.error("⚠️ **Data Mismatch** - No common datasets found between baseline and scenario.")
            return
        
        # Dataset selection
        selected_dataset = st.selectbox(
            "📊 Select Dataset for Comparison:",
            list(common_datasets),
            help="Choose the dataset to compare between baseline and scenario"
        )
        
        if selected_dataset:
            baseline_df = baseline_data[selected_dataset]
            scenario_df = scenario_data[selected_dataset]
            
            # Find common numeric columns
            baseline_numeric = [col for col in baseline_df.columns if baseline_df[col].dtype in ['int64', 'float64']]
            scenario_numeric = [col for col in scenario_df.columns if scenario_df[col].dtype in ['int64', 'float64']]
            common_metrics = list(set(baseline_numeric) & set(scenario_numeric))
            
            if not common_metrics:
                st.warning("⚠️ **No Common Metrics** - No comparable indicators found between datasets.")
                return
            
            # Metric selection
            selected_metric = st.selectbox(
                "📈 Select Indicator for Analysis:",
                common_metrics,
                help="Choose the economic indicator to compare between baseline and scenario"
            )
            
            if selected_metric:
                # Calculate comprehensive risk metrics
                risk_metrics = calculate_risk_metrics(baseline_df, scenario_df, selected_metric)
                
                if not risk_metrics:
                    st.warning(f"⚠️ **Insufficient Data** - Cannot calculate metrics for {selected_metric}.")
                    return
                
                # Risk overview dashboard
                st.markdown("#### 📊 Impact Overview")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    delta_color = "normal" if abs(risk_metrics['percentage_change']) < 5 else "inverse"
                    st.metric(
                        "Impact", 
                        f"{risk_metrics['percentage_change']:+.1f}%",
                        delta=f"{risk_metrics['absolute_change']:+.2f}",
                        delta_color=delta_color
                    )
                
                with col2:
                    vol_delta_color = "inverse" if risk_metrics['volatility_change'] > 0 else "normal"
                    st.metric(
                        "Volatility Change", 
                        f"{risk_metrics['volatility_change_pct']:+.1f}%",
                        delta=f"{risk_metrics['volatility_change']:+.2f}",
                        delta_color=vol_delta_color
                    )
                
                with col3:
                    risk_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
                    st.metric(
                        "Risk Level", 
                        f"{risk_color.get(risk_metrics['risk_level'], '⚪')} {risk_metrics['risk_level']}"
                    )
                
                with col4:
                    baseline_val = risk_metrics['baseline_mean']
                    scenario_val = risk_metrics['scenario_mean']
                    st.metric(
                        "Scenario Value",
                        f"{scenario_val:.2f}",
                        delta=f"vs {baseline_val:.2f} baseline"
                    )
                
                # Detailed comparison chart
                st.markdown("#### 📈 Temporal Comparison")
                
                # Prepare comparison data
                if 'date' in baseline_df.columns and 'date' in scenario_df.columns:
                    # Align dates for proper comparison
                    baseline_clean = baseline_df[['date', selected_metric]].dropna()
                    scenario_clean = scenario_df[['date', selected_metric]].dropna()
                    
                    # Merge on date
                    comparison_df = pd.merge(
                        baseline_clean.rename(columns={selected_metric: 'Baseline'}),
                        scenario_clean.rename(columns={selected_metric: 'Scenario'}),
                        on='date',
                        how='outer'
                    ).sort_values('date')
                    
                    if len(comparison_df) > 0:
                        fig = plot_time_series(
                            comparison_df,
                            y_columns=['Baseline', 'Scenario'],
                            title=f"{selected_metric.replace('_', ' ').title()} - Baseline vs {selected_scenario.replace('_', ' ').title()}",
                            y_axis_title=selected_metric.replace('_', ' ').title()
                        )
                        
                        # Add difference area
                        if 'Baseline' in comparison_df.columns and 'Scenario' in comparison_df.columns:
                            diff_data = comparison_df.dropna()
                            if len(diff_data) > 0:
                                fig.add_trace(go.Scatter(
                                    x=diff_data['date'],
                                    y=diff_data['Scenario'] - diff_data['Baseline'],
                                    mode='lines',
                                    name='Difference (Scenario - Baseline)',
                                    line=dict(color='rgba(255, 0, 0, 0.3)', dash='dot'),
                                    yaxis='y2'
                                ))
                                
                                # Add secondary y-axis for difference
                                fig.update_layout(
                                    yaxis2=dict(
                                        title="Difference",
                                        overlaying='y',
                                        side='right',
                                        showgrid=False
                                    )
                                )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("⚠️ **No Overlapping Data** - Cannot create temporal comparison.")
                else:
                    st.info("ℹ️ **Date Column Missing** - Showing statistical comparison only.")
                
                # Statistical analysis
                st.markdown("#### 📊 Statistical Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div class="metric-card">
                        <h4>📈 Baseline Statistics</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    baseline_stats = baseline_df[selected_metric].describe()
                    st.dataframe(baseline_stats.round(3), use_container_width=True)
                
                with col2:
                    st.markdown("""
                    <div class="metric-card">
                        <h4>🎯 Scenario Statistics</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    scenario_stats = scenario_df[selected_metric].describe()
                    st.dataframe(scenario_stats.round(3), use_container_width=True)
                
                # Risk assessment summary
                st.markdown("#### 🎯 Risk Assessment Summary")
                
                risk_interpretation = ""
                if risk_metrics['risk_level'] == "High":
                    risk_interpretation = f"⚠️ **High Risk Scenario**: The {selected_scenario.replace('_', ' ')} scenario shows significant impact on {selected_metric.replace('_', ' ')} with a {risk_metrics['percentage_change']:+.1f}% change from baseline."
                elif risk_metrics['risk_level'] == "Medium":
                    risk_interpretation = f"⚡ **Medium Risk Scenario**: The {selected_scenario.replace('_', ' ')} scenario shows moderate impact on {selected_metric.replace('_', ' ')} with a {risk_metrics['percentage_change']:+.1f}% change from baseline."
                else:
                    risk_interpretation = f"✅ **Low Risk Scenario**: The {selected_scenario.replace('_', ' ')} scenario shows minimal impact on {selected_metric.replace('_', ' ')} with a {risk_metrics['percentage_change']:+.1f}% change from baseline."
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4>🎯 Key Insights</h4>
                    <p>{risk_interpretation}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Export options
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Export comparison data
                    if 'comparison_df' in locals():
                        csv_data = comparison_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Comparison Data (CSV)",
                            data=csv_data,
                            file_name=f"{selected_scenario}_{selected_metric}_comparison.csv",
                            mime="text/csv"
                        )
                
                with col2:
                    # Export risk metrics
                    risk_df = pd.DataFrame([risk_metrics]).T
                    risk_df.columns = ['Value']
                    risk_csv = risk_df.to_csv()
                    st.download_button(
                        label="📊 Download Risk Metrics (CSV)",
                        data=risk_csv,
                        file_name=f"{selected_scenario}_{selected_metric}_risk_metrics.csv",
                        mime="text/csv"
                    )
