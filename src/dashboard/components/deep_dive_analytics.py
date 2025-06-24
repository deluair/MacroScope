import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from src.dashboard.utils.plotting import plot_time_series, plot_indicator_heatmap

def display_deep_dive_analytics(data: dict[str, pd.DataFrame]):
    """
    Renders an enhanced Deep Dive Analytics page with comprehensive data analysis.

    Args:
        data: A dictionary of DataFrames for detailed analysis.
    """
    if not data:
        st.error("⚠️ **No Data Available** - Unable to perform deep dive analysis.")
        st.info("Please ensure data is loaded properly and try again.")
        return

    # Professional header with context
    st.markdown("""
    <div class="metric-card">
        <h3>🔍 Advanced Analytics Dashboard</h3>
        <p>Comprehensive analysis tools for exploring economic datasets, identifying patterns, and understanding relationships between indicators.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Enhanced dataset selection with metadata
    col1, col2 = st.columns([2, 1])
    
    with col1:
        dataset_names = list(data.keys())
        selected_dataset_name = st.selectbox(
            "📊 Select Dataset for Analysis:", 
            dataset_names,
            help="Choose the economic dataset you want to analyze in detail"
        )
    
    with col2:
        if dataset_names:
            total_datasets = len(dataset_names)
            total_indicators = sum(len([col for col in df.columns if df[col].dtype in ['int64', 'float64']]) for df in data.values())
            st.metric("Available Datasets", total_datasets)
            st.metric("Total Indicators", total_indicators)

    if selected_dataset_name and selected_dataset_name in data:
        df = data[selected_dataset_name]
        
        # Data quality assessment
        st.markdown("---")
        st.markdown(f"### 📈 Dataset: {selected_dataset_name.replace('_', ' ').title()}")
        
        # Data overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(df), help="Number of data points")
        
        with col2:
            numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
            st.metric("Numeric Indicators", len(numeric_cols), help="Number of quantitative measures")
        
        with col3:
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("Data Completeness", f"{100-missing_pct:.1f}%", help="Percentage of non-missing values")
        
        with col4:
            if 'date' in df.columns:
                date_range = (df['date'].max() - df['date'].min()).days
                st.metric("Time Span (Days)", date_range, help="Total period covered")
            else:
                st.metric("Time Span", "N/A", help="No date column found")

        # Data quality insights
        if missing_pct > 10:
            st.warning(f"⚠️ **Data Quality Alert**: {missing_pct:.1f}% of data is missing. Consider data cleaning.")
        elif missing_pct > 0:
            st.info(f"ℹ️ **Data Quality**: {missing_pct:.1f}% missing data detected. Analysis will handle missing values.")
        else:
            st.success("✅ **Data Quality**: Complete dataset with no missing values.")

        # Enhanced metric selection with filtering
        st.markdown("---")
        st.markdown("### 📊 Time Series Analysis")
        
        all_columns = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
        
        if not all_columns:
            st.warning("No numeric columns available for analysis.")
            return
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_metrics = st.multiselect(
                "Select indicators to analyze:", 
                options=all_columns, 
                default=all_columns[:3] if len(all_columns) >= 3 else all_columns,
                help="Choose economic indicators to plot and analyze"
            )
        
        with col2:
            # Analysis options
            show_trend = st.checkbox("Show Trend Lines", value=True)
            normalize_data = st.checkbox("Normalize Data", help="Scale all metrics to 0-100 range")

        if selected_metrics:
            # Prepare data for plotting
            plot_df = df.copy()
            
            if normalize_data:
                for col in selected_metrics:
                    if col in plot_df.columns:
                        col_min = plot_df[col].min()
                        col_max = plot_df[col].max()
                        if col_max != col_min:
                            plot_df[col] = ((plot_df[col] - col_min) / (col_max - col_min)) * 100
            
            fig_ts = plot_time_series(
                plot_df,
                y_columns=selected_metrics,
                title=f"{selected_dataset_name.replace('_', ' ').title()} - Temporal Analysis",
                y_axis_title="Normalized Value (0-100)" if normalize_data else "Value"
            )
            
            # Add trend lines if requested
            if show_trend and 'date' in df.columns:
                for metric in selected_metrics:
                    if metric in df.columns:
                        # Calculate trend line
                        clean_data = df[['date', metric]].dropna()
                        if len(clean_data) > 1:
                            x_numeric = pd.to_numeric(clean_data['date'])
                            z = np.polyfit(x_numeric, clean_data[metric], 1)
                            trend_line = np.poly1d(z)(x_numeric)
                            
                            fig_ts.add_trace(go.Scatter(
                                x=clean_data['date'],
                                y=trend_line,
                                mode='lines',
                                name=f'{metric} Trend',
                                line=dict(dash='dash', width=2),
                                opacity=0.7
                            ))
            
            st.plotly_chart(fig_ts, use_container_width=True)
            
            # Statistical summary
            st.markdown("#### 📋 Statistical Summary")
            summary_stats = df[selected_metrics].describe().round(2)
            st.dataframe(summary_stats, use_container_width=True)
            
        else:
            st.info("👆 Select at least one indicator to display analysis.")

        # Enhanced correlation analysis
        st.markdown("---")
        st.markdown("### 🔗 Correlation Analysis")
        
        if len(all_columns) > 1:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                fig_heatmap = plot_indicator_heatmap(
                    df,
                    columns=all_columns,
                    title=f"Correlation Matrix - {selected_dataset_name.replace('_', ' ').title()}"
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            with col2:
                # Correlation insights
                corr_matrix = df[all_columns].corr()
                
                # Find strongest correlations
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if not pd.isna(corr_val):
                            corr_pairs.append({
                                'pair': f"{corr_matrix.columns[i]} vs {corr_matrix.columns[j]}",
                                'correlation': abs(corr_val),
                                'value': corr_val
                            })
                
                if corr_pairs:
                    corr_pairs.sort(key=lambda x: x['correlation'], reverse=True)
                    
                    st.markdown("""
                    <div class="metric-card">
                        <h4>🔍 Key Correlations</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for i, pair in enumerate(corr_pairs[:3]):
                        strength = "Strong" if pair['correlation'] > 0.7 else "Moderate" if pair['correlation'] > 0.4 else "Weak"
                        direction = "Positive" if pair['value'] > 0 else "Negative"
                        
                        st.markdown(f"""
                        **{i+1}. {strength} {direction}**  
                        {pair['pair'].replace('_', ' ').title()}  
                        Correlation: {pair['value']:.3f}
                        """)
        else:
            st.info("Need at least 2 numeric indicators for correlation analysis.")

        # Enhanced data view with filtering
        st.markdown("---")
        st.markdown("### 📋 Data Explorer")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_rows = st.selectbox("Rows to display:", [10, 25, 50, 100, "All"], index=1)
        
        with col2:
            if 'date' in df.columns:
                sort_by = st.selectbox("Sort by:", ['date'] + all_columns, index=0)
            else:
                sort_by = st.selectbox("Sort by:", all_columns, index=0)
        
        with col3:
            sort_order = st.selectbox("Order:", ["Descending", "Ascending"], index=0)
        
        # Apply sorting and filtering
        display_df = df.copy()
        
        if sort_by in display_df.columns:
            ascending = sort_order == "Ascending"
            display_df = display_df.sort_values(by=sort_by, ascending=ascending)
        
        if show_rows != "All":
            display_df = display_df.head(show_rows)
        
        # Format numeric columns for better display
        for col in display_df.columns:
            if display_df[col].dtype in ['float64']:
                display_df[col] = display_df[col].round(3)
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Download option
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv,
            file_name=f"{selected_dataset_name}_filtered.csv",
            mime="text/csv",
            help="Download the currently displayed data as CSV"
        )
        
    else:
        st.error("⚠️ **Dataset Error** - Selected dataset is not available or corrupted.")
        st.info("Please select a different dataset or check data loading.")
