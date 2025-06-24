import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List

# Professional color palette
COLOR_PALETTE = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e', 
    'success': '#2ca02c',
    'warning': '#d62728',
    'info': '#9467bd',
    'light': '#17becf',
    'dark': '#8c564b',
    'muted': '#e377c2'
}

ECONOMIC_COLORS = {
    'gdp_growth': '#2E8B57',  # Sea Green
    'unemployment_rate': '#DC143C',  # Crimson
    'cpi_inflation': '#FF8C00',  # Dark Orange
    'interest_rate': '#4169E1',  # Royal Blue
    'stock_market': '#9932CC',  # Dark Orchid
    'housing': '#8B4513',  # Saddle Brown
    'trade': '#20B2AA',  # Light Sea Green
    'consumer': '#FF6347'  # Tomato
}

def plot_time_series(df: pd.DataFrame, y_columns: List[str], title: str, y_axis_title: str) -> go.Figure:
    """
    Creates an interactive time series plot with professional styling and missing data handling.

    Args:
        df: DataFrame containing the data, with a 'date' column.
        y_columns: A list of column names to plot on the y-axis.
        title: The title of the chart.
        y_axis_title: The title for the y-axis.

    Returns:
        A Plotly graph_objects Figure.
    """
    fig = go.Figure()
    
    # Handle missing data and create traces
    for i, col in enumerate(y_columns):
        if col not in df.columns:
            continue
            
        # Get color for this metric
        color = ECONOMIC_COLORS.get(col, list(COLOR_PALETTE.values())[i % len(COLOR_PALETTE)])
        
        # Handle missing data
        clean_data = df[['date', col]].dropna()
        
        if len(clean_data) == 0:
            continue
            
        # Create trace with professional styling
        fig.add_trace(go.Scatter(
            x=clean_data['date'], 
            y=clean_data[col], 
            mode='lines+markers',
            name=col.replace('_', ' ').title(),
            line=dict(color=color, width=2.5),
            marker=dict(size=4, color=color),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Date: %{x}<br>' +
                         'Value: %{y:.2f}<br>' +
                         '<extra></extra>'
        ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color='#2c3e50'),
            x=0.5
        ),
        xaxis=dict(
            title='Date',
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            showline=True,
            linecolor='rgba(128,128,128,0.5)'
        ),
        yaxis=dict(
            title=y_axis_title,
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            showline=True,
            linecolor='rgba(128,128,128,0.5)'
        ),
        legend=dict(
            title='Indicators',
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        template='plotly_white',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Arial, sans-serif', size=12, color='#2c3e50'),
        margin=dict(l=60, r=60, t=80, b=60)
    )
    return fig

def plot_indicator_heatmap(df: pd.DataFrame, columns: List[str], title: str) -> go.Figure:
    """
    Creates a professional correlation heatmap with proper missing data handling.

    Args:
        df: DataFrame containing the indicator data.
        columns: A list of columns to include in the correlation matrix.
        title: The title of the heatmap.

    Returns:
        A Plotly graph_objects Figure.
    """
    # Filter columns that exist in the dataframe
    available_columns = [col for col in columns if col in df.columns]
    
    if len(available_columns) < 2:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for correlation analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16, color="gray")
        )
        fig.update_layout(title=title, showlegend=False)
        return fig
    
    # Calculate correlation matrix with proper handling of missing data
    clean_df = df[available_columns].dropna()
    
    if len(clean_df) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data points for correlation analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16, color="gray")
        )
        fig.update_layout(title=title, showlegend=False)
        return fig
    
    corr_matrix = clean_df.corr()
    
    # Create professional heatmap
    fig = px.imshow(
        corr_matrix, 
        text_auto='.2f', 
        aspect="auto",
        color_continuous_scale='RdBu_r',
        range_color=[-1, 1],
        labels=dict(color="Correlation Coefficient"),
        x=[col.replace('_', ' ').title() for col in corr_matrix.columns], 
        y=[col.replace('_', ' ').title() for col in corr_matrix.columns]
    )
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color='#2c3e50'),
            x=0.5
        ),
        font=dict(family='Arial, sans-serif', size=12, color='#2c3e50'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=100, r=100, t=80, b=100)
    )
    
    # Update colorbar only if we have heatmap data
    try:
        if len(fig.data) > 0 and hasattr(fig.data[0], 'z') and fig.data[0].z is not None:
            fig.update_coloraxes(
                colorbar=dict(
                    title=dict(text="Correlation", side="right"),
                    tickmode="linear",
                    tick0=-1,
                    dtick=0.5
                )
            )
    except Exception as e:
        # If coloraxes update fails, continue without it
        pass
    
    return fig

def plot_gauge(value: float, title: str, reference_value: float) -> go.Figure:
    """
    Creates a professional gauge chart with intelligent ranges and color coding.

    Args:
        value: The current value of the metric.
        title: The title of the gauge.
        reference_value: A reference value (e.g., previous period, target).

    Returns:
        A Plotly graph_objects Figure.
    """
    # Handle missing or invalid data
    if pd.isna(value) or pd.isna(reference_value):
        fig = go.Figure()
        fig.add_annotation(
            text="Data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=14, color="gray")
        )
        fig.update_layout(
            title=dict(text=title, font=dict(size=16, color='#2c3e50'), x=0.5),
            showlegend=False,
            height=300
        )
        return fig
    
    # Intelligent range calculation based on metric type
    metric_type = title.lower()
    
    if 'gdp' in metric_type or 'growth' in metric_type:
        # GDP growth: typically -5% to +8%
        range_min, range_max = -5, 8
        good_threshold = 2.5
        excellent_threshold = 4.0
    elif 'unemployment' in metric_type:
        # Unemployment: typically 0% to 15%
        range_min, range_max = 0, 15
        good_threshold = 6.0
        excellent_threshold = 4.0
        # For unemployment, lower is better
        value_color = '#2ca02c' if value <= excellent_threshold else '#ff7f0e' if value <= good_threshold else '#d62728'
    elif 'inflation' in metric_type or 'cpi' in metric_type:
        # Inflation: typically -2% to +8%
        range_min, range_max = -2, 8
        good_threshold = 3.0
        excellent_threshold = 2.0
    else:
        # Generic range
        all_values = [value, reference_value]
        range_min = min(all_values) * 0.8 if min(all_values) > 0 else min(all_values) * 1.2
        range_max = max(all_values) * 1.2 if max(all_values) > 0 else max(all_values) * 0.8
        good_threshold = (range_max - range_min) * 0.6 + range_min
        excellent_threshold = (range_max - range_min) * 0.8 + range_min
    
    # Determine gauge color based on value and metric type
    if 'unemployment' not in metric_type:
        if value >= excellent_threshold:
            gauge_color = '#2ca02c'  # Green
        elif value >= good_threshold:
            gauge_color = '#ff7f0e'  # Orange
        else:
            gauge_color = '#d62728'  # Red
    else:
        # Already set above for unemployment
        gauge_color = value_color if 'value_color' in locals() else '#ff7f0e'
    
    # Calculate delta and its color
    delta_value = value - reference_value
    if 'unemployment' in metric_type:
        delta_color = 'green' if delta_value < 0 else 'red'
    else:
        delta_color = 'green' if delta_value > 0 else 'red'
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={
            'text': title,
            'font': {'size': 16, 'color': '#2c3e50'}
        },
        delta={
            'reference': reference_value,
            'increasing': {'color': delta_color},
            'decreasing': {'color': 'red' if delta_color == 'green' else 'green'},
            'font': {'size': 14}
        },
        number={
            'font': {'size': 24, 'color': '#2c3e50'},
            'suffix': '%' if any(word in metric_type for word in ['rate', 'growth', 'inflation']) else ''
        },
        gauge={
            'axis': {
                'range': [range_min, range_max],
                'tickwidth': 1,
                'tickcolor': "darkblue"
            },
            'bar': {'color': gauge_color, 'thickness': 0.8},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [range_min, good_threshold], 'color': 'rgba(255, 0, 0, 0.1)'},
                {'range': [good_threshold, excellent_threshold], 'color': 'rgba(255, 165, 0, 0.1)'},
                {'range': [excellent_threshold, range_max], 'color': 'rgba(0, 128, 0, 0.1)'}
            ] if 'unemployment' not in metric_type else [
                {'range': [range_min, excellent_threshold], 'color': 'rgba(0, 128, 0, 0.1)'},
                {'range': [excellent_threshold, good_threshold], 'color': 'rgba(255, 165, 0, 0.1)'},
                {'range': [good_threshold, range_max], 'color': 'rgba(255, 0, 0, 0.1)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': reference_value
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(family='Arial, sans-serif', color='#2c3e50'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

if __name__ == '__main__':
    # Example usage for testing
    # Create a sample DataFrame
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=100))
    data = {
        'date': dates,
        'gdp_growth': [2.5 + 0.5 * i for i in range(100)],
        'unemployment_rate': [4.5 - 0.02 * i for i in range(100)],
        'cpi_inflation': [2.0 + 0.01 * i for i in range(100)]
    }
    sample_df = pd.DataFrame(data)

    # Test time series plot
    ts_fig = plot_time_series(sample_df, ['gdp_growth', 'unemployment_rate'], 'Key Economic Indicators', 'Value')
    # ts_fig.show()

    # Test heatmap
    heatmap_fig = plot_indicator_heatmap(sample_df, ['gdp_growth', 'unemployment_rate', 'cpi_inflation'], 'Correlation Matrix')
    # heatmap_fig.show()

    # Test gauge
    gauge_fig = plot_gauge(3.2, "GDP Growth", 3.1)
    # gauge_fig.show()

    print("Plotting functions tested successfully (plots not displayed in console).")
