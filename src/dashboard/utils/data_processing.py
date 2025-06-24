import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from src.dashboard.utils.real_data_loader import load_all_real_data

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define the base path to the data directory relative to this script
DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "synthetic"

def validate_dataframe(df: pd.DataFrame, dataset_name: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validates and cleans a DataFrame with comprehensive data quality checks.
    
    Args:
        df: DataFrame to validate
        dataset_name: Name of the dataset for logging
    
    Returns:
        Tuple of (cleaned_dataframe, list_of_issues)
    """
    issues = []
    cleaned_df = df.copy()
    
    # Check for empty DataFrame
    if df.empty:
        issues.append(f"Dataset {dataset_name} is empty")
        return cleaned_df, issues
    
    # Validate date column
    if 'date' in df.columns:
        try:
            cleaned_df['date'] = pd.to_datetime(df['date'], errors='coerce')
            invalid_dates = cleaned_df['date'].isna().sum()
            if invalid_dates > 0:
                issues.append(f"Found {invalid_dates} invalid dates in {dataset_name}")
                # Remove rows with invalid dates
                cleaned_df = cleaned_df.dropna(subset=['date'])
        except Exception as e:
            issues.append(f"Error processing dates in {dataset_name}: {e}")
    
    # Check for duplicate dates
    if 'date' in cleaned_df.columns:
        duplicates = cleaned_df['date'].duplicated().sum()
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate dates in {dataset_name}")
            # Keep the last occurrence of duplicate dates
            cleaned_df = cleaned_df.drop_duplicates(subset=['date'], keep='last')
    
    # Validate numeric columns
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # Check for infinite values
        inf_count = np.isinf(cleaned_df[col]).sum()
        if inf_count > 0:
            issues.append(f"Found {inf_count} infinite values in {dataset_name}.{col}")
            cleaned_df[col] = cleaned_df[col].replace([np.inf, -np.inf], np.nan)
        
        # Check for extreme outliers (beyond 5 standard deviations)
        if len(cleaned_df[col].dropna()) > 10:  # Only check if we have enough data
            mean_val = cleaned_df[col].mean()
            std_val = cleaned_df[col].std()
            if std_val > 0:
                outliers = np.abs((cleaned_df[col] - mean_val) / std_val) > 5
                outlier_count = outliers.sum()
                if outlier_count > 0:
                    issues.append(f"Found {outlier_count} extreme outliers in {dataset_name}.{col}")
    
    # Check missing data percentage
    total_cells = len(cleaned_df) * len(cleaned_df.columns)
    missing_cells = cleaned_df.isnull().sum().sum()
    missing_percentage = (missing_cells / total_cells) * 100
    
    if missing_percentage > 50:
        issues.append(f"High missing data rate in {dataset_name}: {missing_percentage:.1f}%")
    elif missing_percentage > 20:
        issues.append(f"Moderate missing data rate in {dataset_name}: {missing_percentage:.1f}%")
    
    # Sort by date if date column exists
    if 'date' in cleaned_df.columns:
        cleaned_df = cleaned_df.sort_values('date').reset_index(drop=True)
    
    return cleaned_df, issues

def interpolate_missing_data(df: pd.DataFrame, method: str = 'linear') -> pd.DataFrame:
    """
    Intelligently interpolate missing data in economic time series.
    
    Args:
        df: DataFrame with potential missing values
        method: Interpolation method ('linear', 'forward_fill', 'backward_fill')
    
    Returns:
        DataFrame with interpolated values
    """
    cleaned_df = df.copy()
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if cleaned_df[col].isnull().any():
            try:
                if method == 'linear':
                    # Use time-aware interpolation if date column exists and is properly indexed
                    if 'date' in cleaned_df.columns:
                        # Try to set date as index for time interpolation
                        try:
                            temp_df = cleaned_df.copy()
                            temp_df['date'] = pd.to_datetime(temp_df['date'])
                            temp_df = temp_df.set_index('date')
                            temp_df[col] = temp_df[col].interpolate(method='time')
                            cleaned_df[col] = temp_df[col].values
                        except (ValueError, TypeError, KeyError):
                            # Fallback to linear interpolation
                            cleaned_df[col] = cleaned_df[col].interpolate(method='linear')
                    else:
                        cleaned_df[col] = cleaned_df[col].interpolate(method='linear')
                elif method == 'forward_fill':
                    cleaned_df[col] = cleaned_df[col].fillna(method='ffill')
                elif method == 'backward_fill':
                    cleaned_df[col] = cleaned_df[col].fillna(method='bfill')
            except Exception as e:
                logging.warning(f"Error interpolating column {col}: {e}. Using forward fill as fallback.")
                cleaned_df[col] = cleaned_df[col].fillna(method='ffill')
    
    return cleaned_df

def calculate_data_quality_score(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate comprehensive data quality metrics.
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Dictionary with quality metrics
    """
    if df.empty:
        return {'overall_score': 0.0, 'completeness': 0.0, 'consistency': 0.0, 'validity': 0.0}
    
    # Completeness score (percentage of non-null values)
    total_cells = len(df) * len(df.columns)
    non_null_cells = total_cells - df.isnull().sum().sum()
    completeness = (non_null_cells / total_cells) * 100
    
    # Consistency score (based on data types and ranges)
    consistency = 100.0  # Start with perfect score
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        # Check for infinite values
        if np.isinf(df[col]).any():
            consistency -= 10
        
        # Check for extreme outliers
        if len(df[col].dropna()) > 10:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = ((df[col] < (q1 - 3 * iqr)) | (df[col] > (q3 + 3 * iqr))).sum()
            outlier_rate = outliers / len(df) * 100
            if outlier_rate > 5:
                consistency -= min(20, outlier_rate)
    
    # Validity score (based on date consistency and logical ranges)
    validity = 100.0
    if 'date' in df.columns:
        # Check for chronological order
        if not df['date'].is_monotonic_increasing:
            validity -= 20
        
        # Check for reasonable date range (not too far in future/past)
        current_year = datetime.now().year
        min_year = df['date'].dt.year.min()
        max_year = df['date'].dt.year.max()
        
        if min_year < 1900 or max_year > current_year + 5:
            validity -= 15
    
    # Overall score (weighted average)
    overall_score = (completeness * 0.4 + consistency * 0.3 + validity * 0.3)
    
    return {
        'overall_score': round(overall_score, 1),
        'completeness': round(completeness, 1),
        'consistency': round(max(0, consistency), 1),
        'validity': round(max(0, validity), 1)
    }

@st.cache_data
def load_all_data(use_real_data: bool = True, interpolate_missing: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Loads and processes economic datasets with comprehensive data quality management.

    Args:
        use_real_data: If True, loads real data from APIs. If False, loads synthetic data.
        interpolate_missing: If True, interpolates missing values in time series.

    Returns:
        A dictionary where keys are dataset names and values are cleaned DataFrames.
    """
    datasets = {}
    all_issues = []
    
    if use_real_data:
        logger.info("🔄 Loading real data from APIs...")
        try:
            raw_datasets = load_all_real_data()
            if raw_datasets:
                logger.info(f"✅ Successfully loaded {len(raw_datasets)} real datasets")
                
                # Process and validate each dataset
                for name, df in raw_datasets.items():
                    cleaned_df, issues = validate_dataframe(df, name)
                    
                    if interpolate_missing and not cleaned_df.empty:
                        cleaned_df = interpolate_missing_data(cleaned_df)
                    
                    datasets[name] = cleaned_df
                    all_issues.extend(issues)
                    
                    # Log data quality
                    quality_score = calculate_data_quality_score(cleaned_df)
                    logger.info(f"📊 {name} quality score: {quality_score['overall_score']:.1f}%")
                
                # Display data quality summary
                if all_issues:
                    logger.warning(f"⚠️ Found {len(all_issues)} data quality issues")
                    with st.expander("📋 Data Quality Report", expanded=False):
                        for issue in all_issues[:10]:  # Show first 10 issues
                            st.warning(f"• {issue}")
                        if len(all_issues) > 10:
                            st.info(f"... and {len(all_issues) - 10} more issues")
                
                return datasets
            else:
                logger.warning("⚠️ No real data loaded, falling back to synthetic data")
                st.warning("Unable to load real data from APIs. Falling back to synthetic data.")
        except Exception as e:
            logger.error(f"❌ Error loading real data: {e}")
            st.warning(f"Error loading real data: {e}. Falling back to synthetic data.")
    
    # Fallback to synthetic data
    logger.info("🔄 Loading synthetic data from CSV files...")
    required_files = [
        "primary_indicators.csv",
        "regional_economic.csv", 
        "financial_markets.csv",
        "international_trade.csv",
        "alternative_indicators.csv"
    ]

    logger.info(f"📁 Loading data from: {DATA_DIR}")
    
    if not DATA_DIR.exists():
        logger.error(f"❌ Data directory not found: {DATA_DIR}")
        st.error(f"Data directory not found: {DATA_DIR}. Please ensure data is generated.")
        return {}

    for filename in required_files:
        filepath = DATA_DIR / filename
        dataset_name = filename.replace(".csv", "")
        
        try:
            if not filepath.exists():
                logger.warning(f"⚠️ File not found: {filename}, skipping...")
                continue
                
            df = pd.read_csv(filepath)
            logger.info(f"📄 Loaded {filename}: {len(df)} rows, {len(df.columns)} columns")
            
            # Validate and clean the data
            cleaned_df, issues = validate_dataframe(df, dataset_name)
            
            if interpolate_missing and not cleaned_df.empty:
                cleaned_df = interpolate_missing_data(cleaned_df)
            
            datasets[dataset_name] = cleaned_df
            all_issues.extend(issues)
            
            # Log data quality
            quality_score = calculate_data_quality_score(cleaned_df)
            logger.info(f"📊 {dataset_name} quality score: {quality_score['overall_score']:.1f}%")
            
        except FileNotFoundError:
            logger.error(f"❌ File not found: {filename}")
            st.warning(f"Data file not found: {filename}. Continuing with available data.")
        except Exception as e:
            logger.error(f"❌ Error loading {filename}: {e}")
            st.warning(f"Error loading {filename}: {e}. Continuing with available data.")
    
    # Display comprehensive data quality summary
    if datasets:
        total_quality_score = np.mean([calculate_data_quality_score(df)['overall_score'] for df in datasets.values()])
        logger.info(f"📈 Overall data quality score: {total_quality_score:.1f}%")
        
        if all_issues:
            logger.warning(f"⚠️ Found {len(all_issues)} total data quality issues")
            with st.expander("📋 Comprehensive Data Quality Report", expanded=False):
                st.markdown(f"**Overall Quality Score: {total_quality_score:.1f}%**")
                st.markdown(f"**Total Issues Found: {len(all_issues)}**")
                
                for i, issue in enumerate(all_issues[:15], 1):
                    st.warning(f"{i}. {issue}")
                if len(all_issues) > 15:
                    st.info(f"... and {len(all_issues) - 15} more issues")
        else:
            st.success(f"✅ Data loaded successfully with {total_quality_score:.1f}% quality score")
    else:
        logger.error("❌ No datasets loaded successfully")
        st.error("No datasets could be loaded. Please check your data sources.")
    
    return datasets

if __name__ == '__main__':
    # For testing purposes
    all_data = load_all_data()
    if all_data:
        for name, df in all_data.items():
            print(f"\n--- {name} ---")
            print(df.head())
            print(df.info())
