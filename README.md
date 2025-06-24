# 📊 MacroScope - Economic Intelligence Platform

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 🚀 **Advanced Economic Intelligence & Analytics Platform** - Real-time economic data analysis, forecasting, and scenario modeling powered by modern data science tools.

## 🌟 Overview

MacroScope is a comprehensive economic intelligence platform that provides real-time analysis of economic indicators, advanced forecasting capabilities, and scenario modeling tools for professional economic analysis. The platform combines real-time data feeds with synthetic data generation capabilities for robust economic research and analysis.

### ✨ Key Features

- **📈 Real-Time Data Integration**: Connects to FRED, Yahoo Finance, BLS, and other economic data sources
- **🧠 Synthetic Data Generation**: Advanced economic relationship modeling with realistic shock integration
- **📊 Interactive Dashboard**: Professional Streamlit-based interface with executive summaries and deep analytics
- **⚠️ Risk Assessment**: Comprehensive scenario analysis and risk modeling tools
- **🔍 Deep Analytics**: Advanced correlation analysis, trend detection, and statistical insights
- **📄 Report Generation**: Automated markdown and HTML report generation
- **🎯 Scenario Modeling**: Compare baseline vs. economic scenarios (recession, inflation, productivity boom)

## 🏗️ Project Structure

```
MacroScope/
├── 📁 config/                      # Configuration and settings
│   ├── __init__.py
│   ├── logging_config.py
│   └── settings.py                 # API keys and data directories
├── 📁 data/                        # Data storage
│   ├── processed/                  # Cleaned and processed data
│   ├── raw/                       # Raw data from APIs
│   └── synthetic/                 # Generated synthetic datasets
│       ├── high_inflation/        # High inflation scenario
│       ├── productivity_boom/     # Productivity boom scenario
│       └── recession/             # Recession scenario
├── 📁 src/                        # Source code
│   ├── 📁 api/                    # REST API implementation
│   │   ├── main.py               # FastAPI application
│   │   ├── models/               # API data models
│   │   └── routes/               # API endpoints
│   ├── 📁 dashboard/             # Streamlit dashboard
│   │   ├── streamlit_app.py      # Main dashboard application
│   │   ├── components/           # Dashboard components
│   │   │   ├── executive_summary.py
│   │   │   ├── deep_dive_analytics.py
│   │   │   └── risk_assessment.py
│   │   └── utils/                # Dashboard utilities
│   │       ├── data_processing.py
│   │       ├── plotting.py
│   │       ├── report_generator.py
│   │       └── real_data_loader.py
│   ├── 📁 data_generation/       # Synthetic data generation
│   │   ├── synthetic_data_generator.py  # Main generator
│   │   ├── economic_relationships.py   # Economic models
│   │   └── shock_integration.py        # Economic shock modeling
│   ├── 📁 models/                # ML and forecasting models
│   │   ├── forecasting/          # Time series forecasting
│   │   ├── scenarios/            # Scenario modeling
│   │   └── validation/           # Model validation
│   └── 📁 utils/                 # General utilities
│       ├── database.py           # Database connections
│       └── __init__.py
├── 📁 tests/                     # Test suite
├── 📁 docs/                      # Documentation
├── 📁 logs/                      # Application logs
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/deluair/MacroScope.git
cd MacroScope

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Data

```bash
# Generate comprehensive synthetic economic datasets
python -m src.data_generation.synthetic_data_generator

# This creates:
# - Primary economic indicators (GDP, unemployment, inflation)
# - Regional economic data
# - Financial markets data
# - International trade data
# - Alternative indicators (satellite data, social sentiment)
# - Scenario datasets (recession, high inflation, productivity boom)
```

### 3. Launch Dashboard

```bash
# Start the interactive dashboard
streamlit run src/dashboard/streamlit_app.py

# Open your browser to: http://localhost:8501
```

### 4. API Server (Optional)

```bash
# Start the REST API server
python -m src.api.main

# API documentation available at: http://localhost:8000/docs
```

## 📊 Dashboard Features

### Executive Summary
- **Key Economic Indicators**: GDP growth, unemployment, inflation rates
- **Market Overview**: Stock indices, bond yields, currency rates
- **Data Quality Metrics**: Completeness, consistency, validity scores
- **Real-time Status**: Live vs. synthetic data mode indication

### Deep Dive Analytics
- **Time Series Analysis**: Interactive plotting with trend lines
- **Correlation Analysis**: Heatmaps showing relationships between indicators
- **Data Exploration**: Comprehensive dataset browsing and filtering
- **Export Capabilities**: Download analysis results as CSV

### Risk Assessment
- **Scenario Comparison**: Baseline vs. economic scenarios
- **Statistical Analysis**: Detailed comparison metrics
- **Risk Metrics**: Volatility, stress testing, scenario impact analysis
- **Visualization**: Side-by-side scenario plotting

## 🔧 Configuration

### API Keys Setup

Create a `.env` file or set environment variables:

```bash
# Federal Reserve Economic Data
FRED_API_KEY=your_fred_api_key

# Bureau of Labor Statistics
BLS_API_KEY=your_bls_api_key

# Census Bureau
CENSUS_API_KEY=your_census_api_key
```

### Data Sources

The platform supports multiple data sources:

- **FRED (Federal Reserve Economic Data)**: Primary economic indicators
- **Yahoo Finance**: Financial market data
- **Bureau of Labor Statistics**: Employment and labor data
- **Census Bureau**: Demographic and economic statistics
- **Synthetic Generation**: Realistic economic data simulation

## 🧠 Synthetic Data Generation

### Economic Models Implemented

1. **Phillips Curve**: Inflation-unemployment relationship
2. **Taylor Rule**: Central bank interest rate policy
3. **Okun's Law**: GDP growth and unemployment correlation
4. **Purchasing Power Parity**: Exchange rate dynamics
5. **Trade Balance Models**: International trade relationships

### Shock Integration

- **Systematic Shocks**: Monetary and fiscal policy changes
- **External Shocks**: Oil prices, geopolitical events, natural disasters
- **Financial Shocks**: Banking crises, credit events, market volatility
- **Technological Shocks**: Productivity improvements, automation

### Scenario Generation

- **Recession Scenario**: Economic downturn with higher unemployment
- **High Inflation Scenario**: Persistent price pressures
- **Productivity Boom**: Technology-driven growth acceleration

## 📈 Data Quality Features

- **Comprehensive Validation**: Automated data quality checks
- **Missing Data Handling**: Intelligent interpolation methods
- **Outlier Detection**: Statistical anomaly identification
- **Data Completeness Scoring**: Quality metrics and reporting

## 🛠️ Development

### Running Tests

```bash
# Run test suite
pytest tests/

# Run with coverage
pytest tests/ --cov=src
```

### Code Quality

```bash
# Format code
black src/

# Lint code
flake8 src/

# Type checking
mypy src/
```

### Adding New Features

1. **New Data Sources**: Extend `real_data_loader.py`
2. **Custom Indicators**: Modify `synthetic_data_generator.py`
3. **Dashboard Components**: Add to `src/dashboard/components/`
4. **Economic Models**: Enhance `economic_relationships.py`

## 📦 Dependencies

### Core Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **streamlit**: Web dashboard framework
- **plotly**: Interactive visualizations
- **scipy**: Scientific computing

### Data Sources
- **fredapi**: Federal Reserve economic data
- **yfinance**: Yahoo Finance market data
- **requests**: HTTP client for API calls

### Optional Extensions
- **scikit-learn**: Machine learning models
- **statsmodels**: Statistical analysis
- **prophet**: Time series forecasting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Federal Reserve Bank of St. Louis** for FRED API
- **Yahoo Finance** for market data access
- **Bureau of Labor Statistics** for employment data
- **Streamlit Community** for the amazing framework

## 📞 Support

- **Documentation**: See `/docs` directory
- **Issues**: [GitHub Issues](https://github.com/deluair/MacroScope/issues)
- **Discussions**: [GitHub Discussions](https://github.com/deluair/MacroScope/discussions)

---

*Built with ❤️ by the MacroScope team. Empowering economic analysis through data science.*