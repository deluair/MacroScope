from pathlib import Path
import os

# Base project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

class Settings:
    # API Keys
    FRED_API_KEY = os.getenv("FRED_API_KEY", "3a489b24c1a0898c157ed702c70c529f")
    BLS_API_KEY = os.getenv("BLS_API_KEY", "b155c91d886645b98e0d453434f16774")
    CENSUS_API_KEY = os.getenv("CENSUS_API_KEY", "75ca6118b0a4b7c1648255473c89f49c3f635a57")
    EIA_API_KEY = os.getenv("EIA_API_KEY", "mCDQSBcgcodgzpeuKgoZUXOdzNLXwRxGIL57T4g2")
    NOAA_TOKEN = os.getenv("NOAA_TOKEN", "xGuxpvhZhtWCtALWDYSbOzyHyUCDiRJs")
    COMTRADE_API_KEY = os.getenv("COMTRADE_API_KEY", "4bb037cbddd84bdaa9e40816f90592ae")

    # Data Directories
    DATA_DIR = PROJECT_ROOT / "data"
    SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"

    # Default date range for data generation
    START_DATE = "2010-01-01"
    END_DATE = "2023-12-31"

    def __init__(self):
        # Ensure directories exist
        self.SYNTHETIC_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Create settings instance
settings = Settings()
