from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

class Settings:
    # API Keys - Set these as environment variables
    FRED_API_KEY = os.getenv("FRED_API_KEY")
    BLS_API_KEY = os.getenv("BLS_API_KEY")
    CENSUS_API_KEY = os.getenv("CENSUS_API_KEY")
    EIA_API_KEY = os.getenv("EIA_API_KEY")
    NOAA_TOKEN = os.getenv("NOAA_TOKEN")
    COMTRADE_API_KEY = os.getenv("COMTRADE_API_KEY")

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
