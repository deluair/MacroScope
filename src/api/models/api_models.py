from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class TickerRequest(BaseModel):
    """Request model for a single stock ticker."""
    ticker: str
    start_date: str
    end_date: str

class FredSeriesRequest(BaseModel):
    """Request model for a single FRED series ID."""
    series_id: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class BlsSeriesRequest(BaseModel):
    """Request model for multiple BLS series IDs."""
    series_ids: List[str]
    start_year: str
    end_year: str

class CensusRequest(BaseModel):
    """Request model for Census data."""
    fields: List[str]
    geo: Dict[str, str]
    year: int

class ApiResponse(BaseModel):
    """Generic API response model."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
