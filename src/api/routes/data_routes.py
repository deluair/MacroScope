from fastapi import APIRouter, HTTPException
import yfinance as yf
from fredapi import Fred
import requests
import json
import pandas as pd
import logging
from src.api.models.api_models import TickerRequest, FredSeriesRequest, BlsSeriesRequest, CensusRequest, ApiResponse
from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
fred = Fred(api_key=settings.FRED_API_KEY)

@router.post("/data/yfinance", response_model=ApiResponse)
def get_yfinance_data(request: TickerRequest):
    """
    Fetches historical data for a given stock ticker from Yahoo Finance.
    """
    try:
        ticker = yf.Ticker(request.ticker)
        hist = ticker.history(start=request.start_date, end=request.end_date)
        if hist.empty:
            raise HTTPException(status_code=404, detail="No data found for the given ticker and date range.")
        hist.index = hist.index.strftime('%Y-%m-%d')
        return {"success": True, "message": "Data retrieved successfully", "data": hist.to_dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/data/fred", response_model=ApiResponse)
def get_fred_data(request: FredSeriesRequest):
    """
    Fetches data for a given FRED series ID.
    """
    try:
        series_data = fred.get_series(request.series_id, observation_start=request.start_date, observation_end=request.end_date)
        if not series_data.empty:
            series_data.index = series_data.index.strftime('%Y-%m-%d')
            return {"success": True, "message": "Data retrieved successfully", "data": series_data.to_dict()}
        raise HTTPException(status_code=404, detail="No data found for the given FRED series ID and date range.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/data/bls", response_model=ApiResponse)
def get_bls_data(request: BlsSeriesRequest):
    """
    Fetches data for given BLS series IDs using BLS API v2.
    """
    logger.info(f"Received BLS request for series: {request.series_ids}")
    try:
        headers = {'Content-type': 'application/json'}
        data = json.dumps({
            "seriesid": request.series_ids,
            "startyear": request.start_year,
            "endyear": request.end_year,
            "registrationkey": settings.BLS_API_KEY
        })
        
        logger.info("Posting request to BLS API v2")
        response = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data=data, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        
        json_data = response.json()
        
        if json_data.get('status') != 'REQUEST_SUCCEEDED':
            messages = ". ".join([msg['text'] for msg in json_data.get('message', [])])
            logger.error(f"BLS API returned an error: {messages}")
            raise HTTPException(status_code=400, detail=f"BLS API Error: {messages}")

        results = json_data.get('Results', {}).get('series', [])
        if not results:
            logger.warning("No data found for the given BLS series IDs and date range.")
            raise HTTPException(status_code=404, detail="No data found for the given BLS series IDs and date range.")

        all_series_data = {}
        for series in results:
            series_id = series['seriesID']
            df = pd.DataFrame(series['data'])
            df['date'] = pd.to_datetime(df['year'] + '-' + df['period'].str[1:] + '-01')
            df = df.set_index('date')
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            series_dict = df['value'].dropna().to_dict()
            all_series_data[series_id] = {k.strftime('%Y-%m-%d'): v for k, v in series_dict.items()}

        return {"success": True, "message": "Data retrieved successfully", "data": all_series_data}
        
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err} - {response.text}", exc_info=True)
        raise HTTPException(status_code=response.status_code, detail=f"Error from BLS API: {response.text}")
    except requests.exceptions.RequestException as e:
        logger.error(f"An error occurred while calling BLS API: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error calling BLS API: {e}")
    except Exception as e:
        logger.error(f"An error occurred while processing BLS data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@router.post("/data/census", response_model=ApiResponse)
def get_census_data(request: CensusRequest):
    """
    Fetches data from the US Census Bureau using direct API calls.
    Includes a fallback mechanism - if call with API key fails, it will try without key.
    """
    logger.info(f"Received Census request for fields: {request.fields} in year {request.year}")
    try:
        # First attempt with API key
        data = _call_census_api(request, use_key=True)
        return {"success": True, "message": "Data retrieved successfully with API key", "data": data}
    except Exception as e:
        logger.warning(f"First attempt with API key failed: {str(e)}. Trying without key...")
        try:
            # Fallback attempt without API key
            data = _call_census_api(request, use_key=False)
            return {"success": True, "message": "Data retrieved successfully (without API key)", "data": data}
        except Exception as fallback_e:
            logger.error(f"Both attempts to call Census API failed. Original error: {str(e)}, Fallback error: {str(fallback_e)}")
            return {"success": False, "message": f"Failed to retrieve Census data: {str(fallback_e)}", "data": {} }

def _call_census_api(request: CensusRequest, use_key: bool = True):
    """
    Helper function to call Census API with or without a key.
    """
    try:
        get_fields = ",".join(request.fields)
        geo_for = request.geo['for']
        
        # Construct URL with or without key
        if use_key:
            url = f"https://api.census.gov/data/{request.year}/acs/acs5?get={get_fields}&for={geo_for}&key={settings.CENSUS_API_KEY}"
            logger.info("Calling Census API WITH key")
        else:
            url = f"https://api.census.gov/data/{request.year}/acs/acs5?get={get_fields}&for={geo_for}"
            logger.info("Calling Census API WITHOUT key")
        
        logger.info(f"Census API URL: {url}")
        response = requests.get(url)
        logger.info(f"Census API Response Status: {response.status_code}")
        
        # Save response text for debugging
        response_text = response.text
        logger.info(f"Census API Response Text: {response_text}")
        
        if response.status_code != 200:
            logger.error(f"Census API Error Response: {response_text}")
            response.raise_for_status()
        
        try:
            data = response.json()
            if not data or len(data) <= 1:
                logger.warning("No data found for the given Census query.")
                raise ValueError("No data found for the given Census query.")
            
            # Convert to list of dictionaries
            header = data[0]
            records = [dict(zip(header, row)) for row in data[1:]]
            return {"records": records}
        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to decode JSON response: {str(json_err)}. Response text: {response_text}")
            raise ValueError(f"Invalid JSON response from Census API: {str(json_err)}")
    except Exception as e:
        logger.error(f"Error in _call_census_api: {str(e)}", exc_info=True)
        raise
