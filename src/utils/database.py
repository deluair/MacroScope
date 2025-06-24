"""
Database utilities for MacroScope
"""
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Optional, Dict, Any
from datetime import datetime
import logging

from config.settings import settings

logger = logging.getLogger(__name__)

# Database setup
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class EconomicIndicator(Base):
    """Economic indicator data model"""
    __tablename__ = "economic_indicators"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, index=True)
    indicator = Column(String(100), index=True)
    value = Column(Float)
    category = Column(String(50))
    source = Column(String(50))
    meta_data = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ForecastResult(Base):
    """Forecast results data model"""
    __tablename__ = "forecast_results"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, index=True)
    indicator = Column(String(100), index=True)
    model_name = Column(String(50))
    forecast_value = Column(Float)
    confidence_lower = Column(Float)
    confidence_upper = Column(Float)
    confidence_level = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class ModelPerformance(Base):
    """Model performance tracking"""
    __tablename__ = "model_performance"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(50), index=True)
    indicator = Column(String(100), index=True)
    evaluation_date = Column(DateTime, index=True)
    mae = Column(Float)  # Mean Absolute Error
    rmse = Column(Float)  # Root Mean Square Error
    mape = Column(Float)  # Mean Absolute Percentage Error
    r2_score = Column(Float)
    meta_data = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

def get_db() -> Session:
    """Get database session"""
    db = SessionLocal()
    try:
        return db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.close()
        raise

def create_tables():
    """Create all database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise

class DatabaseManager:
    """Database manager for economic data operations"""
    
    def __init__(self):
        self.engine = engine
        self.Session = SessionLocal
    
    def save_economic_data(self, df: pd.DataFrame, category: str, source: str = "synthetic") -> bool:
        """Save economic indicator data to database"""
        try:
            with self.Session() as session:
                for _, row in df.iterrows():
                    for column in df.columns:
                        if column != 'date' and pd.notna(row[column]):
                            indicator = EconomicIndicator(
                                date=row['date'],
                                indicator=column,
                                value=float(row[column]),
                                category=category,
                                source=source
                            )
                            session.add(indicator)
                session.commit()
                logger.info(f"Saved {len(df)} records for category: {category}")
                return True
        except Exception as e:
            logger.error(f"Error saving economic data: {e}")
            return False
    
    def save_forecast_results(self, forecasts: Dict[str, Any]) -> bool:
        """Save forecast results to database"""
        try:
            with self.Session() as session:
                for forecast in forecasts:
                    result = ForecastResult(**forecast)
                    session.add(result)
                session.commit()
                logger.info(f"Saved {len(forecasts)} forecast results")
                return True
        except Exception as e:
            logger.error(f"Error saving forecast results: {e}")
            return False
    
    def get_economic_data(self, 
                         indicator: str, 
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Retrieve economic indicator data"""
        try:
            with self.Session() as session:
                query = session.query(EconomicIndicator).filter(
                    EconomicIndicator.indicator == indicator
                )
                
                if start_date:
                    query = query.filter(EconomicIndicator.date >= start_date)
                if end_date:
                    query = query.filter(EconomicIndicator.date <= end_date)
                
                results = query.order_by(EconomicIndicator.date).all()
                
                data = []
                for result in results:
                    data.append({
                        'date': result.date,
                        'value': result.value,
                        'indicator': result.indicator,
                        'category': result.category
                    })
                
                return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error retrieving economic data: {e}")
            return pd.DataFrame()
    
    def get_forecast_data(self, 
                         indicator: str,
                         model_name: Optional[str] = None) -> pd.DataFrame:
        """Retrieve forecast data"""
        try:
            with self.Session() as session:
                query = session.query(ForecastResult).filter(
                    ForecastResult.indicator == indicator
                )
                
                if model_name:
                    query = query.filter(ForecastResult.model_name == model_name)
                
                results = query.order_by(ForecastResult.date).all()
                
                data = []
                for result in results:
                    data.append({
                        'date': result.date,
                        'forecast_value': result.forecast_value,
                        'confidence_lower': result.confidence_lower,
                        'confidence_upper': result.confidence_upper,
                        'confidence_level': result.confidence_level,
                        'model_name': result.model_name
                    })
                
                return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error retrieving forecast data: {e}")
            return pd.DataFrame()

# Global database manager instance
db_manager = DatabaseManager()

# Initialize database
create_tables() 