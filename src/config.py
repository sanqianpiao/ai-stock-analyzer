"""
Configuration module for AI Stock Analyzer.

Handles loading of environment variables, API keys, and application constants.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for the AI Stock Analyzer."""
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    NEWSAPI_KEY: Optional[str] = os.getenv("NEWSAPI_KEY")
    
    # OpenAI Configuration
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    
    # Application Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    CACHE_DURATION_MINUTES: int = int(os.getenv("CACHE_DURATION_MINUTES", "5"))
    
    # Data Configuration
    DEFAULT_PERIOD: str = os.getenv("DEFAULT_PERIOD", "1mo")
    MAX_NEWS_ARTICLES: int = int(os.getenv("MAX_NEWS_ARTICLES", "10"))
    
    # Technical Analysis Constants
    RSI_PERIOD: int = 14
    SHORT_MA_PERIOD: int = 5
    LONG_MA_PERIOD: int = 20
    
    # Sentiment Analysis
    SENTIMENT_THRESHOLD_POSITIVE: float = 0.3
    SENTIMENT_THRESHOLD_NEGATIVE: float = -0.3
    
    @classmethod
    def validate_api_keys(cls) -> bool:
        """Validate that required API keys are present."""
        missing_keys = []
        
        if not cls.OPENAI_API_KEY:
            missing_keys.append("OPENAI_API_KEY")
        
        if not cls.NEWSAPI_KEY:
            missing_keys.append("NEWSAPI_KEY")
        
        if missing_keys:
            print(f"Missing required API keys: {', '.join(missing_keys)}")
            print("Please set them in your .env file or environment variables.")
            return False
        
        return True
    
    @classmethod
    def get_valid_periods(cls) -> list[str]:
        """Get list of valid periods for stock data fetching."""
        return [
            "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"
        ]


# Create global config instance
config = Config()