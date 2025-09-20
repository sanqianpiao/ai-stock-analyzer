"""
Data Fetcher Module for AI Stock Analyzer.

This module handles fetching stock data using yfinance and news data using NewsAPI.
It provides functions to retrieve historical stock data and recent news headlines
related to specific stock tickers.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import yfinance as yf
from newsapi import NewsApiClient
from .config import config

# Set up logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)


class DataFetcher:
    """Class for fetching stock and news data."""
    
    def __init__(self):
        """Initialize the DataFetcher with API clients."""
        self.news_client = None
        if config.NEWSAPI_KEY:
            try:
                self.news_client = NewsApiClient(api_key=config.NEWSAPI_KEY)
            except Exception as e:
                logger.warning(f"Failed to initialize NewsAPI client: {e}")
    
    def fetch_stock_data(
        self, 
        ticker: str, 
        period: str = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical stock data for a given ticker.
        
        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
            period (str): Time period for data ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            start_date (str, optional): Start date in 'YYYY-MM-DD' format
            end_date (str, optional): End date in 'YYYY-MM-DD' format
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data, or None if failed
        """
        if period is None:
            period = config.DEFAULT_PERIOD
            
        try:
            # Validate period
            if period not in config.get_valid_periods():
                logger.error(f"Invalid period: {period}. Valid periods: {config.get_valid_periods()}")
                return None
            
            # Create ticker object
            stock = yf.Ticker(ticker.upper())
            
            # Fetch data
            if start_date and end_date:
                data = stock.history(start=start_date, end=end_date)
            else:
                data = stock.history(period=period)
            
            if data.empty:
                logger.warning(f"No data found for ticker: {ticker}")
                return None
            
            # Add ticker column for reference
            data['Ticker'] = ticker.upper()
            
            logger.info(f"Successfully fetched {len(data)} records for {ticker.upper()}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching stock data for {ticker}: {e}")
            return None
    
    def get_stock_info(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get basic information about a stock.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            dict: Dictionary with stock information, or None if failed
        """
        try:
            stock = yf.Ticker(ticker.upper())
            info = stock.info
            
            # Extract key information
            key_info = {
                'symbol': info.get('symbol', ticker.upper()),
                'shortName': info.get('shortName', 'N/A'),
                'longName': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'marketCap': info.get('marketCap', 0),
                'currentPrice': info.get('currentPrice', 0),
                'previousClose': info.get('previousClose', 0),
                'beta': info.get('beta', 0),
                'trailingPE': info.get('trailingPE', 0),
                'dividendYield': info.get('dividendYield', 0),
                'volume': info.get('volume', 0),
                'averageVolume': info.get('averageVolume', 0),
            }
            
            logger.info(f"Successfully fetched info for {ticker.upper()}")
            return key_info
            
        except Exception as e:
            logger.error(f"Error fetching stock info for {ticker}: {e}")
            return None
    
    def fetch_news(
        self, 
        ticker: str, 
        max_articles: Optional[int] = None,
        days_back: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Fetch recent news headlines related to a stock ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            max_articles (int, optional): Maximum number of articles to fetch
            days_back (int): Number of days back to search for news
            
        Returns:
            list: List of news articles with title, description, url, and published date
        """
        if max_articles is None:
            max_articles = config.MAX_NEWS_ARTICLES
            
        if not self.news_client:
            logger.warning("NewsAPI client not initialized. Check NEWSAPI_KEY in config.")
            return []
        
        try:
            # Get stock info to use company name in search
            stock = yf.Ticker(ticker.upper())
            info = stock.info
            company_name = info.get('shortName', ticker.upper())
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Search for news articles
            query = f'"{company_name}" OR "{ticker.upper()}"'
            
            articles = self.news_client.get_everything(
                q=query,
                from_param=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='relevancy',
                page_size=max_articles
            )
            
            if not articles or 'articles' not in articles:
                logger.warning(f"No news articles found for {ticker}")
                return []
            
            # Process articles
            processed_articles = []
            for article in articles['articles'][:max_articles]:
                processed_article = {
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'url': article.get('url', ''),
                    'published_at': article.get('publishedAt', ''),
                    'source': article.get('source', {}).get('name', 'Unknown'),
                }
                processed_articles.append(processed_article)
            
            logger.info(f"Successfully fetched {len(processed_articles)} news articles for {ticker.upper()}")
            return processed_articles
            
        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {e}")
            return []
    
    def validate_ticker(self, ticker: str) -> bool:
        """
        Validate if a ticker symbol exists and has data.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            bool: True if ticker is valid, False otherwise
        """
        try:
            stock = yf.Ticker(ticker.upper())
            # Try to fetch minimal data to validate
            data = stock.history(period="1d")
            return not data.empty
        except Exception as e:
            logger.error(f"Error validating ticker {ticker}: {e}")
            return False


# Create global data fetcher instance
data_fetcher = DataFetcher()


# Convenience functions for direct use
def fetch_stock_data(ticker: str, period: str = None) -> Optional[pd.DataFrame]:
    """Convenience function to fetch stock data."""
    return data_fetcher.fetch_stock_data(ticker, period)


def fetch_news(ticker: str, max_articles: Optional[int] = None) -> List[Dict[str, Any]]:
    """Convenience function to fetch news."""
    return data_fetcher.fetch_news(ticker, max_articles)


def validate_ticker(ticker: str) -> bool:
    """Convenience function to validate ticker."""
    return data_fetcher.validate_ticker(ticker)