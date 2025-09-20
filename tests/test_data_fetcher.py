"""
Unit tests for the data_fetcher module.

Tests the functionality of fetching stock data, news articles, and ticker validation
using mocked API responses to ensure reliability without external dependencies.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import json
import os
from datetime import datetime, timedelta

# Import the module under test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from src.data_fetcher import DataFetcher, fetch_stock_data, fetch_news, validate_ticker
    from src.config import config
except ImportError:
    # Handle import errors during testing setup
    pass


class TestDataFetcher:
    """Test class for DataFetcher functionality."""
    
    @pytest.fixture
    def data_fetcher(self):
        """Create a DataFetcher instance for testing."""
        return DataFetcher()
    
    @pytest.fixture
    def mock_stock_data(self):
        """Load mock stock data from JSON file."""
        mock_file_path = os.path.join(os.path.dirname(__file__), '..', 'mocks', 'sample_stock_data.json')
        with open(mock_file_path, 'r') as f:
            return json.load(f)
    
    @pytest.fixture
    def sample_yfinance_data(self):
        """Create sample yfinance data structure."""
        dates = pd.date_range('2024-08-20', periods=5, freq='D')
        data = pd.DataFrame({
            'Open': [224.38, 225.77, 223.88, 220.85, 217.49],
            'High': [226.05, 225.92, 224.12, 222.50, 218.32],
            'Low': [223.01, 222.56, 220.27, 217.71, 216.10],
            'Close': [225.77, 224.72, 224.12, 220.85, 217.49],
            'Volume': [46311900, 54156900, 42084200, 52722800, 54156900]
        }, index=dates)
        return data
    
    @patch('src.data_fetcher.yf.Ticker')
    def test_fetch_stock_data_success(self, mock_ticker, data_fetcher, sample_yfinance_data):
        """Test successful stock data fetching."""
        # Setup mock
        mock_stock = Mock()
        mock_stock.history.return_value = sample_yfinance_data
        mock_ticker.return_value = mock_stock
        
        # Test
        result = data_fetcher.fetch_stock_data('AAPL', '1mo')
        
        # Assertions
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        assert 'Ticker' in result.columns
        assert all(result['Ticker'] == 'AAPL')
        mock_ticker.assert_called_once_with('AAPL')
        mock_stock.history.assert_called_once_with(period='1mo')
    
    @patch('src.data_fetcher.yf.Ticker')
    def test_fetch_stock_data_empty_result(self, mock_ticker, data_fetcher):
        """Test handling of empty stock data result."""
        # Setup mock to return empty DataFrame
        mock_stock = Mock()
        mock_stock.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_stock
        
        # Test
        result = data_fetcher.fetch_stock_data('INVALID', '1mo')
        
        # Assertions
        assert result is None
    
    @patch('src.data_fetcher.yf.Ticker')
    def test_fetch_stock_data_invalid_period(self, mock_ticker, data_fetcher):
        """Test handling of invalid period parameter."""
        # Test with invalid period
        result = data_fetcher.fetch_stock_data('AAPL', 'invalid_period')
        
        # Assertions
        assert result is None
        mock_ticker.assert_not_called()
    
    @patch('src.data_fetcher.yf.Ticker')
    def test_get_stock_info_success(self, mock_ticker, data_fetcher, mock_stock_data):
        """Test successful stock info retrieval."""
        # Setup mock
        mock_stock = Mock()
        mock_stock.info = mock_stock_data['AAPL']['stock_info']
        mock_ticker.return_value = mock_stock
        
        # Test
        result = data_fetcher.get_stock_info('AAPL')
        
        # Assertions
        assert result is not None
        assert isinstance(result, dict)
        assert result['symbol'] == 'AAPL'
        assert result['shortName'] == 'Apple Inc.'
        assert 'marketCap' in result
        mock_ticker.assert_called_once_with('AAPL')
    
    @patch('src.data_fetcher.yf.Ticker')
    def test_get_stock_info_exception(self, mock_ticker, data_fetcher):
        """Test handling of exceptions in stock info retrieval."""
        # Setup mock to raise exception
        mock_ticker.side_effect = Exception("API Error")
        
        # Test
        result = data_fetcher.get_stock_info('AAPL')
        
        # Assertions
        assert result is None
    
    def test_fetch_news_no_client(self, data_fetcher):
        """Test news fetching when NewsAPI client is not initialized."""
        # Ensure news client is None
        data_fetcher.news_client = None
        
        # Test
        result = data_fetcher.fetch_news('AAPL')
        
        # Assertions
        assert result == []
    
    @patch('src.data_fetcher.yf.Ticker')
    def test_fetch_news_success(self, mock_ticker, data_fetcher, mock_stock_data):
        """Test successful news fetching."""
        # Setup mocks
        mock_stock = Mock()
        mock_stock.info = mock_stock_data['AAPL']['stock_info']
        mock_ticker.return_value = mock_stock
        
        mock_news_client = Mock()
        mock_news_response = {
            'articles': [
                {
                    'title': 'Apple Announces New iPhone Features',
                    'description': 'Apple Inc. unveiled new features...',
                    'url': 'https://example.com/apple-news-1',
                    'publishedAt': '2024-08-23T10:30:00Z',
                    'source': {'name': 'Tech News'}
                }
            ]
        }
        mock_news_client.get_everything.return_value = mock_news_response
        data_fetcher.news_client = mock_news_client
        
        # Test
        result = data_fetcher.fetch_news('AAPL', max_articles=1)
        
        # Assertions
        assert len(result) == 1
        assert result[0]['title'] == 'Apple Announces New iPhone Features'
        assert result[0]['source'] == 'Tech News'
        mock_news_client.get_everything.assert_called_once()
    
    @patch('src.data_fetcher.yf.Ticker')
    def test_validate_ticker_valid(self, mock_ticker, data_fetcher, sample_yfinance_data):
        """Test ticker validation with valid ticker."""
        # Setup mock
        mock_stock = Mock()
        mock_stock.history.return_value = sample_yfinance_data
        mock_ticker.return_value = mock_stock
        
        # Test
        result = data_fetcher.validate_ticker('AAPL')
        
        # Assertions
        assert result is True
        mock_ticker.assert_called_once_with('AAPL')
    
    @patch('src.data_fetcher.yf.Ticker')
    def test_validate_ticker_invalid(self, mock_ticker, data_fetcher):
        """Test ticker validation with invalid ticker."""
        # Setup mock to return empty DataFrame
        mock_stock = Mock()
        mock_stock.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_stock
        
        # Test
        result = data_fetcher.validate_ticker('INVALID')
        
        # Assertions
        assert result is False
    
    @patch('src.data_fetcher.yf.Ticker')
    def test_validate_ticker_exception(self, mock_ticker, data_fetcher):
        """Test ticker validation with exception."""
        # Setup mock to raise exception
        mock_ticker.side_effect = Exception("Network error")
        
        # Test
        result = data_fetcher.validate_ticker('AAPL')
        
        # Assertions
        assert result is False


class TestConvenienceFunctions:
    """Test the convenience functions."""
    
    @patch('src.data_fetcher.data_fetcher')
    def test_fetch_stock_data_convenience(self, mock_data_fetcher):
        """Test the convenience function for fetching stock data."""
        mock_data_fetcher.fetch_stock_data.return_value = pd.DataFrame({'Close': [100, 101, 102]})
        
        result = fetch_stock_data('AAPL', '1mo')
        
        assert result is not None
        mock_data_fetcher.fetch_stock_data.assert_called_once_with('AAPL', '1mo')
    
    @patch('src.data_fetcher.data_fetcher')
    def test_fetch_news_convenience(self, mock_data_fetcher):
        """Test the convenience function for fetching news."""
        mock_news = [{'title': 'Test News', 'description': 'Test Description'}]
        mock_data_fetcher.fetch_news.return_value = mock_news
        
        result = fetch_news('AAPL', max_articles=5)
        
        assert result == mock_news
        mock_data_fetcher.fetch_news.assert_called_once_with('AAPL', 5)
    
    @patch('src.data_fetcher.data_fetcher')
    def test_validate_ticker_convenience(self, mock_data_fetcher):
        """Test the convenience function for validating ticker."""
        mock_data_fetcher.validate_ticker.return_value = True
        
        result = validate_ticker('AAPL')
        
        assert result is True
        mock_data_fetcher.validate_ticker.assert_called_once_with('AAPL')


if __name__ == '__main__':
    pytest.main([__file__])