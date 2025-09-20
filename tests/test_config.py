"""
Unit tests for the config module.

Tests configuration loading, validation, and default values.
"""

import pytest
import os
from unittest.mock import patch

# Import the module under test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from src.config import Config, config
except ImportError:
    # Handle import errors during testing setup
    pass


class TestConfig:
    """Test class for Config functionality."""
    
    def test_default_values(self):
        """Test that config has appropriate default values."""
        test_config = Config()
        
        # Test default values
        assert test_config.OPENAI_MODEL == "gpt-4o-mini"
        assert test_config.EMBEDDING_MODEL == "text-embedding-3-small"
        assert test_config.LOG_LEVEL == "INFO"
        assert test_config.CACHE_DURATION_MINUTES == 5
        assert test_config.DEFAULT_PERIOD == "1mo"
        assert test_config.MAX_NEWS_ARTICLES == 10
        assert test_config.RSI_PERIOD == 14
        assert test_config.SHORT_MA_PERIOD == 5
        assert test_config.LONG_MA_PERIOD == 20
    
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test_openai_key',
        'NEWSAPI_KEY': 'test_news_key',
        'OPENAI_MODEL': 'gpt-4',
        'LOG_LEVEL': 'DEBUG'
    })
    def test_environment_variable_loading(self):
        """Test that environment variables are loaded correctly."""
        test_config = Config()
        
        assert test_config.OPENAI_API_KEY == 'test_openai_key'
        assert test_config.NEWSAPI_KEY == 'test_news_key'
        assert test_config.OPENAI_MODEL == 'gpt-4'
        assert test_config.LOG_LEVEL == 'DEBUG'
    
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test_key',
        'NEWSAPI_KEY': 'test_key'
    })
    def test_validate_api_keys_success(self):
        """Test API key validation when keys are present."""
        result = Config.validate_api_keys()
        assert result is True
    
    @patch.dict(os.environ, {}, clear=True)
    def test_validate_api_keys_missing(self):
        """Test API key validation when keys are missing."""
        result = Config.validate_api_keys()
        assert result is False
    
    def test_get_valid_periods(self):
        """Test that valid periods are returned correctly."""
        periods = Config.get_valid_periods()
        
        assert isinstance(periods, list)
        assert "1d" in periods
        assert "1mo" in periods
        assert "1y" in periods
        assert "max" in periods
        assert len(periods) > 5  # Should have multiple periods
    
    def test_sentiment_thresholds(self):
        """Test sentiment analysis thresholds."""
        test_config = Config()
        
        assert test_config.SENTIMENT_THRESHOLD_POSITIVE == 0.3
        assert test_config.SENTIMENT_THRESHOLD_NEGATIVE == -0.3
        assert test_config.SENTIMENT_THRESHOLD_POSITIVE > test_config.SENTIMENT_THRESHOLD_NEGATIVE


class TestGlobalConfig:
    """Test the global config instance."""
    
    def test_global_config_exists(self):
        """Test that global config instance exists."""
        assert config is not None
        assert isinstance(config, Config)


if __name__ == '__main__':
    pytest.main([__file__])