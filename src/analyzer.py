"""
Quantitative Analysis Module for AI Stock Analyzer.

This module provides functions for calculating technical indicators, analyzing stock trends,
and generating quantitative insights from stock price data.
"""

import logging
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from .config import config

# Set up logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """Class for performing technical analysis on stock data."""
    
    def __init__(self):
        """Initialize the TechnicalAnalyzer."""
        self.rsi_period = config.RSI_PERIOD
        self.short_ma_period = config.SHORT_MA_PERIOD
        self.long_ma_period = config.LONG_MA_PERIOD
    
    def calculate_moving_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate simple moving averages for the given periods.
        
        Args:
            data (pd.DataFrame): Stock data with 'Close' column
            
        Returns:
            pd.DataFrame: Data with added MA columns
        """
        try:
            data = data.copy()
            data[f'MA_{self.short_ma_period}'] = data['Close'].rolling(window=self.short_ma_period).mean()
            data[f'MA_{self.long_ma_period}'] = data['Close'].rolling(window=self.long_ma_period).mean()
            
            logger.info(f"Calculated moving averages: {self.short_ma_period}-day and {self.long_ma_period}-day")
            return data
            
        except Exception as e:
            logger.error(f"Error calculating moving averages: {e}")
            return data
    
    def calculate_rsi(self, data: pd.DataFrame, period: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            data (pd.DataFrame): Stock data with 'Close' column
            period (int, optional): RSI calculation period
            
        Returns:
            pd.DataFrame: Data with added RSI column
        """
        if period is None:
            period = self.rsi_period
            
        try:
            data = data.copy()
            delta = data['Close'].diff()
            
            # Calculate gains and losses
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            # Calculate average gains and losses
            avg_gains = gains.rolling(window=period).mean()
            avg_losses = losses.rolling(window=period).mean()
            
            # Calculate RSI
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            data['RSI'] = rsi
            
            logger.info(f"Calculated RSI with {period}-day period")
            return data
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return data
    
    def calculate_bollinger_bands(self, data: pd.DataFrame, period: int = 20, std_dev: float = 2) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Args:
            data (pd.DataFrame): Stock data with 'Close' column
            period (int): Period for moving average
            std_dev (float): Standard deviation multiplier
            
        Returns:
            pd.DataFrame: Data with added Bollinger Band columns
        """
        try:
            data = data.copy()
            
            # Calculate middle band (simple moving average)
            data['BB_Middle'] = data['Close'].rolling(window=period).mean()
            
            # Calculate standard deviation
            rolling_std = data['Close'].rolling(window=period).std()
            
            # Calculate upper and lower bands
            data['BB_Upper'] = data['BB_Middle'] + (rolling_std * std_dev)
            data['BB_Lower'] = data['BB_Middle'] - (rolling_std * std_dev)
            
            # Calculate Bollinger Band position
            data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
            
            logger.info(f"Calculated Bollinger Bands with {period}-day period and {std_dev} std dev")
            return data
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return data
    
    def calculate_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-based indicators.
        
        Args:
            data (pd.DataFrame): Stock data with 'Volume' column
            
        Returns:
            pd.DataFrame: Data with added volume indicators
        """
        try:
            data = data.copy()
            
            # Volume moving average
            data['Volume_MA'] = data['Volume'].rolling(window=self.long_ma_period).mean()
            
            # Volume ratio (current volume vs average)
            data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
            
            # On-Balance Volume (OBV)
            price_change = data['Close'].diff()
            volume_direction = np.where(price_change > 0, data['Volume'], 
                                      np.where(price_change < 0, -data['Volume'], 0))
            data['OBV'] = volume_direction.cumsum()
            
            logger.info("Calculated volume indicators")
            return data
            
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {e}")
            return data
    
    def detect_trend(self, data: pd.DataFrame) -> str:
        """
        Detect overall trend based on moving averages and recent price action.
        
        Args:
            data (pd.DataFrame): Stock data with MA columns
            
        Returns:
            str: Trend description ('Bullish', 'Bearish', 'Sideways')
        """
        try:
            if len(data) < self.long_ma_period:
                return "Insufficient Data"
            
            latest_close = data['Close'].iloc[-1]
            short_ma = data[f'MA_{self.short_ma_period}'].iloc[-1]
            long_ma = data[f'MA_{self.long_ma_period}'].iloc[-1]
            
            # Calculate price change over different periods
            price_change_5d = (data['Close'].iloc[-1] / data['Close'].iloc[-6] - 1) * 100 if len(data) >= 6 else 0
            price_change_20d = (data['Close'].iloc[-1] / data['Close'].iloc[-21] - 1) * 100 if len(data) >= 21 else 0
            
            # Trend determination logic
            if pd.isna(short_ma) or pd.isna(long_ma):
                return "Insufficient Data"
            
            # Strong bullish: price > short MA > long MA and positive momentum
            if latest_close > short_ma > long_ma and price_change_5d > 2:
                return "Strong Bullish"
            
            # Bullish: price > both MAs or short MA > long MA
            elif latest_close > short_ma and short_ma > long_ma:
                return "Bullish"
            
            # Strong bearish: price < short MA < long MA and negative momentum
            elif latest_close < short_ma < long_ma and price_change_5d < -2:
                return "Strong Bearish"
            
            # Bearish: price < both MAs or short MA < long MA
            elif latest_close < short_ma and short_ma < long_ma:
                return "Bearish"
            
            # Sideways: mixed signals or consolidation
            else:
                return "Sideways"
                
        except Exception as e:
            logger.error(f"Error detecting trend: {e}")
            return "Unknown"
    
    def get_support_resistance_levels(self, data: pd.DataFrame, window: int = 20) -> Dict[str, float]:
        """
        Calculate support and resistance levels based on recent highs and lows.
        
        Args:
            data (pd.DataFrame): Stock data with High and Low columns
            window (int): Lookback window for calculation
            
        Returns:
            dict: Dictionary with support and resistance levels
        """
        try:
            if len(data) < window:
                return {"support": data['Low'].min(), "resistance": data['High'].max()}
            
            recent_data = data.tail(window)
            
            # Simple support/resistance calculation
            support = recent_data['Low'].min()
            resistance = recent_data['High'].max()
            
            # Alternative: use pivot points
            typical_price = (data['High'] + data['Low'] + data['Close']).tail(window).mean()
            pivot_point = typical_price
            
            return {
                "support": support,
                "resistance": resistance,
                "pivot_point": pivot_point
            }
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            return {"support": 0, "resistance": 0, "pivot_point": 0}
    
    def analyze_stock(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive technical analysis on stock data.
        
        Args:
            data (pd.DataFrame): Stock data with OHLCV columns
            
        Returns:
            dict: Comprehensive analysis results
        """
        try:
            if data.empty:
                logger.warning("Empty data provided for analysis")
                return {}
            
            # Calculate all indicators
            data_with_indicators = data.copy()
            data_with_indicators = self.calculate_moving_averages(data_with_indicators)
            data_with_indicators = self.calculate_rsi(data_with_indicators)
            data_with_indicators = self.calculate_bollinger_bands(data_with_indicators)
            data_with_indicators = self.calculate_volume_indicators(data_with_indicators)
            
            # Get latest values
            latest = data_with_indicators.iloc[-1]
            
            # Calculate key metrics
            current_price = latest['Close']
            price_change = latest['Close'] - data['Close'].iloc[-2] if len(data) > 1 else 0
            price_change_pct = (price_change / data['Close'].iloc[-2] * 100) if len(data) > 1 else 0
            
            # Get trend
            trend = self.detect_trend(data_with_indicators)
            
            # Get support/resistance
            levels = self.get_support_resistance_levels(data_with_indicators)
            
            # Calculate volatility (standard deviation of returns)
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
            
            analysis_result = {
                "timestamp": datetime.now().isoformat(),
                "ticker": latest.get('Ticker', 'Unknown'),
                "current_price": round(current_price, 2),
                "price_change": round(price_change, 2),
                "price_change_pct": round(price_change_pct, 2),
                "volume": int(latest['Volume']),
                "volume_ratio": round(latest.get('Volume_Ratio', 1.0), 2),
                "trend": trend,
                "technical_indicators": {
                    "rsi": round(latest.get('RSI', 50), 2),
                    "ma_5": round(latest.get(f'MA_{self.short_ma_period}', current_price), 2),
                    "ma_20": round(latest.get(f'MA_{self.long_ma_period}', current_price), 2),
                    "bb_upper": round(latest.get('BB_Upper', current_price), 2),
                    "bb_lower": round(latest.get('BB_Lower', current_price), 2),
                    "bb_position": round(latest.get('BB_Position', 0.5), 2),
                },
                "support_resistance": {
                    "support": round(levels["support"], 2),
                    "resistance": round(levels["resistance"], 2),
                    "pivot_point": round(levels["pivot_point"], 2),
                },
                "risk_metrics": {
                    "volatility_annualized": round(volatility, 2),
                    "beta": "N/A",  # Would need market data to calculate
                },
                "signals": self._generate_trading_signals(data_with_indicators)
            }
            
            logger.info(f"Completed technical analysis for {analysis_result['ticker']}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in stock analysis: {e}")
            return {"error": str(e)}
    
    def _generate_trading_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signals based on technical indicators.
        
        Args:
            data (pd.DataFrame): Data with technical indicators
            
        Returns:
            dict: Trading signals and recommendations
        """
        try:
            latest = data.iloc[-1]
            signals = {"buy": [], "sell": [], "neutral": []}
            
            # RSI signals
            rsi = latest.get('RSI', 50)
            if rsi < 30:
                signals["buy"].append("RSI oversold (< 30)")
            elif rsi > 70:
                signals["sell"].append("RSI overbought (> 70)")
            else:
                signals["neutral"].append(f"RSI neutral ({rsi:.1f})")
            
            # Moving average signals
            price = latest['Close']
            ma_5 = latest.get(f'MA_{self.short_ma_period}')
            ma_20 = latest.get(f'MA_{self.long_ma_period}')
            
            if pd.notna(ma_5) and pd.notna(ma_20):
                if price > ma_5 > ma_20:
                    signals["buy"].append("Price above both MAs - uptrend")
                elif price < ma_5 < ma_20:
                    signals["sell"].append("Price below both MAs - downtrend")
                else:
                    signals["neutral"].append("Mixed MA signals")
            
            # Bollinger Band signals
            bb_position = latest.get('BB_Position', 0.5)
            if bb_position < 0.1:
                signals["buy"].append("Price near lower Bollinger Band")
            elif bb_position > 0.9:
                signals["sell"].append("Price near upper Bollinger Band")
            
            # Volume signals
            volume_ratio = latest.get('Volume_Ratio', 1.0)
            if volume_ratio > 1.5:
                signals["neutral"].append("High volume - confirm other signals")
            elif volume_ratio < 0.5:
                signals["neutral"].append("Low volume - weak conviction")
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}")
            return {"buy": [], "sell": [], "neutral": []}


# Create global analyzer instance
technical_analyzer = TechnicalAnalyzer()


# Convenience functions
def analyze_stock(data: pd.DataFrame) -> Dict[str, Any]:
    """Convenience function for stock analysis."""
    return technical_analyzer.analyze_stock(data)


def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Convenience function to calculate all technical indicators."""
    data_with_indicators = technical_analyzer.calculate_moving_averages(data)
    data_with_indicators = technical_analyzer.calculate_rsi(data_with_indicators)
    data_with_indicators = technical_analyzer.calculate_bollinger_bands(data_with_indicators)
    data_with_indicators = technical_analyzer.calculate_volume_indicators(data_with_indicators)
    return data_with_indicators