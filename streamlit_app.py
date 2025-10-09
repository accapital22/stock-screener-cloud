import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import warnings
import base64
from io import BytesIO
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
import tempfile
import os
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Clarity Pro 9.0 Suite",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        color: #2e86ab;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #2e86ab;
    }
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem !important;
        }
        .section-header {
            font-size: 1.5rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

class OptimizedScreener:
    def __init__(self):
        self.results = []
    
    def load_sp500_symbols(self):
        """Load S&P 500 symbols with robust error handling"""
        try:
            # Try multiple possible data sources
            urls = [
                "https://raw.githubusercontent.com/accapital22/stock-screener-cloud/main/sp500_symbols.csv",
                "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
                "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
            ]
            
            for url in urls:
                try:
                    df = pd.read_csv(url)
                    # Check for common column names that might contain symbols
                    symbol_col = None
                    for col in df.columns:
                        if 'symbol' in col.lower() or 'ticker' in col.lower() or 'Symbol' in col:
                            symbol_col = col
                            break
                    
                    if symbol_col:
                        st.success(f"✅ Loaded S&P 500 symbols from {url.split('/')[-1]}")
                        # Create a standardized dataframe
                        symbols_df = pd.DataFrame({
                            'symbol': df[symbol_col].str.upper().tolist(),
                            'price_range': ['mid'] * len(df)  # Default price range
                        })
                        return symbols_df
                        
                except Exception as e:
                    continue
            
            # If all URLs fail, use fallback
            st.warning("Using fallback S&P 500 symbol list")
            return self.get_fallback_symbols()
            
        except Exception as e:
            st.error(f"Error loading symbols: {str(e)}")
            return self.get_fallback_symbols()
    
    def get_fallback_symbols(self):
        """Provide comprehensive fallback symbol list"""
        fallback_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V',
            'PG', 'UNH', 'HD', 'DIS', 'PYPL', 'NFLX', 'ADBE', 'CRM', 'INTC', 'CSCO',
            'PEP', 'T', 'ABT', 'TMO', 'AVGO', 'COST', 'LLY', 'WMT', 'XOM', 'CVX',
            'MRK', 'PFE', 'ABBV', 'DHR', 'MDT', 'NEE', 'UNP', 'HON', 'RTX', 'LOW',
            'SPGI', 'ORCL', 'TXN', 'QCOM', 'AMGN', 'UPS', 'SBUX', 'BA', 'CAT', 'DE'
        ]
        
        # Categorize by approximate price ranges
        price_ranges = {
            'high': ['AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'LOW', 'SPGI', 'AMGN'],
            'mid': ['AAPL', 'MSFT', 'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'DIS', 'PYPL', 
                   'NFLX', 'ADBE', 'CRM', 'INTC', 'CSCO', 'PEP', 'T', 'ABT', 'TMO', 
                   'AVGO', 'COST', 'LLY', 'WMT', 'XOM', 'CVX', 'MRK', 'PFE', 'ABBV',
                   'DHR', 'MDT', 'NEE', 'UNP', 'HON', 'RTX', 'ORCL', 'TXN', 'QCOM',
                   'UPS', 'SBUX', 'BA', 'CAT', 'DE'],
            'low': []  # Add any low-priced stocks if needed
        }
        
        symbols_data = []
        for symbol in fallback_symbols:
            price_range = 'mid'  # default
            for range_name, symbols in price_ranges.items():
                if symbol in symbols:
                    price_range = range_name
                    break
            symbols_data.append({'symbol': symbol, 'price_range': price_range})
        
        return pd.DataFrame(symbols_data)
    
    def pre_filter_symbols(self, symbols_df, params):
        """Pre-filter symbols based on price ranges"""
        try:
            price_filtered = []
            price_ranges = {
                'low': (0, 30),
                'mid': (30, 300),  
                'high': (300, 5000)
            }
            
            for _, row in symbols_df.iterrows():
                symbol = row['symbol']
                price_range = row.get('price_range', 'mid')
                range_min, range_max = price_ranges.get(price_range, (0, 5000))
                
                if (range_max >= params['min_price'] and range_min <= params['max_price']):
                    price_filtered.append(symbol)
            
            st.success(f"🎯 Pre-filtering: {len(price_filtered)} stocks likely in price range ${params['min_price']}-${params['max_price']}")
            return price_filtered
        except Exception as e:
            st.error(f"Error in pre-filtering: {str(e)}")
            return symbols_df['symbol'].tolist()  # Return all symbols if filtering fails
    
    def calculate_ema(self, prices, period=50):
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=period, adjust=False).mean()
    
    def calculate_sma(self, prices, period=50):
        """Calculate Simple Moving Average"""
        return prices.rolling(window=period).mean()
    
    def calculate_wma(self, prices, period=50):
        """Calculate Weighted Moving Average"""
        def wma_calc(x):
            weights = np.arange(1, len(x) + 1)
            return np.dot(x, weights) / weights.sum()
        
        return prices.rolling(window=period).apply(wma_calc, raw=True)
    
    def calculate_moving_average(self, prices, ma_type, period):
        """Calculate moving average based on type"""
        if ma_type == "REGIONAL":
            return self.calculate_ema(prices, period)
        elif ma_type == "GLOBAL":
            return self.calculate_sma(prices, period)
        elif ma_type == "NATIONAL":
            return self.calculate_wma(prices, period)
        else:
            return self.calculate_ema(prices, period)  # Default to EMA
    
    def estimate_delta(self, stock_price, strike_price, days_to_expiry, option_type='call'):
        """Estimate option delta"""
        try:
            moneyness = stock_price / strike_price
            if option_type == 'call':
                if moneyness > 1.1: return 0.9
                elif moneyness > 1.0: return 0.7
                elif moneyness > 0.9: return 0.5
                else: return 0.3
            else:
                if moneyness < 0.9: return 0.9
                elif moneyness < 1.0: return 0.7
                elif moneyness < 1.1: return 0.5
                else: return 0.3
        except:
            return 0.5
    
    def get_detailed_options_data(self, symbol, current_price, params):
        """Get detailed options information with robust error handling"""
        try:
            stock = yf.Ticker(symbol)
            expirations = stock.options
            
            if not expirations:
                return {'has_valid_options': False, 'options_count': 0, 'details': []}
            
            valid_options = []
            
            for exp_date in expirations[:2]:  # Check first 2 expirations
                try:
                    exp_dt = datetime.strptime(exp_date, '%Y-%m-%d')
                    days_until_exp = (exp_dt - datetime.now()).days
                    
                    if not (params['min_days_to_exp'] <= days_until_exp <= params['max_days_to_exp']):
                        continue
                    
                    options_chain = stock.option_chain(exp_date)
                    calls = options_chain.calls
                    
                    # Ensure required columns exist
                    required_cols = ['lastPrice', 'strike', 'openInterest', 'volume', 'bid', 'ask', 'inTheMoney']
                    if not all(col in calls.columns for col in required_cols):
                        continue
                    
                    # Filter calls by our criteria
                    filtered_calls = calls[
                        (calls['lastPrice'] >= params['min_option_price']) & 
                        (calls['lastPrice'] <= params['max_option_price']) &
                        (calls['openInterest'].fillna(0) > params['min_open_interest']) &
                        (calls['volume'].fillna(0) > 0)
                    ].copy()
                    
                    if len(filtered_calls) > 0:
                        for _, option in filtered_calls.iterrows():
                            estimated_delta = self.estimate_delta(
                                current_price, option['strike'], days_until_exp, 'call'
                            )
                            
                            if params['min_delta'] <= estimated_delta <= params['max_delta']:
                                option_details = {
                                    'symbol': symbol,
                                    'expiration': exp_date,
                                    'days_to_exp': days_until_exp,
                                    'strike': option['strike'],
                                    'option_price': option['lastPrice'],
                                    'bid': option['bid'],
                                    'ask': option['ask'],
                                    'volume': option['volume'],
                                    'open_interest': option['openInterest'],
                                    'implied_volatility': option.get('impliedVolatility', 0),
                                    'delta': estimated_delta,
                                    'in_the_money': option['inTheMoney'],
                                    'premium_value': option['lastPrice'] * 100,
                                    'breakeven': option['strike'] + option['lastPrice']
                                }
                                valid_options.append(option_details)
                    
                except Exception as e:
                    continue
            
            return {
                'has_valid_options': len(valid_options) > 0,
                'options_count': len(valid_options),
                'details': valid_options
            }
            
        except Exception as e:
            return {'has_valid_options': False, 'options_count': 0, 'details': []}
    
    def screen_stock(self, symbol, params):
        """Screen individual stock with options data"""
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1y", interval="1d")  # 1 year history
            
            if len(hist) < params['ma_period']:
                return None, "Insufficient data for moving average"
            
            current_price = hist['Close'].iloc[-1]
            if not (params['min_price'] <= current_price <= params['max_price']):
                return None, f"Price ${current_price:.2f} outside range"
            
            avg_volume = hist['Volume'].tail(20).mean()
            if avg_volume < params['min_volume']:
                return None, f"Volume {avg_volume:,.0f} below requirement"
            
            # Calculate selected moving average
            ma_type = params['ma_type']
            ma_period = params['ma_period']
            
            hist['MA'] = self.calculate_moving_average(hist['Close'], ma_type, ma_period)
            current_ma = hist['MA'].iloc[-1]
            ma_diff_percent = ((current_price - current_ma) / current_ma) * 100
            
            # NEW FILTERING RULE: Stock price must be greater than bank price
            if current_price <= current_ma:
                return None, f"Stock price ${current_price:.2f} ≤ bank price ${current_ma:.2f}"
            
            if abs(ma_diff_percent) > params['ma_threshold']:
                return None, f"{ma_type} diff {ma_diff_percent:.2f}% > threshold"
            
            info = stock.info
            market_cap = info.get('marketCap', 0)
            if market_cap < params['min_market_cap']:
                return None, f"Market cap ${market_cap/1e9:.1f}B below requirement"
            
            # OPTIONS CHECK
            options_data = self.get_detailed_options_data(symbol, current_price, params)
            if not options_data['has_valid_options']:
                return None, "No valid options found"
            
            stock_data = {
                'symbol': symbol,
                'price': current_price,
                'ma': current_ma,
                'ma_type': ma_type,
                'ma_period': ma_period,
                'ma_diff_percent': ma_diff_percent,
                'volume': avg_volume,
                'market_cap': market_cap,
                'options_count': options_data['options_count'],
                'options_details': options_data['details'],
                'chart_data': hist[['Close', 'MA']].tail(30)
            }
            
            return stock_data, "PASS"
            
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def run_optimized_screener(self, params):
    """Run optimized futures screening for bank proximity with cancel support"""
    try:
        symbols_df = self.load_futures_symbols()
        
        st.info(f"📊 Loaded {len(symbols_df)} futures contracts")
        
        # Filter by category if specified
        if params.get('category_filter'):
            symbols_df = symbols_df[symbols_df['category'].isin(params['category_filter'])]
            st.info(f"🎯 Filtered to {len(symbols_df)} contracts in selected categories")
        
        qualified_symbols = symbols_df['symbol'].tolist()
        
        if not qualified_symbols:
            st.error("No futures contracts passed category filtering")
            return []
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # ADD CANCEL CHECK LOOP
        for i, symbol in enumerate(qualified_symbols):
            # Check if cancellation was requested
            if st.session_state.cancel_futures_screening:
                st.warning("🛑 Futures screening cancelled by user")
                st.session_state.futures_screening_active = False
                st.session_state.cancel_futures_screening = False
                return results
            
            status_text.text(f"🔍 Screening {symbol} ({i+1}/{len(qualified_symbols)})...")
            result, message = self.screen_futures_contract(symbol, params)
            
            if result:
                results.append(result)
                st.success(f"✅ {symbol}: {message}")
            
            progress_bar.progress((i + 1) / len(qualified_symbols))
            time.sleep(0.3)
        
        status_text.text("Screening complete!")
        st.session_state.futures_screening_active = False
        
        if not results:
            st.error("❌ No futures contracts passed screening criteria!")
            st.info("💡 Try relaxing your price range or bank proximity threshold")
        
        return results
        
    except Exception as e:
        st.session_state.futures_screening_active = False
        st.error(f"Error in screening process: {str(e)}")
        return []

class OptimizedFuturesScreener:
    def __init__(self):
        self.results = []
    
    def load_futures_symbols(self):
        """Load major futures symbols"""
        try:
            # Major futures contracts with their Yahoo Finance symbols
            futures_data = [
                            # Equity Index Futures (Multiple Variations)
                            {'symbol': 'ES=F', 'name': 'S&P 500 E-Mini Front Month', 'category': 'equity', 'exchange': 'CME', 'variations': ['ES1!', 'ESc1', '^GSPC']},
                            {'symbol': 'NQ=F', 'name': 'NASDAQ E-Mini Front Month', 'category': 'equity', 'exchange': 'CME', 'variations': ['NQ1!', 'NQc1', '^NDX']},
                            {'symbol': 'YM=F', 'name': 'Dow E-Mini Front Month', 'category': 'equity', 'exchange': 'CBOT', 'variations': ['YM1!', 'YMc1', '^DJI']},
                            {'symbol': 'RTY=F', 'name': 'Russell 2000 E-Mini Front Month', 'category': 'equity', 'exchange': 'CME', 'variations': ['RTY1!', 'RTYc1', '^RUT']},
                            
                            # Commodity Futures - Energy
                            {'symbol': 'CL=F', 'name': 'Crude Oil WTI Front Month', 'category': 'energy', 'exchange': 'NYMEX', 'variations': ['CL1!', 'CLc1']},
                            {'symbol': 'NG=F', 'name': 'Natural Gas Front Month', 'category': 'energy', 'exchange': 'NYMEX', 'variations': ['NG1!', 'NGc1']},
                            {'symbol': 'RB=F', 'name': 'RBOB Gasoline Front Month', 'category': 'energy', 'exchange': 'NYMEX', 'variations': ['RB1!', 'RBc1']},
                            {'symbol': 'HO=F', 'name': 'Heating Oil Front Month', 'category': 'energy', 'exchange': 'NYMEX', 'variations': ['HO1!', 'HOc1']},
                            
                            # Commodity Futures - Metals
                            {'symbol': 'GC=F', 'name': 'Gold Front Month', 'category': 'metals', 'exchange': 'COMEX', 'variations': ['GC1!', 'GCc1']},
                            {'symbol': 'SI=F', 'name': 'Silver Front Month', 'category': 'metals', 'exchange': 'COMEX', 'variations': ['SI1!', 'SIc1']},
                            {'symbol': 'HG=F', 'name': 'Copper Front Month', 'category': 'metals', 'exchange': 'COMEX', 'variations': ['HG1!', 'HGc1']},
                            {'symbol': 'PL=F', 'name': 'Platinum Front Month', 'category': 'metals', 'exchange': 'NYMEX', 'variations': ['PL1!', 'PLc1']},
                            {'symbol': 'PA=F', 'name': 'Palladium Front Month', 'category': 'metals', 'exchange': 'NYMEX', 'variations': ['PA1!', 'PAc1']},
                            
                            # Commodity Futures - Grains
                            {'symbol': 'ZC=F', 'name': 'Corn Front Month', 'category': 'grains', 'exchange': 'CBOT', 'variations': ['ZC1!', 'ZCc1']},
                            {'symbol': 'ZW=F', 'name': 'Wheat Front Month', 'category': 'grains', 'exchange': 'CBOT', 'variations': ['ZW1!', 'ZWc1']},
                            {'symbol': 'ZS=F', 'name': 'Soybeans Front Month', 'category': 'grains', 'exchange': 'CBOT', 'variations': ['ZS1!', 'ZSc1']},
                            {'symbol': 'ZM=F', 'name': 'Soybean Meal Front Month', 'category': 'grains', 'exchange': 'CBOT', 'variations': ['ZM1!', 'ZMc1']},
                            {'symbol': 'ZL=F', 'name': 'Soybean Oil Front Month', 'category': 'grains', 'exchange': 'CBOT', 'variations': ['ZL1!', 'ZLc1']},
                            
                            # Currency Futures
                            {'symbol': '6E=F', 'name': 'Euro FX Front Month', 'category': 'fx', 'exchange': 'CME', 'variations': ['6E1!', '6Ec1']},
                            {'symbol': '6J=F', 'name': 'Japanese Yen Front Month', 'category': 'fx', 'exchange': 'CME', 'variations': ['6J1!', '6Jc1']},
                            {'symbol': '6B=F', 'name': 'British Pound Front Month', 'category': 'fx', 'exchange': 'CME', 'variations': ['6B1!', '6Bc1']},
                            {'symbol': '6C=F', 'name': 'Canadian Dollar Front Month', 'category': 'fx', 'exchange': 'CME', 'variations': ['6C1!', '6Cc1']},
                            {'symbol': '6A=F', 'name': 'Australian Dollar Front Month', 'category': 'fx', 'exchange': 'CME', 'variations': ['6A1!', '6Ac1']},
                            {'symbol': '6S=F', 'name': 'Swiss Franc Front Month', 'category': 'fx', 'exchange': 'CME', 'variations': ['6S1!', '6Sc1']},
                            
                            # Interest Rate Futures
                            {'symbol': 'ZN=F', 'name': '10-Year T-Note Front Month', 'category': 'rates', 'exchange': 'CBOT', 'variations': ['ZN1!', 'ZNc1', '^TNX']},
                            {'symbol': 'ZB=F', 'name': '30-Year T-Bond Front Month', 'category': 'rates', 'exchange': 'CBOT', 'variations': ['ZB1!', 'ZBc1', '^TYX']},
                            {'symbol': 'ZF=F', 'name': '5-Year T-Note Front Month', 'category': 'rates', 'exchange': 'CBOT', 'variations': ['ZF1!', 'ZFc1']},
                            {'symbol': 'ZT=F', 'name': '2-Year T-Note Front Month', 'category': 'rates', 'exchange': 'CBOT', 'variations': ['ZT1!', 'ZTc1']},
                            {'symbol': 'GE=F', 'name': 'Eurodollar Front Month', 'category': 'rates', 'exchange': 'CME', 'variations': ['GE1!', 'GEc1']},
                            
                            # Volatility Futures
                            {'symbol': 'VX=F', 'name': 'VIX Futures Front Month', 'category': 'volatility', 'exchange': 'CBOE', 'variations': ['VX1!', 'VXc1', '^VIX']},
                            
                            # Crypto Futures
                            {'symbol': 'BTC=F', 'name': 'Bitcoin CME Futures', 'category': 'crypto', 'exchange': 'CME', 'variations': ['BTC-USD', 'BTCUSDT', 'XBTUSD']},
                            {'symbol': 'ETH=F', 'name': 'Ethereum CME Futures', 'category': 'crypto', 'exchange': 'CME', 'variations': ['ETH-USD', 'ETHUSDT']},
                            
                            # Additional Major Commodities
                            {'symbol': 'KC=F', 'name': 'Coffee Front Month', 'category': 'softs', 'exchange': 'ICE', 'variations': ['KC1!', 'KCc1']},
                            {'symbol': 'CT=F', 'name': 'Cotton Front Month', 'category': 'softs', 'exchange': 'ICE', 'variations': ['CT1!', 'CTc1']},
                            {'symbol': 'SB=F', 'name': 'Sugar Front Month', 'category': 'softs', 'exchange': 'ICE', 'variations': ['SB1!', 'SBc1']},
                            {'symbol': 'CC=F', 'name': 'Cocoa Front Month', 'category': 'softs', 'exchange': 'ICE', 'variations': ['CC1!', 'CCc1']}
]
            
            symbols_df = pd.DataFrame(futures_data)
            st.success(f"✅ Loaded {len(futures_data)} major futures contracts")
            return symbols_df
            
        except Exception as e:
            st.error(f"Error loading futures symbols: {str(e)}")
            return self.get_fallback_futures_symbols()
    
    def get_fallback_futures_symbols(self):
        """Provide fallback futures symbol list"""
        fallback_futures = [
            {'symbol': 'ES=F', 'name': 'S&P 500 E-Mini Front Month', 'category': 'equity', 'exchange': 'CME'},
            {'symbol': 'NQ=F', 'name': 'NASDAQ E-Mini Front Month', 'category': 'equity', 'exchange': 'CME'},
            {'symbol': 'CL=F', 'name': 'Crude Oil WTI Front Month', 'category': 'energy', 'exchange': 'NYMEX'},
            {'symbol': 'GC=F', 'name': 'Gold Front Month', 'category': 'metals', 'exchange': 'COMEX'},
            {'symbol': '6E=F', 'name': 'Euro FX Front Month', 'category': 'fx', 'exchange': 'CME'},
            {'symbol': 'ZN=F', 'name': '10-Year T-Note Front Month', 'category': 'rates', 'exchange': 'CBOT'},
            # Crypto Futures
            {'symbol': 'BTC=F', 'name': 'Bitcoin CME Futures', 'category': 'crypto', 'exchange': 'CME'},
            {'symbol': 'ETH=F', 'name': 'Ethereum CME Futures', 'category': 'crypto', 'exchange': 'CME'},
        ]
        return pd.DataFrame(fallback_futures)
    
    def calculate_ema(self, prices, period=50):
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=period, adjust=False).mean()
    
    def calculate_sma(self, prices, period=50):
        """Calculate Simple Moving Average"""
        return prices.rolling(window=period).mean()
    
    def calculate_wma(self, prices, period=50):
        """Calculate Weighted Moving Average"""
        def wma_calc(x):
            weights = np.arange(1, len(x) + 1)
            return np.dot(x, weights) / weights.sum()
        
        return prices.rolling(window=period).apply(wma_calc, raw=True)
    
    def calculate_moving_average(self, prices, ma_type, period):
        """Calculate moving average based on type"""
        if ma_type == "REGIONAL":
            return self.calculate_ema(prices, period)
        elif ma_type == "GLOBAL":
            return self.calculate_sma(prices, period)
        elif ma_type == "NATIONAL":
            return self.calculate_wma(prices, period)
        else:
            return self.calculate_ema(prices, period)
    
    def screen_futures_contract(self, symbol, params):
        """Screen individual futures contract for bank proximity"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y", interval="1d")
            
            if len(hist) < params['ma_period']:
                return None, "Insufficient data"
            
            current_price = hist['Close'].iloc[-1]
            if not (params['min_price'] <= current_price <= params['max_price']):
                return None, f"Price ${current_price:.2f} outside range"
            
            avg_volume = hist['Volume'].tail(20).mean()
            if avg_volume < params['min_volume']:
                return None, f"Volume {avg_volume:,.0f} below requirement"
            
            # Calculate selected moving average
            ma_type = params['ma_type']
            ma_period = params['ma_period']
            
            hist['MA'] = self.calculate_moving_average(hist['Close'], ma_type, ma_period)
            current_ma = hist['MA'].iloc[-1]
            ma_diff_percent = ((current_price - current_ma) / current_ma) * 100
            
            if abs(ma_diff_percent) > params['ma_threshold']:
                return None, f"Bank diff {ma_diff_percent:.2f}% > threshold"
            
            futures_data = {
                'symbol': symbol,
                'name': ticker.info.get('shortName', symbol),
                'price': current_price,
                'ma': current_ma,
                'ma_type': ma_type,
                'ma_period': ma_period,
                'ma_diff_percent': ma_diff_percent,
                'volume': avg_volume,
                'chart_data': hist[['Close', 'MA']].tail(30)
            }
            
            return futures_data, "PASS"
            
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def run_optimized_screener(self, params):
    """Run optimized screening with pre-filtering and cancel support"""
    try:
        symbols_df = self.load_sp500_symbols()
        
        # Debug: Show what columns we have
        st.write(f"📋 Loaded data with columns: {list(symbols_df.columns)}")
        
        if 'symbol' not in symbols_df.columns:
            st.error("❌ 'symbol' column not found in the data. Available columns: " + str(list(symbols_df.columns)))
            # Try to use first column as symbols
            first_col = symbols_df.columns[0]
            symbols_df = symbols_df.rename(columns={first_col: 'symbol'})
            st.info(f"🔄 Using '{first_col}' as symbol column")
        
        all_symbols = symbols_df['symbol'].tolist()
        st.info(f"📊 Loaded {len(all_symbols)} S&P 500 symbols")
        
        qualified_symbols = self.pre_filter_symbols(symbols_df, params)
        
        if not qualified_symbols:
            st.error("No stocks passed initial price range filtering")
            return []
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Add longer delay to avoid rate limiting
        delay_between_requests = 0.3  # Increased from 0.1 to 0.3 seconds
        
        # ADD CANCEL CHECK LOOP
        for i, symbol in enumerate(qualified_symbols):
            # Check if cancellation was requested
            if st.session_state.cancel_stock_screening:
                st.warning("🛑 Stock screening cancelled by user")
                st.session_state.stock_screening_active = False
                st.session_state.cancel_stock_screening = False
                return results
            
            status_text.text(f"🔍 Screening {symbol} ({i+1}/{len(qualified_symbols)})...")
            result, message = self.screen_stock(symbol, params)
            
            # ONLY SHOW PASSING SYMBOLS
            if result:
                results.append(result)
                st.success(f"✅ {symbol}: {message} | Options: {result['options_count']}")
            
            progress_bar.progress((i + 1) / len(qualified_symbols))
            time.sleep(delay_between_requests)
        
        status_text.text("Screening complete!")
        st.session_state.stock_screening_active = False
        
        # Show summary
        if not results:
            st.error("❌ No stocks passed screening criteria!")
            st.info("💡 Try relaxing your screening criteria (price range, volume, options filters)")
        
        return results
        
    except Exception as e:
        st.session_state.stock_screening_active = False
        st.error(f"Error in screening process: {str(e)}")
        return []

def display_options_details(selected_stock):
    """Display detailed options information"""
    if not selected_stock['options_details']:
        st.info("No options details available for this stock.")
        return
    
    st.markdown("### 📊 Options Chain Details")
    
    # Convert options details to DataFrame
    options_data = []
    for option in selected_stock['options_details']:
        options_data.append({
            'Expiration': option['expiration'],
            'Days to Exp': option['days_to_exp'],
            'Strike': f"${option['strike']}",
            'Option Price': f"${option['option_price']:.2f}",
            'Bid/Ask': f"${option['bid']:.2f}/${option['ask']:.2f}",
            'Volume': f"{option['volume']:,}",
            'Open Interest': f"{option['open_interest']:,}",
            'Delta': f"{option['delta']:.2f}",
            'ITM': 'Yes' if option['in_the_money'] else 'No',
            'Premium/Contract': f"${option['premium_value']:.2f}",
            'Breakeven': f"${option['breakeven']:.2f}"
        })
    
    options_df = pd.DataFrame(options_data)
    
    # Display options in tabs
    tab1, tab2 = st.tabs(["📋 Options Table", "🎯 Options Analysis"])
    
    with tab1:
        st.dataframe(options_df, use_container_width=True)
        
        # Download options data
        csv = options_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Options Data",
            data=csv,
            file_name=f"{selected_stock['symbol']}_options.csv",
            mime="text/csv"
        )
    
    with tab2:
        # Options analysis
        st.markdown("#### Options Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_premium = sum(opt['premium_value'] for opt in selected_stock['options_details'])
            st.metric("Total Premium Value", f"${total_premium:.2f}")
        with col2:
            avg_delta = np.mean([opt['delta'] for opt in selected_stock['options_details']])
            st.metric("Average Delta", f"{avg_delta:.2f}")
        with col3:
            itm_count = sum(1 for opt in selected_stock['options_details'] if opt['in_the_money'])
            st.metric("ITM Options", itm_count)
        with col4:
            total_oi = sum(opt['open_interest'] for opt in selected_stock['options_details'])
            st.metric("Total Open Interest", f"{total_oi:,}")

def create_tradingview_watchlist(symbols):
    """Create TradingView watchlist content"""
    if not symbols:
        return None
    
    # Format symbols for TradingView (comma-separated)
    watchlist_content = ",".join(symbols)
    
    return watchlist_content

def create_tradingview_multichart_url(symbols):
    """Create TradingView multi-chart URL"""
    if not symbols:
        return None
    
    # TradingView multi-chart URL format
    base_url = "https://www.tradingview.com/chart/"
    symbols_param = ",".join([f"NYSE:{s}" for s in symbols])
    url = f"{base_url}?symbol={symbols_param}"
    
    return url

def create_tradingview_watchlist_file(symbols):
    """Create a watchlist file that can be imported into TradingView"""
    if not symbols:
        return None
    
    # TradingView watchlist format (simple text file with one symbol per line)
    watchlist_content = "\n".join(symbols)
    
    return watchlist_content

def create_futures_tradingview_watchlist(symbols):
    """Create TradingView watchlist content for futures"""
    if not symbols:
        return None
    
    # Format futures symbols for TradingView (comma-separated)
    watchlist_content = ",".join(symbols)
    
    return watchlist_content

def create_futures_tradingview_watchlist_file(symbols):
    """Create a watchlist file for futures that can be imported into TradingView"""
    if not symbols:
        return None
    
    # TradingView watchlist format for futures
    watchlist_content = "\n".join(symbols)
    
    return watchlist_content

def get_bank_color(ma_type):
    """Get color for Bank Type"""
    colors = {
        "GLOBAL": "#ffa726",  # Orange
        "NATIONAL": "#fb0707",  # Red
        "REGIONAL": "#01f90b"   # Green
    }
    return colors.get(ma_type, "#ff7f0e")

def get_bank_name(ma_period):
    """Get Bank name for period"""
    bank_names = {
        33: "JP Morgan",
        50: "Barclays", 
        198: "BlackRock"
    }
    return bank_names.get(ma_period, f"Bank {ma_period}")

def sort_results_for_pdf(results):
    """Sort results for PDF report with the specified hierarchy"""
    if not results:
        return []
    
    # Flatten the results to include option details for sorting
    flattened_results = []
    for stock in results:
        if stock['options_details']:
            for option in stock['options_details']:
                flattened_results.append({
                    'symbol': stock['symbol'],
                    'stock_price': stock['price'],
                    'bank_price': stock['ma'],
                    'bank_diff_percent': stock['ma_diff_percent'],
                    'option_strike': option['strike'],
                    'option_price': option['option_price'],
                    'delta': option['delta'],
                    'open_interest': option['open_interest'],
                    'bid': option['bid'],
                    'ask': option['ask'],
                    'expiration': option['expiration'],
                    'days_to_exp': option['days_to_exp'],
                    'itm': option['in_the_money']
                })
    
    # Convert to DataFrame for easier sorting
    df = pd.DataFrame(flattened_results)
    
    if len(df) == 0:
        return []
    
    # Apply the sorting hierarchy: Delta (desc), Option_Price (asc), Symbol (asc)
    df_sorted = df.sort_values(['delta', 'option_price', 'symbol'], 
                              ascending=[False, True, True])
    
    return df_sorted.to_dict('records')

def create_pdf_report(results, params):
    """Create a PDF report of screening results in landscape mode"""
    try:
        from reportlab.lib.pagesizes import letter, landscape
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        
        # Add bank options mapping
        bank_options = {
            33: "JP Morgan",
            50: "Barclays", 
            198: "BlackRock"
        }
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=landscape(letter))
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title = Paragraph(f"Clarity Screener Pro 9.0 - Screening Report", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 0.2*inch))
        
        # Screening Parameters
        params_text = Paragraph(f"Screening Parameters:", styles['Heading2'])
        story.append(params_text)
        
        param_data = [
            ['Parameter', 'Value', 'Parameter', 'Value'],
            ['Price Range', f"${params['min_price']} - ${params['max_price']}", 'Bank Type', params['ma_type']],
            ['Banks', bank_options.get(params['ma_period'], params['ma_period']), 'Bank Threshold', f"{params['ma_threshold']}%"],
            ['Min Volume', f"{params['min_volume']:,.0f}", 'Market Cap', f"${params['min_market_cap']/1e9:.1f}B+"],
            ['Options Days', f"{params['min_days_to_exp']} - {params['max_days_to_exp']}", 'Option Price', f"${params['min_option_price']} - ${params['max_option_price']}"],
            ['Delta Range', f"{params['min_delta']} - {params['max_delta']}", 'Min OI', f"{params['min_open_interest']:,}"]
        ]
        
        param_table = Table(param_data, colWidths=[1.5*inch, 2*inch, 1.5*inch, 2*inch])
        param_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
        ]))
        story.append(param_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Results Summary
        summary = Paragraph(f"Results Summary: {len(results)} Stocks Found", styles['Heading2'])
        story.append(summary)
        
        if results:
            # Sort results with the new hierarchy
            sorted_results = sort_results_for_pdf(results)
            
            if sorted_results:
                # Prepare detailed results data with new column order and grouping
                results_data = [[
                    'Stock', 'Stock_Price', 'Bank_Price', 'Option_Strike', 'Option_Price', 
                    'Delta', 'Open_Interest', 'Bid/Ask', 'Expiration', 'Days_To_Exp', 'ITM', 'Bank_Diff'
                ]]
                
                current_delta = None
                for i, result in enumerate(sorted_results):
                    # Add blank row when delta changes (with dark grey background)
                    if current_delta is not None and result['delta'] != current_delta:
                        results_data.append([''] * 12)  # Blank row for separation
                    
                    current_delta = result['delta']
                    
                    results_data.append([
                        result['symbol'],
                        f"${result['stock_price']:.2f}",
                        f"${result['bank_price']:.2f}",
                        f"${result['option_strike']}",
                        f"${result['option_price']:.2f}",
                        f"{result['delta']:.2f}",
                        f"{result['open_interest']:,}",
                        f"${result['bid']:.2f}/${result['ask']:.2f}",
                        result['expiration'],
                        str(result['days_to_exp']),
                        'Yes' if result['itm'] else 'No',
                        f"{result['bank_diff_percent']:+.2f}%"
                    ])
                
                # Create table with appropriate column widths for landscape
                col_widths = [0.5*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.7*inch, 
                             0.5*inch, 0.8*inch, 0.9*inch, 0.9*inch, 0.6*inch, 0.4*inch, 0.6*inch]
                
                results_table = Table(results_data, colWidths=col_widths)
                
                # Create table style with dark grey separator rows
                table_style = [
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 7),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('FONTSIZE', (0, 1), (-1, -1), 6),
                    ('ROWBREAK', (0, 25), (-1, -1), 'AFTER'),
                ]
                
                # Add dark grey background for separator rows
                for i, row in enumerate(results_data):
                    if all(cell == '' for cell in row):  # This is a separator row
                        table_style.append(('BACKGROUND', (0, i), (-1, i), colors.darkgrey))
                
                results_table.setStyle(TableStyle(table_style))
                story.append(results_table)
        
        # Footer
        story.append(Spacer(1, 0.2*inch))
        footer = Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal'])
        story.append(footer)
        
        doc.build(story)
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        st.error(f"Error creating PDF: {str(e)}")
        return None

def create_futures_pdf_report(results, params):
    """Create a PDF report of futures screening results"""
    try:
        from reportlab.lib.pagesizes import letter, landscape
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        
        bank_options = {
            33: "JP Morgan",
            50: "Barclays", 
            198: "BlackRock"
        }
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=landscape(letter))
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title = Paragraph(f"Clarity Futures Pro - Bank Proximity Report", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 0.2*inch))
        
        # Screening Parameters
        params_text = Paragraph(f"Screening Parameters:", styles['Heading2'])
        story.append(params_text)
        
        param_data = [
            ['Parameter', 'Value', 'Parameter', 'Value'],
            ['Price Range', f"${params['min_price']} - ${params['max_price']}", 'Bank Type', params['ma_type']],
            ['Banks', bank_options.get(params['ma_period'], params['ma_period']), 'Bank Threshold', f"{params['ma_threshold']}%"],
            ['Min Volume', f"{params['min_volume']:,.0f}", 'Categories', ', '.join(params.get('category_filter', ['All']))],
        ]
        
        param_table = Table(param_data, colWidths=[1.5*inch, 2*inch, 1.5*inch, 2*inch])
        param_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(param_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Results Summary
        summary = Paragraph(f"Results Summary: {len(results)} Futures Contracts Found", styles['Heading2'])
        story.append(summary)
        
        if results:
            # Sort by bank proximity (closest to bank price first)
            results_sorted = sorted(results, key=lambda x: abs(x['ma_diff_percent']))
            
            results_data = [[
                'Contract', 'Price', 'Bank_Price', 'Bank_Diff', 'Volume', 'Category'
            ]]
            
            for result in results_sorted:
                # Get category from original data
                category = "N/A"
                try:
                    screener = OptimizedFuturesScreener()
                    all_symbols = screener.load_futures_symbols()
                    symbol_info = all_symbols[all_symbols['symbol'] == result['symbol']].iloc[0]
                    category = symbol_info['category']
                except:
                    pass
                
                results_data.append([
                    result['symbol'],
                    f"${result['price']:.2f}",
                    f"${result['ma']:.2f}",
                    f"{result['ma_diff_percent']:+.2f}%",
                    f"{result['volume']:,.0f}",
                    category
                ])
            
            results_table = Table(results_data, colWidths=[1.2*inch, 1*inch, 1*inch, 0.8*inch, 1*inch, 0.8*inch])
            results_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            story.append(results_table)
        
        # Footer
        story.append(Spacer(1, 0.2*inch))
        footer = Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal'])
        story.append(footer)
        
        doc.build(story)
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        st.error(f"Error creating PDF: {str(e)}")
        return None

def send_email_with_attachment(pdf_buffer, recipient_emails, report_type="stocks"):
    """Send email with PDF attachment to multiple recipients"""
    try:
        # Email configuration - UPDATE THESE WITH YOUR EMAIL CREDENTIALS
        smtp_server = "smtp.gmail.com"  # For Gmail, adjust for other providers
        smtp_port = 587
        sender_email = "erndollars@gmail.com"  # UPDATE THIS
        sender_password = "ccpg yrqt kbor fuwk"  # UPDATE THIS (use app password for Gmail)
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = ", ".join(recipient_emails)  # Join multiple recipients
        
        if report_type == "stocks":
            msg['Subject'] = f"Clarity Screener Pro Report - {datetime.now().strftime('%Y-%m-%d')}"
        else:
            msg['Subject'] = f"Clarity Futures Pro Report - {datetime.now().strftime('%Y-%m-%d')}"
        
        # Email body
        if report_type == "stocks":
            body = f"""
            Clarity Screener Pro 9.0 Screening Report
            
            Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            This report contains the latest screening results from Clarity Screener Pro.
            
            Best regards,
            Clarity Screener Pro Team
            """
        else:
            body = f"""
            Clarity Futures Pro 9.0 Screening Report
            
            Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            This report contains the latest futures screening results from Clarity Futures Pro.
            
            Best regards,
            Clarity Futures Pro Team
            """
            
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach PDF
        pdf_attachment = MIMEBase('application', 'octet-stream')
        pdf_attachment.set_payload(pdf_buffer.getvalue())
        encoders.encode_base64(pdf_attachment)
        
        if report_type == "stocks":
            filename = f'clarity_screener_report_{datetime.now().strftime("%Y%m%d_%H%M")}.pdf'
        else:
            filename = f'clarity_futures_report_{datetime.now().strftime("%Y%m%d_%H%M")}.pdf'
            
        pdf_attachment.add_header(
            'Content-Disposition',
            f'attachment; filename={filename}'
        )
        msg.attach(pdf_attachment)
        
        # Send email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, recipient_emails, text)
        server.quit()
        
        return True
        
    except Exception as e:
        st.error(f"Error sending email: {str(e)}")
        return False
def main():
    st.markdown('<h1 class="main-header">📈 Clarity Pro 9.0 Suite</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced Stock & Futures Screening Platform")
    
    # Initialize session state to preserve results
    if 'stock_results' not in st.session_state:
        st.session_state.stock_results = None
    if 'stock_params' not in st.session_state:
        st.session_state.stock_params = None
    if 'futures_results' not in st.session_state:
        st.session_state.futures_results = None
    if 'futures_params' not in st.session_state:
        st.session_state.futures_params = None
    
    # ADD CANCEL FLAGS HERE
    if 'stock_screening_active' not in st.session_state:
        st.session_state.stock_screening_active = False
    if 'futures_screening_active' not in st.session_state:
        st.session_state.futures_screening_active = False
    if 'cancel_stock_screening' not in st.session_state:
        st.session_state.cancel_stock_screening = False
    if 'cancel_futures_screening' not in st.session_state:
        st.session_state.cancel_futures_screening = False
    
    # Initialize screeners
    stock_screener = OptimizedScreener()
    futures_screener = OptimizedFuturesScreener()
    
    # Create tabs for Stock and Futures screening
    tab1, tab2 = st.tabs(["📊 Stock Screener", "⚡ Futures Screener"])
    
    with tab1:
        st.markdown('<div class="section-header">Stock Options Screener</div>', unsafe_allow_html=True)
        st.markdown("### S&P 500 Screening with Options Data")
        
        # Stock Sidebar parameters
        st.sidebar.header("🎯 Stock Screening Parameters")
        
        with st.sidebar.expander("Price & Volume", expanded=True):
            min_price = st.slider("Min Price", 10, 200, 30, key="stock_min_price")
            max_price = st.slider("Max Price", 50, 500, 300, key="stock_max_price")
            min_volume = st.slider("Min Volume (M)", 1, 10, 2, key="stock_min_volume") * 1000000
        
        with st.sidebar.expander("Bank Settings", expanded=True):
            ma_type = st.selectbox("Bank Type", 
                                  ["REGIONAL", "NATIONAL", "GLOBAL"], 
                                  index=0,
                                  key="stock_ma_type",
                                  help="REGIONAL: Exponential Moving Average (recent prices weighted more)\nNATIONAL: Weighted Moving Average (linear weights)\nGLOBAL: Simple Moving Average (equal weight)")
            
            # Create mapping for display names (show only bank names, but use numbers in code)
            bank_display_to_code = {
                "JP Morgan": 33,
                "Barclays": 50,
                "BlackRock": 198
            }
            
            bank_display = st.selectbox("Banks", ["JP Morgan", "Barclays", "BlackRock"], index=0, key="stock_bank")
            ma_period = bank_display_to_code[bank_display]
            ma_threshold = st.slider("Bank Proximity %", 2.5, 10.0, 5.0, 0.5, key="stock_ma_threshold")
        
        with st.sidebar.expander("Market Cap", expanded=True):
            min_market_cap = st.selectbox("Min Market Cap", 
                                        [("Large Cap", 10e9), ("Mid Cap", 2e9), ("Small Cap", 300e6)], 
                                        format_func=lambda x: x[0],
                                        key="stock_market_cap")[1]
        
        with st.sidebar.expander("Options Filters", expanded=True):
            min_days_to_exp = st.slider("Min Days to Expiry", 7, 30, 7, key="stock_min_days")
            max_days_to_exp = st.slider("Max Days to Expiry", 14, 60, 30, key="stock_max_days")
            min_option_price = st.slider("Min Option Price", 0.10, 2.00, 0.50, 0.10, key="stock_min_option")
            max_option_price = st.slider("Max Option Price", 1.00, 5.00, 2.50, 0.10, key="stock_max_option")
            min_delta = st.slider("Min Delta", 0.1, 0.8, 0.4, 0.1, key="stock_min_delta")
            max_delta = st.slider("Max Delta", 0.5, 1.0, 1.0, 0.1, key="stock_max_delta")
            min_open_interest = st.slider("Min Open Interest", 1000, 10000, 3000, 500, key="stock_min_oi")
        
        stock_params = {
            'min_price': min_price,
            'max_price': max_price,
            'min_volume': min_volume,
            'ma_type': ma_type,
            'ma_period': ma_period,
            'ma_threshold': ma_threshold,
            'min_market_cap': min_market_cap,
            'min_days_to_exp': min_days_to_exp,
            'max_days_to_exp': max_days_to_exp,
            'min_option_price': min_option_price,
            'max_option_price': max_option_price,
            'min_delta': min_delta,
            'max_delta': max_delta,
            'min_open_interest': min_open_interest
        }
        
        # Main stock screening button - REPLACE THIS SECTION
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("🚀 Run Stock Screener", type="primary", use_container_width=True, key="stock_screener_btn"):
                st.session_state.stock_screening_active = True
                st.session_state.cancel_stock_screening = False
                with st.spinner("Loading S&P 500 symbols and pre-filtering..."):
                    results = stock_screener.run_optimized_screener(stock_params)
                
                # Store results in session state
                st.session_state.stock_results = results
                st.session_state.stock_params = stock_params
        
        with col2:
            if st.session_state.stock_screening_active:
                if st.button("🛑 Cancel Stock Screening", type="secondary", use_container_width=True, key="cancel_stock_btn"):
                    st.session_state.cancel_stock_screening = True
                    st.session_state.stock_screening_active = False
                    st.rerun()
            
            if results:
                st.success(f"🎉 Found {len(results)} qualifying stocks with valid options!")
                
                # Email Report Section
                st.markdown("### 📧 Stock Report Options")
                
                # Create PDF report
                pdf_buffer = create_pdf_report(results, params)
                
                if pdf_buffer:
                    # Create a clean card-like layout
                    with st.container():
                        st.markdown("#### Report Options")
                        
                        # Main action buttons in a row
                        action_col1, action_col2, action_col3 = st.columns([2, 3, 2])
                        
                        with action_col1:
                            # Download PDF button
                            st.download_button(
                                label="📄 Download PDF",
                                data=pdf_buffer,
                                file_name=f"clarity_screener_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                                key="stock_download_pdf"
                            )
                        
                        with action_col2:
                            # Email input - show placeholder but always send to accapital22@gmail.com
                            recipient_email = st.text_input(
                                "Email address to send report:", 
                                value="your-email@example.com",
                                placeholder="name@company.com",
                                key="stock_email_input",
                                label_visibility="visible",
                                help="Report will always be sent to our team. Add your email to also receive a copy."
                            )
                        
                        with action_col3:
                            st.write("")  # Spacer
                            st.write("")  # Spacer
                            # Send Email button - BLUE and prominent
                            email_sent = st.button(
                                "📧 Send Stock Report", 
                                type="primary",  # Blue button
                                use_container_width=True,
                                key="stock_send_email"
                            )
                        
                        # Handle email sending
                        if email_sent:
                            # Always send to accapital22@gmail.com
                            default_recipient = "accapital22@gmail.com"
                            recipients = [default_recipient]
                            
                            # Check if user provided a valid additional email
                            user_email = recipient_email.strip()
                            if user_email and user_email != "your-email@example.com" and "@" in user_email:
                                recipients.append(user_email)
                                display_email = user_email
                                success_message = f"✅ Stock report successfully sent to {display_email}!"
                            else:
                                success_message = "✅ Stock report successfully sent!"
                            
                            with st.spinner("📤 Sending stock report..."):
                                if send_email_with_attachment(pdf_buffer, recipients, "stocks"):
                                    st.success(success_message)
                                else:
                                    st.error("❌ Failed to send email. Please check your email configuration.")
                    
                    # Add some spacing
                    st.markdown("---")
                
                # Extract symbols for TradingView - Apply same sorting logic to dashboard
                sorted_for_dashboard = sort_results_for_pdf(results)
                if sorted_for_dashboard:
                    # Get unique symbols in sorted order for dashboard display
                    seen_symbols = set()
                    passing_symbols = []
                    for result in sorted_for_dashboard:
                        if result['symbol'] not in seen_symbols:
                            passing_symbols.append(result['symbol'])
                            seen_symbols.add(result['symbol'])
                else:
                    passing_symbols = [r['symbol'] for r in results]
                
                # Results table for dashboard - Apply same sorting logic and column order
                if sorted_for_dashboard:
                    # Create dashboard results with same sorting and proper column order
                    dashboard_data = []
                    seen_symbols = set()
                    for result in sorted_for_dashboard:
                        if result['symbol'] not in seen_symbols:
                            # Find the original stock data
                            stock_data = next(r for r in results if r['symbol'] == result['symbol'])
                            dashboard_data.append({
                                'Stock': result['symbol'],
                                'Stock_Price': f"${result['stock_price']:.2f}",
                                'Bank_Price': f"${result['bank_price']:.2f}",
                                'Option_Strike': f"${result['option_strike']}",
                                'Option_Price': f"${result['option_price']:.2f}",
                                'Delta': f"{result['delta']:.2f}",
                                'Open_Interest': f"{result['open_interest']:,}",
                                'Bid/Ask': f"${result['bid']:.2f}/${result['ask']:.2f}",
                                'Expiration': result['expiration'],
                                'Days_To_Exp': str(result['days_to_exp']),
                                'ITM': 'Yes' if result['itm'] else 'No',
                                'Bank_Diff': f"{result['bank_diff_percent']:+.2f}%"
                            })
                            seen_symbols.add(result['symbol'])
                    
                    results_df = pd.DataFrame(dashboard_data)
                else:
                    # Fallback if no sorted results
                    results_df = pd.DataFrame([{
                        'Stock': r['symbol'],
                        'Stock_Price': f"${r['price']:.2f}",
                        'Bank_Price': f"${r['ma']:.2f}",
                        'Bank_Diff': f"{r['ma_diff_percent']:+.2f}%",
                        'Options_Found': r['options_count']
                    } for r in results])
                
                st.dataframe(results_df, use_container_width=True)
                
                # TradingView Integration Section
                st.markdown("### 📊 TradingView Integration")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Copy to clipboard option
                    watchlist_text = create_tradingview_watchlist(passing_symbols)
                    st.text_area("📋 Copy to Clipboard", 
                               value=watchlist_text, 
                               height=100,
                               help="Copy these symbols and paste into TradingView watchlist")
                
                with col2:
                    # Download watchlist file
                    watchlist_file = create_tradingview_watchlist_file(passing_symbols)
                    st.download_button(
                        label="📥 Download Watchlist File",
                        data=watchlist_file,
                        file_name="tradingview_watchlist.txt",
                        mime="text/plain",
                        help="Download as .txt file and import into TradingView"
                    )
                
                with col3:
                    # Multi-chart link
                    multi_chart_url = create_tradingview_multichart_url(passing_symbols[:8])
                    if multi_chart_url:
                        st.markdown(f"[🔄 Open Multi-Chart]({multi_chart_url})", unsafe_allow_html=True)
                        st.caption("Opens first 8 symbols in TradingView multi-chart")
                
                st.markdown("""
                **How to import into TradingView:**
                1. **Copy Method**: Copy the symbols above and paste into a new TradingView watchlist
                2. **File Method**: Download the .txt file and import via TradingView Watchlist settings
                3. **Multi-Chart**: Click the link to open multiple charts simultaneously
                """)
                
                # Detailed analysis section
                st.markdown("### 📊 Detailed Analysis")
                
                # Let user select a stock for detailed view
                selected_symbol = st.selectbox(
                    "Select stock for detailed analysis", 
                    [r['symbol'] for r in results],
                    key="stock_selector"
                )
                
                if selected_symbol:
                    selected_stock = next(r for r in results if r['symbol'] == selected_symbol)
                    
                    # Create tabs for different analysis views
                    tab1, tab2 = st.tabs(["📈 Price Chart", "📊 Options Details"])
                    
                    with tab1:
                        # Create price chart with color-coded Bank Type
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=selected_stock['chart_data'].index,
                            y=selected_stock['chart_data']['Close'],
                            name='Close Price',
                            line=dict(color='#fff59d', width=2)
                        ))
                        
                        # Get Bank color and name for the legend
                        bank_color = get_bank_color(selected_stock['ma_type'])
                        bank_name = get_bank_name(selected_stock['ma_period'])
                        
                        fig.add_trace(go.Scatter(
                            x=selected_stock['chart_data'].index,
                            y=selected_stock['chart_data']['MA'],
                            name=f'{bank_name}',
                            line=dict(color=bank_color, width=2, dash='dash')
                        ))
                        fig.update_layout(
                            title=f"{selected_symbol} Price Chart (Last 30 Days)",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab2:
                        # Display detailed options information
                        display_options_details(selected_stock)
            else:
                st.warning("No stocks passed all screening criteria with valid options")
    
    with tab2:
        st.markdown('<div class="section-header">Futures Screener</div>', unsafe_allow_html=True)
        st.markdown("### Futures Market Bank Proximity Screener")
        
        # Futures Sidebar parameters
        st.sidebar.header("🎯 Futures Screening Parameters")
        
        with st.sidebar.expander("Futures Price & Volume", expanded=True):
            futures_min_price = st.slider("Min Price", 1, 500, 1, key="futures_min_price")
            futures_max_price = st.slider("Max Price", 50000, 250000, 250000, key="futures_max_price")
            futures_min_volume = st.slider("Min Volume (K)", 1, 100, 1, key="futures_min_volume") * 1000
        
        with st.sidebar.expander("Futures Bank Settings", expanded=True):
            futures_ma_type = st.selectbox("Bank Type", 
                                          ["REGIONAL", "NATIONAL", "GLOBAL"], 
                                          index=0,
                                          key="futures_ma_type")
            
            futures_bank_display_to_code = {
                "JP Morgan": 33,
                "Barclays": 50,
                "BlackRock": 198
            }
            
            futures_bank_display = st.selectbox("Banks", ["JP Morgan", "Barclays", "BlackRock"], index=0, key="futures_bank")
            futures_ma_period = futures_bank_display_to_code[futures_bank_display]
            futures_ma_threshold = st.slider("Bank Proximity %", 1.0, 25.0, 5.0, 0.5, key="futures_ma_threshold")
        
        with st.sidebar.expander("Futures Categories", expanded=True):
            futures_categories = st.multiselect(
                "Select Categories",
                ["equity", "energy", "metals", "grains", "softs", "fx", "rates", "volatility", "crypto"],
                default=["equity", "energy", "metals", "fx", "crypto", "volatility","softs"],
                key="futures_categories"
            )
        
        futures_params = {
            'min_price': futures_min_price,
            'max_price': futures_max_price,
            'min_volume': futures_min_volume,
            'ma_type': futures_ma_type,
            'ma_period': futures_ma_period,
            'ma_threshold': futures_ma_threshold,
            'category_filter': futures_categories
        }
        
        # Main futures screening button - REPLACE THIS SECTION
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("🚀 Run Futures Screener", type="primary", use_container_width=True, key="futures_screener_btn"):
                st.session_state.futures_screening_active = True
                st.session_state.cancel_futures_screening = False
                with st.spinner("Loading futures contracts and screening..."):
                    results = futures_screener.run_optimized_screener(futures_params)
                
                st.session_state.futures_results = results
                st.session_state.futures_params = futures_params
        
        with col2:
            if st.session_state.futures_screening_active:
                if st.button("🛑 Cancel Futures Screening", type="secondary", use_container_width=True, key="cancel_futures_btn"):
                    st.session_state.cancel_futures_screening = True
                    st.session_state.futures_screening_active = False
                    st.rerun()
        
        # Display futures results
        if st.session_state.futures_results is not None:
            results = st.session_state.futures_results
            params = st.session_state.futures_params
            
            if results:
                st.success(f"🎉 Found {len(results)} qualifying futures contracts near bank prices!")
                
                # Sort by closest to bank price
                results_sorted = sorted(results, key=lambda x: abs(x['ma_diff_percent']))
                
                # Add separator and header for futures list
                st.markdown("---")
                st.markdown("### 📈 Futures Contracts Near Bank Prices")
                
                # Results table
                results_data = []
                for result in results_sorted:
                    # Get category
                    category = "N/A"
                    try:
                        all_symbols = futures_screener.load_futures_symbols()
                        symbol_info = all_symbols[all_symbols['symbol'] == result['symbol']].iloc[0]
                        category = symbol_info['category']
                        name = symbol_info['name']
                    except:
                        name = result.get('name', result['symbol'])
                    
                    results_data.append({
                        'Contract': result['symbol'],
                        'Name': name,
                        'Price': f"${result['price']:.2f}",
                        'Bank_Price': f"${result['ma']:.2f}",
                        'Bank_Diff': f"{result['ma_diff_percent']:+.2f}%",
                        'Volume': f"{result['volume']:,.0f}",
                        'Category': category
                    })
                
                futures_df = pd.DataFrame(results_data)
                st.dataframe(futures_df, use_container_width=True)
                
                # Futures TradingView Integration Section
                st.markdown("### 📊 Futures TradingView Integration")
                
                futures_symbols = [r['symbol'] for r in results_sorted]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Copy to clipboard option for futures
                    futures_watchlist_text = create_futures_tradingview_watchlist(futures_symbols)
                    st.text_area("📋 Copy Futures to Clipboard", 
                               value=futures_watchlist_text, 
                               height=100,
                               help="Copy these futures symbols and paste into TradingView watchlist",
                               key="futures_clipboard")
                
                with col2:
                    # Download futures watchlist file
                    futures_watchlist_file = create_futures_tradingview_watchlist_file(futures_symbols)
                    st.download_button(
                        label="📥 Download Futures Watchlist",
                        data=futures_watchlist_file,
                        file_name="tradingview_futures_watchlist.txt",
                        mime="text/plain",
                        help="Download as .txt file and import into TradingView",
                        key="futures_download"
                    )
                
                # Futures PDF Report Section
                st.markdown("### 📧 Futures Report Options")
                
                futures_pdf_buffer = create_futures_pdf_report(results, params)
                
                if futures_pdf_buffer:
                    futures_col1, futures_col2 = st.columns(2)
                    
                    with futures_col1:
                        st.download_button(
                            label="📄 Download Futures PDF",
                            data=futures_pdf_buffer,
                            file_name=f"futures_bank_proximity_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            key="futures_pdf_download"
                        )
                    
                    with futures_col2:
                        futures_email_sent = st.button(
                            "📧 Send Futures Report", 
                            type="primary",
                            use_container_width=True,
                            key="futures_email_btn"
                        )
                    
                    # Handle futures email sending
                    if futures_email_sent:
                        # Always send to accapital22@gmail.com
                        default_recipient = "accapital22@gmail.com"
                        recipients = [default_recipient]
                        
                        with st.spinner("📤 Sending futures report..."):
                            if send_email_with_attachment(futures_pdf_buffer, recipients, "futures"):
                                st.success("✅ Futures report successfully sent!")
                            else:
                                st.error("❌ Failed to send futures email. Please check your email configuration.")
                
                # Futures detailed analysis
                st.markdown("### 📊 Futures Detailed Analysis")
                
                selected_futures_symbol = st.selectbox(
                    "Select futures contract for detailed analysis", 
                    [r['symbol'] for r in results],
                    key="futures_selector"
                )
                
                if selected_futures_symbol:
                    selected_future = next(r for r in results if r['symbol'] == selected_futures_symbol)
                    
                    # Display contract info without chart imagery
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Price", f"${selected_future['price']:.2f}")
                    with col2:
                        st.metric("Bank Price", f"${selected_future['ma']:.2f}")
                    with col3:
                        st.metric("Bank Difference", f"{selected_future['ma_diff_percent']:+.2f}%")
                    with col4:
                        st.metric("Average Volume", f"{selected_future['volume']:,.0f}")
                    
                    # Display additional contract information
                    try:
                        all_symbols = futures_screener.load_futures_symbols()
                        symbol_info = all_symbols[all_symbols['symbol'] == selected_futures_symbol].iloc[0]
                        st.info(f"**Contract Details:** {symbol_info['name']} | Category: {symbol_info['category']} | Exchange: {symbol_info['exchange']}")
                    except:
                        st.info(f"**Contract:** {selected_futures_symbol}")
            else:
                st.warning("No futures contracts passed the bank proximity screening")
    
    st.markdown("---")

    try:
        from PIL import Image
        
        # Load image
        image = Image.open('images/Call_Long_Bulls_Logo.png')
        
        # Center the image using columns
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.image(image, width=300)
            
    except FileNotFoundError:
        st.warning("⚠️ Logo image not found. Please check the file path.")
    except Exception as e:
        st.warning(f"⚠️ Could not load image: {e}")

if __name__ == "__main__":
    main()









