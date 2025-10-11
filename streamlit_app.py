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
                'timeframe_interval': '1d',
                'timeframe_period': '1y',
                'timeframe_label': 'Daily (1Y)',
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
    
    def screen_stock_with_timeframe(self, symbol, params, timeframe):
        """Screen individual stock with specific timeframe"""
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period=timeframe['period'], interval=timeframe['interval'])
            
            if len(hist) < params['ma_period']:
                return None, f"Insufficient {timeframe['label']} data for moving average"
            
            current_price = hist['Close'].iloc[-1]
            if not (params['min_price'] <= current_price <= params['max_price']):
                return None, f"Price ${current_price:.2f} outside range"
            
            # Use different volume calculation for different timeframes
            if timeframe['interval'] == '1d':
                avg_volume = hist['Volume'].tail(20).mean()  # 20 days for daily
            elif timeframe['interval'] == '1wk':
                avg_volume = hist['Volume'].tail(12).mean()  # 12 weeks for weekly
            else:  # monthly
                avg_volume = hist['Volume'].tail(6).mean()   # 6 months for monthly
                
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
            
            # OPTIONS CHECK (only for daily timeframe to avoid rate limiting)
            options_data = {'has_valid_options': False, 'options_count': 0, 'details': []}
            if timeframe['interval'] == '1d':  # Only check options for daily timeframe
                options_data = self.get_detailed_options_data(symbol, current_price, params)
            
            if not options_data['has_valid_options'] and timeframe['interval'] == '1d':
                return None, "No valid options found"
            
            stock_data = {
                'symbol': symbol,
                'timeframe_interval': timeframe['interval'],
                'timeframe_period': timeframe['period'],
                'timeframe_label': timeframe['label'],
                'price': current_price,
                'ma': current_ma,
                'ma_type': ma_type,
                'ma_period': ma_period,
                'ma_diff_percent': ma_diff_percent,
                'volume': avg_volume,
                'market_cap': market_cap,
                'options_count': options_data['options_count'],
                'options_details': options_data['details'],
                'chart_data': hist[['Close', 'MA']].tail(30 if timeframe['interval'] == '1d' else 24)
            }
            
            return stock_data, "PASS"
            
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def run_optimized_screener(self, params):
        """Run optimized screening with multiple timeframes and cancel support"""
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
            
            # Define multiple timeframes to screen
            timeframes = [
                {'interval': '1d', 'period': '1y', 'label': 'Daily (1Y)'},
                {'interval': '1wk', 'period': '2y', 'label': 'Weekly (2Y)'},
                {'interval': '1mo', 'period': '5y', 'label': 'Monthly (5Y)'}
            ]
            
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_symbols = len(qualified_symbols) * len(timeframes)
            current_progress = 0
            
            for timeframe in timeframes:
                st.info(f"🕐 Screening with {timeframe['label']} timeframe...")
                
                for i, symbol in enumerate(qualified_symbols):
                    # Check if cancellation was requested
                    if st.session_state.cancel_stock_screening:
                        st.warning("🛑 Stock screening cancelled by user")
                        st.session_state.stock_screening_active = False
                        st.session_state.cancel_stock_screening = False
                        return results
                    
                    status_text.text(f"🔍 Screening {symbol} ({timeframe['label']}) - {i+1}/{len(qualified_symbols)}...")
                    result, message = self.screen_stock_with_timeframe(symbol, params, timeframe)
                    
                    # ONLY SHOW PASSING SYMBOLS
                    if result:
                        results.append(result)
                        st.success(f"✅ {symbol} ({timeframe['label']}): {message} | Options: {result['options_count']}")
                    
                    current_progress += 1
                    progress_bar.progress(current_progress / total_symbols)
                    time.sleep(0.3)  # Rate limiting
            
            status_text.text("Screening complete!")
            st.session_state.stock_screening_active = False
            
            # Show summary
            if not results:
                st.error("❌ No stocks passed screening criteria!")
                st.info("💡 Try relaxing your screening criteria (price range, volume, options filters)")
            else:
                # Group results by timeframe for summary
                timeframe_counts = {}
                for result in results:
                    tf = result['timeframe_label']
                    timeframe_counts[tf] = timeframe_counts.get(tf, 0) + 1
                
                st.success(f"🎉 Found {len(results)} total qualifying stocks across all timeframes!")
                for tf, count in timeframe_counts.items():
                    st.info(f"   • {tf}: {count} stocks")
            
            return results
            
        except Exception as e:
            st.session_state.stock_screening_active = False
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
                'timeframe_interval': '1d',
                'timeframe_period': '1y',
                'timeframe_label': 'Daily (1Y)',
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
    
    def screen_futures_with_timeframe(self, symbol, params, timeframe):
        """Screen individual futures contract with specific timeframe"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=timeframe['period'], interval=timeframe['interval'])
            
            if len(hist) < params['ma_period']:
                return None, f"Insufficient {timeframe['label']} data"
            
            current_price = hist['Close'].iloc[-1]
            if not (params['min_price'] <= current_price <= params['max_price']):
                return None, f"Price ${current_price:.2f} outside range"
            
            # Adjust volume calculation based on timeframe
            if timeframe['interval'] == '1d':
                avg_volume = hist['Volume'].tail(20).mean()  # 20 days
            elif timeframe['interval'] == '1wk':
                avg_volume = hist['Volume'].tail(12).mean()  # 12 weeks
            else:  # monthly
                avg_volume = hist['Volume'].tail(6).mean()   # 6 months
                
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
                'timeframe_interval': timeframe['interval'],
                'timeframe_period': timeframe['period'],
                'timeframe_label': timeframe['label'],
                'name': ticker.info.get('shortName', symbol),
                'price': current_price,
                'ma': current_ma,
                'ma_type': ma_type,
                'ma_period': ma_period,
                'ma_diff_percent': ma_diff_percent,
                'volume': avg_volume,
                'chart_data': hist[['Close', 'MA']].tail(30 if timeframe['interval'] == '1d' else 24)
            }
            
            return futures_data, "PASS"
            
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def run_optimized_screener(self, params):
        """Run optimized futures screening with multiple timeframes and cancel support"""
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
            
            # Define multiple timeframes for futures
            timeframes = [
                {'interval': '1d', 'period': '1y', 'label': 'Daily (1Y)'},
                {'interval': '1wk', 'period': '2y', 'label': 'Weekly (2Y)'},
                {'interval': '1mo', 'period': '5y', 'label': 'Monthly (5Y)'}
            ]
            
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_symbols = len(qualified_symbols) * len(timeframes)
            current_progress = 0
            
            for timeframe in timeframes:
                st.info(f"🕐 Screening with {timeframe['label']} timeframe...")
                
                for i, symbol in enumerate(qualified_symbols):
                    # Check if cancellation was requested
                    if st.session_state.cancel_futures_screening:
                        st.warning("🛑 Futures screening cancelled by user")
                        st.session_state.futures_screening_active = False
                        st.session_state.cancel_futures_screening = False
                        return results
                    
                    status_text.text(f"🔍 Screening {symbol} ({timeframe['label']}) - {i+1}/{len(qualified_symbols)}...")
                    result, message = self.screen_futures_with_timeframe(symbol, params, timeframe)
                    
                    if result:
                        results.append(result)
                        st.success(f"✅ {symbol} ({timeframe['label']}): {message}")
                    
                    current_progress += 1
                    progress_bar.progress(current_progress / total_symbols)
                    time.sleep(0.3)
            
            status_text.text("Screening complete!")
            st.session_state.futures_screening_active = False
            
            if not results:
                st.error("❌ No futures contracts passed screening criteria!")
                st.info("💡 Try relaxing your price range or bank proximity threshold")
            else:
                # Group results by timeframe for summary
                timeframe_counts = {}
                for result in results:
                    tf = result['timeframe_label']
                    timeframe_counts[tf] = timeframe_counts.get(tf, 0) + 1
                
                st.success(f"🎉 Found {len(results)} total qualifying contracts across all timeframes!")
                for tf, count in timeframe_counts.items():
                    st.info(f"   • {tf}: {count} contracts")
            
            return results
            
        except Exception as e:
            st.session_state.futures_screening_active = False
            st.error(f"Error in screening process: {str(e)}")
            return []
