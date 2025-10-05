import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Stock Screener Pro - Cloud",
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
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem !important;
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
                "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
                "https://raw.githubusercontent.com/accapital22/stock-screener-cloud/main/sp500_symbols.csv",
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
        return prices.ewm(span=period, adjust=False).mean()
    
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
            hist = stock.history(period="3mo")
            
            if len(hist) < params['ema_period']:
                return None, "Insufficient data for EMA"
            
            current_price = hist['Close'].iloc[-1]
            if not (params['min_price'] <= current_price <= params['max_price']):
                return None, f"Price ${current_price:.2f} outside range"
            
            avg_volume = hist['Volume'].tail(20).mean()
            if avg_volume < params['min_volume']:
                return None, f"Volume {avg_volume:,.0f} below requirement"
            
            hist['EMA'] = self.calculate_ema(hist['Close'], params['ema_period'])
            current_ema = hist['EMA'].iloc[-1]
            ema_diff_percent = ((current_price - current_ema) / current_ema) * 100
            
            if abs(ema_diff_percent) > params['ema_threshold']:
                return None, f"EMA diff {ema_diff_percent:.2f}% > threshold"
            
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
                'ema': current_ema,
                'ema_diff_percent': ema_diff_percent,
                'volume': avg_volume,
                'market_cap': market_cap,
                'options_count': options_data['options_count'],
                'options_details': options_data['details'],
                'chart_data': hist[['Close', 'EMA']].tail(30)
            }
            
            return stock_data, "PASS"
            
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def run_optimized_screener(self, params):
        """Run optimized screening with pre-filtering"""
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
            
            for i, symbol in enumerate(qualified_symbols):
                status_text.text(f"🔍 Screening {symbol} ({i+1}/{len(qualified_symbols)})...")
                result, message = self.screen_stock(symbol, params)
                
                # ONLY SHOW PASSING SYMBOLS
                if result:
                    results.append(result)
                    st.success(f"✅ {symbol}: {message} | Options: {result['options_count']}")
                
                progress_bar.progress((i + 1) / len(qualified_symbols))
                time.sleep(delay_between_requests)
            
            status_text.text("Screening complete!")
            
            # Show summary
            if not results:
                st.error("❌ No stocks passed screening criteria!")
                st.info("💡 Try relaxing your screening criteria (price range, volume, options filters)")
            
            return results
            
        except Exception as e:
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

def main():
    st.markdown('<h1 class="main-header">📈 Stock Screener Pro - Complete</h1>', unsafe_allow_html=True)
    st.markdown("### Full S&P 500 Screening with Options Data")
    
    # Initialize session state to preserve results
    if 'screening_results' not in st.session_state:
        st.session_state.screening_results = None
    if 'screening_params' not in st.session_state:
        st.session_state.screening_params = None
    
    # Initialize screener
    screener = OptimizedScreener()
    
    # Sidebar parameters
    st.sidebar.header("🎯 Screening Parameters")
    
    with st.sidebar.expander("Price & Volume", expanded=True):
        min_price = st.slider("Min Price", 10, 200, 30)
        max_price = st.slider("Max Price", 50, 500, 300)
        min_volume = st.slider("Min Volume (M)", 1, 10, 2) * 1000000
    
    with st.sidebar.expander("EMA Settings", expanded=True):
        ema_period = st.selectbox("EMA Period", [20, 50, 200], index=1)
        ema_threshold = st.slider("EMA Proximity %", 0.5, 5.0, 2.0, 0.1)
    
    with st.sidebar.expander("Market Cap", expanded=True):
        min_market_cap = st.selectbox("Min Market Cap", 
                                    [("Large Cap", 10e9), ("Mid Cap", 2e9), ("Small Cap", 300e6)], 
                                    format_func=lambda x: x[0])[1]
    
    with st.sidebar.expander("Options Filters", expanded=True):
        min_days_to_exp = st.slider("Min Days to Expiry", 7, 30, 7)
        max_days_to_exp = st.slider("Max Days to Expiry", 14, 60, 30)
        min_option_price = st.slider("Min Option Price", 0.10, 2.00, 0.50, 0.10)
        max_option_price = st.slider("Max Option Price", 1.00, 5.00, 2.50, 0.10)
        min_delta = st.slider("Min Delta", 0.1, 0.8, 0.4, 0.1)
        max_delta = st.slider("Max Delta", 0.5, 1.0, 1.0, 0.1)
        min_open_interest = st.slider("Min Open Interest", 1000, 10000, 3000, 500)
    
    params = {
        'min_price': min_price,
        'max_price': max_price,
        'min_volume': min_volume,
        'ema_period': ema_period,
        'ema_threshold': ema_threshold,
        'min_market_cap': min_market_cap,
        'min_days_to_exp': min_days_to_exp,
        'max_days_to_exp': max_days_to_exp,
        'min_option_price': min_option_price,
        'max_option_price': max_option_price,
        'min_delta': min_delta,
        'max_delta': max_delta,
        'min_open_interest': min_open_interest
    }
    
    # Main screening button
    if st.button("🚀 Run Complete Screener", type="primary", use_container_width=True):
        with st.spinner("Loading S&P 500 symbols and pre-filtering..."):
            results = screener.run_optimized_screener(params)
        
        # Store results in session state
        st.session_state.screening_results = results
        st.session_state.screening_params = params
    
    # Display results from session state (if they exist)
    if st.session_state.screening_results is not None:
        results = st.session_state.screening_results
        params = st.session_state.screening_params
        
        if results:
            st.success(f"🎉 Found {len(results)} qualifying stocks with valid options!")
            
            # Extract symbols for TradingView
            passing_symbols = [r['symbol'] for r in results]
            
            # Results table
            results_df = pd.DataFrame([{
                'Symbol': r['symbol'],
                'Price': f"${r['price']:.2f}",
                f'EMA_{params["ema_period"]}': f"${r['ema']:.2f}",
                'EMA Diff': f"{r['ema_diff_percent']:+.2f}%",
                'Volume': f"{r['volume']:,.0f}",
                'Market Cap': f"${r['market_cap']/1e9:.1f}B",
                'Options Found': r['options_count']
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
                multi_chart_url = create_tradingview_multichart_url(passing_symbols[:4])  # Limit to 4 for multi-chart
                if multi_chart_url:
                    st.markdown(f"[🔄 Open Multi-Chart]({multi_chart_url})", unsafe_allow_html=True)
                    st.caption("Opens first 4 symbols in TradingView multi-chart")
            
            st.markdown("""
            **How to import into TradingView:**
            1. **Copy Method**: Copy the symbols above and paste into a new TradingView watchlist
            2. **File Method**: Download the .txt file and import via TradingView Watchlist settings
            3. **Multi-Chart**: Click the link to open multiple charts simultaneously
            """)
            
            # Detailed analysis section
            st.markdown("### 📊 Detailed Analysis")
            
            # Let user select a stock for detailed view - THIS IS NOW STATE-FUL
            selected_symbol = st.selectbox(
                "Select stock for detailed analysis", 
                [r['symbol'] for r in results],
                key="stock_selector"  # Important: Add a key to make this widget stateful
            )
            
            if selected_symbol:
                selected_stock = next(r for r in results if r['symbol'] == selected_symbol)
                
                # Create tabs for different analysis views
                tab1, tab2 = st.tabs(["📈 Price Chart", "📊 Options Details"])
                
                with tab1:
                    # Create price chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=selected_stock['chart_data'].index,
                        y=selected_stock['chart_data']['Close'],
                        name='Close Price',
                        line=dict(color='#1f77b4', width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        x=selected_stock['chart_data'].index,
                        y=selected_stock['chart_data']['EMA'],
                        name=f'EMA {params["ema_period"]}',
                        line=dict(color='#ff7f0e', width=2, dash='dash')
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
    
    st.markdown("---")
    st.markdown("""
    ### 🚀 Complete Features:
    - **Full S&P 500** screening with pre-filtering
    - **Detailed options chain** analysis
    - **Options criteria**: price, delta, open interest, expiration
    - **TradingView integration** for easy watchlist creation
    - **State persistence** - results don't reset when selecting symbols
    - **Mobile-optimized** cloud deployment
    - **Efficient**: Only screens likely candidates
    
    ### 🔧 State Fix Applied:
    - **Session state** preserves screening results between interactions
    - **Widget keys** prevent reset when selecting different symbols
    - **Persistent data** allows you to explore results without re-screening
    """)

if __name__ == "__main__":
    main()