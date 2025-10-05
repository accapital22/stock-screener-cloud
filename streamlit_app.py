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
        """Load S&P 500 symbols from CSV with price range info"""
        try:
            # For Streamlit Cloud, we'll use a URL to a hosted CSV
            # You can also upload this file to your GitHub repo
            url = "https://raw.githubusercontent.com/accapital22/stock-screener-cloud/main/sp500_symbols.csv"
            df = pd.read_csv(url)
            return df
        except:
            # Fallback to minimal list if CSV fails
            st.warning("Using fallback symbol list - upload sp500_symbols.csv to GitHub for full S&P 500")
            return pd.DataFrame({
                'symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V'],
                'price_range': ['high', 'high', 'high', 'high', 'high', 'high', 'high', 'mid', 'mid', 'high']
            })
    
    def pre_filter_symbols(self, symbols_df, params):
        """Pre-filter symbols based on known price ranges before API calls"""
        price_filtered = []
        
        # Price range mapping
        price_ranges = {
            'low': (0, 30),      # Stocks typically <$30
            'mid': (30, 300),    # Stocks typically $30-$300  
            'high': (300, 5000)  # Stocks typically >$300
        }
        
        for _, row in symbols_df.iterrows():
            symbol = row['symbol']
            price_range = row.get('price_range', 'mid')  # Default to mid
            
            range_min, range_max = price_ranges.get(price_range, (0, 5000))
            
            # Check if this stock's typical range overlaps with our desired range
            if (range_max >= params['min_price'] and range_min <= params['max_price']):
                price_filtered.append(symbol)
        
        st.success(f"🎯 Pre-filtering: {len(price_filtered)} stocks likely in price range ${params['min_price']}-${params['max_price']}")
        return price_filtered
    
    def calculate_ema(self, prices, period=50):
        return prices.ewm(span=period, adjust=False).mean()
    
    def screen_stock(self, symbol, params):
        """Screen individual stock - only called for pre-filtered symbols"""
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="3mo")
            
            if len(hist) < params['ema_period']:
                return None, "Insufficient data for EMA"
            
            # Final price check (in case our pre-filter was wrong)
            current_price = hist['Close'].iloc[-1]
            if not (params['min_price'] <= current_price <= params['max_price']):
                return None, f"Price ${current_price:.2f} outside range ${params['min_price']}-${params['max_price']}"
            
            # Volume check
            avg_volume = hist['Volume'].tail(20).mean()
            if avg_volume < params['min_volume']:
                return None, f"Volume {avg_volume:,.0f} below {params['min_volume']:,}"
            
            # EMA calculation
            hist['EMA'] = self.calculate_ema(hist['Close'], params['ema_period'])
            current_ema = hist['EMA'].iloc[-1]
            ema_diff_percent = ((current_price - current_ema) / current_ema) * 100
            
            if abs(ema_diff_percent) > params['ema_threshold']:
                return None, f"EMA diff {ema_diff_percent:.2f}% > {params['ema_threshold']}%"
            
            # Market cap check
            info = stock.info
            market_cap = info.get('marketCap', 0)
            if market_cap < params['min_market_cap']:
                return None, f"Market cap ${market_cap/1e9:.1f}B < ${params['min_market_cap']/1e9}B"
            
            stock_data = {
                'symbol': symbol,
                'price': current_price,
                'ema': current_ema,
                'ema_diff_percent': ema_diff_percent,
                'volume': avg_volume,
                'market_cap': market_cap,
                'chart_data': hist[['Close', 'EMA']].tail(30)
            }
            
            return stock_data, "PASS"
            
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def run_optimized_screener(self, params):
        """Run optimized screening with pre-filtering"""
        # Load symbols
        symbols_df = self.load_sp500_symbols()
        all_symbols = symbols_df['symbol'].tolist()
        
        st.info(f"📊 Loaded {len(all_symbols)} S&P 500 symbols")
        
        # Pre-filter symbols
        qualified_symbols = self.pre_filter_symbols(symbols_df, params)
        
        if not qualified_symbols:
            st.error("No stocks passed initial price range filtering")
            return []
        
        # Screen only pre-filtered symbols
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(qualified_symbols):
            status_text.text(f"🔍 Screening {symbol}...")
            result, message = self.screen_stock(symbol, params)
            
            if result:
                results.append(result)
                st.success(f"✅ {symbol}: {message}")
            else:
                st.error(f"❌ {symbol}: {message}")
            
            progress_bar.progress((i + 1) / len(qualified_symbols))
            time.sleep(0.1)
        
        status_text.text("Screening complete!")
        return results

def main():
    st.markdown('<h1 class="main-header">📈 Stock Screener Pro - Optimized</h1>', unsafe_allow_html=True)
    
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
    
    params = {
        'min_price': min_price,
        'max_price': max_price,
        'min_volume': min_volume,
        'ema_period': ema_period,
        'ema_threshold': ema_threshold,
        'min_market_cap': min_market_cap
    }
    
    # Main screening button
    if st.button("🚀 Run Optimized Screener", type="primary", use_container_width=True):
        with st.spinner("Loading S&P 500 symbols and pre-filtering..."):
            results = screener.run_optimized_screener(params)
        
        # Display results
        if results:
            st.success(f"🎉 Found {len(results)} qualifying stocks!")
            
            # Results table
            results_df = pd.DataFrame([{
                'Symbol': r['symbol'],
                'Price': f"${r['price']:.2f}",
                f'EMA_{params["ema_period"]}': f"${r['ema']:.2f}",
                'EMA Diff': f"{r['ema_diff_percent']:+.2f}%",
                'Volume': f"{r['volume']:,.0f}",
                'Market Cap': f"${r['market_cap']/1e9:.1f}B"
            } for r in results])
            
            st.dataframe(results_df, use_container_width=True)
            
            # Chart for first result
            if results:
                first = results[0]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=first['chart_data'].index, y=first['chart_data']['Close'], name='Price'))
                fig.add_trace(go.Scatter(x=first['chart_data'].index, y=first['chart_data']['EMA'], name=f'EMA {params["ema_period"]}'))
                fig.update_layout(title=f"{first['symbol']} Chart", height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No stocks passed all screening criteria")
    
    st.markdown("---")
    st.markdown("""
    ### 🚀 Optimization Features:
    - **Pre-filtering**: Only screens stocks likely in your price range
    - **Full S&P 500**: Uses complete index (via CSV)
    - **Efficient**: Avoids unnecessary API calls
    - **Faster**: Skips stocks that obviously don't qualify
    """)

if __name__ == "__main__":
    main()
