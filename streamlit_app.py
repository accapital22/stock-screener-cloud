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
    page_title="Stock Screener Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
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

class StreamlitScreener:
    def __init__(self):
        self.results = []
    
    def calculate_ema(self, prices, period=50):
        return prices.ewm(span=period, adjust=False).mean()
    
    def get_sp500_symbols(self):
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V']
    
    def screen_stock(self, symbol, params):
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="3mo")
            
            if len(hist) < 50:
                return None, "Insufficient data"
            
            current_price = hist['Close'].iloc[-1]
            if not (30 <= current_price <= 300):
                return None, f"Price ${current_price:.2f} outside range"
            
            avg_volume = hist['Volume'].tail(20).mean()
            if avg_volume < 2000000:
                return None, f"Volume {avg_volume:,.0f} below 2M"
            
            hist['EMA_50'] = self.calculate_ema(hist['Close'], 50)
            current_ema = hist['EMA_50'].iloc[-1]
            ema_diff_percent = ((current_price - current_ema) / current_ema) * 100
            
            if abs(ema_diff_percent) > 2:
                return None, f"EMA diff {ema_diff_percent:.2f}% > 2%"
            
            info = stock.info
            market_cap = info.get('marketCap', 0)
            if market_cap < 2000000000:
                return None, f"Market cap ${market_cap/1e9:.1f}B below $2B"
            
            # All criteria passed
            stock_data = {
                'symbol': symbol,
                'price': current_price,
                'ema_50': current_ema,
                'ema_diff_percent': ema_diff_percent,
                'volume': avg_volume,
                'market_cap': market_cap,
                'chart_data': hist[['Close', 'EMA_50']].tail(30)
            }
            
            return stock_data, "PASS"
            
        except Exception as e:
            return None, f"Error: {str(e)}"

def main():
    st.markdown('<h1 class="main-header">📈 Stock Screener Pro</h1>', unsafe_allow_html=True)
    st.markdown("### Cloud Deployment Test Version")
    
    # Initialize screener
    screener = StreamlitScreener()
    
    # Simple interface for testing
    if st.button("🚀 Run Quick Screener", type="primary"):
        with st.spinner("Screening stocks..."):
            symbols = screener.get_sp500_symbols()
            results = []
            
            for symbol in symbols:
                result, message = screener.screen_stock(symbol, {})
                if result:
                    results.append(result)
            
            # Display results
            if results:
                st.success(f"✅ Found {len(results)} qualifying stocks!")
                
                # Create results table
                results_df = pd.DataFrame([{
                    'Symbol': r['symbol'],
                    'Price': f"${r['price']:.2f}",
                    'EMA 50': f"${r['ema_50']:.2f}", 
                    'EMA Diff': f"{r['ema_diff_percent']:+.2f}%",
                    'Volume': f"{r['volume']:,.0f}",
                    'Market Cap': f"${r['market_cap']/1e9:.1f}B"
                } for r in results])
                
                st.dataframe(results_df, use_container_width=True)
                
                # Show chart for first result
                if results:
                    first_stock = results[0]
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=first_stock['chart_data'].index,
                        y=first_stock['chart_data']['Close'],
                        name='Close Price',
                        line=dict(color='#1f77b4', width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        x=first_stock['chart_data'].index,
                        y=first_stock['chart_data']['EMA_50'],
                        name='EMA 50',
                        line=dict(color='#ff7f0e', width=2, dash='dash')
                    ))
                    fig.update_layout(
                        title=f"{first_stock['symbol']} Price Chart",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No stocks passed the screening criteria.")

    st.markdown("---")
    st.markdown("""
    ### 🌐 About This Deployment
    This is a simplified version deployed to Streamlit Cloud. 
    - **Mobile-friendly** design
    - **24/7 access** from any device
    - **Auto-scaling** cloud infrastructure
    """)

if __name__ == "__main__":
    main()