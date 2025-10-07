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
    page_title="Clarity Screener Pro 9.0",
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

def create_pdf_report(results, params):
    """Create a PDF report of screening results in landscape mode"""
    try:
        from reportlab.lib.pagesizes import letter, landscape
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        
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
            ['Banks', params['ma_period'], 'Bank Threshold', f"{params['ma_threshold']}%"],
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
            # Prepare detailed results data with options information
            results_data = [[
                'Stock', 'Stock_Price', f"{params['ma_type']}_{params['ma_period']}", 
                'EMA_Diff', 'Option_Strike', 'Option_Price', 'Bid/Ask', 
                'Open_Interest', 'Delta', 'Expiration', 'Days_to_Exp', 'ITM'
            ]]
            
            for result in results:
                # Get the first valid option for each stock
                if result['options_details']:
                    option = result['options_details'][0]  # Take first option
                    results_data.append([
                        result['symbol'],
                        f"${result['price']:.2f}",
                        f"${result['ma']:.2f}",
                        f"{result['ma_diff_percent']:+.2f}%",
                        f"${option['strike']}",
                        f"${option['option_price']:.2f}",
                        f"${option['bid']:.2f}/${option['ask']:.2f}",
                        f"{option['open_interest']:,}",
                        f"{option['delta']:.2f}",
                        option['expiration'],
                        str(option['days_to_exp']),
                        'Yes' if option['in_the_money'] else 'No'
                    ])
            
            # Create table with appropriate column widths for landscape
            col_widths = [0.6*inch, 0.8*inch, 1.0*inch, 0.7*inch, 0.8*inch, 
                         0.8*inch, 1.0*inch, 0.9*inch, 0.5*inch, 1.0*inch, 0.7*inch, 0.5*inch]
            
            results_table = Table(results_data, colWidths=col_widths)
            results_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 7),
                ('ROWBREAK', (0, 20), (-1, -1), 'AFTER'),
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

def send_email_with_attachment(pdf_buffer, recipient_email="accapital22@gmail.com"):
    """Send email with PDF attachment"""
    try:
        # Email configuration - UPDATE THESE WITH YOUR EMAIL CREDENTIALS
        smtp_server = "smtp.gmail.com"  # For Gmail, adjust for other providers
        smtp_port = 587
        sender_email = "erndollars@gmail.com"  # UPDATE THIS
        sender_password = "ccpg yrqt kbor fuwk"  # UPDATE THIS (use app password for Gmail)
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = f"Clarity Screener Pro Report - {datetime.now().strftime('%Y-%m-%d')}"
        
        # Email body
        body = f"""
        Clarity Screener Pro 9.0 Screening Report
        
        Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        This report contains the latest screening results from Clarity Screener Pro.
        
        Best regards,
        Clarity Screener Pro Team
        """
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach PDF
        pdf_attachment = MIMEBase('application', 'octet-stream')
        pdf_attachment.set_payload(pdf_buffer.getvalue())
        encoders.encode_base64(pdf_attachment)
        pdf_attachment.add_header(
            'Content-Disposition',
            f'attachment; filename=clarity_screener_report_{datetime.now().strftime("%Y%m%d_%H%M")}.pdf'
        )
        msg.attach(pdf_attachment)
        
        # Send email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, recipient_email, text)
        server.quit()
        
        return True
        
    except Exception as e:
        st.error(f"Error sending email: {str(e)}")
        return False

def main():
    st.markdown('<h1 class="main-header">📈 Clarity Screener Pro 9.0</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced S&P 500 Screening with Options Data")
    
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
    
    with st.sidebar.expander("Bank Settings", expanded=True):
        ma_type = st.selectbox("Bank Type", 
                              ["REGIONAL", "NATIONAL", "GLOBAL"], 
                              index=0,
                              help="REGIONAL: Exponential Moving Average (recent prices weighted more)\nNATIONAL: Weighted Moving Average (linear weights)\nGLOBAL: Simple Moving Average (equal weight)")
        
        # Create mapping for display names
        bank_options = {
            33: "33 (JP Morgan)",
            50: "50 (Barclays)", 
            198: "198 (BlackRock)"
        }
        
        ma_period = st.selectbox("Banks", [33, 50, 198], 
                                format_func=lambda x: bank_options[x],
                                index=0)
        ma_threshold = st.slider("Bank Proximity %", 2.5, 10.0, 5.0, 0.5)
    
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
            
                        # Email Report Section
            st.markdown("### 📧 Email Report")
            
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
                            key="download_pdf"
                        )
                    
                    with action_col2:
                        # Email input
                        recipient_email = st.text_input(
                            "Email address to send report:", 
                            value="your-email@example.com",
                            placeholder="name@company.com",
                            key="email_input",
                            label_visibility="visible"
                        )
                    
                    with action_col3:
                        st.write("")  # Spacer
                        st.write("")  # Spacer
                        # Send Email button - BLUE and prominent
                        email_sent = st.button(
                            "📧 Send Email Report", 
                            type="primary",  # Blue button
                            use_container_width=True,
                            key="send_email"
                        )
                    
                    # Handle email sending
                    if email_sent:
                        if recipient_email and "@" in recipient_email:
                            with st.spinner("📤 Sending email report..."):
                                if send_email_with_attachment(pdf_buffer, recipient_email):
                                    st.success(f"✅ Report successfully sent to {recipient_email}!")
                                else:
                                    st.error("❌ Failed to send email. Please check your email configuration.")
                        else:
                            st.warning("⚠️ Please enter a valid email address")
                
                # Add some spacing
                st.markdown("---")
            
            # Extract symbols for TradingView
            passing_symbols = [r['symbol'] for r in results]
            
            # Results table
            results_df = pd.DataFrame([{
                'Symbol': r['symbol'],
                'Price': f"${r['price']:.2f}",
                f'{r["ma_type"]}_{r["ma_period"]}': f"${r['ma']:.2f}",
                'Bank Diff': f"{r['ma_diff_percent']:+.2f}%",
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
                        y=selected_stock['chart_data']['MA'],
                        name=f'{selected_stock["ma_type"]} {selected_stock["ma_period"]}',
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
