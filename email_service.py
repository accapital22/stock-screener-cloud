import smtplib
import schedule
import time
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import yfinance as yf
import pandas as pd
import numpy as np
import math
import io
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
import warnings
warnings.filterwarnings('ignore')

class CloudEmailService:
    def __init__(self):
        self.config = {
            'from_email': 'erndollars@gmail.com',      # UPDATE THIS
            'to_email': 'accapital22@gmail.com',         # UPDATE THIS  
            'password': 'ccpg yrqt kbor fuwk',           # UPDATE THIS (Gmail app password)
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'dashboard_url': 'https://stock-screener-cloud-vcmt76mjgjwrvh8pcmexoz.streamlit.app/'  # YOUR ACTUAL URL!
        }
        self.screened_count = 0
    
    def calculate_ema(self, prices, period=50):
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=period, adjust=False).mean()
    
    def estimate_delta(self, option_type, stock_price, strike_price, days_to_expiry, volatility=0.3):
        """Estimate option delta using Black-Scholes approximation"""
        try:
            if days_to_expiry <= 0:
                days_to_expiry = 1
            time_to_expiry = days_to_expiry / 365.0
            d1 = (np.log(stock_price / strike_price) + (0.05 + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
            return self.norm_cdf(d1) if option_type == 'call' else self.norm_cdf(d1) - 1
        except:
            return 0.5
    
    def norm_cdf(self, x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    
    def get_enhanced_options_data(self, symbol, current_price):
        """Get enhanced options data with delta estimation and open interest filter"""
        try:
            stock = yf.Ticker(symbol)
            expirations = stock.options
            
            if not expirations:
                return None, "No options available"
            
            valid_expirations = []
            for exp_date in expirations:
                exp_dt = datetime.strptime(exp_date, '%Y-%m-%d')
                days_until_exp = (exp_dt - datetime.now()).days
                if 7 <= days_until_exp <= 30:
                    valid_expirations.append((exp_date, days_until_exp))
            
            if not valid_expirations:
                return None, "No expirations in 1-4 week range"
            
            valid_expirations.sort(key=lambda x: x[1])
            options_results = []
            
            for exp_date, days_to_exp in valid_expirations[:2]:
                try:
                    options_chain = stock.option_chain(exp_date)
                    calls = options_chain.calls
                    
                    valid_calls = calls[
                        (calls['lastPrice'] >= 0.50) & 
                        (calls['lastPrice'] <= 2.50) &
                        (calls['volume'].fillna(0) > 0) &
                        (calls['openInterest'].fillna(0) > 3000)
                    ].copy()
                    
                    if len(valid_calls) == 0:
                        continue
                    
                    valid_calls['estimated_delta'] = valid_calls.apply(
                        lambda row: self.estimate_delta('call', current_price, row['strike'], days_to_exp), axis=1
                    )
                    
                    high_delta_calls = valid_calls[
                        (valid_calls['estimated_delta'] >= 0.4) & 
                        (valid_calls['estimated_delta'] <= 1.0)
                    ]
                    
                    itm_calls = calls[
                        (calls['inTheMoney'] == True) &
                        (calls['lastPrice'] >= 0.50) & 
                        (calls['lastPrice'] <= 2.50) &
                        (calls['openInterest'].fillna(0) > 3000)
                    ]
                    
                    all_valid_calls = pd.concat([high_delta_calls, itm_calls]).drop_duplicates()
                    
                    if len(all_valid_calls) > 0:
                        for _, option in all_valid_calls.iterrows():
                            options_results.append({
                                'symbol': symbol,
                                'stock_price': current_price,
                                'strike': option['strike'],
                                'option_price': option['lastPrice'],
                                'bid': option['bid'],
                                'ask': option['ask'],
                                'volume': option['volume'],
                                'open_interest': option['openInterest'],
                                'in_the_money': option['inTheMoney'],
                                'estimated_delta': getattr(option, 'estimated_delta', 0.8 if option['inTheMoney'] else 0.5),
                                'expiration': exp_date,
                                'days_to_expiry': days_to_exp
                            })
                    
                except Exception as e:
                    continue
            
            if options_results:
                return options_results, f"Found {len(options_results)} valid options"
            else:
                return None, "No options meeting criteria"
                
        except Exception as e:
            return None, f"Options error: {str(e)}"
    
    def check_market_cap_category(self, market_cap):
        if market_cap >= 200e9: return "Mega-Cap"
        elif market_cap >= 10e9: return "Large-Cap"
        elif market_cap >= 2e9: return "Mid-Cap"
        else: return "Small-Cap"
    
    def screen_stock(self, symbol):
        try:
            self.screened_count += 1
            print(f"🔍 [{self.screened_count}] Analyzing {symbol}...")
            
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
            
            options_data, options_msg = self.get_enhanced_options_data(symbol, current_price)
            if not options_data:
                return None, f"Options: {options_msg}"
            
            stock_data = {
                'Symbol': symbol,
                'Price': current_price,
                'EMA_50': current_ema,
                'EMA_Diff_Pct': ema_diff_percent,
                'Avg_Volume': avg_volume,
                'Market_Cap': market_cap,
                'Market_Cap_Category': self.check_market_cap_category(market_cap),
                'Options_Count': len(options_data),
                'Options_Details': options_data,
                'Status': '✅ PASSED'
            }
            
            return stock_data, f"PASS | EMA Diff: {ema_diff_percent:+.2f}% | Options: {len(options_data)}"
            
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def format_results_table(self, results):
        """Create a detailed table showing ALL option contracts"""
        table_data = []
        
        for result in results:
            for option in result['Options_Details']:
                table_data.append({
                    'Stock': result['Symbol'],
                    'Stock_Price': f"${result['Price']:.2f}",
                    'EMA_50': f"${result['EMA_50']:.2f}",
                    'EMA_Diff': f"{result['EMA_Diff_Pct']:+.2f}%",
                    'Option_Strike': f"${option['strike']}",
                    'Option_Price': f"${option['option_price']:.2f}",
                    'Bid/Ask': f"${option['bid']:.2f}/${option['ask']:.2f}",
                    'Open_Interest': f"{option['open_interest']:,}",
                    'Delta': f"{option.get('estimated_delta', 0):.2f}",
                    'Expiration': option['expiration'],
                    'Days_to_Exp': option['days_to_expiry'],
                    'ITM': 'Yes' if option['in_the_money'] else 'No'
                })
        
        return pd.DataFrame(table_data)
    
    def run_screener(self, symbols):
        print(f"Screening {len(symbols)} stocks...")
        passed_stocks = []
        
        for symbol in symbols:
            result, message = self.screen_stock(symbol)
            if result:
                print(f"✅ {symbol}: {message}")
                passed_stocks.append(result)
            else:
                print(f"❌ {symbol}: {message}")
            time.sleep(0.3)
        
        return passed_stocks
    
    def create_pdf_report(self, results_df, summary_text):
        """Create an enhanced professional PDF report"""
        try:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=landscape(letter),
                                  topMargin=0.5*inch, bottomMargin=0.5*inch)
            elements = []
            styles = getSampleStyleSheet()
            
            # Header Section
            header_style = styles['Heading1']
            header_style.alignment = 1
            header = Paragraph("DAILY STOCK SCREENER REPORT", header_style)
            elements.append(header)
            elements.append(Spacer(1, 0.1*inch))
            
            # Date and Time
            date_style = styles['Heading3']
            date_style.alignment = 1
            date_text = f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}"
            date_para = Paragraph(date_text, date_style)
            elements.append(date_para)
            elements.append(Spacer(1, 0.2*inch))
            
            # Screening Criteria
            criteria_title = Paragraph("<b>SCREENING CRITERIA</b>", styles['Heading2'])
            elements.append(criteria_title)
            
            criteria_items = [
                "• 50 EMA within 2% proximity",
                "• Stock price: $30 - $300", 
                "• Average volume: >2 million shares",
                "• Market cap: Mid to Large Cap (>$2B)",
                "• Options price: $0.50 - $2.50",
                "• Options delta: 0.4 - 1.0",
                "• Options open interest: >3,000",
                "• Options expiration: 1-4 weeks"
            ]
            
            for item in criteria_items:
                elements.append(Paragraph(item, styles['Normal']))
            
            elements.append(Spacer(1, 0.2*inch))
            
            # Results Summary
            summary_title = Paragraph("<b>SCREENING SUMMARY</b>", styles['Heading2'])
            elements.append(summary_title)
            elements.append(Paragraph(summary_text, styles['Normal']))
            elements.append(Spacer(1, 0.3*inch))
            
            # Results Table
            if not results_df.empty:
                results_title = Paragraph("<b>QUALIFIED STOCKS & OPTIONS</b>", styles['Heading2'])
                elements.append(results_title)
                elements.append(Spacer(1, 0.1*inch))
                
                # Prepare table data
                table_data = [results_df.columns.tolist()]
                for _, row in results_df.iterrows():
                    table_data.append(row.tolist())
                
                # Create table with auto column widths
                col_widths = [doc.width / len(results_df.columns)] * len(results_df.columns)
                table = Table(table_data, colWidths=col_widths, repeatRows=1)
                
                # Table styling
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                
                elements.append(table)
            
            elements.append(Spacer(1, 0.3*inch))
            
            # Footer
            footer_text = "Automatically generated by Stock Screener - For educational purposes only"
            footer = Paragraph(f"<i><font size=8>{footer_text}</font></i>", styles['Italic'])
            elements.append(footer)
            
            # Build PDF
            doc.build(elements)
            pdf_value = buffer.getvalue()
            buffer.close()
            
            email_buffer = io.BytesIO()
            email_buffer.write(pdf_value)
            email_buffer.seek(0)
            
            print("✅ PDF report created successfully")
            return email_buffer
            
        except Exception as e:
            print(f"❌ Enhanced PDF creation failed: {e}")
            return io.BytesIO()
    
    def send_daily_alert(self):
        """Send daily mobile-friendly email with dashboard link AND PDF attachment"""
        try:
            print("🔄 Running automated screening for daily email...")
            
            # Run the screener to get results for PDF
            symbols = self.get_sp500_symbols()
            results = self.run_screener(symbols)
            
            # Create results table and summary
            if results:
                results_table = self.format_results_table(results)
                summary = f"Found {len(results)} stocks with {len(results_table)} total option contracts meeting all criteria."
                print(f"📊 Screening complete: {summary}")
            else:
                results_table = pd.DataFrame()
                summary = "No stocks met all screening criteria today."
                print("📊 Screening complete: No qualifying stocks found")
            
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.config['from_email']
            msg['To'] = self.config['to_email']
            msg['Subject'] = f"📱 Stock Screener Ready - {datetime.now().strftime('%m/%d')}"
            
            # Create HTML email content
            html = f"""
            <html>
            <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; color: white; text-align: center;">
                    <h1 style="margin: 0;">📈 Stock Screener Pro</h1>
                    <p style="margin: 10px 0 0 0;">Daily Screening Complete</p>
                </div>
                
                <div style="padding: 20px;">
                    <h2>🚀 Your Dashboard is Ready</h2>
                    <p>Access your live stock screener from any device:</p>
                    
                    <div style="text-align: center; margin: 30px 0;">
                        <a href="{self.config['dashboard_url']}" 
                           style="background-color: #4CAF50; color: white; padding: 15px 30px; text-decoration: none; border-radius: 25px; font-size: 18px; display: inline-block;">
                            📲 Open Dashboard
                        </a>
                    </div>
                    
                    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 20px 0;">
                        <h3>📊 Screening Results</h3>
                        <p><strong>{summary}</strong></p>
                        <p><strong>Mobile:</strong> Tap the button above</p>
                        <p><strong>Desktop:</strong> <a href="{self.config['dashboard_url']}">{self.config['dashboard_url']}</a></p>
                        <p><strong>Time:</strong> {datetime.now().strftime('%I:%M %p EST')}</p>
                    </div>
                    
                    <div style="background-color: #e8f4fd; padding: 15px; border-radius: 10px; margin: 20px 0;">
                        <h3>📎 PDF Report Attached</h3>
                        <p>A detailed PDF report with all qualified stocks and options is attached to this email.</p>
                    </div>
                    
                    <p style="color: #666; font-size: 12px; text-align: center;">
                        This is an automated message. The dashboard is available 24/7.
                    </p>
                </div>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(html, 'html'))
            
            # CREATE PDF ATTACHMENT
            if not results_table.empty:
                pdf_buffer = self.create_pdf_report(results_table, summary)
                
                # Create PDF attachment
                attachment = MIMEBase('application', 'pdf')
                attachment.set_payload(pdf_buffer.getvalue())
                encoders.encode_base64(attachment)
                attachment.add_header('Content-Disposition', 
                                    'attachment', 
                                    filename=f"stock_screener_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
                msg.attach(attachment)
                
                print("✅ PDF report attached to email")
            else:
                print("ℹ️ No results to attach - skipping PDF attachment")
            
            # Send email
            server = smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port'])
            server.starttls()
            server.login(self.config['from_email'], self.config['password'])
            server.send_message(msg)
            server.quit()
            
            print(f"✅ Mobile alert with PDF sent at {datetime.now()}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to send alert: {e}")
            return False
    
    def get_sp500_symbols(self):
        """Get S&P 500 symbols for screening"""
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 
            'V', 'PG', 'UNH', 'HD', 'DIS', 'PYPL', 'NFLX', 'ADBE', 'CRM', 'INTC', 
            'CSCO', 'PEP', 'T', 'ABT', 'TMO', 'COST', 'AVGO', 'TXN', 'LLY', 'WMT',
            'XOM', 'CVX', 'MRK', 'PFE', 'ABBV', 'DHR', 'MDT', 'BMY', 'UPS', 'SBUX',
            'AMGN', 'HON', 'RTX', 'LOW', 'BA', 'CAT', 'DE', 'MMM', 'GE', 'F',
            'GS', 'NKE', 'IBM', 'AMD', 'QCOM', 'TGT', 'LMT', 'BKNG', 'NOW', 'INTU'
        ]
    
    def start_service(self):
        """Start the daily email service"""
        print("📧 Starting Cloud Email Service...")
        print("⏰ Scheduled: Daily at 8:00 PM EST")
        print("🌐 Dashboard URL:", self.config['dashboard_url'])
        print("📎 Feature: PDF reports with screening results attached")
        
        # Schedule daily at 8:00 PM
        schedule.every().day.at("20:00").do(self.send_daily_alert)
        
        # Send immediate test email
        print("Sending test email with PDF...")
        self.send_daily_alert()
        
        print("🕒 Email scheduler is running. Press Ctrl+C to stop.")
        
        while True:
            schedule.run_pending()
            time.sleep(60)

if __name__ == "__main__":
    service = CloudEmailService()
    service.start_service()