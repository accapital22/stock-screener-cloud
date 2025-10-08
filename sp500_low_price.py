import yfinance as yf
import pandas as pd
import time
from datetime import datetime

def get_sp500_low_price_stocks():
    print("🔄 Loading S&P 500 symbols...")
    
    # Get current S&P 500 constituents from Wikipedia
    try:
        sp500_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S&P_500_companies')
        sp500_df = sp500_table[0]
        symbols = sp500_df['Symbol'].tolist()
        print(f"✅ Loaded {len(symbols)} S&P 500 symbols")
    except:
        print("❌ Failed to load from Wikipedia, using fallback list")
        symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V',
            'PG', 'UNH', 'HD', 'DIS', 'PYPL', 'NFLX', 'ADBE', 'CRM', 'INTC', 'CSCO',
            'PEP', 'T', 'ABT', 'TMO', 'AVGO', 'COST', 'LLY', 'WMT', 'XOM', 'CVX',
            'MRK', 'PFE', 'ABBV', 'DHR', 'MDT', 'NEE', 'UNP', 'HON', 'RTX', 'LOW',
            'SPGI', 'ORCL', 'TXN', 'QCOM', 'AMGN', 'UPS', 'SBUX', 'BA', 'CAT', 'DE'
        ]

    low_price_stocks = []
    failed_symbols = []
    
    print("🔄 Fetching current prices...")
    
    for i, symbol in enumerate(symbols):
        try:
            # Add delay to avoid rate limiting
            if i > 0:
                time.sleep(0.1)
            
            # Get stock data
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1d")
            
            if len(hist) > 0:
                current_price = hist['Close'].iloc[-1]
                
                if current_price <= 300:
                    stock_data = {
                        'Symbol': symbol,
                        'Company_Name': info.get('longName', info.get('shortName', 'N/A')),
                        'Current_Price': round(current_price, 2),
                        'Market_Cap_Billions': round(info.get('marketCap', 0) / 1e9, 2) if info.get('marketCap') else 'N/A',
                        'Sector': info.get('sector', 'N/A'),
                        'Industry': info.get('industry', 'N/A'),
                        'Volume': f"{hist['Volume'].iloc[-1]:,}" if 'Volume' in hist.columns else 'N/A',
                        'PE_Ratio': info.get('trailingPE', 'N/A'),
                        'Day_Change_Percent': round(((current_price - hist['Open'].iloc[-1]) / hist['Open'].iloc[-1]) * 100, 2) if 'Open' in hist.columns else 'N/A'
                    }
                    low_price_stocks.append(stock_data)
                    
                    print(f"✅ {symbol}: ${current_price:.2f}")
                else:
                    print(f"❌ {symbol}: ${current_price:.2f} (above $300)")
            else:
                failed_symbols.append(symbol)
                print(f"⚠️ {symbol}: No data")
                
        except Exception as e:
            failed_symbols.append(symbol)
            print(f"❌ {symbol}: Error - {str(e)}")
    
    # Create DataFrame and save to CSV
    if low_price_stocks:
        df = pd.DataFrame(low_price_stocks)
        
        # Sort by price (lowest first)
        df = df.sort_values('Current_Price')
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"sp500_stocks_under_300_{timestamp}.csv"
        
        # Save to CSV
        df.to_csv(filename, index=False)
        
        print(f"\n🎉 SUCCESS: Found {len(low_price_stocks)} stocks priced ≤ $300")
        print(f"💾 File saved as: {filename}")
        print(f"📊 Summary:")
        print(f"   - Lowest price: ${df['Current_Price'].min():.2f}")
        print(f"   - Highest price: ${df['Current_Price'].max():.2f}")
        print(f"   - Average price: ${df['Current_Price'].mean():.2f}")
        
        if failed_symbols:
            print(f"⚠️ Failed to process {len(failed_symbols)} symbols")
            
        # Show top 5 lowest priced stocks
        print(f"\n📈 Top 5 Lowest Priced Stocks:")
        for _, row in df.head().iterrows():
            print(f"   {row['Symbol']}: ${row['Current_Price']} - {row['Company_Name']}")
            
    else:
        print("❌ No stocks found priced ≤ $300")
    
    return low_price_stocks

# Run the function
if __name__ == "__main__":
    get_sp500_low_price_stocks()
