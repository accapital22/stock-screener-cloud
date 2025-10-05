import smtplib
import schedule
import time
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

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
    
    def send_daily_alert(self):
        """Send daily mobile-friendly email with dashboard link"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config['from_email']
            msg['To'] = self.config['to_email']
            msg['Subject'] = f"📱 Stock Screener Ready - {datetime.now().strftime('%m/%d')}"
            
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
                        <h3>📊 Quick Access</h3>
                        <p><strong>Mobile:</strong> Tap the button above</p>
                        <p><strong>Desktop:</strong> <a href="{self.config['dashboard_url']}">{self.config['dashboard_url']}</a></p>
                        <p><strong>Time:</strong> {datetime.now().strftime('%I:%M %p EST')}</p>
                    </div>
                    
                    <p style="color: #666; font-size: 12px; text-align: center;">
                        This is an automated message. The dashboard is available 24/7.
                    </p>
                </div>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(html, 'html'))
            
            # Send email
            server = smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port'])
            server.starttls()
            server.login(self.config['from_email'], self.config['password'])
            server.send_message(msg)
            server.quit()
            
            print(f"✅ Mobile alert sent at {datetime.now()}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to send alert: {e}")
            return False
    
    def start_service(self):
        """Start the daily email service"""
        print("📧 Starting Cloud Email Service...")
        print("⏰ Scheduled: Daily at 8:00 PM EST")
        print("🌐 Dashboard URL:", self.config['dashboard_url'])
        
        # Schedule daily at 8:00 PM
        schedule.every().day.at("20:00").do(self.send_daily_alert)
        
        # Send immediate test email
        print("Sending test email...")
        self.send_daily_alert()
        
        print("🕒 Email scheduler is running. Press Ctrl+C to stop.")
        
        while True:
            schedule.run_pending()
            time.sleep(60)

if __name__ == "__main__":
    service = CloudEmailService()
    service.start_service()