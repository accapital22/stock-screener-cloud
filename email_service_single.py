# email_service_single.py - Run once and exit
from email_service import CloudEmailService

print("🏃 Running one-time screening...")
service = CloudEmailService()
service.send_daily_alert()
print("✅ Screening complete, exiting.")