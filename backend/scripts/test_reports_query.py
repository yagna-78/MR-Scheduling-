
import requests
import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

BASE_URL = "http://localhost:8000"

print("--- 1. Testing DB Connectivity ---")
try:
    res = supabase.table("master_schedule").select("*").limit(1).execute()
    print("DB OK.")
except Exception as e:
    print(f"DB Error: {e}")
    exit()

print("\n--- 2. Direct DB Query for Report Params ---")
# Simulate: Activity Report, Start 29-01-2026 (aka 2026-01-29), MR_W1_2
mr_id = "MR_W1_2" 
date = "2026-01-29"

res = supabase.table("master_schedule").select("*").eq("mr_id", mr_id).eq("date", date).execute()
print(f"Direct Query Found: {len(res.data)} rows for {mr_id} on {date}")
if len(res.data) > 0:
    print(f"Sample: {res.data[0]['customer_name']}")

print("\n--- 3. Testing Compliance Params (Month 1, 2026) ---")
start_date = "2026-01-01"
end_date = "2026-01-31"

res = supabase.table("master_schedule").select("*").eq("mr_id", mr_id).gte("date", start_date).lte("date", end_date).execute()
print(f"Direct Compliance Query Found: {len(res.data)} rows in Jan 2026")

