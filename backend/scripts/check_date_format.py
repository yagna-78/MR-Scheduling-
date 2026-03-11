
import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

print("--- Checking Date Formats in master_schedule ---")
res = supabase.table("master_schedule").select("date").limit(20).execute()
dates = [r['date'] for r in res.data]
print(f"Sample dates: {dates}")

print("\n--- Testing Report Logic ---")
start = "2026-01-01"
end = "2026-02-01"
print(f"Querying range: {start} to {end}")

try:
    res = supabase.table("master_schedule").select("*").gte("date", start).lte("date", end).limit(5).execute()
    print(f"Found {len(res.data)} rows.")
except Exception as e:
    print(f"Query failed: {e}")
