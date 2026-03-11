
import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

date_target = "2026-01-29"
print(f"--- Checking tasks for date {date_target} ---")

res = supabase.table("master_schedule").select("mr_id, date").eq("date", date_target).limit(10).execute()

if res.data:
    print(f"✅ Found tasks on this date for: {[r['mr_id'] for r in res.data]}")
else:
    print("❌ No tasks found for ANYONE on this date.")
    
    # Check date format in DB by listing any date
    res_any = supabase.table("master_schedule").select("date").limit(1).execute()
    if res_any.data:
        print(f"Sample date in DB: '{res_any.data[0]['date']}'")
