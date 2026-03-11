
import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Search params
mr_id = "MR_W1_1"
date_to_check = "2026-01-29"

print(f"Checking Master Schedule for {mr_id} on {date_to_check}...")

try:
    # 1. Fetch exactly matching row
    res = supabase.table("master_schedule").select("*").eq("mr_id", mr_id).eq("date", date_to_check).execute()
    
    print(f"\n✅ Exact match count: {len(res.data)}")
    if res.data:
        print(f"Sample task: {res.data[0]['customer_name']} - {res.data[0]['status']}")
    else:
        # 2. Check ANY tasks for this MR
        print("\n❌ No tasks for this date.")
        res_any = supabase.table("master_schedule").select("date").eq("mr_id", mr_id).limit(5).execute()
        print(f"Found other dates for {mr_id}: {[t['date'] for t in res_any.data]}")

except Exception as e:
    print(f"Error: {e}")
