
import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

target_mr = "MR_W1_1"

print(f"--- Checking tasks for {target_mr} ---")
res = supabase.table("master_schedule").select("date, activity_id").eq("mr_id", target_mr).limit(10).execute()

if not res.data:
    print("❌ No tasks found for this MR ID.")
    # Try with lowercase
    res_lower = supabase.table("master_schedule").select("*").eq("mr_id", target_mr.lower()).limit(1).execute()
    if res_lower.data:
        print("✅ Found with lowercase ID!")
    else:
         print("❌ Checked lowercase too, nothing.")
else:
    print(f"✅ Found {len(res.data)} tasks. Sample dates:")
    for row in res.data:
        print(f"Date: '{row['date']}' (Type: {type(row['date'])})")
