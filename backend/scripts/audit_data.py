
import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

print("--- Data Audit Report ---")

# 1. Get all MRs
try:
    users_res = supabase.table("users").select("mr_id, first_name, last_name").execute()
    users = users_res.data
    users.sort(key=lambda x: x['mr_id'])
    
    print(f"Found {len(users)} MRs in User Directory.\n")
    print(f"{'MR ID':<15} {'Name':<25} {'Schedule Tasks':<15} {'Activity Logs':<15}")
    print("-" * 75)

    for user in users:
        mr_id = user['mr_id']
        name = f"{user.get('first_name', '')} {user.get('last_name', '')}".strip()
        
        # Count Master Schedule
        sched_res = supabase.table("master_schedule").select("*", count="exact").eq("mr_id", mr_id).execute()
        sched_count = sched_res.count
        
        # Count Activities
        act_res = supabase.table("activities").select("*", count="exact").eq("mr_id", mr_id).execute()
        act_count = act_res.count
        
        print(f"{mr_id:<15} {name:<25} {sched_count:<15} {act_count:<15}")

except Exception as e:
    print(f"Error during audit: {e}")
