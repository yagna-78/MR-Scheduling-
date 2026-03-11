
import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

target_date = "2026-01-29"

print(f"--- DIAGNOSTICS FOR {target_date} ---")

# 1. Fetch Users
users = supabase.table("users").select("mr_id, first_name, last_name").execute().data
print(f"Total Users: {len(users)}")

# 2. Check Coverage
print(f"{'Name':<25} {'MR ID':<15} {'Tasks Today'}")
print("-" * 55)

for u in users:
    name = f"{u.get('first_name', '')} {u.get('last_name', '')}".strip()
    mr_id = u['mr_id']
    
    # Count tasks
    res = supabase.table("master_schedule").select("*", count="exact").eq("mr_id", mr_id).eq("date", target_date).execute()
    count = res.count
    
    print(f"{name:<25} {mr_id:<15} {count}")

print("\n--- CHECKING FOR COMMON NAMES ---")
names_to_check = ["Amit", "Narendra", "Shah", "Soni"]
for n in names_to_check:
    print(f"Searching for '{n}'...")
    matches = [u for u in users if n.lower() in (u.get('first_name','')+u.get('last_name','')).lower()]
    for m in matches:
        print(f"  FOUND: {m['first_name']} {m['last_name']} ({m['mr_id']})")

