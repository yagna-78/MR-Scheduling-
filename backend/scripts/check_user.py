
import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

print("Checking for MR_W1_1...")
try:
    # Explicitly select mr_id column
    res = supabase.table("users").select("*").eq("mr_id", "MR_W1_1").execute()
    if res.data:
        print(f"✅ FOUND: {res.data[0]}")
    else:
        print("❌ NOT FOUND")
        # List all to see what's there
        all_users = supabase.table("users").select("mr_id").execute()
        print(f"Available IDs: {[u['mr_id'] for u in all_users.data]}")
except Exception as e:
    print(f"Error: {e}")
