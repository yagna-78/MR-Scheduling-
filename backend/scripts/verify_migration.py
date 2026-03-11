
import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("Error: Env vars missing")
    exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

tables = ["users", "contacts", "activities", "master_schedule"]

print("--- Supabase Table Verification ---")
for t in tables:
    try:
        res = supabase.table(t).select("id", count="exact").execute()
        count = res.count
        print(f"✅ {t}: {count} rows")
    except Exception as e:
        print(f"❌ {t}: Error - {e}")
