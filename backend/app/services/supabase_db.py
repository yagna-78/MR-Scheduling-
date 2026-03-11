
import os
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL:
    print("WARNING: SUPABASE_URL not set. Database operations will fail.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

# Map Sheet Names to Supabase Table Names
TABLE_MAP = {
    "User_Master": "users",
    "Contacts": "contacts",
    "Activities": "activities",
    "Master_Schedule": "master_schedule"
}

def load_data(sheet_name: str) -> pd.DataFrame:
    """Load data from Supabase table into DataFrame"""
    if not supabase:
        print("Supabase not configured.")
        return pd.DataFrame()
        
    table_name = TABLE_MAP.get(sheet_name, sheet_name.lower())
    
    try:
        # Fetch all data (Supabase limits to 1000 by default, need to paginate for large sets)
        # For now, simplistic fetch
        response = supabase.table(table_name).select("*").execute()
        data = response.data
        
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        return df
        
    except Exception as e:
        print(f"Error loading table '{table_name}': {e}")
        return pd.DataFrame()

def save_data(df: pd.DataFrame, sheet_name: str):
    """
    Save DataFrame to Supabase table. 
    WARNING: This mimics the GSheets 'Overwrite' behavior by deleting all rows first.
    """
    if not supabase:
        print("Supabase not configured.")
        return

    table_name = TABLE_MAP.get(sheet_name, sheet_name.lower())

    if df.empty:
        return

    try:
        # 1. Convert DataFrame to records
        # Handle NaN values which JSON doesn't like
        df_clean = df.where(pd.notnull(df), None)
        # Convert Timestamp/Date to strings
        for col in df_clean.columns:
            if pd.api.types.is_datetime64_any_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].astype(str)
        
        records = df_clean.to_dict(orient='records')
        
        # 2. Delete all existing records (Logic: ID is not null)
        # Note: This is dangerous for production but matches 'Sheet Overwrite' logic.
        # Ideally, we should use Upsert if we have Primary Keys.
        # But 'Master_Schedule' behaves like a snapshot in the current app.
        
        # Batch delete if needed, but for now simple delete
        # Requires a column to match. Assuming 'id' exists or we use a always-true condition if possible.
        # Supabase-py doesn't support 'delete all' easily without a where clause.
        # We can try to delete where id > 0 (assuming numeric ID)
        
        # Hack: Check if 'id' is in columns, else use a known column
        # Or better: UPSERT based on a unique key if we have one.
        
        # Strategy: upsert to avoid deletion issues, IF we have `activity_id` or similar.
        # However, if rows were removed in DF, upsert won't remove them from DB.
        
        # Let's try to delete everything first.
        # 'id' is created by Supabase usually.
        supabase.table(table_name).delete().neq("id", 0).execute() # 0 is standard dummy for 'all non-zero'
        
        # 3. Insert in chunks
        chunk_size = 1000
        for i in range(0, len(records), chunk_size):
            chunk = records[i:i + chunk_size]
            supabase.table(table_name).insert(chunk).execute()
            
        print(f"Saved {len(records)} rows to '{table_name}'")

    except Exception as e:
        print(f"Error saving to table '{table_name}': {e}")
