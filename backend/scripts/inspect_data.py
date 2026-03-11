
import os
import sys
import pandas as pd

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.supabase_db import supabase

def inspect_day():
    print("Inspecting activities for 2026-01-30...")
    
    # Filter for the specific day and roughly the MR if possible, or just search all
    response = supabase.table("activities").select("*").eq("date", "2026-01-30").execute()
    data = response.data
    
    if not data:
        print("No data for 2026-01-30")
        return

    df = pd.DataFrame(data)
    print(f"Total rows for 2026-01-30: {len(df)}")
    
    print("\nSample of data (MR_ID, Customer_ID, Local Time):")
    print(df[['mr_id', 'customer_id', 'start_time']].head(20))
    
    # Check counts per MR
    print("\nCounts per MR:")
    print(df['mr_id'].value_counts())

if __name__ == "__main__":
    inspect_day()
