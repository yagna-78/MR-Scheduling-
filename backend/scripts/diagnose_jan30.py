
import os
import sys
import pandas as pd

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.supabase_db import supabase

def diagnose_jan30():
    print("Diagnosing Jan 30 in master_schedule...")
    
    # Fetch all rows for this date
    response = supabase.table("master_schedule").select("*").eq("date", "2026-01-30").execute()
    data = response.data
    
    if not data:
        print("No data in master_schedule for 2026-01-30")
        return

    df = pd.DataFrame(data)
    print(f"Total rows on 2026-01-30: {len(df)}")
    
    # Counts by MR
    print("\nCounts by MR ID:")
    print(df['mr_id'].value_counts())
    
    # Show sample IDs for top MR
    top_mr = df['mr_id'].mode()[0]
    print(f"\nSample IDs for {top_mr}:")
    print(df[df['mr_id'] == top_mr][['id', 'activity_id', 'customer_id']].head(10))

if __name__ == "__main__":
    diagnose_jan30()
