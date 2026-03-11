
import os
import sys
import pandas as pd
from collections import Counter

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.supabase_db import supabase

def analyze_duplicates(table_name):
    print(f"\nAnalyzing table: {table_name}...")
    
    try:
        # Fetch all data (handling pagination if needed, but for now assuming < 10000 or using limits)
        # We'll fetch just IDs and key fields to identify duplicates
        response = supabase.table(table_name).select("*").execute()
        data = response.data
        
        if not data:
            print(f"No data found in {table_name}")
            return

        df = pd.DataFrame(data)
        print(f"Total rows: {len(df)}")
        
        if 'activity_id' in df.columns:
            # Check by activity_id
            duplicates = df[df.duplicated(subset=['activity_id'], keep=False)]
            duplicate_count = len(duplicates)
            unique_ids = duplicates['activity_id'].nunique()
            print(f"Duplicates by activity_id: {duplicate_count} rows ({unique_ids} unique IDs)")
            
            if duplicate_count > 0:
                print("Example duplicates:")
                print(duplicates.sort_values('activity_id').head(10)[['id', 'activity_id', 'date', 'mr_id']])
        
        # Check by composite key just in case activity_id is unique but content is same
        if {'mr_id', 'date', 'customer_id'}.issubset(df.columns):
            composite_dups = df[df.duplicated(subset=['mr_id', 'date', 'customer_id'], keep=False)]
            print(f"Duplicates by (mr_id, date, customer_id): {len(composite_dups)} rows")

    except Exception as e:
        print(f"Error: {e}")

def main():
    if not supabase:
        print("Supabase client not initialized. Check env vars.")
        return

    analyze_duplicates("master_schedule")
    analyze_duplicates("activities")

if __name__ == "__main__":
    main()
