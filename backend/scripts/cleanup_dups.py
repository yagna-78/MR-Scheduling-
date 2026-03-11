
import os
import sys
import pandas as pd
from datetime import datetime

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.supabase_db import supabase

def fetch_all_rows(table_name):
    """Fetch all rows using pagination"""
    all_data = []
    chunk_size = 1000
    start = 0
    
    while True:
        print(f"Fetching {table_name} rows {start} to {start+chunk_size}...")
        response = supabase.table(table_name).select("*").range(start, start + chunk_size - 1).execute()
        data = response.data
        if not data:
            break
        all_data.extend(data)
        if len(data) < chunk_size:
            break
        start += chunk_size
        
    return all_data

def cleanup_table(table_name):
    print(f"\n--- Cleaning {table_name} ---")
    
    # 1. Fetch all data
    data = fetch_all_rows(table_name)
    print(f"Total rows in DB: {len(data)}")
    
    if not data:
        return

    df = pd.DataFrame(data)
    
    # Define columns that constitute a "duplicate" visit
    # Same MR, Same Customer, Same Date
    dedup_cols = ['mr_id', 'date', 'customer_id']
    
    # Verify cols exist
    if not set(dedup_cols).issubset(df.columns):
        print(f"Skipping {table_name}, missing columns.")
        return

    # 2. Identify duplicates
    # We want to KEEP the one with the latest ID (assuming higher ID = newer) or created_at
    # Let's sort by id descending so the first one we see is the latest
    if 'id' in df.columns:
        df = df.sort_values('id', ascending=True) # Keep first (oldest) or last?
        # Usually duplicates happen because scripts ran multiple times.
        # If we keep the first one, we keep the original. 
        # If we keep the last one, we keep the latest generate.
        # Let's keep the LAST one (latest imported/generated) just in case logic changed.
        # Valid strategy: keep 'last'
    
    duplicates = df[df.duplicated(subset=dedup_cols, keep='last')]
    
    if duplicates.empty:
        print("No duplicates found to clean.")
        return
        
    ids_to_remove = duplicates['id'].tolist()
    print(f"Found {len(ids_to_remove)} duplicate rows to remove.")
    
    # 3. Delete in batches
    batch_size = 100
    deleted_count = 0
    
    print("Deleting duplicates...")
    for i in range(0, len(ids_to_remove), batch_size):
        batch_ids = ids_to_remove[i:i + batch_size]
        try:
            supabase.table(table_name).delete().in_('id', batch_ids).execute()
            deleted_count += len(batch_ids)
            print(f"Deleted {deleted_count}/{len(ids_to_remove)}", end='\r')
        except Exception as e:
            print(f"Error deleting batch: {e}")
            
    print(f"\nSuccessfully deleted {deleted_count} duplicate rows from {table_name}.")

def main():
    if not supabase:
        print("Supabase client not initialized.")
        return

    # Clean valid tables
    cleanup_table("activities")
    cleanup_table("master_schedule")

if __name__ == "__main__":
    main()
