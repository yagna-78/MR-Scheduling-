
import os
import sys
import pandas as pd
from collections import defaultdict

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.supabase_db import supabase

MAX_VISITS_PER_DAY = 15

def fetch_all_rows(table_name):
    """Fetch all rows using pagination"""
    all_data = []
    chunk_size = 1000
    start = 0
    
    while True:
        print(f"Fetching {table_name} rows {start} to {start+chunk_size}...", end='\r')
        response = supabase.table(table_name).select("id, mr_id, date").range(start, start + chunk_size - 1).execute()
        data = response.data
        if not data:
            break
        all_data.extend(data)
        if len(data) < chunk_size:
            break
        start += chunk_size
    print() # Newline
    return all_data

def prune_table(table_name, date_col='date'):
    print(f"\n--- Pruning {table_name} ---")
    
    # Fetch all data using pagination
    data = fetch_all_rows(table_name)
    
    if not data:
        print("No data found.")
        return

    df = pd.DataFrame(data)
    print(f"Total rows: {len(df)}")
    
    # Group by MR and Date
    groups = df.groupby(['mr_id', date_col])
    
    ids_to_delete = []
    
    for (mr_id, date), group in groups:
        if len(group) > MAX_VISITS_PER_DAY:
            # Keep first MAX_VISITS_PER_DAY items (by ID assumes order?)
            # We'll just take the head
            # Identify IDs to drop
            # Sort by ID to remove "newer" duplicates/generations?
            # Or shuffle? Let's exact sort by ID.
            sorted_group = group.sort_values('id')
            
            # Keep the first N
            keep_ids = set(sorted_group.head(MAX_VISITS_PER_DAY)['id'])
            
            # Remove the rest
            all_ids = set(sorted_group['id'])
            drop_ids = list(all_ids - keep_ids)
            ids_to_delete.extend(drop_ids)
            
    if not ids_to_delete:
        print("No pruning needed. All days within limits.")
        return
        
    print(f"Found {len(ids_to_delete)} excessive rows to delete (Limit: {MAX_VISITS_PER_DAY}/day).")
    
    # Delete in batches
    batch_size = 100
    deleted_count = 0
    
    print("Deleting excessive rows...")
    for i in range(0, len(ids_to_delete), batch_size):
        batch_ids = ids_to_delete[i:i + batch_size]
        try:
            supabase.table(table_name).delete().in_('id', batch_ids).execute()
            deleted_count += len(batch_ids)
            print(f"Deleted {deleted_count}/{len(ids_to_delete)}", end='\r')
        except Exception as e:
            print(f"\nError deleting batch: {e}")
            
    print(f"\nSuccessfully pruned {deleted_count} rows from {table_name}.")

def main():
    if not supabase:
        print("Supabase client not initialized.")
        return

    # Prune master_schedule
    # Note: Using 'date' column
    prune_table("master_schedule")
    
    # Prune activities
    # Note: Also 'date' column, though format differs, grouping still works by unique string
    prune_table("activities")

if __name__ == "__main__":
    main()
