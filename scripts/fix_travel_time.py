
import os
import sys
import math

# Add backend directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

from app.services.supabase_db import supabase

def calculate_time(dist_km):
    # Avg speed for city traffic: 25 km/h
    speed_kmh = 25
    return int((dist_km / speed_kmh) * 60)

def fix_travel_data():
    if not supabase:
        print("Supabase connection failed.")
        return

    print("Fetching master_schedule...")
    # Fetch all rows (paginate if needed, but for now assuming < 1000 or fetching max)
    response = supabase.table("master_schedule").select("*").execute()
    data = response.data
    
    if not data:
        print("No data found.")
        return

    updates = []
    
    print(f"Checking {len(data)} rows...")
    
    count_fixed = 0
    
    for row in data:
        has_change = False
        
        # Check Travel Duration
        t_dur = row.get('travel_duration_min')
        dist = row.get('distance_km')
        
        # If travel duration is missing (None, empty string, or 0)
        if not t_dur and dist:
             # Fix based on distance
             try:
                 d_val = float(dist)
                 new_time = calculate_time(d_val)
                 row['travel_duration_min'] = new_time
                 has_change = True
                 count_fixed += 1
             except ValueError:
                 pass
        
        # If distance is missing? (Optional, but user focused on travel time)
        
        if has_change:
            updates.append(row)

    if not updates:
        print("No rows needed fixing.")
        return

    print(f"Fixing {len(updates)} rows...")
    
    # Update in DB
    # Since we don't have a reliable single primary key that we know is unique and persistent (id is there but we need to be careful),
    # and upsert requires a primary key constraint. 
    # 'id' is likely the PK.
    
    # We will upsert using 'id'.
    try:
        # split into chunks
        chunk_size = 100
        for i in range(0, len(updates), chunk_size):
            chunk = updates[i:i + chunk_size]
            response = supabase.table("master_schedule").upsert(chunk).execute()
            print(f"Upserted items {i} to {i+len(chunk)}")
            
        print("✅ Fix Complete! Refresh your dashboard.")
        
    except Exception as e:
        print(f"Error updating DB: {e}")

if __name__ == "__main__":
    fix_travel_data()
