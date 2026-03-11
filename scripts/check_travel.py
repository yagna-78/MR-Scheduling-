
import os
import sys

# Add backend directory to sys.path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

from app.services.supabase_db import supabase

def check_travel_time():
    print("Fetching master_schedule...")
    response = supabase.table("master_schedule").select("activity_id, travel_duration_min").execute()
    data = response.data
    
    if not data:
        print("No data in master_schedule")
        return

    total = len(data)
    with_travel = sum(1 for item in data if item.get('travel_duration_min') is not None)
    
    print(f"Total rows: {total}")
    print(f"Rows with travel_duration_min: {with_travel}")
    print("First 20 items:")
    for item in data[:20]:
        print(item)

if __name__ == "__main__":
    check_travel_time()
