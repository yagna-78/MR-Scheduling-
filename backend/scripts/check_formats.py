
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.services.supabase_db import supabase

def inspect_dates():
    print("Checking date formats...")
    
    # Fetch a sample
    response = supabase.table("activities").select("date").limit(50).execute()
    data = response.data
    if data:
        print("Activities Dates sample:", [d['date'] for d in data[:5]])
    else:
        print("No activities data found.")
        
    response = supabase.table("master_schedule").select("date").limit(50).execute()
    data = response.data
    if data:
        print("Master Schedule Dates sample:", [d['date'] for d in data[:5]])

if __name__ == "__main__":
    inspect_dates()
