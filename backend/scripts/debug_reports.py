
import os
import sys
import pandas as pd

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.supabase_db import supabase

def test_query(mr_id):
    print(f"\nTesting query for mr_id='{mr_id}'...")
    
    query = supabase.table("master_schedule").select("*")
    
    if mr_id and mr_id.lower() != 'admin':
        clean_mr_id = mr_id.strip()
        print(f"Applying filter: ilike('mr_id', '{clean_mr_id}')")
        query = query.ilike("mr_id", clean_mr_id)
        
    query = query.eq("date", "2026-01-30")
    
    response = query.execute()
    data = response.data
    print(f"Result count: {len(data)}")
    
    if len(data) > 0:
        print("First 3 MR IDs found:")
        print([x['mr_id'] for x in data[:3]])

def main():
    # Test strict match
    test_query("MR_N1_3")
    
    # Test lowercase match
    test_query("mr_n1_3")
    
    # Test empty (should be all 114)
    test_query("")

if __name__ == "__main__":
    main()
