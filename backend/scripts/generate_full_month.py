
import os
import uuid
import random
import datetime
from datetime import timedelta
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Configuration to populate Whole Month
# From Jan 1 to Feb 28 to cover all bases
START_DATE = datetime.date(2026, 1, 1)
END_DATE = datetime.date(2026, 2, 28)

print(f"--- 🚀 Starting WHOLE MONTH Data Generation ({START_DATE} to {END_DATE}) ---")

# 1. Fetch Users
users = supabase.table("users").select("*").execute().data
print(f"Found {len(users)} MRs.")

# 2. Fetch Template tasks
template_tasks = supabase.table("master_schedule").select("*").limit(50).execute().data
if not template_tasks:
    print("❌ No template data found!")
    exit()

def generate_day_schedule(mr, date_str, pool):
    tasks = []
    # Skip Sundays
    if datetime.datetime.strptime(date_str, "%Y-%m-%d").weekday() == 6:
        return []

    count = random.randint(5, 12)
    start_lat = mr.get('starting_latitude') or 23.0
    start_lon = mr.get('starting_longitude') or 72.5
    
    for i in range(count):
        template = random.choice(pool)
        
        task = {
            "mr_id": mr['mr_id'],
            "team": mr.get('team', 'General'),
            "zone": mr.get('zone', 'North'),
            "date": date_str,
            "activity_id": f"FULL_{mr['mr_id']}_{date_str.replace('-','')}_{i}",
            "status": "Done" if date_str < "2026-01-29" else "Pending", # Past is Done, Future is Pending
            "customer_id": template.get('customer_id', f"C_{i}"),
            "customer_name": f"Client {chr(65+i)}{i} - {template.get('customer_name', 'Doc')[:10]}",
            "activity_type": "Visit",
            "locality": template.get('locality', "Area"),
            "latitude": start_lat + random.uniform(-0.03, 0.03),
            "longitude": start_lon + random.uniform(-0.03, 0.03),
            "start_time": f"{9+i}:00",
            "end_time": f"{9+i}:30"
        }
        tasks.append(task)
    return tasks

# 3. Iterate Dates
curr = START_DATE
total_inserted = 0

while curr <= END_DATE:
    date_str = curr.isoformat()
    # Check if date already has bulk data (heuristic: > 50 rows total)
    check = supabase.table("master_schedule").select("id", count="exact").eq("date", date_str).limit(1).execute()
    
    if check.count > 100:
        print(f"Skipping {date_str} (already data present)")
    else:
        print(f"Generating for {date_str}...")
        daily_batch = []
        for mr in users:
            daily_batch.extend(generate_day_schedule(mr, date_str, template_tasks))
        
        # Insert in chunks of 500
        for i in range(0, len(daily_batch), 500):
            try:
                supabase.table("master_schedule").insert(daily_batch[i:i+500]).execute()
            except Exception as e:
                print(f"Error inserting chunk on {date_str}: {e}")
        
        total_inserted += len(daily_batch)

    curr += timedelta(days=1)

print(f"\n✅ Generation Complete! Added {total_inserted} new tasks for full timeline.")
