
import os
import uuid
import random
import datetime
from datetime import timedelta
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Configuration
TARGET_DATE = datetime.date.today().isoformat() # "2026-01-29" in your context presumably, or actual today
# Note: User's system time is 2026-01-29.
TARGET_DATE = "2026-01-29" 
REPORT_MONTH_START = "2026-01-01"

print(f"--- 🚀 Starting Bulk Data Generation for {TARGET_DATE} ---")

# 1. Fetch all MRs
users = supabase.table("users").select("*").execute().data
print(f"Found {len(users)} MRs.")

# 2. Fetch a pool of sample customers/tasks to clone from
# We need 'good' data to copy structure from.
template_tasks = supabase.table("master_schedule").select("*").limit(50).execute().data
if not template_tasks:
    print("❌ No template data found in master_schedule to clone from!")
    exit()

def generate_schedule_for_mr(mr, tasks_pool):
    print(f"Generating Schedule for {mr['mr_id']}...")
    new_tasks = []
    
    # Create 8 tasks for the day
    # Use MR's starting location or default
    start_lat = mr.get('starting_latitude') or 23.0
    start_lon = mr.get('starting_longitude') or 72.5
    
    for i in range(8):
        # Pick a random template
        template = random.choice(tasks_pool)
        
        # Jitter location slightly (approx 1-5km)
        lat_offset = random.uniform(-0.05, 0.05)
        lon_offset = random.uniform(-0.05, 0.05)
        
        task = {
            "mr_id": mr['mr_id'],
            "team": mr.get('team', 'General'),
            "zone": mr.get('zone', 'North'),
            "date": TARGET_DATE,
            "activity_id": f"GEN_{mr['mr_id']}_{uuid.uuid4().hex[:6]}",
            "status": "Pending" if i > 3 else "Done", # Mix statuses
            
            # Copy generic fields from template or defaults
            "customer_id": template.get('customer_id', f"CUST_{i}"),
            "customer_name": template.get('customer_name', f"Doctor {i}"),
            "customer_status": random.choice(["Key", "Target", "Regular"]),
            "activity_type": random.choice(["Visit", "Call", "Camp"]),
            "locality": template.get('locality', "City Center"),
            "duration_min": 30,
            
            # Location
            "latitude": start_lat + lat_offset,
            "longitude": start_lon + lon_offset,
            
            # Time (simple sequential)
            "start_time": f"{10+i}:00",
            "end_time": f"{10+i}:30"
        }
        new_tasks.append(task)
        
    return new_tasks

# 3. Main Loop
total_inserted = 0
for mr in users:
    mr_id = mr['mr_id']
    
    # Check if they already have data for today
    existing = supabase.table("master_schedule").select("id", count="exact").eq("mr_id", mr_id).eq("date", TARGET_DATE).execute()
    if existing.count > 0:
        print(f"⏩ Skipping {mr_id} (already has {existing.count} tasks today)")
        continue

    # Generate
    tasks = generate_schedule_for_mr(mr, template_tasks)
    
    # Insert
    if tasks:
        try:
            supabase.table("master_schedule").insert(tasks).execute()
            total_inserted += len(tasks)
        except Exception as e:
            print(f"❌ Failed to insert for {mr_id}: {e}")

print(f"\n✅ Generation Complete! Added {total_inserted} new tasks across all MRs.")
