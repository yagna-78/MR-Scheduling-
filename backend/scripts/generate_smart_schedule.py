
import os
import uuid
import random
import datetime
from datetime import timedelta
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# ---------------- CONFIGURATION ----------------
START_DATE = datetime.date(2026, 1, 1)
END_DATE = datetime.date(2026, 2, 28)

# Doctor/Hospital Directory (Simulated DB of real locations in Ahmedabad)
# In real prod these would come from the 'customers' or 'doctors' table
DOCTOR_DB = [
    {"name": "Dr. Aarav Patel", "locality": "Navrangpura", "lat": 23.0365, "lon": 72.5611, "spec": "Cardio"},
    {"name": "Apollo Hospitals", "locality": "Bhat", "lat": 23.1168, "lon": 72.6369, "spec": "Multi"},
    {"name": "Dr. Sneha Shah", "locality": "Satellite", "lat": 23.0300, "lon": 72.5200, "spec": "Pedia"},
    {"name": "Zydus Hospital", "locality": "Thaltej", "lat": 23.0640, "lon": 72.5080, "spec": "Multi"},
    {"name": "Dr. Rohan Mehta", "locality": "Maninagar", "lat": 22.9960, "lon": 72.6030, "spec": "Ortho"},
    {"name": "Sterling Hospital", "locality": "Gurukul", "lat": 23.0487, "lon": 72.5312, "spec": "Multi"},
    {"name": "Dr. Priya Desai", "locality": "Vastrapur", "lat": 23.0425, "lon": 72.5256, "spec": "Derma"},
    {"name": "Civil Hospital", "locality": "Asarwa", "lat": 23.0532, "lon": 72.6033, "spec": "Govt"},
    {"name": "Dr. Vikram Singh", "locality": "Gota", "lat": 23.1000, "lon": 72.5400, "spec": "GP"},
    {"name": "CIMS Hospital", "locality": "Sola", "lat": 23.0782, "lon": 72.5098, "spec": "Cardiac"},
    {"name": "Dr. Anita Roy", "locality": "Bopal", "lat": 23.0289, "lon": 72.4633, "spec": "Gyno"},
    {"name": "Shelby Hospital", "locality": "SG Highway", "lat": 23.0240, "lon": 72.5080, "spec": "Ortho"},
    {"name": "Dr. Rajesh Kumar", "locality": "Naroda", "lat": 23.0685, "lon": 72.6533, "spec": "GP"},
    {"name": "Lifeline Clinic", "locality": "Chandkheda", "lat": 23.1090, "lon": 72.5850, "spec": "Clinic"},
    {"name": "Dr. Meera Joshi", "locality": "Paldi", "lat": 23.0120, "lon": 72.5610, "spec": "ENT"}
]

# Activity Types with weights
ACTIVITY_TYPES = ["Visit", "Visit", "Visit", "Call", "Camp", "Meeting"]

# ---------------- LOGIC ----------------

print(f"--- 🏥 Generating INDUSTRY-GRADE Schedule ({START_DATE} to {END_DATE}) ---")

# 1. Fetch Users
users = supabase.table("users").select("*").execute().data
print(f"Found {len(users)} MRs to schedule.")

# 2. Advanced Scheduling Algorithm
# Rules:
# - No duplicates visits to same doc on same day.
# - Reasonable travel path (clustered by locality in a day if possible).
# - Variable start/end times.
# - Random cancellations or reschedules.

def generate_smart_daily_schedule(mr, date_obj):
    # Skip Sundays
    if date_obj.weekday() == 6:
        return []

    # Decide today's zone/focus area for the MR
    # (In real life MRs visit specific areas on specific days)
    # Pick a random "anchor" location from DOCTOR_DB to focus on today
    anchor = random.choice(DOCTOR_DB)
    
    # Filter doctors near the anchor (within ~5-8km approx difference in lat/lon) to minimize travel
    # Simple lat/lon diff here (0.05 is roughly 5km)
    nearby_docs = [
        d for d in DOCTOR_DB 
        if abs(d['lat'] - anchor['lat']) < 0.08 and abs(d['lon'] - anchor['lon']) < 0.08
    ]
    
    # If not enough nearby, augment with randoms
    if len(nearby_docs) < 8:
        nearby_docs.extend(random.sample(DOCTOR_DB, min(len(DOCTOR_DB), 8 - len(nearby_docs))))
        
    # Shuffle and pick workload (6-9 visits for realism, user requested max 8-9)
    workload_count = random.randint(6, 9)
    selected_docs = random.sample(nearby_docs, min(len(nearby_docs), workload_count))
    
    daily_tasks = []
    
    start_hour = 9 + random.choice([0, 0.5]) # Starts 9 or 9:30
    current_time_min = start_hour * 60 
    
    for i, doc in enumerate(selected_docs):
        # Calculate duration
        duration = random.choice([15, 20, 30, 45]) # mins
        
        # Format times
        start_h = int(current_time_min // 60)
        start_m = int(current_time_min % 60)
        end_time_min = current_time_min + duration
        end_h = int(end_time_min // 60)
        end_m = int(end_time_min % 60)
        
        time_str_start = f"{start_h:02d}:{start_m:02d}"
        time_str_end = f"{end_h:02d}:{end_m:02d}"
        
        # Determine status based on date
        is_past = date_obj < datetime.date.today()
        if is_past:
             status_roll = random.random()
             if status_roll > 0.9: status = "Cancelled"
             elif status_roll > 0.8: status = "Pending" # Forgot to update
             else: status = "Completed"
        elif date_obj == datetime.date.today():
             # Today: Mix of Done and Pending depending on time of day (simulated)
             status = "Pending"
        else:
             status = "Planned"
             
        # Jitter location slightly so markers don't overlap perfectly if multiple MRs visit same hospital
        loc_jitter = 0.0005
        task_distance = round(random.uniform(1.0, 15.0), 2)
        # Calculate travel time (Avg city speed ~25 km/h)
        speed_kmh = 25
        travel_time = int((task_distance / speed_kmh) * 60)

        task = {
            "mr_id": mr['mr_id'],
            "team": mr.get('team', 'General'),
            "zone": mr.get('zone', 'North'),
            "date": date_obj.isoformat(),
            "activity_id": f"SMART_{mr['mr_id']}_{uuid.uuid4().hex[:8]}",
            "status": status,
            "customer_id": f"DOC_{doc['name'][:3].upper()}_{random.randint(100,999)}",
            "customer_name": doc['name'],
            "contact_person": "Reception" if "Hospital" in doc['name'] else doc['name'],
            "customer_status": "Key" if i < 2 else "Regular", # Kept as standard MR metric
            "activity_type": random.choice(ACTIVITY_TYPES),
            "locality": doc['locality'],
            "latitude": doc['lat'] + random.uniform(-loc_jitter, loc_jitter),
            "longitude": doc['lon'] + random.uniform(-loc_jitter, loc_jitter),
            "start_time": time_str_start,
            "end_time": time_str_end,
            "duration_min": duration,
            "distance_km": task_distance,
            "travel_duration_min": travel_time
        }
        
        daily_tasks.append(task)
        
        # Travel time to next (15-45 mins)
        current_time_min = end_time_min + travel_time + random.randint(5, 10) # Travel + Buffer

    return daily_tasks


# 3. Execution (Clear & Re-seed)
# Note: Deleting existing "SMART" or "FULL" tags to avoid overlap if running multiple times? 
# For safety, we just append. Or we could clear range.
# Let's simple insert.

curr = START_DATE
total_tasks = 0

batch_buffer = []

while curr <= END_DATE:
    # print(f"Processing {curr}...")
    for mr in users:
        tasks = generate_smart_daily_schedule(mr, curr)
        batch_buffer.extend(tasks)
        
    # Flush every 5 days (~500-800 rows)
    if len(batch_buffer) > 500:
        try:
             # Delete existing data in this date range to prevent doubling up (15-18 tasks)
             dates_in_batch = list(set(t['date'] for t in batch_buffer))
             # Naive delete: Delete all for these dates before insert
             # In prod this is slow, but fine for script
             for d in dates_in_batch:
                 supabase.table("master_schedule").delete().eq("date", d).execute()

             # Insert
             supabase.table("master_schedule").insert(batch_buffer).execute()
             print(f"✅ Flushed {len(batch_buffer)} tasks up to {curr}")
             total_tasks += len(batch_buffer)
             batch_buffer = []
        except Exception as e:
            print(f"Error inserting: {e}")
            batch_buffer = [] # Clear to prevent stuck loop

    curr += timedelta(days=1)

# Final Flush
if batch_buffer:
    supabase.table("master_schedule").insert(batch_buffer).execute()
    total_tasks += len(batch_buffer)

print(f"\n🎉 DONE. Generated {total_tasks} industry-grade schedule entries.")
