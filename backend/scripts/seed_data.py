
import os
import uuid
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

source_mr = "MR_W2_1"
target_mr = "MR_W1_1"
target_date = "2026-01-29"

print(f"--- Cloning tasks from {source_mr} to {target_mr} for {target_date} ---")

# 1. Get tasks
res = supabase.table("master_schedule").select("*").eq("mr_id", source_mr).eq("date", target_date).execute()
tasks = res.data

if not tasks:
    print("❌ Source MR has no tasks either!")
    exit()

print(f"Found {len(tasks)} tasks to clone.")

new_tasks = []
for task in tasks:
    new_task = task.copy()
    del new_task['id'] # Let DB generate ID
    del new_task['created_at'] # Let DB generate timestamp
    new_task['mr_id'] = target_mr
    new_task['activity_id'] = f"{task['activity_id']}_CLONE_{uuid.uuid4().hex[:4]}"
    new_tasks.append(new_task)

# 2. Insert
try:
    data, count = supabase.table("master_schedule").insert(new_tasks).execute()
    print(f"✅ Successfully inserted {len(new_tasks)} tasks for {target_mr}")
except Exception as e:
    print(f"❌ Error inserting: {e}")
