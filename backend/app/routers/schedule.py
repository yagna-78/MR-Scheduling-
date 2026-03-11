
# app/routers/schedule.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime
import traceback

from app.services.supabase_db import load_data, save_data, supabase

router = APIRouter(prefix="/schedule", tags=["Schedule"])


class StatusUpdate(BaseModel):
    activity_id: str
    status: str  # Expected: "Done", "Pending", etc.


@router.get("/daily/{mr_id}/{date}", response_model=List[Dict[str, Any]])
def get_daily_schedule(mr_id: str, date: str):
    """
    Fetch all scheduled activities for a specific MR on a given date.
    Uses efficient DB-side filtering.
    """
    print(f"[SCHEDULE GET] Requested for mr_id='{mr_id}', date='{date}'")

    try:
        if not supabase:
             raise HTTPException(status_code=500, detail="Database connection failed")

        # 1. Efficient DB Filter for Master Schedule
        query = supabase.table("master_schedule").select("*")
        
        # Handle Admin requesting ALL data for a date
        if mr_id.lower() != 'admin':
            query = query.eq("mr_id", mr_id)
            
        query = query.eq("date", date)
        
        response = query.execute()
        data_master = response.data or []

        # 2. Fetch from Activities (Past/Completed data)
        query_act = supabase.table("activities").select("*")
        if mr_id.lower() != 'admin':
            query_act = query_act.eq("mr_id", mr_id)
        query_act = query_act.eq("date", date)
        
        response_act = query_act.execute()
        data_activities = response_act.data or []

        # Process Activities Data
        if data_activities:
            # Rename activity_status to status
            for item in data_activities:
                if 'activity_status' in item:
                    item['status'] = item.pop('activity_status')
        
        # Combine Data (Activities take precedence if duplicates exist)
        # We convert to DF immediately to handle deduplication easily
        
        # --- ENRICHMENT STEP ---
        # Activities table lacks 'travel_duration_min', 'distance_km', 'suggested_talking_points'
        # We fetch these from master_schedule if available.
        if data_activities and data_master:
            master_map = {row['activity_id']: row for row in data_master}
            enrich_keys = ['travel_duration_min', 'distance_km', 'suggested_talking_points']
            
            for row in data_activities:
                if 'activity_id' in row and row['activity_id'] in master_map:
                    m_row = master_map[row['activity_id']]
                    for key in enrich_keys:
                        # Only fill if missing or empty in activity row
                        if key not in row or row[key] is None or row[key] == "":
                             if key in m_row:
                                 row[key] = m_row[key]
        # -----------------------

        df_master = pd.DataFrame(data_master)
        df_activities = pd.DataFrame(data_activities)
        
        dfs = []
        if not df_activities.empty:
            dfs.append(df_activities)
        if not df_master.empty:
            dfs.append(df_master)
            
        if not dfs:
            print("[SCHEDULE GET] No tasks found in DB query.")
            return []
            
        df = pd.concat(dfs, ignore_index=True)

        print(f"[SCHEDULE GET] DB returned {len(df)} rows (merged)")

        # Deduplicate to prevent multiple cards for same activity
        # Priority is kept by order of concatenation (Activities first)
        if 'activity_id' in df.columns:
            df = df.drop_duplicates(subset=['activity_id'], keep='first')

        # --- Safe Contacts merge (phone, segment, customer_name) ---
        contacts = load_data("Contacts")
        if not contacts.empty:
            
            phone_col = None
            segment_col = None
            name_col = None

            # Phone column search
            phone_keywords = ['phone', 'mobile', 'contact', 'tel', 'cell', 'number']
            for col in contacts.columns:
                lower = col.lower().strip()
                if any(kw in lower for kw in phone_keywords):
                    phone_col = col
                    break

            # Segment column search
            segment_keywords = ['segment', 'category', 'type', 'group', 'class']
            for col in contacts.columns:
                lower = col.lower().strip()
                if any(kw in lower for kw in segment_keywords):
                    segment_col = col
                    break
            
            # Name column search (usually contact_name)
            name_keywords = ['contact_name', 'name', 'customer_name']
            for col in contacts.columns:
                 lower = col.lower().strip()
                 if any(kw in lower for kw in name_keywords):
                     name_col = col
                     break

            # Apply merge only if columns exist and customer_id is present
            if 'customer_id' in df.columns and 'contact_id' in contacts.columns: 
                # Normalize Contact DF keys
                contacts.columns = [c.lower() for c in contacts.columns]
                
                # Create helper lookup dicts
                contacts_idx = contacts.set_index('contact_id')
                
                if phone_col and phone_col.lower() in contacts.columns:
                    try:
                        contacts_dict = contacts_idx[phone_col.lower()].to_dict()
                        df['phone'] = df['customer_id'].map(contacts_dict).fillna('N/A')
                    except Exception as e:
                        df['phone'] = 'N/A'
                else:
                    df['phone'] = 'N/A'

                if segment_col and segment_col.lower() in contacts.columns:
                    try:
                        contacts_dict = contacts_idx[segment_col.lower()].to_dict()
                        df['segment'] = df['customer_id'].map(contacts_dict).fillna('General')
                    except Exception as e:
                        df['segment'] = 'General'
                else:
                    df['segment'] = 'General'
                    
                # Fix missing customer_name (from activities)
                if 'customer_name' not in df.columns:
                    df['customer_name'] = None # Initialize if completely missing
                
                if name_col and name_col.lower() in contacts.columns:
                    try:
                        name_dict = contacts_idx[name_col.lower()].to_dict()
                        # Only fill if missing (NaN or None)
                        df['customer_name'] = df['customer_name'].fillna(df['customer_id'].map(name_dict))
                    except Exception as e:
                        print(f"[WARNING] Name merge failed: {e}")
                        
            else:
                 df['phone'] = 'N/A'
                 df['segment'] = 'General'
        else:
            df['phone'] = 'N/A'
            df['segment'] = 'General'
            
        # Nan handling for JSON
        df = df.fillna('')
        
        # Determine Lat/Long if missing (e.g. from activities if they don't capture it?)
        # Actually activities has lat/long. 
        # But if missing, could fallback to contacts lat/long?
        # For now, trust the data.

        # Return result
        return df.to_dict(orient="records")

    except Exception as e:
        print(f"[SCHEDULE GET ERROR] {type(e).__name__}: {str(e)}")
        # Dump trace for debugging
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.put("/status")
def update_status(update: StatusUpdate):
    """
    Update task status (used by drag & drop + buttons).
    Uses efficient DB-side update.
    """
    print(f"[STATUS PUT] START - activity_id='{update.activity_id}', status='{update.status}'")

    try:
        # Direct DB Update
        response = supabase.table("master_schedule").update({"status": update.status}).eq("activity_id", update.activity_id).execute()
        
        if not response.data:
             raise HTTPException(status_code=404, detail=f"Activity ID '{update.activity_id}' not found or update failed")

        return {"message": f"Task {update.activity_id} updated to {update.status}"}

    except Exception as e:
        print(f"[STATUS PUT CRASH] {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")