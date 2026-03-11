from fastapi import APIRouter, HTTPException
from typing import Optional, Dict, Any
import pandas as pd
from app.services.supabase_db import load_data, save_data
from app.services.logic import run_schedule_logic_for_single_mr
import traceback

router = APIRouter(prefix="/admin", tags=["Admin"])

# --------------------------------------------------
# GET /admin/mrs - List of MRs for dropdown (ADMIN ONLY)
# --------------------------------------------------
@router.get("/mrs")
def get_all_mrs():
    print("[GET MRS] STARTED - Fetching MR list for dropdown")

    try:
        df = load_data("User_Master")
        print(f"[GET MRS] User_Master loaded - {len(df)} rows")

        if df.empty:
            print("[GET MRS] Sheet is empty - returning []")
            return []

        # Find mr_id column (flexible)
        mr_id_col = None
        for col in df.columns:
            if 'mr_id' in col.lower() or 'mrid' in col.lower():
                mr_id_col = col
                break

        if not mr_id_col:
            print("[GET MRS] No mr_id column found - returning []")
            return []

        # Clean mr_id
        df[mr_id_col] = df[mr_id_col].astype(str).str.strip()
        df = df[df[mr_id_col].notna() & (df[mr_id_col] != '') & (df[mr_id_col] != 'nan')]

        # Build display name
        name = ''
        if 'name' in df.columns:
            name = df['name'].fillna('')
        elif 'first_name' in df.columns or 'last_name' in df.columns:
            first = df.get('first_name', pd.Series([''] * len(df))).fillna('')
            last = df.get('last_name', pd.Series([''] * len(df))).fillna('')
            name = (first + ' ' + last).str.strip()

        # Fallback to mr_id if name empty
        df['display_name'] = name.where(name != '', df[mr_id_col])

        result = df[[mr_id_col, 'display_name']].rename(columns={mr_id_col: 'mr_id'}).to_dict(orient="records")

        print(f"[GET MRS] SUCCESS - Returning {len(result)} MRs")
        return result

    except Exception as e:
        print(f"[GET MRS] CRASHED - {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        return []  # Never crash - just return empty list


# --------------------------------------------------
# POST /admin/generate-schedule
# --------------------------------------------------
@router.post("/generate-schedule")
def generate_schedule():
    try:
        users = load_data("User_Master")
        contacts = load_data("Contacts")
        activities = load_data("Activities")
        
        all_schedules = []
        current_date = pd.Timestamp.now().normalize()
        
        for _, user in users.iterrows():
            # Flexible column name for MR ID
            mr_id = user.get('mr_id') or user.get('MR_ID') or user.get('Mr_id')
            if not mr_id:
                continue
            sched = run_schedule_logic_for_single_mr(
                mr_id, users, contacts, activities, current_date
            )
            if not sched.empty:
                all_schedules.append(sched)
        
        if all_schedules:
            final = pd.concat(all_schedules, ignore_index=True)
            final['date'] = final['date'].astype(str)
            save_data(final, "Master_Schedule")
            return {"message": f"Schedule generated for {len(all_schedules)} MRs!"}
        
        return {"message": "No schedule generated"}
    except Exception as e:
        print(f"[GENERATE SCHEDULE ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------
# GET /admin/table/{table_name}
# --------------------------------------------------
@router.get("/table/{table_name}")
def get_table_data(table_name: str, page: int = 1, page_size: int = 20):
    """
    Fetch paginated data from any table (User_Master, Contacts, etc.).
    Returns { "data": [...], "total": N }
    """
    try:
        df = load_data(table_name)
        total = len(df)
        
        # Pagination logic
        if page < 1:
            page = 1
        start = (page - 1) * page_size
        end = start + page_size
        
        # Safe slicing
        paginated = df.iloc[start:end]
        
        # Fill NaNs to avoid JSON serialization errors
        paginated = paginated.fillna('') 
        
        # Optional debug
        print(f"[ADMIN TABLE] {table_name} | page={page}, size={page_size}, returned={len(paginated)}, total={total}")
        
        return {
            "data": paginated.to_dict(orient="records"),
            "total": total,
            "page": page,
            "page_size": page_size
        }
    except Exception as e:
        print(f"[ADMIN TABLE ERROR] {table_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))