
# app/routers/reports.py
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime, timedelta
import calendar
from app.services.supabase_db import supabase

router = APIRouter(prefix="/reports", tags=["Reports"])


@router.get("/debug-health")
def debug_reports_health():
    """Check if reports router can access DB"""
    try:
        res = supabase.table("master_schedule").select("count", count="exact").limit(1).execute()
        return {"status": "ok", "total_rows": res.count, "timestamp": datetime.now().isoformat()}
    except Exception as e:
         return {"status": "error", "detail": str(e)}

def get_db_data(table="master_schedule", mr_id=None, start_date=None, end_date=None, year=None, month=None):
    """
    Helper to fetch data efficiently from DB.
    """
    if not supabase:
        raise HTTPException(status_code=500, detail="Database connection failed")

    query = supabase.table(table).select("*")
    
    # 1. MR Filter
    if mr_id and mr_id.lower() != 'admin':
        # Robust filtering: strip whitespace and use case-insensitive match
        clean_mr_id = mr_id.strip()
        if clean_mr_id:
            # We use distinct logic to handle potential casing issues
            query = query.ilike("mr_id", clean_mr_id) 

    # 2. Date Range
    if start_date and end_date:
        query = query.gte("date", start_date.strip()).lte("date", end_date.strip())
    
    # 3. Month/Year Logic
    if year and month:
        # Calculate start and end of month
        s_date = f"{year}-{month:02d}-01"
        _, last_day = calendar.monthrange(year, month)
        e_date = f"{year}-{month:02d}-{last_day}"
        query = query.gte("date", s_date).lte("date", e_date)

    response = query.execute()
    data = response.data
    
    if not data:
        return pd.DataFrame() # Empty DF
        
    return pd.DataFrame(data)


@router.get("/activity")
def get_activity_report(start_date: str, end_date: str, mr_id: Optional[str] = None):
    print(f"[ACTIVITY REPORT] START - start={start_date}, end={end_date}, mr_id={mr_id or 'all'}")

    try:
        # Efficient DB Fetch
        df = get_db_data(mr_id=mr_id, start_date=start_date, end_date=end_date)
        
        if df.empty:
            return {"data": [], "total_activities": 0, "completed_activities": 0}

        # Deduplicate
        if 'activity_id' in df.columns:
            df = df.drop_duplicates(subset=['activity_id'])
            
        total_activities = len(df)
        
        # Count Completed
        if 'status' in df.columns:
             completed_mask = df['status'].astype(str).str.strip().str.lower().isin(['done', 'completed'])
             completed_activities = completed_mask.sum()
        else:
             completed_activities = 0

        # Group by Date
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Aggregate
        daily = df.groupby(df['date'].dt.strftime('%Y-%m-%d')).agg(
            activity_count=('activity_id', 'count'),
            completed_activity_count=('status', lambda x: x.astype(str).str.strip().str.lower().isin(['done', 'completed']).sum())
        ).reset_index()
        
        return {
            "data": daily.rename(columns={'date': 'date'}).to_dict(orient="records"),
            "total_activities": int(total_activities),
            "completed_activities": int(completed_activities)
        }

    except Exception as e:
        print(f"[ACTIVITY ERROR] {str(e)}")
        # import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compliance", response_model=List[Dict[str, Any]])
def get_compliance_report(month: int, year: int, mr_id: Optional[str] = None):
    try:
        df = get_db_data(mr_id=mr_id, year=year, month=month)
        if df.empty:
            return []

        # Group by Customer Name to see how many days they were visited
        # Compliance logic: usually "Visited / Target Visit Frequency"
        # Current logic seems to be just listing customers visited in that month.
        
        visits = df.groupby('customer_name')['date'].nunique().reset_index(name='visit_count')
        
        # visits['sr_no'] = range(1, len(visits) + 1) # Removed
        visits['monthly_range'] = f"{calendar.month_name[month]} {year}"
        # If logical 'compliance' needs target data, that's missing. 
        # For now we return what the frontend expects: visits count. 

        return visits.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/customer-behaviour", response_model=List[Dict[str, Any]])
def get_customer_behaviour_report(month: int, year: int, mr_id: Optional[str] = None):
    try:
        df = get_db_data(mr_id=mr_id, year=year, month=month)
        if df.empty:
            return []

        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.day_name()
        
        # Pivot: Rows=Customer, Cols=DayOfWeek, Value=Count
        pivot = pd.pivot_table(
            df,
            index='customer_name',
            columns='day_of_week',
            values='activity_id', # or just count rows
            aggfunc='count',
            fill_value=0
        )
        
        # Ensure all days exist
        days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        for day in days:
            if day not in pivot.columns:
                pivot[day] = 0
                
        # Reorder
        pivot = pivot[days]
        pivot['total_activities'] = pivot.sum(axis=1)
        
        pivot.reset_index(inplace=True)
        # pivot['sr_no'] = range(1, len(pivot) + 1) # Removed
        
        return pivot.to_dict(orient="records")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/travel", response_model=List[Dict[str, Any]])
def get_travel_report(month: int, year: int, mr_id: Optional[str] = None):
    try:
        df = get_db_data(mr_id=mr_id, year=year, month=month)
        if df.empty:
            return []
            
        # Ensure distance is numeric
        if 'distance_km' not in df.columns:
             df['distance_km'] = 0
        df['distance_km'] = pd.to_numeric(df['distance_km'], errors='coerce').fillna(0)
        
        # Group by Date
        grouped = df.groupby('date')['distance_km'].sum().reset_index()
        
        grouped['date'] = pd.to_datetime(grouped['date']).dt.strftime('%Y-%m-%d')
        grouped['travel_distance_km'] = grouped['distance_km'].round(2)
        # grouped['sr_no'] = range(1, len(grouped) + 1) # Removed as per user request
        
        return grouped[['date', 'travel_distance_km']].to_dict(orient="records")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))