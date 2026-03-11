# app/routers/auth.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import jwt
from datetime import datetime, timedelta
from app.services.supabase_db import load_data
import pandas as pd

router = APIRouter(prefix="/auth", tags=["Auth"])

SECRET_KEY = "yash-dcom-2026-ahmedabad"

class LoginRequest(BaseModel):
    username: str
    password: str

@router.post("/login")
def login(request: LoginRequest):
    username = request.username.strip()
    password = request.password.strip()

    # ADMIN check
    if username == "ADMIN" and password == "ADMIN":
        token = jwt.encode({
            "user_id": "ADMIN",
            "role": "admin",
            "name": "Administrator",
            "exp": datetime.utcnow() + timedelta(hours=24)
        }, SECRET_KEY, algorithm="HS256")
        return {"token": token, "role": "admin", "name": "Administrator"}

    # Normal MR check
    try:
        users = load_data("User_Master")
        if users.empty:
            print("[AUTH] User_Master sheet is empty")
            raise HTTPException(status_code=401, detail="User database is empty")

        # Flexible MR_ID column search
        mr_id_col = None
        for col in users.columns:
            if 'mr_id' in col.lower() or 'mrid' in col.lower() or 'user_id' in col.lower():
                mr_id_col = col
                break
        
        if not mr_id_col:
            print("[AUTH] Critical: No mr_id column found in User_Master")
            raise HTTPException(status_code=500, detail="Server misconfiguration: User ID column not found")

        # Normalize column for comparison
        users[mr_id_col] = users[mr_id_col].astype(str).str.strip()
        
        # Check credentials (password assumed to be same as username for now)
        user = users[users[mr_id_col] == username]

        if user.empty:
            print(f"[AUTH] Login failed: User '{username}' not found")
            raise HTTPException(status_code=401, detail="Invalid credentials")
            
        if password != username:
            print(f"[AUTH] Login failed: Password mismatch for '{username}'")
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Flexible Name construction
        if 'name' in user.columns:
            full_name = user['name'].iloc[0]
        elif 'first_name' in user.columns or 'last_name' in user.columns:
            first = user.get('first_name', pd.Series([''])).iloc[0]
            last = user.get('last_name', pd.Series([''])).iloc[0]
            full_name = f"{first} {last}".strip()
        else:
            full_name = username

        # Generate Token
        token = jwt.encode({
            "user_id": username,
            "role": "mr",
            "name": str(full_name) or username,
            "exp": datetime.utcnow() + timedelta(hours=24)
        }, SECRET_KEY, algorithm="HS256")
        
        print(f"[AUTH] Success: {username} logged in")
        return {"token": token, "role": "mr", "name": str(full_name) or username}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[AUTH ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail="Authentication system error")