
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

# Create the app instance
app = FastAPI(title="Dcom MR Scheduling", version="2.0")

# Allow React frontend (Vercel + Localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for debugging, restrict in production later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
from app.routers import auth, schedule, reports, admin

app.include_router(auth.router)
app.include_router(schedule.router)
app.include_router(reports.router)
app.include_router(admin.router)

# Root endpoint check
@app.get("/")
def home():
    return {"message": "Dcom API Running - Backend is LIVE"}

# Test endpoint
@app.get("/test")
def test():
    return {"message": "CORS and API are working correctly!"}