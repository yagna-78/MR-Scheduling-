# MR Portal 🏥

> A full-stack Medical Representative scheduling and management system utilizing hybrid AI algorithms and real-time routing to optimize daily visit plans.

## 📖 Project Overview

### The Problem

Managing and prioritizing daily schedules for medical representatives (MRs) across vast geographical territories is incredibly complex. Manual planning often leads to sub-optimal routing, neglected high-value contacts, and inefficient use of time, ultimately hurting sales and engagement.

### Why It Was Built

MR Portal was built to automate and optimize this complex scheduling process. By blending rule-based business logic with machine learning (XGBoost) and real-time road distance APIs (OSRM), the platform provides a deterministic, data-driven approach to medical sales management.

### Real-World Use Case

A pharmaceutical or medical device company assigns zones to its Medical Representatives. Using MR Portal, an admin can generate a full month's schedule for all MRs with a single click. The system automatically prioritizes the most valuable doctors, groups visits geographically, considers real-world traffic and road distances, and optimizes the MR's workday within standard business hours.

## ✨ Key Features

- **Hybrid AI Scheduling Algorithm:** Combines rule-based business scoring with an XGBoost machine learning model to prioritize contacts based on engagement status, past visits, and spatial patterns.
- **Smart Route Optimization:** Integrates with the Open Source Routing Machine (OSRM) to calculate realistic road-distance and drive time.
- **Interactive Map Dashboard:** Real-time visual tracking of planned, completed, and cancelled visits using a React-Leaflet map interface.
- **Comprehensive Reporting Engine:** Generates detailed compliance, travel, and customer behavior reports with one-click CSV export capabilities.
- **Role-Based Access Control:** Distinct workflows featuring tailored dashboards for MRs (execution) and Admins (schedule generation, analytics, and data inspection).

## 🏗️ System Architecture

The overarching system architecture dictates a modern, decoupled client-server model:

- **Frontend (Client):** A dynamic React SPA built with Vite and Tailwind CSS. It communicates securely with the backend via REST APIs and renders interactive map views using Leaflet.
- **Backend (API Layer):** A high-performance FastAPI server running on Uvicorn. It handles multi-router endpoints, JWT authentication, and hosts the complex scheduling AI logic.
- **Database (Data Layer):** Supabase (PostgreSQL) acts as the primary data store. Strict Row-Level Security policies ensure safe and scalable data isolation across central tables (`users`, `contacts`, `activities`, and `master_schedule`).
- **External Dependencies:** XGBoost handles predictive ML scoring and OSRM API provides spatial routing logic.

## 💻 Technical Stack

**Programming Languages:** Python, JavaScript, SQL

**Frontend:**

- React 19 / Vite
- Tailwind CSS (Styling)
- React Router (Navigation)
- React-Leaflet & Leaflet (Maps)
- dnd-kit (Drag-and-Drop Kanban)

**Backend:**

- Python 3.9+
- FastAPI & Uvicorn (REST API generation)
- Supabase (PostgreSQL Database)
- Pandas (Data processing & matrices)
- PyJWT (Authentication)

**Machine Learning & Infrastructure:**

- XGBoost & Scikit-learn (Machine Learning scoring)
- OSRM API (Road-distance Routing)

## 🧠 AI / ML Details

The intelligence of the platform is driven by a highly orchestrated **9-step scheduling pipeline** that relies on a hybrid scoring system:

- **Models Used:** XGBoost regression model trained dynamically.
- **Feature Engineering:** Features include LabelEncoded business segments, engagement statuses, historical referrals, lifetime visit counts, 90-day visit frequency, and geographic clustering vectors (Latitude/Longitude).
- **Training Pipeline:** A deterministic rule-based score (awarding points for neglected contacts, strategic referral segments, etc.) is quantified first. The XGBoost model is then trained on-the-fly utilizing these rules as labels, allowing it to seamlessly learn non-linear interactions and spatial proximity phenomena that flat rules might ignore.
- **Evaluation & Optimization:** The final priority score is a **50/50 fusion** of the rule-based score and the XGBoost prediction. Top-tier targets are sequenced using OSRM to validate that physical travel times strictly align with standard 10:00 AM to 7:00 PM operational hours.

## 📂 Project Structure

```text
mr-project/
├── backend/
│   ├── app/
│   │   ├── routers/        # API Endpoints (auth, schedule, reports, admin)
│   │   ├── services/       # Core ML scheduling logic, routing, DB services
│   │   └── main.py         # FastAPI application entry point
│   ├── scripts/            # Data seeders, ML simulators, migration tools
│   ├── requirements.txt    # Python dependencies
│   └── supabase_schema.sql # Database DDL
├── frontend/
│   ├── src/
│   │   ├── components/     # UI Components (Map, Kanban, Sidebar)
│   │   ├── pages/          # Views (Dashboard, Reports, Admin)
│   │   └── lib/            # API network integration
│   ├── package.json        # Node dependencies
│   └── tailwind.config.js  # UI Framework configuration
└── README.md
```

## 🚀 Installation Guide

### Prerequisites

- Node.js (v18+)
- Python (v3.9+)
- Supabase Account

### 1. Database Setup

Run the SQL script provided in `backend/supabase_schema.sql` within your Supabase SQL Editor to provision the tables and necessary Row-Level Security policies.

### 2. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup Environment Variables
echo SUPABASE_URL=your_supabase_url > .env
echo SUPABASE_KEY=your_supabase_anon_key >> .env

# Start the FastAPI server
uvicorn app.main:app --reload
```

The backend will be automatically available at `http://localhost:8000`.

### 3. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install NPM dependencies
npm install

# Start the Vite development server
npm run dev
```

The frontend will initialize and be available at `http://localhost:5173`.

## 💡 Usage

### Generating Schedules (Admin Workflow)

1. Log in via the interface using Admin credentials (`Username: ADMIN`, `Password: ADMIN`).
2. Navigate to the **Admin Console** tab.
3. Click on **Generate Schedule**. The FastAPI backend will immediately analyze all zone contacts, score them via XGBoost arrays, route transit coordinates via OSRM, and populate exactly targeted daily itineraries for the upcoming 30 days.
4. Review generated outputs in the `Master_Schedule` Dataset Inspector table.

![Schedule Generation Process](schedule%20generation.png)

### Daily MR Workflow

1. Log in via your assigned MR credentials.
2. View the centralized **Dashboard** to see the uniquely optimized route mapped dynamically on the Leaflet interface.
3. Manage statuses effectively using the drag-and-drop Kanban board (Planned → Completed columns).
4. Analyze personal historical adherence and travel metrics natively through the **Reports** tab.

### Example API Request

Easily retrieve the granular schedule for a specific MR programmatically:

```bash
curl -X GET "http://localhost:8000/schedule/daily/MR_W1_1/2023-10-25" \
     -H "Authorization: Bearer <YOUR_JWT_TOKEN>"
```

### Application Outputs (Placeholder)

*(Insert screenshots highlighting the Map Dashboard, Kanban Task Board, Reports View, and Admin Dataset Inspector here to provide visual context).*

---
*Architected for scale. Built for efficiency.*
