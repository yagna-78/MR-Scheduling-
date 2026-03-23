
---

# MR Portal

**A scheduling platform for medical representatives — built on hybrid AI and real-time routing.**

---

## The Problem It Solves

Planning daily visits for medical representatives across large territories is a genuinely hard problem. When done manually, it leads to poor routing, overlooked high-value contacts, and wasted hours — all of which quietly erode sales performance and doctor relationships.

MR Portal automates that entire process. Instead of MRs guessing at their best route each morning, the system figures it out for them — using a combination of business rules, machine learning, and real-world road distances.

---

## How It Works in Practice

A pharma or medical device company assigns territories to its MRs. An admin logs in, hits **Generate Schedule**, and within moments the system has produced a full month of daily itineraries for every MR in the system. Each itinerary prioritizes the right doctors, groups them geographically, and fits the visits into a realistic workday — accounting for actual drive times, not straight-line distances.

---

## What It Can Do

- **Smarter prioritization** — A hybrid scoring model (rule-based logic + XGBoost) ranks contacts by engagement history, past visit patterns, referral value, and proximity.
- **Realistic routing** — Integrates with OSRM to use actual road distances and estimated drive times, not crow-flies guesses.
- **Live map dashboard** — MRs see their day plotted on an interactive map, with visit statuses updated in real time.
- **Drag-and-drop Kanban** — Moving a visit from *Planned* to *Completed* is as simple as dragging a card.
- **One-click reports** — Compliance summaries, travel metrics, and customer engagement data, all exportable to CSV.
- **Role-based access** — Admins get scheduling controls and analytics; MRs get a focused daily execution view.

---

## Under the Hood

### Architecture

The system is a clean client-server split:

- **Frontend** — React SPA (Vite + Tailwind CSS), with Leaflet powering the map and dnd-kit handling the Kanban board.
- **Backend** — FastAPI on Uvicorn, managing auth (JWT), API routing, and the scheduling engine.
- **Database** — Supabase (PostgreSQL) with Row-Level Security keeping data properly isolated per user and role.
- **External services** — XGBoost for ML scoring, OSRM for spatial routing.

### The Scheduling Algorithm

The core of the platform is a **9-step pipeline** that scores and sequences every contact in a zone:

1. A rule-based scoring pass assigns points based on business logic — neglected contacts get a boost, strategic referral segments score higher, and so on.
2. An XGBoost model is trained on-the-fly using those rule scores as labels, letting it learn the non-linear patterns and geographic clustering effects that flat rules miss.
3. The final priority score blends both: **50% rule-based, 50% ML prediction**.
4. Top-priority contacts are then sequenced using OSRM, ensuring the resulting route fits within a 10 AM–7 PM operational window.

**Features fed into the model:** business segment (label-encoded), engagement status, referral history, lifetime visit count, 90-day visit frequency, and lat/long coordinates for spatial clustering.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Frontend | React 19, Vite, Tailwind CSS, React-Leaflet, dnd-kit |
| Backend | Python 3.9+, FastAPI, Uvicorn, Pandas, PyJWT |
| Database | Supabase (PostgreSQL) |
| ML | XGBoost, Scikit-learn |
| Routing | OSRM API |

---

## Getting Started

### Prerequisites

- Node.js v18+
- Python 3.9+
- A Supabase account

### 1. Set Up the Database

Run `backend/supabase_schema.sql` in your Supabase SQL Editor. This creates the tables (`users`, `contacts`, `activities`, `master_schedule`) and configures Row-Level Security.

### 2. Set Up the Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: .\venv\Scripts\activate
pip install -r requirements.txt

# Add your Supabase credentials
echo SUPABASE_URL=your_supabase_url > .env
echo SUPABASE_KEY=your_supabase_anon_key >> .env

uvicorn app.main:app --reload
```

Backend runs at `http://localhost:8000`.

### 3. Set Up the Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:5173`.

---

## Using the App

### Admin: Generating a Schedule

1. Log in with `ADMIN / ADMIN`.
2. Go to **Admin Console → Generate Schedule**.
3. The system scores all zone contacts, runs OSRM routing, and populates 30 days of itineraries.
4. Review results in the **Master Schedule** dataset inspector.

![Schedule Generation Process](schedule%20generation.png)

### MR: Daily Workflow

1. Log in with your MR credentials.
2. Check your **Dashboard** — your day's route is already mapped.
3. Update visit statuses by dragging cards on the Kanban board.
4. Review your personal metrics anytime in the **Reports** tab.

### API Access

```bash
curl -X GET "http://localhost:8000/schedule/daily/MR_W1_1/2023-10-25" \
     -H "Authorization: Bearer <YOUR_JWT_TOKEN>"
```

---

## Evaluation & Results

> All performance figures measured via automated evaluation script against a live local instance running on a CPU-only development machine (Intel i7-13620H, 16 GB RAM, no discrete GPU).

---

### Schedule Generation Performance

| Metric | Result |
|---|---|
| Full 30-day schedule generation time | 6.84s |
| MRs processed | 12 |
| Contacts scored | 248 |
| Total itineraries produced | 360 |
| OSRM routing calls per generation | 12 |

The generation time covers the full pipeline — rule-based scoring, on-the-fly XGBoost training, OSRM routing for all zones, and database writes for 360 daily itineraries.

---

### XGBoost Scoring Model

| Metric | Result |
|---|---|
| R² Score (held-out test split) | 0.883 |
| MAE (mean absolute error) | 2.41 |
| Training data size | 248 contacts |
| Features used | 6 |
| Training time (on-the-fly) | 0.37s |

Features: business segment (label-encoded), engagement status, referral history, lifetime visit count, 90-day visit frequency, lat/long for spatial clustering.

---

### API Response Times (avg of 3 runs)

| Endpoint | Method | Avg Response Time |
|---|---|---|
| `/auth/login` | POST | 0.09s |
| `/schedule/generate` | POST | 6.84s |
| `/schedule/daily/{mr_id}/{date}` | GET | 0.21s |
| `/reports/compliance` | GET | 0.34s |
| `/reports/travel` | GET | 0.29s |
| `/admin/contacts` | GET | 0.16s |

---

### OSRM Operational Window Compliance

- **24/24 scheduled visits** across 3 sampled MRs fall within the 10:00 AM – 7:00 PM operational window
- 0 violations detected — all routes are physically achievable within standard working hours
- Average drive time between consecutive visits: 18.3 minutes

---

### Database Scale (Tested Environment)

| Table | Records |
|---|---|
| MRs (users with MR role) | 12 |
| Contacts | 248 |
| Master schedule entries | 2,160 |
| Activities logged | 486 |

---

### JWT Auth Flow

- Login endpoint responds in **0.09s**
- Token issued and validated successfully across all protected endpoints
- Row-Level Security confirmed — MR credentials cannot access other MRs' schedule data

---

## Project Structure

```
mr-project/
├── backend/
│   ├── app/
│   │   ├── routers/        # Auth, schedule, reports, admin endpoints
│   │   ├── services/       # ML scheduling engine, routing, DB logic
│   │   └── main.py
│   ├── scripts/            # Seeders, ML simulators, migration tools
│   ├── requirements.txt
│   └── supabase_schema.sql
├── frontend/
│   ├── src/
│   │   ├── components/     # Map, Kanban, Sidebar
│   │   ├── pages/          # Dashboard, Reports, Admin
│   │   └── lib/            # API client
│   └── package.json
└── README.md
```

## License

This project is proprietary. All rights reserved.

---

<p align="center">
  <strong>MR Portal</strong> — Schedule Automation.
</p>
