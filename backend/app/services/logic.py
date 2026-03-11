# app/services/logic.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import uuid
import re
import pytesseract
from PIL import Image, ImageEnhance
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Your OSRM and constants
OSRM_BASE_URL = "http://router.project-osrm.org/route/v1/driving/"
# ... REST OF YOUR ORIGINAL FILE EXACTLY AS IS ...

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import datetime as dt
import requests
import uuid
import re
import pytesseract
from PIL import Image, ImageEnhance
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
from app.services.supabase_db import load_data, save_data  # <--- IMPORT THE NEW DATABASE FILE

warnings.filterwarnings('ignore')

# ─── CONFIGURATION ───
# Note: On Streamlit Cloud, you don't set the tesseract path. 
# It is handled by packages.txt. We keep this check for local windows testing.
import os
if os.name == 'nt': 
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

OSRM_BASE_URL = "http://router.project-osrm.org/route/v1/driving/"

# ─── CONSTANTS ───
SPECIALITIES = [
    "Orthopedics", "Orthopaedic", "Multi-Specialty", "Multispeciality", "General Medicine",
    "Primary Care", "Nursing Home", "Maternity", "Ophthalmology", "Eye Care", "General",
    "Medical College", "Ayurveda", "Oncology", "Cancer", "Gynecology", "Gynaecology",
    "Pediatrics", "Paediatrics", "Government", "Dental", "Cardiology", "Neurology",
    "ICU", "Super Speciality", "Holistic", "Neonatal"
]

LOCALITIES = [
    "Gota", "Ghatlodiya", "Science City", "Vaishnodevi", "Chandlodiya", "Ognaj", "Maninagar",
    "Ghodasar", "Isanpur", "Vatva", "Vastral", "Odhav", "Bapunagar", "Naroda Road", "Saraspur",
    "Rakhial", "Ellisbridge", "Sola", "Satellite", "Ghuma", "Paldi", "Vasna", "Vastrapur",
    "Memnagar", "Thaltej", "SG Highway", "Bopal", "Naroda", "Naranpura", "Ranip", "Navrangpura",
    "Kubernagar", "Nikol", "Vadaj", "Hebatpur", "Zundal", "Ambli", "Ramdevnagar", "Lavanya",
    "Jivraj Park", "Gurukul", "Mithakhali"
]

LOC_ZONE_MAP = {
    'Bapunagar': 'East', 'Bopal': 'West', 'Chandlodiya': 'West', 'Ellisbridge': 'West',
    'Ghatlodiya': 'West', 'Ghodasar': 'South', 'Ghuma': 'West', 'Gota': 'West',
    'Isanpur': 'South', 'Maninagar': 'South', 'Memnagar': 'West', 'Naroda Road': 'East',
    'Nikol': 'East', 'Odhav': 'East', 'Ognaj': 'West', 'Paldi': 'West', 'Rakhial': 'East',
    'SG Highway': 'West', 'Saraspur': 'East', 'Satellite': 'West', 'Science City': 'West',
    'Sola': 'West', 'Thaltej': 'West', 'Vaishnodevi': 'West', 'Vasna': 'West',
    'Vastral': 'East', 'Vastrapur': 'West', 'Vatva': 'South'
}

PHONE_PATTERN = re.compile(r'^\+91\s?\d{5}\s?\d{5}$|^\d{10}$')
EMAIL_PATTERN = re.compile(r'^[\w\.-]+@[\w\.-]+\.\w{2,}$')

# ─── OCR FUNCTIONS (Unchanged) ───
def preprocess_image(pil_image):
    img = pil_image.convert('L')
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = ImageEnhance.Sharpness(img).enhance(1.5)
    img = img.resize((int(img.width * 2.5), int(img.height * 2.5)), Image.Resampling.LANCZOS)
    arr = np.array(img)
    binary = np.where(arr > 130, 255, 0).astype(np.uint8)
    return Image.fromarray(binary)

def extract_name(text):
    patterns = [r'(?:Dr\.?|Doctor|DR|Prof\.?|Prof)?\s*([A-Za-z\s\.\']{4,50})(?:\s*(?:MD|MS|MDS|DM|DNB|MBBS|BHMS|BAMS|PhD))?', r'([A-Z][a-zA-Z\s\.]{3,50})']
    for p in patterns:
        m = re.search(p, text, re.I | re.M)
        if m: return m.group(1).strip().title()
    return "Unknown Doctor"

def extract_phone(text):
    patterns = [r'(?:\+91|91|0)?[\s.-]*?(\d{3,5})[\s.-]*?(\d{3,5})[\s.-]*?(\d{3,5})?', r'\(?(\d{3})\)?[\s.-]*?(\d{3})[\s.-]*?(\d{4})']
    for pattern in patterns:
        for m in re.finditer(pattern, text):
            digits = ''.join(g for g in m.groups() if g)
            if 10 <= len(digits) <= 12: return f"+91 {digits[-10:-5]} {digits[-5:]}"
    return ""

def extract_email(text):
    m = re.search(r'[\w\.-]+@[\w\.-]+\.\w{2,}', text, re.I)
    return m.group(0) if m else ""

def extract_speciality(text):
    text_lower = text.lower()
    for spec in SPECIALITIES:
        if re.search(rf'\b{re.escape(spec.lower())}\b', text_lower): return spec
    return "General"

def extract_locality(text):
    text_lower = text.lower()
    for loc in LOCALITIES:
        if re.search(rf'\b{re.escape(loc.lower())}\b', text_lower): return loc
    return "Unknown"

def extract_address(text):
    m = re.search(r'(?:Address|Clinic|Hosp|Hospital|Opp\.?|Nr\.?|Near|Opp|Rd|Road|Circle|Nagar|Society)[\s:]*([A-Za-z0-9\s\.,\-\/\(\)]{20,300})(?=\s*(?:Phone|Mobile|Tel|Email|Website|\d{6}|$))', text, re.I | re.DOTALL | re.M)
    return m.group(1).strip().replace('\n', ' ') if m else ""

# ─── LOGIC FUNCTIONS ───
def get_travel_distance(lat1, lon1, lat2, lon2):
    url = f"{OSRM_BASE_URL}{lon1},{lat1};{lon2},{lat2}?overview=false"
    try:
        r = requests.get(url, timeout=3)
        data = r.json()
        if data.get('code') == 'Ok' and data.get('routes'):
            route = data['routes'][0]
            return round(route['distance'] / 1000, 2), round(route['duration'] / 60, 1)
    except: pass
    return 5.0, 15.0

def predict_status(ref):
    if ref == 0: return 'Unaware'
    elif 1 <= ref <= 3: return 'Exploring'
    elif 4 <= ref <= 10: return 'Engaged'
    else: return 'Champion'

# def run_schedule_logic_for_single_mr(selected_mr_id, users_df, contacts_df, activities_df, current_date_obj):
#     # 1. Get MR Details
#     mr_info = users_df[users_df['mr_id'] == selected_mr_id]
#     if mr_info.empty: return pd.DataFrame()
    
#     mr_zone = str(mr_info['zone'].iloc[0]).upper()
    
#     # 2. Filter Contacts by Zone
#     if 'Zone' not in contacts_df.columns: return pd.DataFrame()
#     contacts = contacts_df[contacts_df['Zone'].str.upper() == mr_zone].copy()
#     if contacts.empty: return pd.DataFrame()

#     # 3. Merge Activity Data (Referrals, Visits)
#     latest_per_customer = activities_df.sort_values('date').groupby('customer_id').tail(1)[['customer_id', 'referrals_count', 'visit_count']]
#     contacts = contacts.merge(
#         latest_per_customer.rename(columns={'customer_id': 'cust_id_latest'}),
#         left_on='Contact_id', right_on='cust_id_latest', how='left'
#     ).drop(columns=['cust_id_latest'], errors='ignore')

#     contacts['referrals_count'] = contacts['referrals_count'].fillna(0).astype(int)
#     contacts['visit_count'] = contacts['visit_count'].fillna(0).astype(int)
#     contacts['current_status'] = contacts['referrals_count'].apply(predict_status)

#     # 4. Calculate Days Since Last Visit
#     last_visit = activities_df.groupby('customer_id')['date'].max().reset_index()
#     last_visit['days_since_last_visit'] = (current_date_obj - last_visit['date']).dt.days
#     contacts = contacts.merge(
#         last_visit.rename(columns={'customer_id': 'cust_id_visit'}),
#         left_on='Contact_id', right_on='cust_id_visit', how='left'
#     ).drop(columns=['cust_id_visit'], errors='ignore')
#     contacts['days_since_last_visit'] = contacts['days_since_last_visit'].fillna(365)

#     # 5. Calculate Recent Activity (Last 90 Days)
#     recent_visits = activities_df[activities_df['date'] > current_date_obj - timedelta(days=90)]
#     visit_count_90 = recent_visits.groupby('customer_id').size().reset_index(name='visit_count_last_90')
#     contacts = contacts.merge(
#         visit_count_90.rename(columns={'customer_id': 'cust_id_90'}),
#         left_on='Contact_id', right_on='cust_id_90', how='left'
#     ).drop(columns=['cust_id_90'], errors='ignore')
#     contacts['visit_count_last_90'] = contacts['visit_count_last_90'].fillna(0)

#     # 6. Rule-Based Scoring
#     def rule_priority(row):
#         score = 0
#         if row['current_status'] == 'Unaware': score += 5
#         elif row['current_status'] == 'Exploring': score += 3
#         elif row['current_status'] == 'Engaged': score += 2
#         if row['days_since_last_visit'] > 60: score += 4
#         if row['visit_count'] < 3: score += 3
#         if row['Segment'] in ['Peripheral Supporter', 'Silent Referrer']: score += 2
#         return score

#     contacts['rule_score'] = contacts.apply(rule_priority, axis=1)

#     # 7. XGBoost Scoring
#     le_segment = LabelEncoder()
#     le_status = LabelEncoder()
#     contacts['Segment_encoded'] = le_segment.fit_transform(contacts['Segment'])
#     contacts['Status_encoded'] = le_status.fit_transform(contacts['current_status'])

#     features = ['Segment_encoded', 'Status_encoded', 'referrals_count', 'visit_count', 'days_since_last_visit', 'visit_count_last_90', 'Latitude', 'Longitude']
#     X = contacts[features]
#     y = contacts['rule_score']

#     model = XGBRegressor()
#     model.fit(X, y)
#     contacts['xgb_score'] = model.predict(X)
    
#     # 8. Final Prioritization
#     contacts['priority_score'] = 0.5 * contacts['rule_score'] + 0.5 * contacts['xgb_score']
#     contacts = contacts.sort_values('priority_score', ascending=False)

#     # 9. Schedule Generation Loop
#     predicted_activities = []
#     activity_types = ['Doctor Visit', 'Phone Call', 'Follow-up', 'Presentation']
#     type_probs = {
#         'Unaware': [0.4, 0.3, 0.2, 0.1], 'Exploring': [0.3, 0.3, 0.3, 0.1],
#         'Engaged': [0.2, 0.2, 0.3, 0.3], 'Champion': [0.1, 0.1, 0.2, 0.6]
#     }
#     duration_ranges = {
#         'Unaware': range(30, 46, 5), 'Exploring': range(25, 41, 5),
#         'Engaged': range(20, 36, 5), 'Champion': range(15, 31, 5)
#     }

#     start_date = current_date_obj + timedelta(days=1)
#     end_date = current_date_obj + timedelta(days=30)
    
#     # Get Start Location
#     try:
#         team = mr_info['team'].iloc[0]
#         start_lat = mr_info['starting_latitude'].iloc[0]
#         start_lon = mr_info['starting_longitude'].iloc[0]
#     except:
#         team = "General"
#         start_lat, start_lon = 23.0, 72.5

#     for day in pd.date_range(start=start_date, end=end_date):
#         if day.weekday() >= 5: continue  # Skip Weekends

#         # Pick top 8 contacts for the day
#         daily_pool = contacts.sample(frac=0.3 if len(contacts)>10 else 1.0).sort_values('priority_score', ascending=False).head(8)

#         current_time = dt.datetime.combine(day.date(), dt.time(10, 0))
#         current_lat, current_lon = start_lat, start_lon
        
#         for _, cust in daily_pool.iterrows():
#             # OSRM Calculation (This was calculated but lost in your old code)
#             dist, dur = get_travel_distance(current_lat, current_lon, cust.Latitude, cust.Longitude)
            
#             probs = type_probs.get(cust.current_status, [0.25]*4)
#             act_type = np.random.choice(activity_types, p=probs)
#             duration_min = int(np.random.choice(duration_ranges.get(cust.current_status, range(20,36,5))))

#             estimated_end = current_time + timedelta(minutes=int(dur) + duration_min)
#             if estimated_end.time() > dt.time(19, 0): continue

#             current_time += timedelta(minutes=int(dur))
#             start_str = current_time.strftime('%H:%M')
#             end_time = current_time + timedelta(minutes=duration_min)
#             end_str = end_time.strftime('%H:%M')
            
#             talking_points = {
#                 'Unaware': "Introduce hospital specialties & benefits",
#                 'Exploring': "Share success stories & referral process",
#                 'Engaged': "Discuss collaboration opportunities",
#                 'Champion': "Thank for referrals & explore joint activities"
#             }.get(cust.current_status, "General follow-up")

#             # SAVE DATA: Now explicitly adding the missing columns
#             predicted_activities.append({
#                 'activity_id': f"ACT_{selected_mr_id}_{uuid.uuid4().hex[:6]}",
#                 'mr_id': selected_mr_id,
#                 'team': team,
#                 'zone': mr_zone,
#                 'customer_id': cust.Contact_id,
#                 'customer_name': cust.Contact_name,
#                 'contact_person': cust.Contact_name, # Added Doctor Name
#                 'customer_status': cust.current_status,
#                 'activity_type': act_type,
#                 'locality': cust.Locality,
#                 'date': day.date(),
#                 'start_time': start_str,
#                 'end_time': end_str,
#                 'duration_min': duration_min,       # Appointment Duration
#                 'travel_duration_min': dur,         # Added Travel Time
#                 'distance_km': dist,                # Added Distance
#                 'latitude': cust.Latitude,          # Added Lat
#                 'longitude': cust.Longitude,        # Added Long
#                 'suggested_talking_points': talking_points,
#                 'status': 'Pending'
#             })

#             current_time = end_time
#             current_lat, current_lon = cust.Latitude, cust.Longitude

#     return pd.DataFrame(predicted_activities)
def run_schedule_logic_for_single_mr(selected_mr_id, users_df, contacts_df, activities_df, current_date_obj):
    print(f"[SCHEDULE] Starting for MR: {selected_mr_id}")
    
    # Force current_date_obj to Timestamp
    current_date_obj = pd.Timestamp(current_date_obj)
    
    # 1. Get MR Details
    mr_info = users_df[users_df['mr_id'] == selected_mr_id]
    if mr_info.empty: 
        print("[SCHEDULE] No MR info found")
        return pd.DataFrame()
    
    mr_zone = str(mr_info['zone'].iloc[0]).upper()
    
    # 2. Filter Contacts by Zone
    if 'Zone' not in contacts_df.columns: 
        print("[SCHEDULE] No 'Zone' column")
        return pd.DataFrame()
    
    contacts = contacts_df[contacts_df['Zone'].str.upper() == mr_zone].copy()
    if contacts.empty: 
        print("[SCHEDULE] No contacts in zone")
        return pd.DataFrame()

    # CRITICAL FIX: Convert ALL date columns to datetime BEFORE any calculation
    if 'date' in activities_df.columns:
        print("[SCHEDULE] Converting activities 'date' to datetime")
        activities_df['date'] = pd.to_datetime(activities_df['date'], errors='coerce')
        invalid = activities_df['date'].isna().sum()
        print(f"[SCHEDULE] Dropped {invalid} invalid dates")
        activities_df = activities_df.dropna(subset=['date'])

    # 3. Merge Activity Data (Referrals, Visits)
    latest_per_customer = activities_df.sort_values('date').groupby('customer_id').tail(1)[['customer_id', 'referrals_count', 'visit_count']]
    contacts = contacts.merge(
        latest_per_customer.rename(columns={'customer_id': 'cust_id_latest'}),
        left_on='Contact_id', right_on='cust_id_latest', how='left'
    ).drop(columns=['cust_id_latest'], errors='ignore')

    contacts['referrals_count'] = contacts['referrals_count'].fillna(0).astype(int)
    contacts['visit_count'] = contacts['visit_count'].fillna(0).astype(int)
    contacts['current_status'] = contacts['referrals_count'].apply(predict_status)

    # 4. Calculate Days Since Last Visit - NOW SAFE
    last_visit = activities_df.groupby('customer_id')['date'].max().reset_index()
    last_visit['days_since_last_visit'] = (current_date_obj - last_visit['date']).dt.days
    contacts = contacts.merge(
        last_visit.rename(columns={'customer_id': 'cust_id_visit'}),
        left_on='Contact_id', right_on='cust_id_visit', how='left'
    ).drop(columns=['cust_id_visit'], errors='ignore')
    contacts['days_since_last_visit'] = contacts['days_since_last_visit'].fillna(365)

    # 5. Calculate Recent Activity (Last 90 Days) - NOW SAFE
    recent_visits = activities_df[activities_df['date'] > current_date_obj - timedelta(days=90)]
    visit_count_90 = recent_visits.groupby('customer_id').size().reset_index(name='visit_count_last_90')
    contacts = contacts.merge(
        visit_count_90.rename(columns={'customer_id': 'cust_id_90'}),
        left_on='Contact_id', right_on='cust_id_90', how='left'
    ).drop(columns=['cust_id_90'], errors='ignore')
    contacts['visit_count_last_90'] = contacts['visit_count_last_90'].fillna(0)

    # 6. Rule-Based Scoring
    def rule_priority(row):
        score = 0
        if row['current_status'] == 'Unaware': score += 5
        elif row['current_status'] == 'Exploring': score += 3
        elif row['current_status'] == 'Engaged': score += 2
        if row['days_since_last_visit'] > 60: score += 4
        if row['visit_count'] < 3: score += 3
        if row['Segment'] in ['Peripheral Supporter', 'Silent Referrer']: score += 2
        return score

    contacts['rule_score'] = contacts.apply(rule_priority, axis=1)

    # 7. XGBoost Scoring
    le_segment = LabelEncoder()
    le_status = LabelEncoder()
    contacts['Segment_encoded'] = le_segment.fit_transform(contacts['Segment'])
    contacts['Status_encoded'] = le_status.fit_transform(contacts['current_status'])

    features = ['Segment_encoded', 'Status_encoded', 'referrals_count', 'visit_count', 'days_since_last_visit', 'visit_count_last_90', 'Latitude', 'Longitude']
    X = contacts[features]
    y = contacts['rule_score']

    model = XGBRegressor()
    model.fit(X, y)
    contacts['xgb_score'] = model.predict(X)
    
    # 8. Final Prioritization (this was missing earlier - add it)
    contacts['priority_score'] = 0.5 * contacts['rule_score'] + 0.5 * contacts['xgb_score']
    contacts = contacts.sort_values('priority_score', ascending=False)

    # 9. Schedule Generation Loop
    predicted_activities = []
    activity_types = ['Doctor Visit', 'Phone Call', 'Follow-up', 'Presentation']
    type_probs = {
        'Unaware': [0.4, 0.3, 0.2, 0.1], 'Exploring': [0.3, 0.3, 0.3, 0.1],
        'Engaged': [0.2, 0.2, 0.3, 0.3], 'Champion': [0.1, 0.1, 0.2, 0.6]
    }
    duration_ranges = {
        'Unaware': range(30, 46, 5), 'Exploring': range(25, 41, 5),
        'Engaged': range(20, 36, 5), 'Champion': range(15, 31, 5)
    }

    start_date = current_date_obj + timedelta(days=1)
    end_date = current_date_obj + timedelta(days=30)
    
    # Get Start Location
    try:
        team = mr_info['team'].iloc[0]
        start_lat = mr_info['starting_latitude'].iloc[0]
        start_lon = mr_info['starting_longitude'].iloc[0]
    except:
        team = "General"
        start_lat, start_lon = 23.0, 72.5

    for day in pd.date_range(start=start_date, end=end_date):
        if day.weekday() >= 5: continue  # Skip Weekends

        # Pick top 8 contacts for the day
        daily_pool = contacts.sample(frac=0.3 if len(contacts)>10 else 1.0).sort_values('priority_score', ascending=False).head(8)

        current_time = dt.datetime.combine(day.date(), dt.time(10, 0))
        current_lat, current_lon = start_lat, start_lon
        
        for _, cust in daily_pool.iterrows():
            dist, dur = get_travel_distance(current_lat, current_lon, cust.Latitude, cust.Longitude)
            
            probs = type_probs.get(cust.current_status, [0.25]*4)
            act_type = np.random.choice(activity_types, p=probs)
            duration_min = int(np.random.choice(duration_ranges.get(cust.current_status, range(20,36,5))))

            estimated_end = current_time + timedelta(minutes=int(dur) + duration_min)
            if estimated_end.time() > dt.time(19, 0): continue

            current_time += timedelta(minutes=int(dur))
            start_str = current_time.strftime('%H:%M')
            end_time = current_time + timedelta(minutes=duration_min)
            end_str = end_time.strftime('%H:%M')
            
            talking_points = {
                'Unaware': "Introduce hospital specialties & benefits",
                'Exploring': "Share success stories & referral process",
                'Engaged': "Discuss collaboration opportunities",
                'Champion': "Thank for referrals & explore joint activities"
            }.get(cust.current_status, "General follow-up")

            predicted_activities.append({
                'activity_id': f"ACT_{selected_mr_id}_{uuid.uuid4().hex[:6]}",
                'mr_id': selected_mr_id,
                'team': team,
                'zone': mr_zone,
                'customer_id': cust.Contact_id,
                'customer_name': cust.Contact_name,
                'contact_person': cust.Contact_name,
                'customer_status': cust.current_status,
                'activity_type': act_type,
                'locality': cust.Locality,
                'date': day.date(),
                'start_time': start_str,
                'end_time': end_str,
                'duration_min': duration_min,
                'travel_duration_min': dur,
                'distance_km': dist,
                'latitude': cust.Latitude,
                'longitude': cust.Longitude,
                'suggested_talking_points': talking_points,
                'status': 'Pending'
            })

            current_time = end_time
            current_lat, current_lon = cust.Latitude, cust.Longitude

    print(f"[SCHEDULE] Generated {len(predicted_activities)} activities for MR {selected_mr_id}")
    return pd.DataFrame(predicted_activities)