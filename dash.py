import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import os
import uuid
import re
import warnings
import datetime as dt
import joblib
import shutil

from PIL import Image
import cv2
import pytesseract
from pytesseract import Output

from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="MR Dashboard", layout="wide")

CONTACTS_PATH = "Contacts.csv"
CONTACTS_DB = "contacts_db.csv"   # ✅ SINGLE SOURCE OF TRUTH

ACTIVITIES_PATH = "ref_activities_dec_2025_WITH_STATUS.csv"
USERS_PATH = "User_Master.csv"

OSRM_BASE_URL = "http://router.project-osrm.org/route/v1/driving/"

# Detect tesseract dynamically (Cloud + Local safe)
tesseract_path = shutil.which("tesseract")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

SPECIALITIES = [
    "General", "Orthopedics", "Gynecology", "Pediatrics",
    "Cardiology", "Neurology", "Oncology", "Dental"
]

LOCALITIES = [
    "Gota", "Ghatlodiya", "Science City", "Vaishnodevi",
    "Chandlodiya", "Sola", "Satellite", "Bopal"
]

LOC_ZONE_MAP = {
    "Gota": "West",
    "Science City": "West",
    "Vaishnodevi": "West",
    "Satellite": "West",
    "Bopal": "West"
}

PHONE_PATTERN = re.compile(r"\+91\s?\d{5}\s?\d{5}")
EMAIL_PATTERN = re.compile(r"[\w\.-]+@[\w\.-]+\.\w{2,}")

# ─────────────────────────────────────────────────────────────
# LOAD XGBOOST MODEL (INFERENCE ONLY)
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_xgb_artifacts():
    model = XGBRegressor()
    model.load_model("xgb_priority_model.json")
    le_segment = joblib.load("segment_encoder.pkl")
    le_status = joblib.load("status_encoder.pkl")
    return model, le_segment, le_status

xgb_model, le_segment, le_status = load_xgb_artifacts()

# ─────────────────────────────────────────────────────────────
# OCR HELPERS (IMPROVED & STABLE)
# ─────────────────────────────────────────────────────────────

def preprocess_image(pil_image):
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 2
    )

    return Image.fromarray(thresh)

OCR_CONFIG = (
    "--oem 3 --psm 4 -l eng "
    "-c preserve_interword_spaces=1 "
    "-c tessedit_char_whitelist="
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    ".@,+-:/() "
)

def extract_text_safe(image):
    data = pytesseract.image_to_data(
        image,
        config=OCR_CONFIG,
        output_type=Output.DICT
    )

    words = []
    for i, txt in enumerate(data["text"]):
        try:
            conf = int(data["conf"][i])
            if conf > 50 and len(txt.strip()) > 1:
                words.append(txt)
        except:
            pass

    return " ".join(words)

def extract_name(text):
    tokens = text.split()
    for t in tokens:
        if "dr" in t.lower():
            return text.replace(t, "").strip().title()
    return "Unknown Doctor"

def extract_phone(text):
    m = PHONE_PATTERN.search(text)
    return m.group(0) if m else ""

def extract_email(text):
    m = EMAIL_PATTERN.search(text)
    return m.group(0) if m else ""

def extract_locality(text):
    for loc in LOCALITIES:
        if loc.lower() in text.lower():
            return loc
    return "Unknown"

# ─────────────────────────────────────────────────────────────
# SIDEBAR (OVERVIEW REMOVED)
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("MR Dashboard")
    page = st.selectbox(
        "Select Section",
        [
            "OCR Add Contacts",
            "View Contacts",
            "Generate Schedule"
        ]
    )

st.title("Medical Representative Dashboard")

# ─────────────────────────────────────────────────────────────
# OCR ADD CONTACTS
# ─────────────────────────────────────────────────────────────

if page == "OCR Add Contacts":

    uploaded_files = st.file_uploader(
        "Upload visiting card images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        temp_entries = []

        base_df = (
            pd.read_csv(CONTACTS_DB)
            if os.path.exists(CONTACTS_DB)
            else pd.read_csv(CONTACTS_PATH)
        )

        for idx, file in enumerate(uploaded_files):
            with st.expander(f"Card: {file.name}", expanded=True):
                img = Image.open(file)
                processed = preprocess_image(img)
                raw_text = extract_text_safe(processed)

                st.image(img, width=250)
                st.text_area("OCR Text", raw_text, height=120)

                name = extract_name(raw_text)
                phone = extract_phone(raw_text)
                email = extract_email(raw_text)
                locality = extract_locality(raw_text)

                col1, col2 = st.columns(2)

                name = col1.text_input("Name", name, key=f"name_{idx}")
                phone = col1.text_input("Phone", phone, key=f"phone_{idx}")
                email = col1.text_input("Email", email, key=f"email_{idx}")
                speciality = col1.selectbox("Speciality", SPECIALITIES, key=f"spec_{idx}")

                locality = col2.selectbox("Locality", LOCALITIES, key=f"loc_{idx}")
                zone = LOC_ZONE_MAP.get(locality, "Unknown")
                segment = col2.selectbox(
                    "Segment",
                    ["Growth Catalyst", "Key Influencer", "Silent Referrer"],
                    key=f"seg_{idx}"
                )

                if st.button("Add Entry", key=f"add_{idx}"):
                    temp_entries.append({
                        "Contact_id": f"CONT_{uuid.uuid4().hex[:8].upper()}",
                        "Contact_name": name,
                        "ph_no": phone,
                        "Contact_email": email,
                        "Speciality": speciality,
                        "Locality": locality,
                        "Zone": zone,
                        "Segment": segment,
                        "Latitude": "",
                        "Longitude": ""
                    })
                    st.success("Added to batch")

        if temp_entries:
            st.subheader("Batch Preview")
            st.dataframe(pd.DataFrame(temp_entries))

            if st.button("Save All to Database", type="primary"):
                updated = pd.concat(
                    [base_df, pd.DataFrame(temp_entries)],
                    ignore_index=True
                )
                updated.to_csv(CONTACTS_DB, index=False)

                st.success(f"{len(temp_entries)} contacts saved")
                st.experimental_rerun()

# ─────────────────────────────────────────────────────────────
# VIEW CONTACTS
# ─────────────────────────────────────────────────────────────

elif page == "View Contacts":
    st.subheader("Contacts Database")

    df = (
        pd.read_csv(CONTACTS_DB)
        if os.path.exists(CONTACTS_DB)
        else pd.read_csv(CONTACTS_PATH)
    )

    st.dataframe(df)

# ─────────────────────────────────────────────────────────────
# GENERATE SCHEDULE (UNCHANGED PLACEHOLDER)
# ─────────────────────────────────────────────────────────────
def generate_schedule(selected_mr_id):
    activities = pd.read_csv(ACTIVITIES_PATH)
    activities['date'] = pd.to_datetime(activities['date'], errors='coerce')

    contacts = pd.read_csv(CONTACTS_PATH)
    users = pd.read_csv(USERS_PATH)
    contacts['Zone'] = contacts['Zone'].str.upper()

    current_date = pd.to_datetime('2025-12-31')

    latest_per_customer = activities.sort_values('date').groupby('customer_id').tail(1)[['customer_id', 'referrals_count', 'visit_count']]
    contacts = contacts.merge(latest_per_customer, left_on='Contact_id', right_on='customer_id', how='left')
    contacts['referrals_count'] = contacts['referrals_count'].fillna(0).astype(int)
    contacts['visit_count'] = contacts['visit_count'].fillna(0).astype(int)
    contacts['current_status'] = contacts['referrals_count'].apply(predict_status)

    last_visit = activities.groupby('customer_id')['date'].max().reset_index()
    last_visit['days_since_last_visit'] = (current_date - last_visit['date']).dt.days
    contacts = contacts.merge(last_visit[['customer_id', 'days_since_last_visit']], left_on='Contact_id', right_on='customer_id', how='left')
    contacts['days_since_last_visit'] = contacts['days_since_last_visit'].fillna(365)

    recent_visits = activities[activities['date'] > current_date - timedelta(days=90)]
    visit_count_90 = recent_visits.groupby('customer_id').size().reset_index(name='visit_count_last_90')
    contacts = contacts.merge(visit_count_90, left_on='Contact_id', right_on='customer_id', how='left')
    contacts['visit_count_last_90'] = contacts['visit_count_last_90'].fillna(0)

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
    # Encode using PRE-FITTED encoders
    contacts['Segment_encoded'] = contacts['Segment'].map(
        lambda x: le_segment.transform([x])[0] if x in le_segment.classes_ else 0
    )

    contacts['Status_encoded'] = contacts['current_status'].map(
        lambda x: le_status.transform([x])[0] if x in le_status.classes_ else 0
    )

    features = [
        'Segment_encoded',
        'Status_encoded',
        'referrals_count',
        'visit_count',
        'days_since_last_visit',
        'visit_count_last_90',
        'Latitude',
        'Longitude'
    ]

    X = contacts[features]

# ✅ Predict using PRE-TRAINED model
    contacts['xgb_score'] = xgb_model.predict(X)

# Hybrid priority (same as notebook)
    contacts['priority_score'] = (
        0.5 * contacts['rule_score'] +
        0.5 * contacts['xgb_score']
    )

    contacts['priority_score'] = 0.5 * contacts['rule_score'] + 0.5 * contacts['xgb_score']

    if selected_mr_id:
        mr_zone = users[users['mr_id'] == selected_mr_id]['zone'].iloc[0].upper()
        contacts = contacts[contacts['Zone'] == mr_zone]

    contacts = contacts.sort_values('priority_score', ascending=False)

    predicted_activities = []
    activity_types = ['Doctor Visit', 'Phone Call', 'Follow-up', 'Presentation']
    type_probs = {
        'Unaware': [0.4, 0.3, 0.2, 0.1],
        'Exploring': [0.3, 0.3, 0.3, 0.1],
        'Engaged': [0.2, 0.2, 0.3, 0.3],
        'Champion': [0.1, 0.1, 0.2, 0.6]
    }
    duration_ranges = {
        'Unaware': range(30, 46, 5),
        'Exploring': range(25, 41, 5),
        'Engaged': range(20, 36, 5),
        'Champion': range(15, 31, 5)
    }
    status_transition = {'Unaware': 'Exploring', 'Exploring': 'Engaged', 'Engaged': 'Champion', 'Champion': 'Champion'}

    activity_id_counter = 1
    start_date = current_date + timedelta(days=1)
    end_date = current_date + timedelta(days=30)

    mr_info = users[users['mr_id'] == selected_mr_id]
    if mr_info.empty:
        st.error("MR not found")
        return pd.DataFrame()

    mr_id = selected_mr_id
    team = mr_info['team'].iloc[0]
    zone = mr_info['zone'].iloc[0]
    start_lat = mr_info['starting_latitude'].iloc[0]
    start_lon = mr_info['starting_longitude'].iloc[0]

    for day in pd.date_range(start=start_date, end=end_date):
        if day.weekday() >= 5: continue

        daily_pool = contacts.sample(frac=0.1).sort_values('priority_score', ascending=False).head(8)

        current_time = dt.datetime.combine(day.date(), dt.time(9, 0))
        current_lat, current_lon = start_lat, start_lon
        previous_locality = 'Home'

        for _, cust in daily_pool.iterrows():
            dist, dur = get_travel_distance(current_lat, current_lon, cust.Latitude, cust.Longitude)
            current_time += timedelta(minutes=int(dur))

            probs = type_probs.get(cust.current_status, [0.25]*4)
            act_type = np.random.choice(activity_types, p=probs)

            duration_min = int(np.random.choice(duration_ranges.get(cust.current_status, range(20,36,5))))

            start_str = current_time.strftime('%H:%M')
            end_time = current_time + timedelta(minutes=duration_min)
            end_str = end_time.strftime('%H:%M')

            gap_days = cust.days_since_last_visit
            reason_parts = []
            if gap_days > 90: reason_parts.append(f"Long gap ({int(gap_days)} days)")
            if cust.current_status == 'Unaware': reason_parts.append("Unaware - needs intro")
            if cust.Segment in ['Peripheral Supporter', 'Silent Referrer']: reason_parts.append("Growth segment")
            priority_reason = "; ".join(reason_parts) or "Maintenance"

            talking_points = {
                'Unaware': "Introduce hospital specialties & benefits",
                'Exploring': "Share success stories & referral process",
                'Engaged': "Discuss collaboration opportunities",
                'Champion': "Thank for referrals & explore joint activities"
            }.get(cust.current_status, "General follow-up")

            last_date_str = (current_date - timedelta(days=gap_days)).strftime('%b %d, %Y') if gap_days < 365 else "Never visited"

            is_high_value = 'Yes' if cust.referrals_count > 10 else 'No'

            predicted_activities.append({
                'activity_id': f"ACT_{str(activity_id_counter).zfill(7)}",
                'mr_id': mr_id,
                'team': team,
                'zone': zone,
                'customer_id': cust.Contact_id,
                'customer_status': cust.current_status,
                'activity_type': act_type,
                'locality': cust.Locality,
                'date': day.date(),
                'start_time': start_str,
                'end_time': end_str,
                'duration_min': duration_min,
                'Latitude': cust.Latitude,
                'Longitude': cust.Longitude,
                'travel_km': dist,
                'travel_min': dur,
                'expected_next_status': status_transition.get(cust.current_status, 'Champion'),
                'priority_reason': priority_reason,
                'suggested_talking_points': talking_points,
                'last_visit_date': last_date_str,
                'total_referrals_so_far': cust.referrals_count,
                'travel_from_previous': previous_locality,
                'is_high_value': is_high_value
            })

            activity_id_counter += 1
            current_time = end_time
            current_lat, current_lon = cust.Latitude, cust.Longitude
            previous_locality = cust.Locality

    return pd.DataFrame(predicted_activities)
    
elif page == "View Past Activities":
    st.subheader("Past Activities")
    df = pd.read_csv(ACTIVITIES_PATH)
    st.dataframe(df)

elif page == "View Users":
    st.subheader("MR Users")
    df = pd.read_csv(USERS_PATH)
    st.dataframe(df)

elif page == "Generate Schedule":
    st.subheader("Generate MR Schedule")
    users = pd.read_csv(USERS_PATH)
    mr_list = users['mr_id'].tolist()
    selected_mr = st.selectbox("Select MR", mr_list)

    if st.button("Generate Schedule"):
        with st.spinner("Generating..."):
            df = generate_schedule(selected_mr)
        st.dataframe(df.head(20))
        st.download_button("Download CSV", df.to_csv(index=False), f"schedule_{selected_mr}.csv")

st.markdown("---")
st.caption("Dashboard • Ahmedabad • January 2026")
