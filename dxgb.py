import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import os
import joblib
import uuid
import re
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
import datetime as dt
import os.path
import shutil

warnings.filterwarnings('ignore')

@st.cache_resource
def load_xgb_artifacts():
    model = XGBRegressor()
    model.load_model("xgb_priority_model.json")

    le_segment = joblib.load("segment_encoder.pkl")
    le_status = joblib.load("status_encoder.pkl")

    return model, le_segment, le_status

xgb_model, le_segment, le_status = load_xgb_artifacts()
# ─── Safe LOG_PATH (no warning anymore) ─────────────────────────────────
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ocr_logs.txt")

# Create log file if it doesn't exist
if not os.path.exists(LOG_PATH):
    try:
        open(LOG_PATH, 'a').close()
    except:
        pass  # silent fail - logging will be skipped if can't create

# ─── Configuration ──────────────────────────────────────────────────────
tesseract_path = shutil.which("tesseract")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

CONTACTS_PATH = "Contacts.csv"
UPDATED_CONTACTS = "updated_contacts.csv"
ACTIVITIES_PATH = "ref_activities_dec_2025_WITH_STATUS.csv"
USERS_PATH = "User_Master.csv"
OSRM_BASE_URL = "http://router.project-osrm.org/route/v1/driving/"

# ─── OCR Constants ──────────────────────────────────────────────────────
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

# ─── Page Config & Styling ──────────────────────────────────────────────
st.set_page_config(page_title="MR Dashboard", layout="wide")

st.markdown("""
    <style>
    .main .block-container {padding: 2rem 3rem;}
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #334155;
    }
    section[data-testid="stSidebar"] * {color: #e2e8f0 !important;}
    .card {
        background: white;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        padding: 1.6rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 1.8rem;
    }
    div[data-testid="metric-container"] {
        background-color: #f0f9ff !important;
        border: 2px solid #60a5fa !important;
        border-radius: 12px !important;
    }
    </style>
""", unsafe_allow_html=True)

# ─── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("MR Dashboard")
    st.caption("Ahmedabad • 2026")

    page = st.selectbox("Select Section", [
        "Overview",
        "OCR Add Contacts",
        "View Contacts",
        "View Users",
        "View Past Activities",
        "Generate Schedule"
    ])

# ─── OCR Helper Functions ───────────────────────────────────────────────
def log_extraction(filename, raw_text):
    timestamp = dt.datetime.now().isoformat()
    try:
        with open(LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] File: {filename}\n{raw_text}\n{'-'*50}\n")
    except:
        pass  # silent fail if can't write

def validate_data(data):
    errors = []
    if not data['Contact_name'].strip():
        errors.append("Name is required")
    if not data['ph_no'].strip():
        errors.append("Phone number is required")
    elif not PHONE_PATTERN.match(data['ph_no'].replace(' ', '')):
        errors.append("Invalid phone format")
    if data['Contact_email'] and not EMAIL_PATTERN.match(data['Contact_email']):
        errors.append("Invalid email format")
    if not data['Locality'] or data['Locality'] == 'Unknown':
        errors.append("Locality is required")
    return errors

def check_duplicate(contacts, phone, name):
    if phone and phone.strip():
        # Escape special chars to prevent regex error
        escaped = re.escape(phone.strip())
        dup_phone = contacts[contacts['ph_no'].str.contains(escaped, na=False, regex=True)]
        if not dup_phone.empty:
            return f"Duplicate phone: {dup_phone.iloc[0]['Contact_name']}"

    if name and name.strip():
        dup_name = contacts[contacts['Contact_name'].str.lower() == name.lower()]
        if not dup_name.empty:
            return f"Similar name: {dup_name.iloc[0]['Contact_name']}"

    return None

def preprocess_image(pil_image):
    img = pil_image.convert('L')
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = ImageEnhance.Sharpness(img).enhance(1.5)
    img = img.resize((int(img.width * 2.5), int(img.height * 2.5)), Image.Resampling.LANCZOS)
    arr = np.array(img)
    binary = np.where(arr > 130, 255, 0).astype(np.uint8)
    return Image.fromarray(binary)

def extract_name(text):
    patterns = [
        r'(?:Dr\.?|Doctor|DR|Prof\.?|Prof)?\s*([A-Za-z\s\.\']{4,50})(?:\s*(?:MD|MS|MDS|DM|DNB|MBBS|BHMS|BAMS|PhD))?',
        r'([A-Z][a-zA-Z\s\.]{3,50})'
    ]
    for p in patterns:
        m = re.search(p, text, re.I | re.M)
        if m:
            return m.group(1).strip().title()
    return "Unknown Doctor"

def extract_phone(text):
    patterns = [
        r'(?:\+91|91|0)?[\s.-]*?(\d{3,5})[\s.-]*?(\d{3,5})[\s.-]*?(\d{3,5})?',
        r'\(?(\d{3})\)?[\s.-]*?(\d{3})[\s.-]*?(\d{4})'
    ]
    for pattern in patterns:
        for m in re.finditer(pattern, text):
            digits = ''.join(g for g in m.groups() if g)
            if 10 <= len(digits) <= 12:
                return f"+91 {digits[-10:-5]} {digits[-5:]}"
    return ""

def extract_email(text):
    m = re.search(r'[\w\.-]+@[\w\.-]+\.\w{2,}', text, re.I)
    return m.group(0) if m else ""

def extract_speciality(text):
    text_lower = text.lower()
    for spec in SPECIALITIES:
        if re.search(rf'\b{re.escape(spec.lower())}\b', text_lower):
            return spec
    return "General"

def extract_locality(text):
    text_lower = text.lower()
    for loc in LOCALITIES:
        if re.search(rf'\b{re.escape(loc.lower())}\b', text_lower):
            return loc
    return "Unknown"

def extract_address(text):
    m = re.search(r'(?:Address|Clinic|Hosp|Hospital|Opp\.?|Nr\.?|Near|Opp|Rd|Road|Circle|Nagar|Society)[\s:]*([A-Za-z0-9\s\.,\-\/\(\)]{20,300})(?=\s*(?:Phone|Mobile|Tel|Email|Website|\d{6}|$))', 
                  text, re.I | re.DOTALL | re.M)
    return m.group(1).strip().replace('\n', ' ') if m else ""

# ─── Travel Time with OSRM ──────────────────────────────────────────────
def get_travel_distance(lat1, lon1, lat2, lon2):
    url = f"{OSRM_BASE_URL}{lon1},{lat1};{lon2},{lat2}?overview=false"
    try:
        r = requests.get(url, timeout=6)
        data = r.json()
        if data.get('code') == 'Ok' and data.get('routes'):
            route = data['routes'][0]
            dist_km = route['distance'] / 1000
            dur_min = route['duration'] / 60
            return round(dist_km, 2), round(dur_min, 1)
    except:
        pass
    return 5.0, 15.0

# ─── Status & Schedule Generation (UNCHANGED from your notebook) ────────
def predict_status(ref):
    if ref == 0: return 'Unaware'
    elif 1 <= ref <= 3: return 'Exploring'
    elif 4 <= ref <= 10: return 'Engaged'
    else: return 'Champion'

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

# ─── Dashboard ──────────────────────────────────────────────────────────
st.title("Medical Representative Dashboard")
st.markdown(f"**Ahmedabad •**")

if page == "Overview":
    st.subheader("Overview")
    st.info("Use sidebar to navigate")

elif page == "OCR Add Contacts":
    st.subheader("Add New Doctors via Visiting Card (OCR)")

    uploaded_files = st.file_uploader(
        "Upload card photos",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True
    )

    if uploaded_files:
        temp_entries = []
        existing = pd.read_csv(CONTACTS_PATH)

        for idx, file in enumerate(uploaded_files):
            with st.expander(f"Card: {file.name}", expanded=(idx == 0)):
                img = Image.open(file)
                processed = preprocess_image(img)
                raw_text = pytesseract.image_to_string(processed, config='--oem 3 --psm 6 -l eng')
                log_extraction(file.name, raw_text)

                st.image(img, caption="Original", width=300)
                st.text_area("Raw OCR Text", raw_text, height=120)

                extracted = {
                    'Contact_name': extract_name(raw_text),
                    'ph_no': extract_phone(raw_text),
                    'Address': extract_address(raw_text),
                    'Contact_email': extract_email(raw_text),
                    'Speciality': extract_speciality(raw_text),
                    'Locality': extract_locality(raw_text),
                    'Zone': LOC_ZONE_MAP.get(extract_locality(raw_text), 'Unknown'),
                    'Segment': 'Growth Catalyst',
                    'Contact_id': f"CONT_AHD_REAL_{uuid.uuid4().hex[:8].upper()}",
                    'Latitude': '',
                    'Longitude': ''
                }

                col1, col2 = st.columns(2)
                name = col1.text_input("Name", extracted['Contact_name'], key=f"name_{idx}")
                phone = col1.text_input("Phone", extracted['ph_no'], key=f"phone_{idx}")
                email = col1.text_input("Email", extracted['Contact_email'], key=f"email_{idx}")
                speciality = col1.selectbox("Speciality", SPECIALITIES,
                                          index=SPECIALITIES.index(extracted['Speciality']) if extracted['Speciality'] in SPECIALITIES else 0,
                                          key=f"spec_{idx}")

                address = col2.text_area("Address", extracted['Address'], height=80, key=f"addr_{idx}")
                locality = col2.selectbox("Locality", LOCALITIES,
                                        index=LOCALITIES.index(extracted['Locality']) if extracted['Locality'] in LOCALITIES else 0,
                                        key=f"loc_{idx}")
                zone = col2.selectbox("Zone", ["West", "East", "South", "North", "Unknown"],
                                    index=["West", "East", "South", "North", "Unknown"].index(extracted['Zone']) if extracted['Zone'] in ["West", "East", "South", "North"] else 4,
                                    key=f"zone_{idx}")
                segment = col2.selectbox("Segment", ["Growth Catalyst", "Key Influencer", "Silent Referrer", "Peripheral Supporter"],
                                       index=0, key=f"seg_{idx}")

                col_lat, col_lon = st.columns(2)
                lat = col_lat.text_input("Latitude", extracted['Latitude'], key=f"lat_{idx}")
                lng = col_lon.text_input("Longitude", extracted['Longitude'], key=f"lng_{idx}")

                entry = {
                    'Contact_name': name,
                    'ph_no': phone,
                    'Address': address,
                    'Latitude': lat,
                    'Longitude': lng,
                    'Zone': zone,
                    'Speciality': speciality,
                    'Locality': locality,
                    'Contact_id': extracted['Contact_id'],
                    'Segment': segment,
                    'Contact_email': email
                }

                errors = validate_data(entry)
                if errors:
                    st.warning("Validation issues:\n" + "\n".join(errors))
                else:
                    dup_msg = check_duplicate(existing, phone, name)
                    if dup_msg:
                        st.warning(dup_msg)
                        if st.checkbox("Add anyway (possible duplicate)", key=f"dupchk_{idx}"):
                            temp_entries.append(entry)
                            st.success("Added!")
                    else:
                        if st.button("Add Entry", key=f"addbtn_{idx}", type="primary"):
                            temp_entries.append(entry)
                            st.success("Added to batch!")

        if temp_entries:
            st.subheader("Batch Preview")
            st.dataframe(pd.DataFrame(temp_entries))

            if st.button("Save All to Database", type="primary"):
                current = pd.read_csv(UPDATED_CONTACTS) if os.path.exists(UPDATED_CONTACTS) else pd.read_csv(CONTACTS_PATH)
                updated = pd.concat([current, pd.DataFrame(temp_entries)], ignore_index=True)
                updated.to_csv(UPDATED_CONTACTS, index=False)
                st.success(f"Added {len(temp_entries)} new contacts!")
                st.balloons()

elif page == "View Contacts":
    st.subheader("Contacts Database")
    df = pd.read_csv(UPDATED_CONTACTS if os.path.exists(UPDATED_CONTACTS) else CONTACTS_PATH)
    st.dataframe(df)

elif page == "View Users":
    st.subheader("MR Users")
    df = pd.read_csv(USERS_PATH)
    st.dataframe(df)

elif page == "View Past Activities":
    st.subheader("Past Activities")
    df = pd.read_csv(ACTIVITIES_PATH)
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
