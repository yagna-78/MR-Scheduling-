import streamlit as st
import pandas as pd
import numpy as np
import os
import uuid
import requests
import datetime as dt
from datetime import timedelta
import joblib
import shutil
import cv2
import pytesseract
from pytesseract import Output
from PIL import Image
from xgboost import XGBRegressor

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config("MR CRM Dashboard", layout="wide")

CONTACTS_DB = "Contacts.csv"
USERS_PATH = "User_Master.csv"
HIST_ACTIVITIES = "ref_activities_dec_2025_WITH_STATUS.csv"
MASTER_SCHEDULE = "predicted_schedule_all_mrs_jan_2026.csv"

OSRM_BASE_URL = "http://router.project-osrm.org/route/v1/driving/"

# OCR binary
tesseract_path = shutil.which("tesseract")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

#safe read csv
def safe_read_csv(path, required=True):
    if not os.path.exists(path):
        if required:
            st.error(f"Required file missing: {path}")
            st.stop()
        return pd.DataFrame()
    return pd.read_csv(path)

# ─────────────────────────────────────────────
# LOAD ML
# ─────────────────────────────────────────────
@st.cache_resource
def load_ml():
    model = XGBRegressor()
    model.load_model("xgb_priority_model.json")
    le_segment = joblib.load("segment_encoder.pkl")
    le_status = joblib.load("status_encoder.pkl")
    return model, le_segment, le_status


xgb_model, le_segment, le_status = load_ml()

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def predict_status(ref):
    if ref == 0:
        return "Unaware"
    elif 1 <= ref <= 3:
        return "Exploring"
    elif 4 <= ref <= 10:
        return "Engaged"
    return "Champion"


def get_travel_distance(lat1, lon1, lat2, lon2):
    try:
        url = f"{OSRM_BASE_URL}{lon1},{lat1};{lon2},{lat2}?overview=false"
        r = requests.get(url, timeout=5)
        data = r.json()
        if data.get("code") == "Ok":
            route = data["routes"][0]
            return round(route["distance"] / 1000, 2), round(route["duration"] / 60, 1)
    except Exception:
        pass
    return 5.0, 15.0

# ─────────────────────────────────────────────
# OCR
# ─────────────────────────────────────────────
def preprocess_image(img):
    g = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    g = cv2.bilateralFilter(g, 9, 75, 75)
    t = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 2
    )
    return Image.fromarray(t)


def extract_text(img):
    data = pytesseract.image_to_data(
        img,
        output_type=Output.DICT,
        config="--oem 3 --psm 4"
    )
    return " ".join(
        w for w, c in zip(data["text"], data["conf"])
        if w.strip() and int(c) > 50
    )

# ─────────────────────────────────────────────
# LOGIN
# ─────────────────────────────────────────────
def login():
    st.title("Login")

    users = pd.read_csv(USERS_PATH)
    role = st.selectbox("Role", ["MR", "ADMIN"])

    if role == "MR":
        mr = st.selectbox("MR ID", users["mr_id"].tolist())
        if st.button("Login"):
            st.session_state["role"] = "MR"
            st.session_state["mr_id"] = mr
            st.rerun()
    else:
        if st.button("Login as Admin"):
            st.session_state["role"] = "ADMIN"
            st.rerun()

# ─────────────────────────────────────────────
# SCHEDULE GENERATION (ADMIN)
# ─────────────────────────────────────────────
def generate_all_mr_schedule():
    users = safe_read_csv(USERS_PATH)
    contacts = safe_read_csv(CONTACTS_DB)
    activities = safe_read_csv(HIST_ACTIVITIES)


    activities["date"] = pd.to_datetime(activities["date"], errors="coerce")
    today = pd.to_datetime("2025-12-31")

    result = []
    act_id = 1

    for _, mr in users.iterrows():
        mr_contacts = contacts[
            contacts["Zone"].str.upper() == mr["zone"].upper()
        ].copy()

        latest = (
            activities.sort_values("date")
            .groupby("customer_id")
            .tail(1)
        )

        mr_contacts = mr_contacts.merge(
            latest[["customer_id", "referrals_count", "visit_count"]],
            left_on="Contact_id",
            right_on="customer_id",
            how="left"
        ).fillna(0)

        mr_contacts["current_status"] = mr_contacts["referrals_count"].apply(predict_status)

        mr_contacts["Segment_encoded"] = mr_contacts["Segment"].apply(
            lambda x: le_segment.transform([x])[0] if x in le_segment.classes_ else 0
        )

        mr_contacts["Status_encoded"] = mr_contacts["current_status"].apply(
            lambda x: le_status.transform([x])[0] if x in le_status.classes_ else 0
        )

        features = mr_contacts[
            ["Segment_encoded", "Status_encoded", "referrals_count", "visit_count"]
        ]

        mr_contacts["priority"] = xgb_model.predict(features)
        mr_contacts = mr_contacts.sort_values("priority", ascending=False)

        start = today + timedelta(days=1)
        end = today + timedelta(days=30)

        for d in pd.date_range(start, end):
            if d.weekday() >= 5:
                continue

            daily = mr_contacts.head(8)
            time = dt.datetime.combine(d.date(), dt.time(9, 0))

            for _, c in daily.iterrows():
                result.append({
                    "activity_id": f"ACT_{act_id:07d}",
                    "mr_id": mr["mr_id"],
                    "date": d.date(),
                    "activity_type": "Doctor Visit",
                    "customer_id": c["Contact_id"],
                    "customer_status": c["current_status"],
                    "locality": c.get("Locality", ""),
                    "start_time": time.strftime("%H:%M"),
                    "end_time": (time + timedelta(minutes=30)).strftime("%H:%M")
                })
                time += timedelta(minutes=45)
                act_id += 1

    pd.DataFrame(result).to_csv(MASTER_SCHEDULE, index=False)

# ─────────────────────────────────────────────
# MR KANBAN
# ─────────────────────────────────────────────
def mr_kanban():
    st.subheader("My Activities")

    if not os.path.exists(MASTER_SCHEDULE):
        st.warning("Schedule not generated yet. Please contact Admin.")
        return

    df = pd.read_csv(MASTER_SCHEDULE)

    selected_date = st.date_input("Select Date")

    df = df[
        (df["mr_id"] == st.session_state["mr_id"]) &
        (pd.to_datetime(df["date"]).dt.date == selected_date)
    ]

    cols = st.columns(3)

    for col, title in zip(cols, ["Planned", "In Progress", "Completed"]):
        with col:
            st.markdown(f"### {title}")
            for _, r in df.iterrows():
                st.info(
                    f"{r['activity_type']}\n"
                    f"Customer: {r['customer_id']}\n"
                    f"{r['start_time']} - {r['end_time']}"
                )

# ─────────────────────────────────────────────
# CONTACTS
# ─────────────────────────────────────────────
def view_contacts():
    df = pd.read_csv(CONTACTS_DB)

    if st.session_state["role"] == "MR":
        users = pd.read_csv(USERS_PATH)
        zone = users.loc[
            users["mr_id"] == st.session_state["mr_id"], "zone"
        ].iloc[0]
        df = df[df["Zone"].str.upper() == zone.upper()]

    st.dataframe(df)

# ─────────────────────────────────────────────
# OCR ADD CONTACT
# ─────────────────────────────────────────────
def add_contact_ocr():
    st.subheader("Add Contact via OCR")

    files = st.file_uploader(
        "Upload visiting cards",
        type=["jpg", "png"],
        accept_multiple_files=True
    )

    if not files:
        return

    base = pd.read_csv(CONTACTS_DB)

    for f in files:
        img = Image.open(f)
        processed = preprocess_image(img)
        text = extract_text(processed)

        st.image(img, width=200)

        name = st.text_input("Name", value=text[:30], key=f.name)

        if st.button(f"Save {f.name}"):
            base = pd.concat([
                base,
                pd.DataFrame([{
                    "Contact_id": f"CONT_{uuid.uuid4().hex[:8]}",
                    "Contact_name": name,
                    "Zone": "West"
                }])
            ])

    base.to_csv(CONTACTS_DB, index=False)

# ─────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────
if "role" not in st.session_state:
    login()
    st.stop()

with st.sidebar:
    if st.session_state["role"] == "ADMIN":
        page = st.radio("Menu", ["Generate Schedule", "Contacts"])
    else:
        page = st.radio("Menu", ["Kanban", "Contacts", "Add Contact"])

if page == "Generate Schedule":
    generate_all_mr_schedule()
    st.success("Master schedule generated")

elif page == "Kanban":
    mr_kanban()

elif page == "Contacts":
    view_contacts()

elif page == "Add Contact":
    add_contact_ocr()
