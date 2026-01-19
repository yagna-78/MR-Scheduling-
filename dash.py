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