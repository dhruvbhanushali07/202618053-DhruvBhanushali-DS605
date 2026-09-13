import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="NYC Airbnb Price Predictor",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# DESIGN TOKENS
# ==================================================
# Ink navy for structure, warm paper for content, brass as the
# single accent — evokes an appraisal report rather than a generic
# SaaS dashboard. One serif (Fraunces) carries the headline voice,
# Inter handles every label, input, and number.

INK = "#0F1B2B"
INK_SOFT = "#16273C"
PAPER = "#F6F5F1"
PAPER_RAISED = "#FFFFFF"
RULE = "#DCD8CE"
TEXT_PRIMARY = "#16202B"
TEXT_MUTED = "#5B6472"
BRASS = "#A9762F"
BRASS_DARK = "#8B5F22"

# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown(
    f"""
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, sans-serif;
    }}

    /* ==========================================
       APP BACKGROUND
       ========================================== */

    .stApp {{
        background-color: {PAPER};
    }}

    [data-testid="stMainBlockContainer"] {{
        max-width: 900px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }}

    /* ==========================================
       SIDEBAR
       ========================================== */

    [data-testid="stSidebar"] {{
        background-color: {INK} !important;
        border-right: 1px solid {INK_SOFT};
    }}

    [data-testid="stSidebar"] > div:first-child {{
        padding: 2rem 1.5rem;
    }}

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        font-family: 'Fraunces', serif;
        color: #F2EFE7 !important;
        font-weight: 500 !important;
        letter-spacing: 0.01em;
    }}

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] li {{
        color: #B9C0CC !important;
        font-size: 14.5px;
        line-height: 1.55;
    }}

    [data-testid="stSidebar"] strong {{
        color: #F2EFE7 !important;
    }}

    [data-testid="stSidebar"] hr {{
        border: none;
        border-top: 1px solid {INK_SOFT};
        margin: 1.4rem 0;
    }}

    .sidebar-kicker {{
        color: {BRASS} !important;
        font-size: 12.5px;
        font-weight: 600;
        letter-spacing: 0.04em;
        margin-bottom: 0.3rem;
    }}

    /* ==========================================
       HEADER
       ========================================== */

    .app-header {{
        margin-bottom: 0.4rem;
    }}

    .app-header h1 {{
        font-family: 'Fraunces', serif;
        font-weight: 500;
        font-size: 2.6rem;
        color: {TEXT_PRIMARY};
        margin: 0 0 0.35rem 0;
        line-height: 1.15;
    }}

    .app-header p {{
        font-size: 16px;
        color: {TEXT_MUTED};
        margin: 0;
        max-width: 60ch;
    }}

    /* ==========================================
       INFO NOTICE
       ========================================== */

    [data-testid="stAlert"] {{
        background-color: {PAPER_RAISED} !important;
        border: 1px solid {RULE} !important;
        border-left: 3px solid {BRASS} !important;
        border-radius: 4px !important;
        box-shadow: none !important;
    }}

    [data-testid="stAlert"] p {{
        color: {TEXT_PRIMARY} !important;
        font-size: 14.5px !important;
    }}

    /* ==========================================
       SECTION TITLES
       ========================================== */

    .section-title {{
        font-family: 'Fraunces', serif;
        font-size: 19px;
        font-weight: 500;
        color: {TEXT_PRIMARY};
        border-bottom: 1px solid {RULE};
        padding-bottom: 0.5rem;
        margin: 2.2rem 0 1.1rem 0;
    }}

    /* ==========================================
       GROUPED SECTION CARDS
       ========================================== */

    [data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: {PAPER_RAISED};
        border: 1px solid {RULE} !important;
        border-radius: 6px;
        padding: 0.5rem 0.25rem;
    }}

    /* ==========================================
       LABELS & INPUTS
       ========================================== */

    label, .stSlider label {{
        color: {TEXT_PRIMARY} !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
    }}

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    .stNumberInput input {{
        background-color: {PAPER} !important;
        border: 1px solid {RULE} !important;
        border-radius: 4px !important;
        color: {TEXT_PRIMARY} !important;
    }}

    .stSlider [data-baseweb="slider"] div[role="slider"] {{
        background-color: {BRASS} !important;
    }}

    .stSlider [data-testid="stTickBar"] {{
        display: none;
    }}

    /* ==========================================
       PREDICT BUTTON
       ========================================== */

    .stButton > button {{
        background-color: {INK} !important;
        color: #F2EFE7 !important;
        border: none !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        padding: 0.7rem 1rem !important;
        transition: background-color 0.15s ease;
    }}

    .stButton > button:hover {{
        background-color: {INK_SOFT} !important;
        color: #FFFFFF !important;
    }}

    /* ==========================================
       PREDICTION RESULT
       ========================================== */

    [data-testid="stMetric"] {{
        background-color: {PAPER_RAISED} !important;
        border: 1px solid {RULE} !important;
        border-left: 3px solid {BRASS} !important;
        border-radius: 6px !important;
        padding: 1.6rem 1.8rem !important;
    }}

    [data-testid="stMetricLabel"] {{
        color: {TEXT_MUTED} !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
    }}

    [data-testid="stMetricValue"] {{
        color: {TEXT_PRIMARY} !important;
        font-family: 'Fraunces', serif !important;
        font-size: 40px !important;
        font-weight: 500 !important;
    }}

    /* ==========================================
       DIVIDERS & FOOTER
       ========================================== */

    hr {{
        border: none;
        border-top: 1px solid {RULE};
        margin: 2.4rem 0;
    }}

    .app-footer {{
        color: {TEXT_MUTED};
        font-size: 13px;
        text-align: center;
    }}

    </style>
    """,
    unsafe_allow_html=True
)

# ==================================================
# LOAD TRAINED MODEL
# ==================================================

@st.cache_resource
def load_model():
    model_path = os.path.join(
        os.path.dirname(__file__),
        "airbnb_price_pipeline.pkl"
    )
    return joblib.load(model_path)

model = load_model()

# ==================================================
# HEADER
# ==================================================

st.markdown(
    """
    <div class="app-header">
        <h1>NYC Airbnb Price Predictor</h1>
        <p>Estimate the nightly rate of a listing from its location,
        room type, and booking history using a trained regression model.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.info(
    "This model was trained on the Airbnb NYC 2019 dataset. "
    "Its estimate reflects that historical market and should not be "
    "read as a current 2026 price."
)

# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown('<div class="sidebar-kicker">Project</div>', unsafe_allow_html=True)
    st.markdown("### NYC Airbnb Price Predictor")

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-kicker">About the model</div>', unsafe_allow_html=True)
    st.write(
        "A Random Forest Regressor trained on a log-transformed "
        "target price, with the full preprocessing pipeline stored "
        "alongside the model."
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-kicker">Details</div>', unsafe_allow_html=True)
    st.markdown(
        """
        - **Model** — Random Forest Regressor
        - **Target** — Nightly price (log-transformed)
        - **Dataset** — Airbnb NYC, 2019
        - **Evaluated with** — MAE, RMSE, R²
        """
    )

# ==================================================
# LOCATION
# ==================================================

st.markdown('<div class="section-title">Location</div>', unsafe_allow_html=True)

with st.container(border=True):
    col1, col2, col3 = st.columns(3)

    with col1:
        neighbourhood_group = st.selectbox(
            "Neighbourhood group",
            ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
        )

    with col2:
        neighbourhood_options = [
            "Williamsburg", "Bedford-Stuyvesant", "Harlem", "Bushwick",
            "Upper West Side", "Hell's Kitchen", "East Village",
            "Upper East Side", "Crown Heights", "Midtown"
        ]
        neighbourhood = st.selectbox("Neighbourhood", neighbourhood_options)

    with col3:
        room_type = st.selectbox(
            "Room type",
            ["Entire home/apt", "Private room", "Shared room"]
        )

    st.markdown("")
    col1, col2 = st.columns(2)

    with col1:
        latitude = st.number_input(
            "Latitude", min_value=40.4, max_value=41.0,
            value=40.7180, step=0.0001, format="%.6f"
        )

    with col2:
        longitude = st.number_input(
            "Longitude", min_value=-74.3, max_value=-73.6,
            value=-73.9950, step=0.0001, format="%.6f"
        )

# ==================================================
# LISTING DETAILS
# ==================================================

st.markdown('<div class="section-title">Listing details</div>', unsafe_allow_html=True)

with st.container(border=True):
    col1, col2, col3 = st.columns(3)

    with col1:
        minimum_nights = st.number_input(
            "Minimum nights", min_value=1, max_value=365, value=3, step=1
        )

    with col2:
        number_of_reviews = st.number_input(
            "Number of reviews", min_value=0, max_value=1000, value=10, step=1
        )

    with col3:
        reviews_per_month = st.number_input(
            "Reviews per month", min_value=0.0, max_value=100.0, value=1.0, step=0.1
        )

    st.markdown("")
    last_review_year = st.selectbox(
        "Last review year",
        ["No review", 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
    )

    if last_review_year == "No review":
        last_review_year = np.nan

# ==================================================
# HOST & AVAILABILITY
# ==================================================

st.markdown('<div class="section-title">Host &amp; availability</div>', unsafe_allow_html=True)

with st.container(border=True):
    col1, col2 = st.columns(2)

    with col1:
        calculated_host_listings_count = st.number_input(
            "Host's number of listings", min_value=1, max_value=500, value=1, step=1
        )

    with col2:
        availability_365 = st.slider(
            "Availability (days per year)", min_value=0, max_value=365, value=200
        )

# ==================================================
# BUILD INPUT DATAFRAME
# ==================================================

input_data = pd.DataFrame({
    "neighbourhood_group": [neighbourhood_group],
    "neighbourhood": [neighbourhood],
    "latitude": [latitude],
    "longitude": [longitude],
    "room_type": [room_type],
    "minimum_nights": [minimum_nights],
    "number_of_reviews": [number_of_reviews],
    "reviews_per_month": [reviews_per_month],
    "calculated_host_listings_count": [calculated_host_listings_count],
    "availability_365": [availability_365],
    "last_review_year": [last_review_year]
})

# ==================================================
# PREDICTION
# ==================================================

st.markdown("<hr>", unsafe_allow_html=True)

predict_button = st.button(
    "Predict nightly price",
    type="primary",
    use_container_width=True
)

if predict_button:

    try:
        prediction_log = model.predict(input_data)
        predicted_price = np.expm1(prediction_log[0])

        st.markdown('<div class="section-title">Estimate</div>', unsafe_allow_html=True)

        st.metric(
            label="Estimated nightly price",
            value=f"${predicted_price:,.2f}"
        )

        st.caption(
            "Generated by the Random Forest model on a log-transformed "
            "target, using last review year as an additional feature."
        )

    except Exception as e:
        st.error("Something went wrong while generating the prediction.")
        st.exception(e)

# ==================================================
# FOOTER
# ==================================================

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    '<div class="app-footer">M.Sc. Data Science · Airbnb Price Prediction Project</div>',
    unsafe_allow_html=True
)