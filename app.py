import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    m = joblib.load("best_model.pkl")
    if not hasattr(m, 'multi_class'):
        m.multi_class = 'auto'
    return m

model = load_model()

def safe_predict_proba(m, X):
    """Works even if predict_proba fails due to sklearn version mismatch."""
    try:
        return m.predict_proba(X)[0]
    except AttributeError:
        score = m.decision_function(X)[0]
        churn_p = 1 / (1 + np.exp(-score))
        return np.array([1 - churn_p, churn_p])

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 Customer Churn Predictor")
st.markdown("Fill in the customer details below to predict whether they are likely to churn.")
st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("prediction_form"):

    col1, col2, col3 = st.columns(3)

    # ── Column 1: Demographics & Account ─────────────────────────────────────
    with col1:
        st.subheader("👤 Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])

    # ── Column 2: Services ────────────────────────────────────────────────────
    with col2:
        st.subheader("📡 Services")
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox(
            "Multiple Lines", ["No", "No phone service", "Yes"]
        )
        internet_service = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online Security", ["No", "No internet service", "Yes"]
        )
        online_backup = st.selectbox(
            "Online Backup", ["No", "No internet service", "Yes"]
        )
        device_protection = st.selectbox(
            "Device Protection", ["No", "No internet service", "Yes"]
        )

    # ── Column 3: Streaming, Contract & Charges ───────────────────────────────
    with col3:
        st.subheader("💳 Contract & Billing")
        tech_support = st.selectbox(
            "Tech Support", ["No", "No internet service", "Yes"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV", ["No", "No internet service", "Yes"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies", ["No", "No internet service", "Yes"]
        )
        contract = st.selectbox(
            "Contract Type", ["Month-to-month", "One year", "Two year"]
        )
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
                "Mailed check",
            ],
        )
        monthly_charges = st.number_input(
            "Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0, step=0.5
        )
        total_charges = st.number_input(
            "Total Charges ($)", min_value=0.0, max_value=10000.0, value=800.0, step=10.0
        )

    submitted = st.form_submit_button("🔍 Predict Churn", use_container_width=True)

# ── Feature engineering ───────────────────────────────────────────────────────
def build_feature_vector(
    gender, senior_citizen, partner, dependents, tenure,
    phone_service, tech_support, paperless_billing,
    monthly_charges, total_charges,
    device_protection, internet_service, streaming_tv, streaming_movies,
    payment_method, online_security, online_backup, multiple_lines, contract
):
    """
    Constructs a 44-element feature vector matching the model's training columns:
    ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
     'PhoneService', 'TechSupport', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
     'DeviceProtection_No', 'DeviceProtection_No internet service', 'DeviceProtection_Yes',
     'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
     'StreamingTV_No', 'StreamingTV_No internet service', 'StreamingTV_Yes',
     'StreamingMovies_No', 'StreamingMovies_No internet service', 'StreamingMovies_Yes',
     'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
     'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
     'OnlineSecurity_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
     'OnlineBackup_No', 'OnlineBackup_No internet service', 'OnlineBackup_Yes',
     'MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes',
     'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',   # duplicate cols
     'DeviceProtection_No', 'DeviceProtection_No internet service', 'DeviceProtection_Yes', # duplicate cols
     'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year']
    """

    def yn(val):
        return 1 if val == "Yes" else 0

    # Scalar features
    f_gender         = 1 if gender == "Male" else 0
    f_senior         = yn(senior_citizen)
    f_partner        = yn(partner)
    f_dependents     = yn(dependents)
    f_tenure         = tenure
    f_phone          = yn(phone_service)
    f_tech_support   = 1 if tech_support == "Yes" else 0
    f_paperless      = yn(paperless_billing)
    f_monthly        = monthly_charges
    f_total          = total_charges

    # One-hot: DeviceProtection
    dp_no  = 1 if device_protection == "No" else 0
    dp_nis = 1 if device_protection == "No internet service" else 0
    dp_yes = 1 if device_protection == "Yes" else 0

    # One-hot: InternetService
    is_dsl   = 1 if internet_service == "DSL" else 0
    is_fiber = 1 if internet_service == "Fiber optic" else 0
    is_no    = 1 if internet_service == "No" else 0

    # One-hot: StreamingTV
    stv_no  = 1 if streaming_tv == "No" else 0
    stv_nis = 1 if streaming_tv == "No internet service" else 0
    stv_yes = 1 if streaming_tv == "Yes" else 0

    # One-hot: StreamingMovies
    sm_no  = 1 if streaming_movies == "No" else 0
    sm_nis = 1 if streaming_movies == "No internet service" else 0
    sm_yes = 1 if streaming_movies == "Yes" else 0

    # One-hot: PaymentMethod
    pm_bt  = 1 if payment_method == "Bank transfer (automatic)" else 0
    pm_cc  = 1 if payment_method == "Credit card (automatic)" else 0
    pm_ec  = 1 if payment_method == "Electronic check" else 0
    pm_mc  = 1 if payment_method == "Mailed check" else 0

    # One-hot: OnlineSecurity
    os_no  = 1 if online_security == "No" else 0
    os_nis = 1 if online_security == "No internet service" else 0
    os_yes = 1 if online_security == "Yes" else 0

    # One-hot: OnlineBackup
    ob_no  = 1 if online_backup == "No" else 0
    ob_nis = 1 if online_backup == "No internet service" else 0
    ob_yes = 1 if online_backup == "Yes" else 0

    # One-hot: MultipleLines
    ml_no  = 1 if multiple_lines == "No" else 0
    ml_nps = 1 if multiple_lines == "No phone service" else 0
    ml_yes = 1 if multiple_lines == "Yes" else 0

    # One-hot: Contract
    ct_mtm = 1 if contract == "Month-to-month" else 0
    ct_1yr = 1 if contract == "One year" else 0
    ct_2yr = 1 if contract == "Two year" else 0

    # Build as DataFrame with exact column names the model was trained on
    columns = list(model.feature_names_in_)
    values = [
        f_gender, f_senior, f_partner, f_dependents, f_tenure,
        f_phone, f_tech_support, f_paperless, f_monthly, f_total,
        dp_no, dp_nis, dp_yes,
        is_dsl, is_fiber, is_no,
        stv_no, stv_nis, stv_yes,
        sm_no, sm_nis, sm_yes,
        pm_bt, pm_cc, pm_ec, pm_mc,
        os_no, os_nis, os_yes,
        ob_no, ob_nis, ob_yes,
        ml_no, ml_nps, ml_yes,
        is_dsl, is_fiber, is_no,    # duplicate InternetService cols
        dp_no, dp_nis, dp_yes,      # duplicate DeviceProtection cols
        ct_mtm, ct_1yr, ct_2yr,
    ]
    return pd.DataFrame([values], columns=columns)

# ── Prediction & result display ───────────────────────────────────────────────
if submitted:
    X = build_feature_vector(
        gender, senior_citizen, partner, dependents, tenure,
        phone_service, tech_support, paperless_billing,
        monthly_charges, total_charges,
        device_protection, internet_service, streaming_tv, streaming_movies,
        payment_method, online_security, online_backup, multiple_lines, contract
    )

    prediction = model.predict(X)[0]
    proba = safe_predict_proba(model, X)
    churn_prob = proba[1] * 100
    stay_prob  = proba[0] * 100

    st.divider()
    st.subheader("🎯 Prediction Result")

    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        if prediction == 1:
            st.error("⚠️ **Customer is likely to CHURN**")
        else:
            st.success("✅ **Customer is likely to STAY**")

    with res_col2:
        st.metric(label="🔴 Churn Probability", value=f"{churn_prob:.1f}%")

    with res_col3:
        st.metric(label="🟢 Retention Probability", value=f"{stay_prob:.1f}%")

    # Probability bar
    st.markdown("#### Churn Risk Level")
    st.progress(int(churn_prob))

    # Risk interpretation
    if churn_prob < 30:
        st.info("🟢 **Low Risk** — Customer is very likely to stay. No immediate action needed.")
    elif churn_prob < 60:
        st.warning("🟡 **Medium Risk** — Consider a retention offer or follow-up.")
    else:
        st.error("🔴 **High Risk** — Immediate retention action recommended!")

    # Key factors hint
    st.divider()
    st.subheader("📌 Key Input Summary")
    summary_cols = st.columns(4)
    summary_cols[0].metric("Tenure", f"{tenure} months")
    summary_cols[1].metric("Monthly Charges", f"${monthly_charges:.2f}")
    summary_cols[2].metric("Contract", contract)
    summary_cols[3].metric("Internet Service", internet_service)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Model: Logistic Regression | Dataset: Telco Customer Churn | Built with Streamlit 🚀")
