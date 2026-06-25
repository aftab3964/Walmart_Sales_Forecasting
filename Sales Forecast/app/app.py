import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Walmart Sales Forecasting",
    page_icon="📈",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = pickle.load(open("../models/sales_model.pkl", "rb"))
importance = pd.read_csv("../models/feature_importance.csv")

# -----------------------------
# Title
# -----------------------------
st.title("📈 Walmart Sales Forecasting Dashboard")

st.markdown("""
Predict **Weekly Sales** using a Machine Learning model trained on historical Walmart sales data.

---
""")

# =============================
# Sidebar
# =============================

st.sidebar.header("Store Information")

store = st.sidebar.number_input(
    "Store Number",
    min_value=1,
    max_value=50,
    value=1
)

dept = st.sidebar.number_input(
    "Department",
    min_value=1,
    max_value=100,
    value=1
)

size = st.sidebar.number_input(
    "Store Size",
    value=150000
)

store_type = st.sidebar.selectbox(
    "Store Type",
    ["A","B","C"]
)

holiday = st.sidebar.selectbox(
    "Holiday Week",
    ["No","Yes"]
)

st.sidebar.header("Date")

year = st.sidebar.selectbox(
    "Year",
    [2010,2011,2012]
)

month = st.sidebar.slider(
    "Month",
    1,
    12,
    6
)

week = st.sidebar.slider(
    "Week",
    1,
    52,
    25
)

# -----------------------------
# Feature Encoding
# -----------------------------

holiday = 1 if holiday=="Yes" else 0

type_B = 0
type_C = 0

if store_type=="B":
    type_B=1

elif store_type=="C":
    type_C=1

# -----------------------------
# Input Data
# -----------------------------

input_df = pd.DataFrame([[
    store,
    dept,
    size,
    holiday,
    year,
    month,
    week,
    type_B,
    type_C
]],columns=[
    "Store",
    "Dept",
    "Size",
    "IsHoliday",
    "Year",
    "Month",
    "Week",
    "Type_B",
    "Type_C"
])

# =============================
# Prediction
# =============================

if st.button("Predict Weekly Sales", use_container_width=True):

    prediction = model.predict(input_df)[0]

    st.success("Prediction Completed Successfully")

    st.metric(
        label="Predicted Weekly Sales",
        value=f"${prediction:,.2f}"
    )

    st.markdown("---")

    col1,col2 = st.columns(2)

    with col1:

        st.subheader("Input Summary")

        summary = pd.DataFrame({
            "Feature":[
                "Store",
                "Department",
                "Store Size",
                "Store Type",
                "Holiday",
                "Year",
                "Month",
                "Week"
            ],
            "Value":[
                store,
                dept,
                size,
                store_type,
                "Yes" if holiday else "No",
                year,
                month,
                week
            ]
        })

        st.table(summary)

    with col2:

        st.subheader("Top Feature Importance")

        top = importance.sort_values(
            "Importance",
            ascending=False
        ).head(10)

        fig,ax = plt.subplots(figsize=(8,5))

        ax.barh(
            top["Feature"],
            top["Importance"]
        )

        ax.set_xlabel("Importance")

        ax.set_ylabel("Feature")

        ax.set_title("Top 10 Important Features")

        ax.invert_yaxis()

        st.pyplot(fig)

    st.markdown("---")

    st.subheader("Business Insight")

    if prediction > 30000:

        st.success("""
Expected sales are **very high**.

Business Recommendation:

- Increase inventory.
- Schedule additional staff.
- Ensure product availability.
- Prepare promotional campaigns.
""")

    elif prediction > 15000:

        st.info("""
Expected sales are **moderate**.

Business Recommendation:

- Maintain regular inventory.
- Monitor demand.
- Continue planned operations.
""")

    else:

        st.warning("""
Expected sales are **relatively low**.

Business Recommendation:

- Review promotions.
- Optimize staffing.
- Investigate store performance.
""")

st.markdown("---")

st.subheader("About the Model")

st.write("""
**Dataset:** Walmart Weekly Sales Forecasting

**Machine Learning Model:** Random Forest Regressor

**Evaluation Metrics**

- MAE: 1340.54
- RMSE: 3508.54
- R² Score: 0.976

This dashboard demonstrates how machine learning can support retail demand forecasting and assist managers in making inventory and staffing decisions.
""")