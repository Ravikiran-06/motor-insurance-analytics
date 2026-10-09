
import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Motor Insurance Claims & Policy Analytics",
    page_icon="🚗",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
CLEANED_DIR = BASE_DIR / "data" / "cleaned"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(LOG_DIR / "pipeline.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("motor_insurance_analytics")


# ============================================================
# 2. LOAD THE FIVE DATASETS
# ============================================================
REQUIRED_COLUMNS = {
    "customers": [
        "customer_id", "customer_name", "gender", "date_of_birth",
        "age", "city", "state", "occupation"
    ],
    "vehicles": [
        "vehicle_id", "customer_id", "vehicle_type", "vehicle_make",
        "vehicle_model", "vehicle_year", "fuel_type", "vehicle_value"
    ],
    "policies": [
        "policy_id", "customer_id", "vehicle_id", "policy_start_date",
        "policy_end_date", "policy_type", "premium_amount",
        "coverage_amount", "policy_status", "payment_frequency"
    ],
    "claims": [
        "claim_id", "policy_id", "customer_id", "claim_date",
        "accident_date", "claim_type", "accident_location",
        "claim_amount", "claim_status", "approval_date",
        "settlement_date", "damage_severity"
    ],
    "payments": [
        "payment_id", "claim_id", "policy_id", "payment_date",
        "payment_amount", "payment_status", "payment_method"
    ]
}


@st.cache_data
def load_all_data():
    data = {}

    for table_name, columns in REQUIRED_COLUMNS.items():
        raw_file = RAW_DIR / f"{table_name}.csv"
        cleaned_file = CLEANED_DIR / f"{table_name}_cleaned.csv"

        if cleaned_file.exists():
            file_path = cleaned_file
        elif raw_file.exists():
            file_path = raw_file
        else:
            raise FileNotFoundError(
                f"Missing {table_name} dataset. Expected either:\n"
                f"{raw_file}\nor\n{cleaned_file}"
            )

        df = pd.read_csv(file_path)

        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(
                f"{table_name}.csv is missing these columns: {missing}"
            )

        data[table_name] = df.drop_duplicates().copy()

    return data


# ============================================================
# 3. CLEAN DATA AND CREATE ANALYTICS COLUMNS
# ============================================================
def prepare_data(data):
    data = {name: df.copy() for name, df in data.items()}

    date_columns = {
        "customers": ["date_of_birth"],
        "policies": ["policy_start_date", "policy_end_date"],
        "claims": [
            "claim_date", "accident_date",
            "approval_date", "settlement_date"
        ],
        "payments": ["payment_date"]
    }

    for table_name, columns in date_columns.items():
        for column in columns:
            data[table_name][column] = pd.to_datetime(
                data[table_name][column], errors="coerce"
            )

    numeric_columns = {
        "customers": ["age"],
        "vehicles": ["vehicle_year", "vehicle_value"],
        "policies": ["premium_amount", "coverage_amount"],
        "claims": ["claim_amount"],
        "payments": ["payment_amount"]
    }

    for table_name, columns in numeric_columns.items():
        for column in columns:
            data[table_name][column] = pd.to_numeric(
                data[table_name][column], errors="coerce"
            ).fillna(0)

    text_columns = {
        "customers": [
            "customer_name", "gender", "city", "state", "occupation"
        ],
        "vehicles": [
            "vehicle_type", "vehicle_make", "vehicle_model", "fuel_type"
        ],
        "policies": [
            "policy_type", "policy_status", "payment_frequency"
        ],
        "claims": [
            "claim_type", "accident_location",
            "claim_status", "damage_severity"
        ],
        "payments": ["payment_status", "payment_method"]
    }

    for table_name, columns in text_columns.items():
        for column in columns:
            data[table_name][column] = (
                data[table_name][column]
                .fillna("Unknown")
                .astype(str)
                .str.strip()
            )

    data["policies"]["premium_amount"] = (
        data["policies"]["premium_amount"].clip(lower=0)
    )
    data["policies"]["coverage_amount"] = (
        data["policies"]["coverage_amount"].clip(lower=0)
    )
    data["claims"]["claim_amount"] = (
        data["claims"]["claim_amount"].clip(lower=0)
    )
    data["payments"]["payment_amount"] = (
        data["payments"]["payment_amount"].clip(lower=0)
    )

    policies = data["policies"]
    claims = data["claims"]
    customers = data["customers"]
    vehicles = data["vehicles"]
    payments = data["payments"]

    policies["policy_duration_days"] = (
        policies["policy_end_date"] - policies["policy_start_date"]
    ).dt.days.fillna(0)

    policies["policy_active_flag"] = (
        policies["policy_status"].str.lower().eq("active").astype(int)
    )

    claims["claim_processing_days"] = (
        claims["approval_date"] - claims["claim_date"]
    ).dt.days.fillna(0).clip(lower=0)

    claims["claim_settlement_days"] = (
        claims["settlement_date"] - claims["claim_date"]
    ).dt.days.fillna(0).clip(lower=0)

    customers["customer_age"] = customers["age"]

    current_year = pd.Timestamp.today().year
    vehicles["vehicle_age"] = (
        current_year - vehicles["vehicle_year"]
    ).where(vehicles["vehicle_year"] > 0, np.nan).clip(lower=0)

    policy_details = policies[
        ["policy_id", "premium_amount", "coverage_amount"]
    ]
    claims_with_policy = claims.merge(
        policy_details, on="policy_id", how="left"
    )

    claims["claim_to_premium_ratio"] = (
        claims_with_policy["claim_amount"]
        / claims_with_policy["premium_amount"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    claims["claim_to_coverage_ratio"] = (
        claims_with_policy["claim_amount"]
        / claims_with_policy["coverage_amount"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    claim_amounts = claims[["claim_id", "claim_amount"]]
    payments_with_claim = payments.merge(
        claim_amounts, on="claim_id", how="left"
    )

    payments["paid_claim_ratio"] = (
        payments_with_claim["payment_amount"]
        / payments_with_claim["claim_amount"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    return data


# ============================================================
# 4. KPI CALCULATIONS
# ============================================================
def total_policies(data):
    return data["policies"]["policy_id"].nunique()


def total_premium(data):
    return data["policies"]["premium_amount"].sum()


def average_premium(data):
    return data["policies"]["premium_amount"].mean() if not data["policies"].empty else 0


def total_claims(data):
    return data["claims"]["claim_id"].nunique()


def approved_claims(data):
    return data["claims"]["claim_status"].str.lower().eq("approved").sum()


def rejected_claims(data):
    return data["claims"]["claim_status"].str.lower().eq("rejected").sum()


def total_claim_amount(data):
    return data["claims"]["claim_amount"].sum()


def average_claim_amount(data):
    return data["claims"]["claim_amount"].mean() if not data["claims"].empty else 0


def total_payments(data):
    return data["payments"]["payment_amount"].sum()


def approval_rate(data):
    total = total_claims(data)
    return approved_claims(data) / total * 100 if total else 0


def claim_to_premium_ratio(data):
    premium = total_premium(data)
    return total_claim_amount(data) / premium if premium else 0


def average_settlement_days(data):
    df = data["claims"]
    return df["claim_settlement_days"].mean() if not df.empty else 0


# ============================================================
# 5. CHART HELPERS
# ============================================================
def show_chart(fig):
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def bar_chart(df, category, value, title, horizontal=False):
    if df.empty or category not in df.columns:
        st.info(f"No data available for {title}.")
        return

    if value is None:
        counts = df[category].fillna("Unknown").astype(str).value_counts()
        labels = counts.index.astype(str)
        values = counts.values
    else:
        if value not in df.columns:
            st.info(f"Column '{value}' is not available.")
            return
        grouped = df.groupby(category, dropna=False)[value].sum().sort_values(
            ascending=False
        )
        labels = grouped.index.astype(str)
        values = grouped.values

    fig, ax = plt.subplots(figsize=(7, 4))
    if horizontal:
        ax.barh(labels, values)
        ax.set_xlabel(value or "Count")
        ax.set_ylabel(category)
    else:
        ax.bar(labels, values)
        ax.set_xlabel(category)
        ax.set_ylabel(value or "Count")
        ax.tick_params(axis="x", rotation=30)

    ax.set_title(title)
    show_chart(fig)


def line_chart(df, date_column, value_column, title):
    if df.empty:
        st.info(f"No data available for {title}.")
        return

    temp = df.copy()
    temp[date_column] = pd.to_datetime(temp[date_column], errors="coerce")
    temp = temp.dropna(subset=[date_column])

    if temp.empty:
        st.info(f"No valid dates available for {title}.")
        return

    temp["month"] = temp[date_column].dt.to_period("M").astype(str)

    if value_column is None:
        grouped = temp.groupby("month").size()
    else:
        grouped = temp.groupby("month")[value_column].sum()

    grouped = grouped.sort_index().tail(12)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(grouped.index, grouped.values, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel(value_column or "Number of claims")
    ax.tick_params(axis="x", rotation=35)
    show_chart(fig)


def histogram_chart(df, column, title):
    if df.empty or column not in df.columns:
        st.info(f"No data available for {title}.")
        return

    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        st.info(f"No numeric values available for {title}.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=20, edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    show_chart(fig)


def scatter_chart(df, x, y, title):
    if df.empty or x not in df.columns or y not in df.columns:
        st.info(f"Data needed for {title} is not available.")
        return

    temp = df[[x, y]].copy()
    temp[x] = pd.to_numeric(temp[x], errors="coerce")
    temp[y] = pd.to_numeric(temp[y], errors="coerce")
    temp = temp.dropna()

    if temp.empty:
        st.info(f"No valid values available for {title}.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(temp[x], temp[y], alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    show_chart(fig)


def merge_claims_vehicles(data):
    return data["claims"].merge(
        data["vehicles"], on="customer_id", how="left"
    )


def merge_claims_customers(data):
    return data["claims"].merge(
        data["customers"], on="customer_id", how="left"
    )


def create_insight(df, group_column, value_column=None):
    if df.empty or group_column not in df.columns:
        return "There is not enough data to generate this insight."

    if value_column and value_column in df.columns:
        values = df.groupby(group_column)[value_column].sum()
        description = f"total {value_column}"
    else:
        values = df[group_column].fillna("Unknown").astype(str).value_counts()
        description = "record count"

    if values.empty:
        return "There is not enough data to generate this insight."

    top_group = values.idxmax()
    top_value = values.max()

    return (
        f"**Main finding:** {top_group} has the highest {description} "
        f"({top_value:,.2f}).\n\n"
        "**Business meaning:** This category contributes the most to the "
        "current filtered results.\n\n"
        "**Recommendation:** Review this category in more detail and "
        "compare its risk, claims, and premium patterns before making "
        "business decisions."
    )


# ============================================================
# 6. LOAD AND PREPARE DATA
# ============================================================
st.title("🚗 Motor Insurance Claims & Policy Analytics")
st.caption("Interactive dashboard for policy, customer, vehicle, claims and payment analysis.")

try:
    raw_data = load_all_data()
    data = prepare_data(raw_data)
except Exception as error:
    logger.exception("Dashboard data loading failed")
    st.error("The dashboard could not load the data.")
    st.code(str(error))
    st.info(
        "Check that all five CSV files are inside the project's "
        "'data/raw' folder and that their column names match the expected schema."
    )
    st.stop()


# ============================================================
# 7. SIDEBAR FILTERS
# ============================================================
st.sidebar.title("🔎 Dashboard Filters")

policy_types = sorted(data["policies"]["policy_type"].dropna().astype(str).unique())
policy_statuses = sorted(data["policies"]["policy_status"].dropna().astype(str).unique())
claim_statuses = sorted(data["claims"]["claim_status"].dropna().astype(str).unique())
claim_types = sorted(data["claims"]["claim_type"].dropna().astype(str).unique())
vehicle_types = sorted(data["vehicles"]["vehicle_type"].dropna().astype(str).unique())

selected_policy_types = st.sidebar.multiselect(
    "Policy Type", policy_types, default=policy_types
)
selected_policy_statuses = st.sidebar.multiselect(
    "Policy Status", policy_statuses, default=policy_statuses
)
selected_claim_statuses = st.sidebar.multiselect(
    "Claim Status", claim_statuses, default=claim_statuses
)
selected_claim_types = st.sidebar.multiselect(
    "Claim Type", claim_types, default=claim_types
)
selected_vehicle_types = st.sidebar.multiselect(
    "Vehicle Type", vehicle_types, default=vehicle_types
)

filtered = {name: df.copy() for name, df in data.items()}

filtered["policies"] = filtered["policies"][
    filtered["policies"]["policy_type"].isin(selected_policy_types)
    & filtered["policies"]["policy_status"].isin(selected_policy_statuses)
].copy()

filtered["vehicles"] = filtered["vehicles"][
    filtered["vehicles"]["vehicle_type"].isin(selected_vehicle_types)
].copy()

filtered["claims"] = filtered["claims"][
    filtered["claims"]["claim_status"].isin(selected_claim_statuses)
    & filtered["claims"]["claim_type"].isin(selected_claim_types)
].copy()

valid_policy_ids = filtered["policies"]["policy_id"]
filtered["claims"] = filtered["claims"][
    filtered["claims"]["policy_id"].isin(valid_policy_ids)
].copy()

valid_customer_ids = filtered["policies"]["customer_id"]
filtered["customers"] = filtered["customers"][
    filtered["customers"]["customer_id"].isin(valid_customer_ids)
].copy()

filtered["vehicles"] = filtered["vehicles"][
    filtered["vehicles"]["customer_id"].isin(valid_customer_ids)
].copy()

valid_claim_ids = filtered["claims"]["claim_id"]
filtered["payments"] = filtered["payments"][
    filtered["payments"]["claim_id"].isin(valid_claim_ids)
].copy()


# ============================================================
# 8. DASHBOARD NAVIGATION
# ============================================================
page = st.sidebar.radio(
    "Choose a dashboard page",
    [
        "Executive Overview",
        "Policy Analysis",
        "Claims Analysis",
        "Customer Analysis",
        "Vehicle Analysis",
        "Premium & Risk",
        "Insights & Actions"
    ]
)


# ============================================================
# 9. EXECUTIVE OVERVIEW
# ============================================================
if page == "Executive Overview":
    st.header("📊 Executive Risk & Operations Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Policies", f"{total_policies(filtered):,}")
    c2.metric("Total Premium", f"${total_premium(filtered):,.2f}")
    c3.metric("Total Claims", f"{total_claims(filtered):,}")
    c4.metric("Total Claim Amount", f"${total_claim_amount(filtered):,.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Approved Claims", f"{approved_claims(filtered):,}")
    c6.metric("Rejected Claims", f"{rejected_claims(filtered):,}")
    c7.metric("Approval Rate", f"{approval_rate(filtered):.1f}%")
    c8.metric("Average Settlement", f"{average_settlement_days(filtered):.1f} days")

    st.subheader("Policy Status")
    bar_chart(
        filtered["policies"], "policy_status", None,
        "Number of Policies by Status"
    )

    st.subheader("Claims Over Time")
    line_chart(
        filtered["claims"], "claim_date", None,
        "Monthly Claim Volume"
    )

    st.subheader("Key Insight")
    st.markdown(
        create_insight(filtered["policies"], "policy_status")
    )


# ============================================================
# 10. POLICY ANALYSIS
# ============================================================
elif page == "Policy Analysis":
    st.header("📜 Policy Portfolio Analysis")

    c1, c2 = st.columns(2)
    c1.metric("Average Premium", f"${average_premium(filtered):,.2f}")
    c2.metric("Total Coverage", f"${filtered['policies']['coverage_amount'].sum():,.2f}")

    c1, c2 = st.columns(2)
    with c1:
        bar_chart(
            filtered["policies"], "policy_status", None,
            "Policy Status Distribution"
        )
    with c2:
        bar_chart(
            filtered["policies"], "policy_type", None,
            "Policies by Type"
        )

    bar_chart(
        filtered["policies"], "policy_type", "premium_amount",
        "Total Premium by Policy Type"
    )

    line_chart(
        filtered["policies"], "policy_start_date", "premium_amount",
        "Monthly Premium Trend"
    )

    st.subheader("Policy Data")
    st.dataframe(filtered["policies"], use_container_width=True)


# ============================================================
# 11. CLAIMS ANALYSIS
# ============================================================
elif page == "Claims Analysis":
    st.header("💥 Claims Lifecycle & Incident Analysis")

    c1, c2, c3 = st.columns(3)
    c1.metric("Average Claim Amount", f"${average_claim_amount(filtered):,.2f}")
    c2.metric("Total Payments", f"${total_payments(filtered):,.2f}")
    c3.metric("Average Settlement", f"{average_settlement_days(filtered):.1f} days")

    c1, c2 = st.columns(2)
    with c1:
        bar_chart(
            filtered["claims"], "claim_status", None,
            "Claims by Status"
        )
    with c2:
        bar_chart(
            filtered["claims"], "claim_type", None,
            "Claims by Type", horizontal=True
        )

    c1, c2 = st.columns(2)
    with c1:
        line_chart(
            filtered["claims"], "claim_date", None,
            "Monthly Claim Volume"
        )
    with c2:
        histogram_chart(
            filtered["claims"], "claim_amount",
            "Claim Amount Distribution"
        )

    bar_chart(
        filtered["claims"], "damage_severity", None,
        "Claims by Damage Severity"
    )

    st.subheader("Claims Data")
    st.dataframe(filtered["claims"], use_container_width=True)


# ============================================================
# 12. CUSTOMER ANALYSIS
# ============================================================
elif page == "Customer Analysis":
    st.header("👤 Customer Demographic Analysis")

    c1, c2 = st.columns(2)
    with c1:
        customer_claims = merge_claims_customers(filtered)
        scatter_chart(
            customer_claims, "customer_age", "claim_amount",
            "Customer Age vs Claim Amount"
        )
    with c2:
        bar_chart(
            customer_claims, "state", None,
            "Claims by State"
        )

    bar_chart(
        customer_claims, "occupation", "claim_amount",
        "Total Claim Amount by Occupation"
    )

    st.subheader("Customer Data")
    st.dataframe(filtered["customers"], use_container_width=True)


# ============================================================
# 13. VEHICLE ANALYSIS
# ============================================================
elif page == "Vehicle Analysis":
    st.header("🚗 Vehicle Analysis")

    claims_vehicles = merge_claims_vehicles(filtered)

    c1, c2 = st.columns(2)
    with c1:
        bar_chart(
            claims_vehicles, "vehicle_type", "claim_amount",
            "Claim Amount by Vehicle Type"
        )
    with c2:
        scatter_chart(
            claims_vehicles, "vehicle_age", "claim_amount",
            "Vehicle Age vs Claim Amount"
        )

    bar_chart(
        filtered["vehicles"], "fuel_type", None,
        "Vehicles by Fuel Type"
    )

    st.subheader("Vehicle Data")
    st.dataframe(filtered["vehicles"], use_container_width=True)


# ============================================================
# 14. PREMIUM & RISK
# ============================================================
elif page == "Premium & Risk":
    st.header("⚖ Premium & Risk Analysis")

    premium = total_premium(filtered)
    claims_amount = total_claim_amount(filtered)
    loss_ratio = claim_to_premium_ratio(filtered)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Premium", f"${premium:,.2f}")
    c2.metric("Total Claim Amount", f"${claims_amount:,.2f}")
    c3.metric("Claims-to-Premium Ratio", f"{loss_ratio:.2f}")

    claims_policy = filtered["claims"].merge(
        filtered["policies"][["policy_id", "policy_type", "premium_amount"]],
        on="policy_id",
        how="left"
    )

    bar_chart(
        claims_policy, "policy_type", "claim_amount",
        "Claim Amount by Policy Type"
    )

    bar_chart(
        filtered["payments"], "payment_status", "payment_amount",
        "Payment Amount by Payment Status"
    )

    bar_chart(
        filtered["claims"], "damage_severity", "claim_amount",
        "Claim Amount by Damage Severity"
    )

    st.caption(
        "The claims-to-premium ratio is total claim amount divided by total "
        "premium. It is a simple portfolio indicator and is not a complete "
        "actuarial profitability measure."
    )


# ============================================================
# 15. INSIGHTS & ACTIONS
# ============================================================
elif page == "Insights & Actions":
    st.header("💡 Insights & Actions")

    st.subheader("Policy Status Insight")
    st.markdown(
        create_insight(filtered["policies"], "policy_status")
    )

    st.subheader("Premium by Policy Type")
    st.markdown(
        create_insight(
            filtered["policies"], "policy_type", "premium_amount"
        )
    )

    st.subheader("Claims by Type")
    st.markdown(
        create_insight(filtered["claims"], "claim_type")
    )

    st.subheader("Claim Amount by Damage Severity")
    st.markdown(
        create_insight(
            filtered["claims"], "damage_severity", "claim_amount"
        )
    )

    st.subheader("Recommended Actions")
    st.markdown(
        "- Review policy categories with high claim amounts.\n"
        "- Monitor claim approval and settlement times.\n"
        "- Compare claims across vehicle types and regions.\n"
        "- Check missing values and unusual claim amounts before reporting.\n"
        "- Validate findings with the insurance business team before "
        "changing pricing or underwriting rules."
    )


# ============================================================
# 16. DOWNLOAD FILTERED DATA
# ============================================================
st.sidebar.divider()
st.sidebar.subheader("Export Data")

download_table = st.sidebar.selectbox(
    "Choose a dataset to download",
    ["customers", "vehicles", "policies", "claims", "payments"]
)

st.sidebar.download_button(
    label="Download selected CSV",
    data=filtered[download_table].to_csv(index=False).encode("utf-8"),
    file_name=f"{download_table}_filtered.csv",
    mime="text/csv"
)

st.sidebar.caption("Motor Insurance Analytics Dashboard")