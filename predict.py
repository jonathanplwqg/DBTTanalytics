import pandas as pd
import numpy as np
import random
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib

st.set_page_config(layout="wide")
st.title("📊 Customer Payment Behavior Insights Dashboard")

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Load data
csv_file_path = "3000_Varied_Product_Transactions.csv"
df = pd.read_csv(csv_file_path, parse_dates=["date_of_purchase", "payment_due"])

# Simulate varied customer behavior
def simulate_payment_date(row):
    behavior_type = random.choices(
        ['early', 'ontime', 'late'],
        weights=[0.2, 0.5, 0.3],
        k=1
    )[0]
    if behavior_type == 'early':
        return row["payment_due"] - timedelta(days=random.randint(3, 10))
    elif behavior_type == 'ontime':
        return row["payment_due"] + timedelta(days=random.randint(-2, 2))
    else:
        return row["payment_due"] + timedelta(days=random.randint(3, 20))

df["payment_date"] = df.apply(simulate_payment_date, axis=1)
df["on_time"] = (df["payment_date"] <= df["payment_due"]).astype(int)
df["days_until_payment"] = (df["payment_date"] - df["date_of_purchase"]).dt.days

# Feature engineering
df["days_to_payment"] = (df["payment_due"] - df["date_of_purchase"]).dt.days
df["day_of_week"] = df["date_of_purchase"].dt.dayofweek
df["month"] = df["date_of_purchase"].dt.month
df["year"] = df["date_of_purchase"].dt.year
df["is_weekend"] = df["day_of_week"] >= 5
df["is_month_end"] = df["date_of_purchase"].dt.day > 25

# User filters
years = sorted(df["year"].dropna().unique())
customers = sorted(df["customer_id"].dropna().unique())
selected_year = st.selectbox("Select Year", [None] + years)
selected_customer = st.selectbox("Select Customer", [None] + customers)

if selected_year and selected_customer:
    filtered_df_for_products = df[(df["year"] == selected_year) & (df["customer_id"] == selected_customer)]
    products = sorted(filtered_df_for_products["product_name"].dropna().unique())
else:
    products = sorted(df["product_name"].dropna().unique())

selected_products = st.multiselect("Select Product(s)", products, key="product_filter")

# Apply filters
df_filtered = df.copy()
if selected_year is not None:
    df_filtered = df_filtered[df_filtered["year"] == selected_year]
if selected_customer is not None:
    df_filtered = df_filtered[df_filtered["customer_id"] == selected_customer]
if selected_products:
    df_filtered = df_filtered[df_filtered["product_name"].isin(selected_products)]

# Payment accuracy heatmap based on filtered data
st.subheader("📊 Payment Accuracy of Filtered Customers")
payment_accuracy_filtered = df_filtered.groupby("customer_id")["on_time"].mean().reset_index()
payment_accuracy_filtered["accuracy_percent"] = payment_accuracy_filtered["on_time"] * 100
pivot_accuracy = payment_accuracy_filtered.set_index("customer_id")[["accuracy_percent"]]

fig_overview, ax_overview = plt.subplots(figsize=(8, max(2, len(pivot_accuracy) * 0.3)))
sns.heatmap(pivot_accuracy,
            annot=True,
            cmap="RdYlGn",
            center=50,
            fmt=".1f",
            linewidths=0.5,
            linecolor='gray',
            cbar_kws={'label': 'Payment Accuracy (%)'},
            ax=ax_overview,
            annot_kws={"fontsize": 8})

ax_overview.set_title("📊 Payment Accuracy (%) per Customer", fontsize=14, pad=12)
ax_overview.set_xlabel("")
ax_overview.set_ylabel("Customer ID")
plt.xticks(rotation=0)
plt.yticks(rotation=0)
st.pyplot(fig_overview)

# Skip ML logic if no data remains
if df_filtered.empty:
    st.warning("No records found for the selected filter(s). Please adjust.")
    st.stop()

# Prepare model input
drop_cols = ["transact_id", "payment_due", "payment_date", "customer_name"]
df_model = df_filtered.drop(columns=[col for col in drop_cols if col in df_filtered.columns])
df_model = pd.get_dummies(df_model, drop_first=True)
datetime_cols = df_model.select_dtypes(include=["datetime64"]).columns

# Regression model
X_reg = df_model.drop(columns=["on_time", "days_until_payment"] + list(datetime_cols))
y_reg = df_filtered["days_until_payment"]
if len(X_reg) >= 2:
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)
    reg = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=10, random_state=42)
    reg.fit(X_train_r, y_train_r)
    reg_all_preds = reg.predict(X_reg)
    df_filtered["predicted_days"] = reg_all_preds
    df_filtered["payment_diff"] = df_filtered["days_until_payment"] - df_filtered["predicted_days"]

# Classification model
df_clf = df_model.drop(columns=["days_until_payment"] + list(datetime_cols))
df_majority = df_clf[df_clf.on_time == 1]
df_minority = df_clf[df_clf.on_time == 0]
if len(df_minority) > 0 and len(df_majority) > 0:
    df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
    df_balanced = pd.concat([df_majority, df_minority_upsampled])
else:
    df_balanced = df_clf.copy()

X_class = df_balanced.drop("on_time", axis=1)
y_class = df_balanced["on_time"]
clf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
clf_model.fit(X_class, y_class)

X_all_class = df_model.drop(columns=["on_time", "days_until_payment"] + list(datetime_cols))
df_filtered["predicted_on_time"] = clf_model.predict(X_all_class)

# --------------------
# TRANSACTION-LEVEL VISUALIZATION
# --------------------

st.subheader("🔍 Detailed Product Transactions")

if not df_filtered.empty:
    df_filtered = df_filtered.sort_values("date_of_purchase")
    df_filtered["on_time_label"] = df_filtered["on_time"].map({1: "On Time", 0: "Late"})

    fig_detail, ax_detail = plt.subplots(figsize=(12, 4))
    for i, row in df_filtered.iterrows():
        ax_detail.text(
            row["date_of_purchase"],
            row["days_until_payment"] + 0.5,
            f"{row['product_name']}\n{row['on_time_label']}\n{row['days_until_payment']}d",
            fontsize=7,
            ha='center',
            va='bottom',
            rotation=0,
            bbox=dict(boxstyle="round,pad=0.2", edgecolor='gray', facecolor='white', alpha=0.5)
        )

    sns.scatterplot(data=df_filtered,
                    x="date_of_purchase",
                    y="days_until_payment",
                    hue="on_time_label",
                    palette={"On Time": "green", "Late": "red"},
                    s=100,
                    ax=ax_detail)

    title_parts = [f"📅 Transactions in {selected_year}" if selected_year else ""]
    if selected_customer:
        title_parts.append(f"Customer: {selected_customer}")
    if selected_products:
        title_parts.append("Products: " + ", ".join(selected_products))

    ax_detail.set_title(" | ".join(title_parts), fontsize=13)
    ax_detail.set_ylabel("Days Until Payment")
    ax_detail.set_xlabel("Transaction Date")
    ax_detail.legend(title="Payment Status")
    plt.xticks(rotation=45)
    st.pyplot(fig_detail)
else:
    st.warning("No transaction data found for the selected filters.")

st.success("✅ Dashboard loaded. Filter to explore accuracy and payment timelines.")