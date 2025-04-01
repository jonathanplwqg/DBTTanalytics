import pandas as pd
import numpy as np
import random
from datetime import timedelta, datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import calendar

# Set page configuration
st.set_page_config(layout="wide", page_title="Payment Behavior Dashboard", page_icon="💰")

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #1E3A8A;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .metric-container {
        background-color: #f0f5ff;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        border-left: 5px solid #1E3A8A;
    }
    .insight-box {
        background-color: #f0f9ff;
        border-radius: 10px;
        padding: 15px;
        margin-top: 10px;
        margin-bottom: 20px;
        border-left: 5px solid #0369a1;
    }
    .recommendation-box {
        background-color: #f0f2ff;
        border-radius: 10px;
        padding: 15px;
        margin-top: 10px;
        margin-bottom: 20px;
        border-left: 5px solid #4f46e5;
    }
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
</style>
""", unsafe_allow_html=True)

# Dashboard title with custom styling
st.markdown("<div class='main-header'>💰 Customer Payment Behavior Intelligence Dashboard</div>", unsafe_allow_html=True)

# Initialize session state for data persistence
if 'df' not in st.session_state:
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Load data
    try:
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
        df["days_overdue"] = np.maximum(0, (df["payment_date"] - df["payment_due"]).dt.days)
        df["days_early"] = np.maximum(0, (df["payment_due"] - df["payment_date"]).dt.days)
        df["payment_status"] = df.apply(
            lambda x: "Early" if x["payment_date"] < x["payment_due"] else 
                     "On Time" if x["payment_date"] == x["payment_due"] else "Late", 
            axis=1
        )

        # Feature engineering
        df["days_to_payment"] = (df["payment_due"] - df["date_of_purchase"]).dt.days
        df["day_of_week"] = df["date_of_purchase"].dt.dayofweek
        df["day_name"] = df["date_of_purchase"].dt.day_name()
        df["month"] = df["date_of_purchase"].dt.month
        df["month_name"] = df["date_of_purchase"].dt.month_name()
        df["quarter"] = df["date_of_purchase"].dt.quarter
        df["year"] = df["date_of_purchase"].dt.year
        df["is_weekend"] = df["day_of_week"] >= 5
        df["is_month_end"] = df["date_of_purchase"].dt.day > 25
        
        # Calculate financial impact
        df["order_value"] = df["product_price"] * df["order_quantity"]
        df["overdue_interest"] = df.apply(
            lambda x: x["order_value"] * (x["days_overdue"] * 0.0005) if x["days_overdue"] > 0 else 0, 
            axis=1
        )
        
        # Add risk score based on payment history
        customer_payment_history = df.groupby("customer_id")["on_time"].mean()
        customer_risk_mapping = {
            customer_id: (1 - on_time_rate) * 100 
            for customer_id, on_time_rate in customer_payment_history.items()
        }
        df["risk_score"] = df["customer_id"].map(customer_risk_mapping)
        
        # Calculate cash flow impact
        df["expected_payment_date"] = df["payment_due"]
        df["cash_flow_impact_days"] = (df["payment_date"] - df["expected_payment_date"]).dt.days
        
        # Store in session state
        st.session_state.df = df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
else:
    # Use data from session state
    df = st.session_state.df

# Sidebar for filters and controls
st.sidebar.markdown("### 🔍 Dashboard Filters")

# User filters
years = sorted(df["year"].dropna().unique())
customers = sorted(df["customer_id"].dropna().unique())

col1, col2 = st.sidebar.columns(2)
with col1:
    selected_year = st.selectbox("Select Year", ["All"] + list(years))
with col2:
    selected_customer = st.selectbox("Select Customer", ["All"] + list(customers))

if selected_year != "All" and selected_customer != "All":
    filtered_df_for_products = df[(df["year"] == selected_year) & (df["customer_id"] == selected_customer)]
elif selected_year != "All":
    filtered_df_for_products = df[df["year"] == selected_year]
elif selected_customer != "All":
    filtered_df_for_products = df[df["customer_id"] == selected_customer]
else:
    filtered_df_for_products = df

products = sorted(filtered_df_for_products["product_name"].dropna().unique())
selected_products = st.sidebar.multiselect("Select Product(s)", products, default=products[:min(3, len(products))])

payment_statuses = ["Early", "On Time", "Late"]
selected_statuses = st.sidebar.multiselect("Payment Status", payment_statuses, default=payment_statuses)

# Apply filters
df_filtered = df.copy()
if selected_year != "All":
    df_filtered = df_filtered[df_filtered["year"] == selected_year]
if selected_customer != "All":
    df_filtered = df_filtered[df_filtered["customer_id"] == selected_customer]
if selected_products:
    df_filtered = df_filtered[df_filtered["product_name"].isin(selected_products)]
if selected_statuses:
    df_filtered = df_filtered[df_filtered["payment_status"].isin(selected_statuses)]

# Skip logic if no data remains
if df_filtered.empty:
    st.warning("No records found for the selected filter(s). Please adjust your selection.")
    st.stop()

# === EXECUTIVE SUMMARY ===
st.markdown("<div class='sub-header'>📊 Executive Summary</div>", unsafe_allow_html=True)

# Key metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    on_time_rate = df_filtered["on_time"].mean() * 100
    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
    st.metric(
        "On-Time Payment Rate", 
        f"{on_time_rate:.1f}%",
        f"{on_time_rate - 70:.1f}%" if on_time_rate > 70 else f"{on_time_rate - 70:.1f}%"
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    avg_days_overdue = df_filtered[df_filtered["days_overdue"] > 0]["days_overdue"].mean()
    avg_days_overdue = avg_days_overdue if not np.isnan(avg_days_overdue) else 0
    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
    st.metric(
        "Avg Days Overdue",
        f"{avg_days_overdue:.1f}",
        f"{5 - avg_days_overdue:.1f}" if avg_days_overdue < 5 else f"{5 - avg_days_overdue:.1f}",
        delta_color="inverse"
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    total_overdue_interest = df_filtered["overdue_interest"].sum()
    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
    st.metric(
        "Interest on Late Payments",
        f"${total_overdue_interest:.2f}"
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col4:
    avg_risk_score = df_filtered["risk_score"].mean()
    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
    st.metric(
        "Avg Risk Score (0-100)",
        f"{avg_risk_score:.1f}",
        f"{30 - avg_risk_score:.1f}" if avg_risk_score < 30 else f"{30 - avg_risk_score:.1f}",
        delta_color="inverse"
    )
    st.markdown("</div>", unsafe_allow_html=True)

# === PAYMENT BEHAVIOR OVERVIEW ===
st.markdown("<div class='sub-header'>🔍 Payment Behavior Overview</div>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Payment Status Distribution", "Timeline Analysis"])

with tab1:
    # Create columns for side-by-side charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Payment status distribution pie chart
        payment_status_counts = df_filtered["payment_status"].value_counts().reset_index()
        payment_status_counts.columns = ["Status", "Count"]
        
        fig = px.pie(
            payment_status_counts, 
            values="Count", 
            names="Status",
            title="Payment Status Distribution",
            color="Status",
            color_discrete_map={
                "Early": "#22c55e",
                "On Time": "#3b82f6",
                "Late": "#ef4444"
            },
            hole=0.4
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            margin=dict(t=50, b=100),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Days to payment distribution
        fig = px.histogram(
            df_filtered,
            x="days_until_payment",
            color="payment_status",
            title="Distribution of Days Until Payment",
            color_discrete_map={
                "Early": "#22c55e",
                "On Time": "#3b82f6",
                "Late": "#ef4444"
            },
            barmode="overlay",
            opacity=0.7,
            marginal="box"
        )
        
        fig.add_vline(
            x=df_filtered["days_to_payment"].median(),
            line_dash="dash",
            line_color="black",
            annotation_text="Median Due Date"
        )
        
        fig.update_layout(
            xaxis_title="Days from Purchase to Payment",
            yaxis_title="Number of Transactions",
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            margin=dict(t=50, b=100),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Timeline analysis
    fig = px.scatter(
        df_filtered,
        x="date_of_purchase",
        y="days_until_payment",
        color="payment_status",
        size="order_value",
        hover_name="product_name",
        hover_data=["customer_id", "days_until_payment", "payment_status", "order_value"],
        title="Payment Timeline Analysis",
        color_discrete_map={
            "Early": "#22c55e",
            "On Time": "#3b82f6",
            "Late": "#ef4444"
        },
        size_max=20
    )
    
    fig.add_hline(
        y=df_filtered["days_to_payment"].median(),
        line_dash="dash",
        line_color="black",
        annotation_text="Median Payment Terms"
    )
    
    fig.update_layout(
        xaxis_title="Purchase Date",
        yaxis_title="Days Until Payment",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Provide insights box
    st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
    st.markdown("**🔍 Key Timeline Insights:**")
    
    late_clusters = df_filtered[df_filtered["payment_status"] == "Late"].shape[0]
    common_late_month = df_filtered[df_filtered["payment_status"] == "Late"]["month_name"].mode()[0] if late_clusters > 0 else "N/A"
    high_value_late = df_filtered[df_filtered["payment_status"] == "Late"]["order_value"].mean() > df_filtered["order_value"].mean()
    
    st.markdown(f"""
    - Found **{late_clusters}** late payment clusters in the selected time period
    - Most late payments occur in **{common_late_month}**
    - Late payments {'tend to be higher value orders' if high_value_late else 'are not correlated with order value'}
    - The payment timeline shows {'consistent patterns' if df_filtered['days_until_payment'].std() < 10 else 'high variability'} in payment behavior
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# === CUSTOMER SEGMENTATION BY PAYMENT BEHAVIOR ===
st.markdown("<div class='sub-header'>👥 Customer Payment Behavior Segmentation</div>", unsafe_allow_html=True)

# Calculate customer segments based on payment behavior
customer_metrics = df.groupby("customer_id").agg({
    "on_time": "mean",
    "days_overdue": "mean",
    "days_early": "mean",
    "order_value": "mean",
    "customer_id": "count"
}).rename(columns={"customer_id": "transaction_count"})

customer_metrics["on_time_rate"] = customer_metrics["on_time"] * 100
customer_metrics["avg_days_overdue"] = customer_metrics["days_overdue"]
customer_metrics["avg_days_early"] = customer_metrics["days_early"]
customer_metrics["avg_order_value"] = customer_metrics["order_value"]

# Create customer segments
def assign_segment(row):
    if row["on_time_rate"] >= 90:
        return "Reliable Payers"
    elif row["on_time_rate"] >= 70:
        return "Mostly Reliable"
    elif row["on_time_rate"] >= 50:
        return "Inconsistent Payers"
    else:
        return "High Risk"

customer_metrics["segment"] = customer_metrics.apply(assign_segment, axis=1)
customer_metrics = customer_metrics.reset_index()

# Display only customers in the filtered dataset
filtered_customers = df_filtered["customer_id"].unique()
filtered_customer_metrics = customer_metrics[customer_metrics["customer_id"].isin(filtered_customers)]

# Create a scatter plot of customer payment behavior
fig = px.scatter(
    filtered_customer_metrics,
    x="avg_days_overdue",
    y="on_time_rate",
    color="segment",
    size="avg_order_value",
    hover_name="customer_id",
    text="customer_id",
    title="Customer Segmentation by Payment Behavior",
    color_discrete_map={
        "Reliable Payers": "#22c55e",
        "Mostly Reliable": "#3b82f6",
        "Inconsistent Payers": "#eab308",
        "High Risk": "#ef4444"
    },
    size_max=25
)

fig.update_layout(
    xaxis_title="Average Days Overdue",
    yaxis_title="On-Time Payment Rate (%)",
    height=500
)

fig.update_traces(
    textposition='top center',
    marker=dict(line=dict(width=1, color='DarkSlateGrey'))
)

# Add quadrant lines
fig.add_hline(
    y=70,
    line_dash="dash",
    line_color="gray"
)

fig.add_vline(
    x=5,
    line_dash="dash",
    line_color="gray"
)

# Add annotations for quadrants
fig.add_annotation(
    x=2.5, y=85,
    text="RELIABLE",
    showarrow=False,
    font=dict(size=14, color="#22c55e", family="Arial Black")
)

fig.add_annotation(
    x=10, y=85,
    text="DELAYED BUT PREDICTABLE",
    showarrow=False,
    font=dict(size=14, color="#3b82f6", family="Arial Black")
)

fig.add_annotation(
    x=2.5, y=35,
    text="INCONSISTENT",
    showarrow=False,
    font=dict(size=14, color="#eab308", family="Arial Black")
)

fig.add_annotation(
    x=10, y=35,
    text="HIGH RISK",
    showarrow=False,
    font=dict(size=14, color="#ef4444", family="Arial Black")
)

st.plotly_chart(fig, use_container_width=True)

# Display customer segment summary
segment_summary = filtered_customer_metrics.groupby("segment").agg({
    "customer_id": "count",
    "on_time_rate": "mean",
    "avg_days_overdue": "mean",
    "avg_days_early": "mean",
    "avg_order_value": "mean",
    "transaction_count": "mean"
}).reset_index()

segment_summary.columns = [
    "Segment", "Customer Count", "Avg On-Time Rate (%)", 
    "Avg Days Overdue", "Avg Days Early", "Avg Order Value ($)",
    "Avg Transaction Count"
]

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("#### Customer Segment Summary")
st.dataframe(segment_summary.style.format({
    "Avg On-Time Rate (%)": "{:.1f}",
    "Avg Days Overdue": "{:.1f}",
    "Avg Days Early": "{:.1f}",
    "Avg Order Value ($)": "${:.2f}",
    "Avg Transaction Count": "{:.1f}"
}), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# Business recommendations based on segments
st.markdown("<div class='recommendation-box'>", unsafe_allow_html=True)
st.markdown("#### 💡 Business Recommendations by Segment")

st.markdown("""
- **Reliable Payers:**
  - Offer premium payment terms or early payment discounts
  - Consider for customer loyalty programs
  - Prioritize for new product launches

- **Mostly Reliable:**
  - Monitor for improvement opportunities
  - Send gentle payment reminders
  - Maintain standard payment terms

- **Inconsistent Payers:**
  - Implement automated payment reminders
  - Consider shorter payment terms
  - Review account regularly

- **High Risk:**
  - Require advance payment or deposits
  - Shorten payment terms significantly
  - Consider credit holds for repeat late payers
""")
st.markdown("</div>", unsafe_allow_html=True)

# === PAYMENT TREND ANALYSIS ===
st.markdown("<div class='sub-header'>📈 Payment Trend Analysis</div>", unsafe_allow_html=True)

# Prepare data for trend analysis
trend_data = df.copy()
trend_data["year_month"] = trend_data["date_of_purchase"].dt.to_period("M").astype(str)

# Aggregate by month
monthly_trends = trend_data.groupby("year_month").agg({
    "on_time": "mean",
    "days_overdue": "mean",
    "days_early": "mean",
    "order_value": "sum",
    "date_of_purchase": "count"
}).reset_index()

monthly_trends.columns = [
    "Year-Month", "On-Time Rate", "Avg Days Overdue", 
    "Avg Days Early", "Total Order Value", "Transaction Count"
]

monthly_trends["On-Time Rate"] = monthly_trends["On-Time Rate"] * 100

# Only show data for the filtered year if a specific year is selected
if selected_year != "All":
    monthly_trends = monthly_trends[monthly_trends["Year-Month"].str.startswith(str(selected_year))]

if not monthly_trends.empty:
    # Create trend visualization
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("On-Time Payment Rate Trend", "Average Days Overdue/Early Trend"),
        shared_xaxes=True,
        vertical_spacing=0.1
    )
    
    # Add on-time rate trend
    fig.add_trace(
        go.Scatter(
            x=monthly_trends["Year-Month"],
            y=monthly_trends["On-Time Rate"],
            mode="lines+markers",
            name="On-Time Rate (%)",
            marker=dict(color="#3b82f6"),
            line=dict(width=3)
        ),
        row=1, col=1
    )
    
    # Add target line
    fig.add_trace(
        go.Scatter(
            x=monthly_trends["Year-Month"],
            y=[80] * len(monthly_trends),
            mode="lines",
            name="Target (80%)",
            line=dict(color="green", width=2, dash="dash")
        ),
        row=1, col=1
    )
    
    # Add days overdue trend
    fig.add_trace(
        go.Scatter(
            x=monthly_trends["Year-Month"],
            y=monthly_trends["Avg Days Overdue"],
            mode="lines+markers",
            name="Avg Days Overdue",
            marker=dict(color="#ef4444"),
            line=dict(width=3)
        ),
        row=2, col=1
    )
    
    # Add days early trend
    fig.add_trace(
        go.Scatter(
            x=monthly_trends["Year-Month"],
            y=monthly_trends["Avg Days Early"],
            mode="lines+markers",
            name="Avg Days Early",
            marker=dict(color="#22c55e"),
            line=dict(width=3)
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        margin=dict(t=60, b=100)
    )
    
    fig.update_xaxes(tickangle=45, row=2, col=1)
    
    fig.update_yaxes(title_text="On-Time Rate (%)", row=1, col=1)
    fig.update_yaxes(title_text="Days", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add insight for trend analysis
    last_3_months = monthly_trends.iloc[-3:] if len(monthly_trends) >= 3 else monthly_trends
    recent_trend = "improving" if last_3_months["On-Time Rate"].is_monotonic_increasing else "declining" if last_3_months["On-Time Rate"].is_monotonic_decreasing else "stable"
    
    st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
    st.markdown("**📈 Payment Trend Insights:**")
    st.markdown(f"""
    - The on-time payment rate is **{recent_trend}** over the most recent periods
    - Highest on-time rate: **{monthly_trends["On-Time Rate"].max():.1f}%** in {monthly_trends.loc[monthly_trends["On-Time Rate"].idxmax(), "Year-Month"]}
    - Lowest on-time rate: **{monthly_trends["On-Time Rate"].min():.1f}%** in {monthly_trends.loc[monthly_trends["On-Time Rate"].idxmin(), "Year-Month"]}
    - Average days overdue shows a **{("decreasing" if last_3_months["Avg Days Overdue"].is_monotonic_decreasing else "increasing" if last_3_months["Avg Days Overdue"].is_monotonic_increasing else "stable")}** trend
    """)
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Insufficient trend data for the selected filters.")

# === CASH FLOW IMPACT ANALYSIS ===
st.markdown("<div class='sub-header'>💵 Cash Flow Impact Analysis</div>", unsafe_allow_html=True)

# Calculate cash flow impact by month
cash_flow_data = df.copy()
cash_flow_data["expected_month"] = cash_flow_data["payment_due"].dt.to_period("M").astype(str)
cash_flow_data["actual_month"] = cash_flow_data["payment_date"].dt.to_period("M").astype(str)
cash_flow_data["delayed_to_next_month"] = cash_flow_data["expected_month"] != cash_flow_data["actual_month"]

# Calculate impact by expected month
cash_flow_impact = cash_flow_data.groupby("expected_month").agg({
    "order_value": "sum",
    "delayed_to_next_month": lambda x: (x == True).sum()
}).reset_index()

cash_flow_impact.columns = ["Expected Month", "Expected Value", "Transactions Delayed"]
cash_flow_impact["Delayed Value"] = cash_flow_data[cash_flow_data["delayed_to_next_month"]].groupby("expected_month")["order_value"].sum().reindex(cash_flow_impact["Expected Month"]).fillna(0).values
cash_flow_impact["Delayed Percentage"] = (cash_flow_impact["Delayed Value"] / cash_flow_impact["Expected Value"] * 100).fillna(0)

# If a specific year is selected, filter the data
if selected_year != "All":
    cash_flow_impact = cash_flow_impact[cash_flow_impact["Expected Month"].str.startswith(str(selected_year))]

if not cash_flow_impact.empty:
    # Create cash flow impact visualization
    fig = go.Figure()
    
    # Add expected value bars
    fig.add_trace(
        go.Bar(
            x=cash_flow_impact["Expected Month"],
            y=cash_flow_impact["Expected Value"],
            name="Expected Cash Inflow",
            marker_color="#3b82f6"
        )
    )
    
    # Add delayed value bars
    fig.add_trace(
        go.Bar(
            x=cash_flow_impact["Expected Month"],
            y=cash_flow_impact["Delayed Value"],
            name="Delayed to Next Period",
            marker_color="#ef4444"
        )
    )
    
    # Add delayed percentage line
    fig.add_trace(
        go.Scatter(
            x=cash_flow_impact["Expected Month"],
            y=cash_flow_impact["Delayed Percentage"],
            mode="lines+markers",
            name="% Delayed",
            yaxis="y2",
            line=dict(color="#eab308", width=3)
        )
    )
    
    fig.update_layout(
        title="Monthly Cash Flow Impact of Payment Delays",
        xaxis=dict(title="Expected Payment Month"),
        yaxis=dict(title="Amount ($)", side="left"),
        yaxis2=dict(
            title="Delayed Percentage (%)",
            side="right",
            overlaying="y",
            range=[0, max(100, cash_flow_impact["Delayed Percentage"].max() * 1.1)]
        ),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(t=60, b=100),
        height=500
    )
    
    fig.update_xaxes(tickangle=45)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add cash flow impact insight
    total_delayed = cash_flow_impact["Delayed Value"].sum()
    avg_delayed_pct = cash_flow_impact["Delayed Percentage"].mean()
    worst_month = cash_flow_impact.loc[cash_flow_impact["Delayed Percentage"].idxmax(), "Expected Month"]
    worst_month_pct = cash_flow_impact["Delayed Percentage"].max()
    
    st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
    st.markdown("**💵 Cash Flow Impact Insights:**")
    st.markdown(f"""
    - Total value delayed to subsequent periods: **${total_delayed:,.2f}**
    - Average monthly delay percentage: **{avg_delayed_pct:.1f}%**
    - Highest impact period: **{worst_month}** with **{worst_month_pct:.1f}%** of expected value delayed
    - This delayed cash creates potential liquidity challenges and may necessitate additional working capital
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Recommendations based on cash flow impact
    st.markdown("<div class='recommendation-box'>", unsafe_allow_html=True)
    st.markdown("**💡 Cash Flow Improvement Recommendations:**")
    st.markdown(f"""
    - **Implement Early Payment Incentives:** Offer discounts of 1-2% for payments made within 10 days
    - **Revise Payment Terms:** Consider shortening payment terms from {df_filtered["days_to_payment"].median():.0f} days to {max(15, df_filtered["days_to_payment"].median() - 15):.0f} days
    - **Automate Payment Reminders:** Send automated reminders at 10, 5, and 1 days before due date
    - **Focus Collection Efforts:** Prioritize the largest {worst_month} receivables to improve cash flow
    """)
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Insufficient cash flow data for the selected filters.")

# === PRODUCT PAYMENT BEHAVIOR ANALYSIS ===
st.markdown("<div class='sub-header'>🛍️ Product-Specific Payment Patterns</div>", unsafe_allow_html=True)

# Analyze payment behavior by product
product_payment = df_filtered.groupby("product_name").agg({
    "on_time": "mean",
    "days_until_payment": "mean",
    "days_overdue": "mean",
    "order_value": ["mean", "sum"],
    "customer_id": "nunique"
}).reset_index()

product_payment.columns = [
    "Product", "On-Time Rate", "Avg Days to Payment", 
    "Avg Days Overdue", "Avg Order Value", "Total Sales", "Unique Customers"
]

product_payment["On-Time Rate"] = product_payment["On-Time Rate"] * 100

if not product_payment.empty and len(product_payment) >= 3:
    # Create product payment behavior visualization
    fig = px.scatter(
        product_payment,
        x="Avg Days to Payment",
        y="On-Time Rate",
        size="Total Sales",
        color="Avg Days Overdue",
        hover_name="Product",
        text="Product",
        title="Payment Behavior by Product",
        color_continuous_scale="Reds",
        size_max=50
    )
    
    fig.update_traces(
        textposition='top center',
        marker=dict(line=dict(width=1, color='DarkSlateGrey'))
    )
    
    fig.update_layout(
        xaxis_title="Average Days to Payment",
        yaxis_title="On-Time Payment Rate (%)",
        coloraxis_colorbar=dict(title="Avg Days<br>Overdue"),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add product insights
    best_product = product_payment.loc[product_payment["On-Time Rate"].idxmax(), "Product"]
    worst_product = product_payment.loc[product_payment["On-Time Rate"].idxmin(), "Product"]
    
    st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
    st.markdown("**🛍️ Product Payment Insights:**")
    st.markdown(f"""
    - Best payment performance: **{best_product}** with **{product_payment.loc[product_payment["On-Time Rate"].idxmax(), "On-Time Rate"]:.1f}%** on-time rate
    - Worst payment performance: **{worst_product}** with **{product_payment.loc[product_payment["On-Time Rate"].idxmin(), "On-Time Rate"]:.1f}%** on-time rate
    - Products with higher prices tend to have **{("better" if pd.Series.corr(product_payment["Avg Order Value"], product_payment["On-Time Rate"]) > 0 else "worse")}** payment performance
    - Product-specific payment behavior variations suggest different customer segments or product-specific terms may be beneficial
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display product payment behavior details
    with st.expander("📊 View Product Payment Details"):
        st.dataframe(product_payment.sort_values("On-Time Rate", ascending=False).style.format({
            "On-Time Rate": "{:.1f}%",
            "Avg Days to Payment": "{:.1f}",
            "Avg Days Overdue": "{:.1f}",
            "Avg Order Value": "${:.2f}",
            "Total Sales": "${:.2f}"
        }), use_container_width=True)
else:
    st.info("Insufficient product data for the selected filters.")

# === SEASONAL PAYMENT BEHAVIOR PATTERNS ===
st.markdown("<div class='sub-header'>📅 Seasonal Payment Behavior Analysis</div>", unsafe_allow_html=True)

# Create monthly/quarterly payment behavior analysis
# By Month
monthly_behavior = df.groupby("month_name").agg({
    "on_time": "mean",
    "days_overdue": "mean",
    "days_early": "mean",
    "order_value": "mean"
}).reset_index()

monthly_behavior["on_time_rate"] = monthly_behavior["on_time"] * 100
monthly_behavior["month_num"] = monthly_behavior["month_name"].map({month: i for i, month in enumerate(calendar.month_name) if month})
monthly_behavior = monthly_behavior.sort_values("month_num")

# By Quarter
quarterly_behavior = df.groupby("quarter").agg({
    "on_time": "mean",
    "days_overdue": "mean",
    "days_early": "mean",
    "order_value": "mean"
}).reset_index()

quarterly_behavior["on_time_rate"] = quarterly_behavior["on_time"] * 100
quarterly_behavior = quarterly_behavior.sort_values("quarter")
quarterly_behavior["quarter_label"] = "Q" + quarterly_behavior["quarter"].astype(str)

# Create tabs for different seasonal views
tab1, tab2 = st.tabs(["Monthly Patterns", "Quarterly Patterns"])

with tab1:
    # Monthly pattern visualization
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("On-Time Payment Rate by Month", "Days Overdue/Early by Month"),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Add on-time rate bars
    fig.add_trace(
        go.Bar(
            x=monthly_behavior["month_name"],
            y=monthly_behavior["on_time_rate"],
            name="On-Time Rate (%)",
            marker_color="#3b82f6"
        ),
        row=1, col=1
    )
    
    # Add days overdue bars
    fig.add_trace(
        go.Bar(
            x=monthly_behavior["month_name"],
            y=monthly_behavior["days_overdue"],
            name="Avg Days Overdue",
            marker_color="#ef4444"
        ),
        row=1, col=2
    )
    
    # Add days early bars
    fig.add_trace(
        go.Bar(
            x=monthly_behavior["month_name"],
            y=monthly_behavior["days_early"],
            name="Avg Days Early",
            marker_color="#22c55e"
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=450,
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(t=60, b=100)
    )
    
    fig.update_xaxes(tickangle=45, row=1, col=1)
    fig.update_xaxes(tickangle=45, row=1, col=2)
    
    fig.update_yaxes(title_text="On-Time Rate (%)", row=1, col=1)
    fig.update_yaxes(title_text="Days", row=1, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add seasonal insights
    best_month = monthly_behavior.loc[monthly_behavior["on_time_rate"].idxmax(), "month_name"]
    worst_month = monthly_behavior.loc[monthly_behavior["on_time_rate"].idxmin(), "month_name"]
    
    st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
    st.markdown("**📅 Monthly Payment Pattern Insights:**")
    st.markdown(f"""
    - Best payment month: **{best_month}** ({monthly_behavior.loc[monthly_behavior["on_time_rate"].idxmax(), "on_time_rate"]:.1f}% on-time)
    - Worst payment month: **{worst_month}** ({monthly_behavior.loc[monthly_behavior["on_time_rate"].idxmin(), "on_time_rate"]:.1f}% on-time)
    - Seasonal variation of **{monthly_behavior["on_time_rate"].max() - monthly_behavior["on_time_rate"].min():.1f}%** points between best and worst months
    - The data shows {"consistent" if monthly_behavior["on_time_rate"].std() < 5 else "significant"} seasonal patterns in payment behavior
    """)
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    # Quarterly pattern visualization
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("On-Time Payment Rate by Quarter", "Days Overdue/Early by Quarter"),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Add on-time rate bars
    fig.add_trace(
        go.Bar(
            x=quarterly_behavior["quarter_label"],
            y=quarterly_behavior["on_time_rate"],
            name="On-Time Rate (%)",
            marker_color="#3b82f6"
        ),
        row=1, col=1
    )
    
    # Add days overdue bars
    fig.add_trace(
        go.Bar(
            x=quarterly_behavior["quarter_label"],
            y=quarterly_behavior["days_overdue"],
            name="Avg Days Overdue",
            marker_color="#ef4444"
        ),
        row=1, col=2
    )
    
    # Add days early bars
    fig.add_trace(
        go.Bar(
            x=quarterly_behavior["quarter_label"],
            y=quarterly_behavior["days_early"],
            name="Avg Days Early",
            marker_color="#22c55e"
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=450,
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(t=60, b=100)
    )
    
    fig.update_xaxes(title_text="Quarter", row=1, col=1)
    fig.update_xaxes(title_text="Quarter", row=1, col=2)
    
    fig.update_yaxes(title_text="On-Time Rate (%)", row=1, col=1)
    fig.update_yaxes(title_text="Days", row=1, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add quarterly insights
    best_quarter = quarterly_behavior.loc[quarterly_behavior["on_time_rate"].idxmax(), "quarter_label"]
    worst_quarter = quarterly_behavior.loc[quarterly_behavior["on_time_rate"].idxmin(), "quarter_label"]
    
    st.markdown("<div class='recommendation-box'>", unsafe_allow_html=True)
    st.markdown("**💡 Seasonal Payment Strategy Recommendations:**")
    st.markdown(f"""
    - **{worst_month}/{worst_quarter} Strategy:** Implement stricter payment terms and more frequent reminders during typically problematic periods
    - **Year-End Planning:** Adjust cash flow forecasts to account for Q4 typically having {"better" if quarterly_behavior.loc[quarterly_behavior["quarter"] == 4, "on_time_rate"].values[0] > quarterly_behavior["on_time_rate"].mean() else "worse"} payment performance
    - **Seasonal Incentives:** Offer early payment incentives during historically problematic months to improve cash flow
    - **Resource Allocation:** Increase collection resources during {worst_month} to mitigate seasonal payment delays
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# === PAYMENT RISK SCORE DASHBOARD ===
st.markdown("<div class='sub-header'>⚠️ Payment Risk Assessment</div>", unsafe_allow_html=True)

# Calculate risk score by customer
risk_data = df.groupby("customer_id").agg({
    "on_time": "mean",
    "days_overdue": "mean",
    "order_value": ["mean", "sum"],
    "date_of_purchase": "max"
}).reset_index()

risk_data.columns = [
    "Customer ID", "On-Time Rate", "Avg Days Overdue", 
    "Avg Order Value", "Total Value", "Last Purchase Date"
]

# Calculate risk score (lower on-time rate and higher days overdue = higher risk)
risk_data["Risk Score"] = ((1 - risk_data["On-Time Rate"]) * 70) + (np.minimum(risk_data["Avg Days Overdue"], 30) / 30 * 30)
risk_data["Risk Category"] = pd.cut(
    risk_data["Risk Score"], 
    bins=[0, 20, 40, 60, 100],
    labels=["Low Risk", "Moderate Risk", "High Risk", "Very High Risk"]
)

# Filter to customers in our filtered dataset
filtered_customers = df_filtered["customer_id"].unique()
filtered_risk_data = risk_data[risk_data["Customer ID"].isin(filtered_customers)]

if not filtered_risk_data.empty:
    # Create risk score visualization
    fig = px.bar(
        filtered_risk_data.sort_values("Risk Score", ascending=False),
        x="Customer ID",
        y="Risk Score",
        color="Risk Category",
        title="Customer Payment Risk Assessment",
        color_discrete_map={
            "Low Risk": "#22c55e",
            "Moderate Risk": "#3b82f6",
            "High Risk": "#eab308",
            "Very High Risk": "#ef4444"
        },
        text="Risk Score"
    )
    
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    
    fig.update_layout(
        xaxis_title="Customer ID",
        yaxis_title="Risk Score (0-100)",
        height=450,
        yaxis=dict(range=[0, 100]),
        xaxis=dict(categoryorder="total descending")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk category distribution
    risk_distribution = filtered_risk_data["Risk Category"].value_counts().reset_index()
    risk_distribution.columns = ["Risk Category", "Count"]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        fig = px.pie(
            risk_distribution, 
            values="Count",
            names="Risk Category",
            color="Risk Category",
            title="Risk Category Distribution",
            color_discrete_map={
                "Low Risk": "#22c55e",
                "Moderate Risk": "#3b82f6",
                "High Risk": "#eab308",
                "Very High Risk": "#ef4444"
            },
            hole=0.4
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        fig.update_layout(
            height=350,
            margin=dict(t=50, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk recommendations
        st.markdown("<div class='recommendation-box'>", unsafe_allow_html=True)
        st.markdown("**⚠️ Risk Management Recommendations:**")
        
        high_risk_count = len(filtered_risk_data[filtered_risk_data["Risk Category"].isin(["High Risk", "Very High Risk"])])
        high_risk_value = filtered_risk_data[filtered_risk_data["Risk Category"].isin(["High Risk", "Very High Risk"])]["Total Value"].sum()
        
        st.markdown(f"""
        - **{high_risk_count} high-risk customers** represent **${high_risk_value:,.2f}** in order value
        - **Action plan for high-risk customers:**
          - Require deposits or advance payment for orders over ${filtered_risk_data["Avg Order Value"].mean():.0f}
          - Implement credit holds for "Very High Risk" customers
          - Assign dedicated account managers to large "High Risk" accounts
          - Consider factoring invoices for highest risk accounts
        
        - **Proactive monitoring:**
          - Weekly review of all "High Risk" and "Very High Risk" accounts
          - Monthly review of payment pattern changes for "Moderate Risk" accounts
          - Quarterly risk score recalculation and strategy adjustment
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Detailed risk data table
    with st.expander("📊 View Detailed Risk Assessment Data"):
        st.dataframe(filtered_risk_data.sort_values("Risk Score", ascending=False).style.format({
            "On-Time Rate": "{:.1%}",
            "Avg Days Overdue": "{:.1f}",
            "Avg Order Value": "${:.2f}",
            "Total Value": "${:.2f}",
            "Risk Score": "{:.1f}"
        }), use_container_width=True)
else:
    st.info("Insufficient risk data for the selected filters.")

# === PAYMENT TERMS OPTIMIZATION ===
st.markdown("<div class='sub-header'>⚙️ Payment Terms Optimization</div>", unsafe_allow_html=True)

# Analyze payment terms effectiveness
terms_analysis = df.copy()
terms_analysis["terms_group"] = pd.cut(
    terms_analysis["days_to_payment"],
    bins=[0, 15, 30, 45, 60, 120],
    labels=["0-15 days", "16-30 days", "31-45 days", "46-60 days", "61+ days"]
)

# Calculate metrics by terms group
terms_effectiveness = terms_analysis.groupby("terms_group").agg({
    "on_time": "mean",
    "days_overdue": "mean",
    "order_value": ["mean", "sum"],
    "customer_id": "nunique"
}).reset_index()

terms_effectiveness.columns = [
    "Payment Terms", "On-Time Rate", "Avg Days Overdue", 
    "Avg Order Value", "Total Order Value", "Customer Count"
]

terms_effectiveness["On-Time Rate"] = terms_effectiveness["On-Time Rate"] * 100

# Create visualization of payment terms effectiveness
fig = px.bar(
    terms_effectiveness,
    x="Payment Terms",
    y="On-Time Rate",
    color="Avg Days Overdue",
    title="Payment Terms Effectiveness Analysis",
    text="On-Time Rate",
    color_continuous_scale="Reds",
    hover_data=["Customer Count", "Avg Order Value", "Total Order Value"]
)

fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")

fig.update_layout(
    xaxis_title="Payment Terms",
    yaxis_title="On-Time Payment Rate (%)",
    coloraxis_colorbar=dict(title="Avg Days<br>Overdue"),
    height=450
)

st.plotly_chart(fig, use_container_width=True)

# Terms optimization insights
best_terms = terms_effectiveness.loc[terms_effectiveness["On-Time Rate"].idxmax(), "Payment Terms"]
worst_terms = terms_effectiveness.loc[terms_effectiveness["On-Time Rate"].idxmin(), "Payment Terms"]

st.markdown("<div class='recommendation-box'>", unsafe_allow_html=True)
st.markdown("**⚙️ Payment Terms Optimization Recommendations:**")
st.markdown(f"""
- **Optimal Terms:** The data suggests **{best_terms}** payment terms yield the best on-time payment rate ({terms_effectiveness.loc[terms_effectiveness["On-Time Rate"].idxmax(), "On-Time Rate"]:.1f}%)
- **Terms to Avoid:** **{worst_terms}** terms show worst performance ({terms_effectiveness.loc[terms_effectiveness["On-Time Rate"].idxmin(), "On-Time Rate"]:.1f}%)

**Recommended Terms Strategy:**
- Standard accounts: {best_terms}
- New customers: 50% advance payment, 50% net 15
- High-risk accounts: Advance payment or net 10 with credit card authorization
- High-value reliable accounts: Consider {terms_effectiveness["Payment Terms"].iloc[1]} with early payment incentives

**Implementation Plan:**
1. Phase in new terms with new customers immediately
2. Transition existing customers over 3-month period
3. Communicate changes with clear rationale and benefits
4. Monitor results and adjust as needed after 90 days
""")
st.markdown("</div>", unsafe_allow_html=True)

# === ML MODELS PERFORMANCE (If used in prediction) ===
if "predicted_days" in df_filtered.columns and "predicted_on_time" in df_filtered.columns:
    st.markdown("<div class='sub-header'>🤖 Payment Prediction Model Performance</div>", unsafe_allow_html=True)
    
    # Calculate model accuracy metrics
    mae = np.mean(np.abs(df_filtered["days_until_payment"] - df_filtered["predicted_days"]))
    accuracy = np.mean(df_filtered["on_time"] == df_filtered["predicted_on_time"]) * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.metric("Days Prediction Error (MAE)", f"{mae:.1f} days")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.metric("On-Time Prediction Accuracy", f"{accuracy:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Create prediction vs actual visualization
    fig = px.scatter(
        df_filtered,
        x="predicted_days",
        y="days_until_payment",
        color="payment_status",
        title="Predicted vs. Actual Payment Days",
        color_discrete_map={
            "Early": "#22c55e",
            "On Time": "#3b82f6",
            "Late": "#ef4444"
        },
        opacity=0.7
    )
    
    # Add perfect prediction line
    fig.add_trace(
        go.Scatter(
            x=[df_filtered["predicted_days"].min(), df_filtered["predicted_days"].max()],
            y=[df_filtered["predicted_days"].min(), df_filtered["predicted_days"].max()],
            mode="lines",
            name="Perfect Prediction",
            line=dict(color="black", width=2, dash="dash")
        )
    )
    
    fig.update_layout(
        xaxis_title="Predicted Days Until Payment",
        yaxis_title="Actual Days Until Payment",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Model insights
    st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
    st.markdown("**🤖 Payment Prediction Insights:**")
    st.markdown(f"""
    - The model predicts payment timing with an average error of **{mae:.1f} days**
    - On-time/late prediction accuracy is **{accuracy:.1f}%**
    - The model tends to {"overestimate" if df_filtered["predicted_days"].mean() > df_filtered["days_until_payment"].mean() else "underestimate"} payment days by {abs(df_filtered["predicted_days"].mean() - df_filtered["days_until_payment"].mean()):.1f} days on average
    - Model performance suggests {"reliable predictions for operational planning" if accuracy > 70 else "the need for more data or model refinement"}
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# === FOOTER WITH LAST UPDATE TIME ===
st.markdown("---")
st.caption(f"Customer Payment Behavior Intelligence Dashboard | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption("Developed for business decision support | Data source: 3000_Varied_Product_Transactions.csv")