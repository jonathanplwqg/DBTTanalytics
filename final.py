import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import calendar
from datetime import datetime

st.set_page_config(layout="wide", page_title="Manufacturing Intelligence Dashboard")
st.title("🏭 Product Manufacturing & Profitability Intelligence Dashboard")

# === LOAD DATA ===
@st.cache_data  # Cache the data loading for better performance
def load_data():
    csv_path = "3000_Varied_Product_Transactions.csv"
    df = pd.read_csv(csv_path, parse_dates=["date_of_purchase", "payment_due"])
    
    # Add time dimensions for better analysis
    df["year"] = df["date_of_purchase"].dt.year
    df["month"] = df["date_of_purchase"].dt.month
    df["quarter"] = df["date_of_purchase"].dt.quarter
    df["month_name"] = df["date_of_purchase"].dt.month_name()
    
    # COMPUTE PROFIT METRICS
    df["total_revenue"] = df["product_price"] * df["order_quantity"]
    df["total_cost"] = df["product_cost"] * df["order_quantity"]
    df["total_profit"] = df["total_revenue"] - df["total_cost"]
    df["profit_margin"] = ((df["product_price"] - df["product_cost"]) / df["product_price"]) * 100
    df["cost_margin"] = ((df["product_price"] - df["product_cost"]) / df["product_cost"]) * 100
    
    return df

df = load_data()

# === SIDEBAR FILTERS ===
st.sidebar.header("Dashboard Filters")
# Year filter
years = sorted(df["year"].unique())
selected_years = st.sidebar.multiselect("Select Years", years, default=years)

# Apply filters
filtered_df = df[df["year"].isin(selected_years)]

if filtered_df.empty:
    st.warning("No data available with the current filter settings.")
    st.stop()

# === EXECUTIVE SUMMARY METRICS ===
st.header("📊 Executive Summary")

# Create three columns for KPIs
col1, col2, col3 = st.columns(3)

with col1:
    total_revenue = filtered_df["total_revenue"].sum()
    prev_period_revenue = df[df["year"].isin([y for y in years if y < max(selected_years)])]["total_revenue"].sum() if len(selected_years) > 0 else 0
    revenue_change = ((total_revenue - prev_period_revenue) / prev_period_revenue) * 100 if prev_period_revenue > 0 else 0
    
    st.metric(
        "Total Revenue", 
        f"${total_revenue:,.2f}", 
        f"{revenue_change:.1f}%" if prev_period_revenue > 0 else None
    )
    
    total_profit = filtered_df["total_profit"].sum()
    prev_period_profit = df[df["year"].isin([y for y in years if y < max(selected_years)])]["total_profit"].sum() if len(selected_years) > 0 else 0
    profit_change = ((total_profit - prev_period_profit) / prev_period_profit) * 100 if prev_period_profit > 0 else 0
    
    st.metric(
        "Total Profit", 
        f"${total_profit:,.2f}", 
        f"{profit_change:.1f}%" if prev_period_profit > 0 else None
    )

with col2:
    avg_margin = (filtered_df["total_profit"].sum() / filtered_df["total_revenue"].sum()) * 100
    prev_period_margin = (prev_period_profit / prev_period_revenue) * 100 if prev_period_revenue > 0 else 0
    margin_change = avg_margin - prev_period_margin if prev_period_revenue > 0 else 0
    
    st.metric(
        "Profit Margin", 
        f"{avg_margin:.2f}%", 
        f"{margin_change:.1f}%" if prev_period_revenue > 0 else None
    )
    
    total_volume = filtered_df["order_quantity"].sum()
    prev_period_volume = df[df["year"].isin([y for y in years if y < max(selected_years)])]["order_quantity"].sum() if len(selected_years) > 0 else 0
    volume_change = ((total_volume - prev_period_volume) / prev_period_volume) * 100 if prev_period_volume > 0 else 0
    
    st.metric(
        "Total Sales Volume", 
        f"{total_volume:,.0f}", 
        f"{volume_change:.1f}%" if prev_period_volume > 0 else None
    )

with col3:
    top_product_by_revenue = filtered_df.groupby('product_name')['total_revenue'].sum().idxmax()
    top_product_by_margin = filtered_df.groupby('product_name')['profit_margin'].mean().idxmax()
    
    st.metric("Top Product by Revenue", top_product_by_revenue)
    st.metric("Top Product by Margin", top_product_by_margin)

# === PROFITABILITY TRENDS ===
st.header("📈 Profitability Trends")

# Create tabs for different time granularities
tab1, tab2 = st.tabs(["Yearly Trends", "Quarterly Trends"])

with tab1:
    # Yearly Trends
    yearly_data = filtered_df.groupby("year").agg({
        "total_revenue": "sum",
        "total_profit": "sum",
        "order_quantity": "sum"
    }).reset_index()
    
    yearly_data["profit_margin"] = (yearly_data["total_profit"] / yearly_data["total_revenue"]) * 100
    
    # Create a plotly figure for better interactivity
    fig = go.Figure()
    
    # Add revenue bars
    fig.add_trace(
        go.Bar(
            x=yearly_data["year"],
            y=yearly_data["total_revenue"],
            name="Revenue",
            marker_color="#636EFA"
        )
    )
    
    # Add profit bars
    fig.add_trace(
        go.Bar(
            x=yearly_data["year"],
            y=yearly_data["total_profit"],
            name="Profit",
            marker_color="#00CC96"
        )
    )
    
    # Add margin line
    fig.add_trace(
        go.Scatter(
            x=yearly_data["year"],
            y=yearly_data["profit_margin"],
            name="Profit Margin %",
            yaxis="y2",
            line=dict(color="#EF553B", width=3)
        )
    )
    
    # Update layout for dual y-axis
    fig.update_layout(
        title="Yearly Revenue, Profit, and Margin Trends",
        xaxis=dict(title="Year"),
        yaxis=dict(title="Amount ($)", side="left"),
        yaxis2=dict(title="Profit Margin (%)", side="right", overlaying="y", rangemode="tozero"),
        legend=dict(x=0.01, y=0.99),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Quarterly Trends
    quarterly_data = filtered_df.groupby(["year", "quarter"]).agg({
        "total_revenue": "sum",
        "total_profit": "sum",
        "order_quantity": "sum"
    }).reset_index()
    
    quarterly_data["profit_margin"] = (quarterly_data["total_profit"] / quarterly_data["total_revenue"]) * 100
    quarterly_data["period"] = quarterly_data["year"].astype(str) + "-Q" + quarterly_data["quarter"].astype(str)
    
    # Create a plotly figure
    fig = go.Figure()
    
    # Add revenue bars
    fig.add_trace(
        go.Bar(
            x=quarterly_data["period"],
            y=quarterly_data["total_revenue"],
            name="Revenue",
            marker_color="#636EFA"
        )
    )
    
    # Add profit bars
    fig.add_trace(
        go.Bar(
            x=quarterly_data["period"],
            y=quarterly_data["total_profit"],
            name="Profit",
            marker_color="#00CC96"
        )
    )
    
    # Add margin line
    fig.add_trace(
        go.Scatter(
            x=quarterly_data["period"],
            y=quarterly_data["profit_margin"],
            name="Profit Margin %",
            yaxis="y2",
            line=dict(color="#EF553B", width=3)
        )
    )
    
    # Update layout for dual y-axis
    fig.update_layout(
        title="Quarterly Revenue, Profit, and Margin Trends",
        xaxis=dict(title="Quarter"),
        yaxis=dict(title="Amount ($)", side="left"),
        yaxis2=dict(title="Profit Margin (%)", side="right", overlaying="y", rangemode="tozero"),
        legend=dict(x=0.01, y=0.99),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# === PRODUCT PORTFOLIO ANALYSIS (BCG MATRIX) ===
st.header("🎯 Product Portfolio Analysis")

# Calculate metrics for BCG matrix
product_metrics = filtered_df.groupby("product_name").agg({
    "total_revenue": "sum",
    "total_profit": "sum",
    "profit_margin": "mean",
    "order_quantity": "sum"
}).reset_index()

# Calculate market share and growth for each product
total_market = product_metrics["total_revenue"].sum()
product_metrics["market_share"] = product_metrics["total_revenue"] / total_market * 100

# If we have multiple years, calculate growth rate
if len(selected_years) > 1:
    # Calculate growth rate for each product
    product_growth = []
    
    for product in product_metrics["product_name"]:
        product_data = filtered_df[filtered_df["product_name"] == product]
        product_yearly = product_data.groupby("year")["total_revenue"].sum().reset_index()
        
        if len(product_yearly) > 1:
            # Calculate year-over-year growth
            product_yearly["growth"] = product_yearly["total_revenue"].pct_change() * 100
            avg_growth = product_yearly["growth"].dropna().mean()
        else:
            avg_growth = 0
            
        product_growth.append(avg_growth)
    
    product_metrics["growth_rate"] = product_growth
else:
    # If only one year, use profit margin as proxy for growth potential
    product_metrics["growth_rate"] = product_metrics["profit_margin"]

# Create BCG Matrix
fig = px.scatter(
    product_metrics,
    x="market_share",
    y="growth_rate",
    size="total_revenue",
    color="profit_margin",
    hover_name="product_name",
    text="product_name",
    log_x=False,  # Using linear scale
    color_continuous_scale=px.colors.sequential.Viridis,
    title="Product Portfolio Matrix (BCG)",
    labels={
        "market_share": "Market Share (%)",
        "growth_rate": "Growth Rate (%)",
        "profit_margin": "Profit Margin (%)"
    },
    height=600
)

# Add quadrant lines
median_share = product_metrics["market_share"].median()
median_growth = product_metrics["growth_rate"].median()

fig.add_shape(type="line", x0=median_share, y0=min(product_metrics["growth_rate"]), 
              x1=median_share, y1=max(product_metrics["growth_rate"]), 
              line=dict(color="Gray", width=1, dash="dash"))

fig.add_shape(type="line", x0=min(product_metrics["market_share"]), y0=median_growth, 
              x1=max(product_metrics["market_share"]), y1=median_growth, 
              line=dict(color="Gray", width=1, dash="dash"))

# Calculate the max and min points for positioning the labels
x_max = product_metrics["market_share"].max() * 1.1
x_min = 0  # We're starting from 0 for better visibility
y_max = product_metrics["growth_rate"].max() * 1.1
y_min = product_metrics["growth_rate"].min() * 1.1

# Add quadrant labels with adjusted positions
# Top-left quadrant (Question Marks)
fig.add_trace(go.Scatter(
    x=[x_min + (median_share - x_min) / 2],
    y=[y_max - (y_max - median_growth) * 0.2],  # Moved down slightly
    mode="text",
    hoverinfo="text",
    hovertext=["High growth, low market share - Evaluate for investment"],
    textfont=dict(color="black", size=12),
    showlegend=False
))

# Top-right quadrant (Stars) - REPOSITIONED to avoid overlap
fig.add_trace(go.Scatter(
    x=[median_share + (x_max - median_share) * 0.7],  # Moved right
    y=[y_max - (y_max - median_growth) * 0.3],  # Moved down
    mode="text",
    hoverinfo="text",
    hovertext=["High growth, high market share - Invest for growth"],
    textfont=dict(color="black", size=12),
    showlegend=False
))

# Bottom-left quadrant (Dogs)
fig.add_trace(go.Scatter(
    x=[x_min + (median_share - x_min) / 2],
    y=[y_min + (median_growth - y_min) * 0.4],  # Moved up slightly
    mode="text",
    hoverinfo="text",
    hovertext=["Low growth, low market share - Consider divesting"],
    textfont=dict(color="black", size=12),
    showlegend=False
))

# Bottom-right quadrant (Cash Cows)
fig.add_trace(go.Scatter(
    x=[median_share + (x_max - median_share) / 2],
    y=[y_min + (median_growth - y_min) * 0.2],  # Moved up slightly
    mode="text",
    hoverinfo="text",
    hovertext=["Low growth, high market share - Harvest for cash"],
    textfont=dict(color="black", size=12),
    showlegend=False
))

# Set the x-axis range explicitly
fig.update_xaxes(range=[x_min, x_max])
fig.update_yaxes(range=[y_min, y_max])

# Configure hover mode to show more user-friendly information
fig.update_layout(
    hovermode="closest",
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_family="Arial"
    )
)

fig.update_traces(textposition='top center')
fig.update_layout(showlegend=False)

st.plotly_chart(fig, use_container_width=True)

with st.expander("📊 View Product Portfolio Data"):
    st.dataframe(product_metrics.sort_values("total_revenue", ascending=False))
    
    st.markdown("""
    **Interpretation Guide:**
    - **Stars:** High growth, high market share (upper right) - Invest for growth
    - **Question Marks:** High growth, low market share (upper left) - Evaluate for investment
    - **Cash Cows:** Low growth, high market share (lower right) - Harvest for cash
    - **Dogs:** Low growth, low market share (lower left) - Consider divesting
    """)





# === SEASONALITY ANALYSIS ===
st.header("🗓️ Seasonality Analysis")

# Create monthly trends
monthly_data = filtered_df.groupby(["year", "month"]).agg({
    "total_revenue": "sum",
    "total_profit": "sum",
    "order_quantity": "sum"
}).reset_index()

# Add month name for better visualization
monthly_data["month_name"] = monthly_data["month"].apply(lambda x: calendar.month_abbr[x])

# Create seasonality visualization
tab1, tab2 = st.tabs(["Monthly Pattern", "Year-over-Year Comparison"])

with tab1:
    # Create aggregated monthly pattern across all years
    monthly_pattern = filtered_df.groupby("month").agg({
        "total_revenue": "sum",
        "total_profit": "sum",
        "order_quantity": "sum"
    }).reset_index()
    
    monthly_pattern["month_name"] = monthly_pattern["month"].apply(lambda x: calendar.month_abbr[x])
    monthly_pattern = monthly_pattern.sort_values("month")
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=monthly_pattern["month_name"],
        y=monthly_pattern["total_revenue"],
        name="Revenue",
        marker_color="#1F77B4"
    ))
    
    fig.add_trace(go.Bar(
        x=monthly_pattern["month_name"],
        y=monthly_pattern["total_profit"],
        name="Profit",
        marker_color="#00CC96"
    ))
    
    fig.add_trace(go.Scatter(
        x=monthly_pattern["month_name"],
        y=monthly_pattern["order_quantity"],
        name="Quantity",
        yaxis="y2",
        mode="lines+markers",
        line=dict(color="#FF7F0E", width=3)
    ))
    
    fig.update_layout(
        title="Monthly Seasonality Pattern (All Years)",
        xaxis=dict(title="Month"),
        yaxis=dict(title="Amount ($)", side="left"),
        yaxis2=dict(title="Quantity", side="right", overlaying="y", rangemode="tozero"),
        legend=dict(x=0.01, y=0.99),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate and show peak months
    peak_revenue_month = monthly_pattern.loc[monthly_pattern["total_revenue"].idxmax(), "month_name"]
    peak_quantity_month = monthly_pattern.loc[monthly_pattern["order_quantity"].idxmax(), "month_name"]
    
    st.markdown(f"""
    **Seasonality Insights:**
    - Peak Revenue Month: **{peak_revenue_month}**
    - Peak Volume Month: **{peak_quantity_month}**
    
    **Business Recommendations:**
    - Plan production capacity increases ahead of peak months
    - Optimize inventory levels to support seasonal demand patterns
    - Consider seasonal pricing strategies to maximize profitability during peak demand periods
    """)

with tab2:
    # Create year-over-year monthly comparison
    if len(selected_years) > 1:
        monthly_comparison = monthly_data.pivot(index="month", columns="year", values="total_revenue").reset_index()
        monthly_comparison["month_name"] = monthly_comparison["month"].apply(lambda x: calendar.month_abbr[x])
        monthly_comparison = monthly_comparison.sort_values("month")
        
        fig = go.Figure()
        
        for year in selected_years:
            fig.add_trace(go.Scatter(
                x=monthly_comparison["month_name"],
                y=monthly_comparison[year],
                mode="lines+markers",
                name=f"Revenue {year}",
                line=dict(width=3)
            ))
        
        fig.update_layout(
            title="Monthly Revenue Comparison by Year",
            xaxis=dict(title="Month"),
            yaxis=dict(title="Revenue ($)"),
            legend=dict(x=0.01, y=0.99),
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select multiple years to view year-over-year monthly comparison.")

# === ENHANCED PRICE SENSITIVITY SIMULATOR ===
st.header("🧪 Price Sensitivity Simulator")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Simulation Controls")
    product_for_simulation = st.selectbox("Select product for price simulation", filtered_df["product_name"].unique())
    price_change = st.slider("Adjust price by (%)", -30, 30, 0)
    elasticity = st.slider("Demand Elasticity (0 = inelastic, >1 = elastic)", 0.0, 3.0, 1.0, 0.1)
    
    # Get product data
    product_df = filtered_df[filtered_df["product_name"] == product_for_simulation]
    current_price = product_df["product_price"].mean()
    current_cost = product_df["product_cost"].mean()
    current_quantity = product_df["order_quantity"].sum()
    current_profit = (current_price - current_cost) * current_quantity
    
    # Calculate new values
    price_multiplier = 1 + (price_change / 100)
    demand_multiplier = 1 - (price_change / 100 * elasticity)
    
    new_price = current_price * price_multiplier
    new_quantity = current_quantity * demand_multiplier
    new_profit = (new_price - current_cost) * new_quantity
    profit_change = new_profit - current_profit
    profit_change_pct = (profit_change / current_profit) * 100 if current_profit > 0 else 0
    
    # Display current vs. simulated metrics
    st.markdown("### Current Metrics")
    st.markdown(f"**Current Price:** ${current_price:.2f}")
    st.markdown(f"**Current Quantity:** {current_quantity:,.0f}")
    st.markdown(f"**Current Profit:** ${current_profit:,.2f}")
    
    st.markdown("### Simulated Metrics")
    st.markdown(f"**New Price:** ${new_price:.2f}")
    st.markdown(f"**Projected Quantity:** {new_quantity:,.0f}")
    st.markdown(f"**Projected Profit:** ${new_profit:,.2f}")
    
    # Show profit impact
    delta_color = "normal" if profit_change >= 0 else "inverse"
    st.metric(
        "Profit Impact", 
        f"${profit_change:,.2f}", 
        f"{profit_change_pct:.1f}%",
        delta_color=delta_color
    )

with col2:
    # Create a range of price changes to visualize
    price_changes = np.arange(-30, 31, 1)
    price_multipliers = 1 + price_changes / 100
    demand_multipliers = 1 - (price_changes / 100 * elasticity)
    
    new_prices = current_price * price_multipliers
    new_quantities = current_quantity * demand_multipliers
    new_profits = (new_prices - current_cost) * new_quantities
    
    # Create dataframe for visualization
    sim_data = pd.DataFrame({
        "Price Change (%)": price_changes,
        "New Price": new_prices,
        "Projected Quantity": new_quantities,
        "Projected Profit": new_profits
    })
    
    # Highlight current selection
    highlight_index = np.where(price_changes == price_change)[0][0]
    
    fig = px.line(
        sim_data, 
        x="Price Change (%)", 
        y=["Projected Profit", "Projected Quantity"],
        title=f"Price Sensitivity Curve - {product_for_simulation}",
        labels={"value": "Value", "variable": "Metric"}
    )
    
    # Add vertical line for current selection
    fig.add_shape(
        type="line",
        x0=price_change,
        y0=0,
        x1=price_change,
        y1=max(sim_data["Projected Profit"].max(), sim_data["Projected Quantity"].max()),
        line=dict(color="Red", width=2, dash="dash")
    )
    
    # Add optimal price point annotation
    optimal_price_change = sim_data.loc[sim_data["Projected Profit"].idxmax(), "Price Change (%)"]
    optimal_profit = sim_data.loc[sim_data["Projected Profit"].idxmax(), "Projected Profit"]
    
    fig.add_annotation(
        x=optimal_price_change,
        y=optimal_profit,
        text=f"Optimal Price: {optimal_price_change:.1f}%",
        showarrow=True,
        arrowhead=1,
        ax=20,
        ay=-40
    )
    
    fig.update_layout(
        xaxis=dict(title="Price Change (%)"),
        yaxis=dict(title="Value"),
        legend=dict(x=0.01, y=0.99),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    if optimal_price_change != price_change:
        st.info(f"🔍 **Profit Optimization Opportunity:** Consider a price change of {optimal_price_change:.1f}% for maximum profitability.")



# === MAKE VS BUY ANALYSIS ===
st.header("🏗️ Make-vs-Buy Strategic Analysis")

# Prepare data for make vs buy analysis
make_buy_data = filtered_df.groupby("product_name").agg({
    "order_quantity": "sum",
    "total_profit": "sum",
    "total_revenue": "sum",
    "total_cost": "sum"
}).reset_index()

make_buy_data["profit_margin"] = make_buy_data["total_profit"] / make_buy_data["total_revenue"] * 100
make_buy_data["volume_percentile"] = make_buy_data["order_quantity"].rank(pct=True) * 100
make_buy_data["margin_percentile"] = make_buy_data["profit_margin"].rank(pct=True) * 100

# Create quadrant thresholds
volume_threshold = 50  # Percentile
margin_threshold = 50  # Percentile

# Create quadrant labels
make_buy_data["quadrant"] = "Unknown"
make_buy_data.loc[(make_buy_data["volume_percentile"] >= volume_threshold) & 
                 (make_buy_data["margin_percentile"] >= margin_threshold), "quadrant"] = "Strategic Make (High Volume, High Margin)"
make_buy_data.loc[(make_buy_data["volume_percentile"] < volume_threshold) & 
                 (make_buy_data["margin_percentile"] >= margin_threshold), "quadrant"] = "Tactical Make (Low Volume, High Margin)"
make_buy_data.loc[(make_buy_data["volume_percentile"] >= volume_threshold) & 
                 (make_buy_data["margin_percentile"] < margin_threshold), "quadrant"] = "Consider Outsourcing (High Volume, Low Margin)"
make_buy_data.loc[(make_buy_data["volume_percentile"] < volume_threshold) & 
                 (make_buy_data["margin_percentile"] < margin_threshold), "quadrant"] = "Buy or Discontinue (Low Volume, Low Margin)"

# Create visualization
fig = px.scatter(
    make_buy_data,
    x="volume_percentile",
    y="margin_percentile",
    size="total_revenue",
    color="quadrant",
    hover_name="product_name",
    text="product_name",
    title="Make vs Buy Strategic Framework",
    labels={
        "volume_percentile": "Volume Percentile",
        "margin_percentile": "Profit Margin Percentile"
    },
    height=600
)

# Add quadrant lines
fig.add_shape(
    type="line",
    x0=volume_threshold,
    y0=0,
    x1=volume_threshold,
    y1=100,
    line=dict(color="Gray", width=1, dash="dash")
)

fig.add_shape(
    type="line",
    x0=0,
    y0=margin_threshold,
    x1=100,
    y1=margin_threshold,
    line=dict(color="Gray", width=1, dash="dash")
)

fig.update_traces(textposition='top center')
fig.update_layout(showlegend=True)

st.plotly_chart(fig, use_container_width=True)

with st.expander("📊 Make vs Buy Framework - Business Logic"):
    st.markdown("""
    ### Make vs Buy Decision Framework
    
    1. **Strategic Make (High Volume, High Margin)**
       - These products represent your core business
       - Invest in manufacturing capabilities and process improvements
       - Focus on quality control and vertical integration
    
    2. **Tactical Make (Low Volume, High Margin)**
       - Premium products with high margins but lower volumes
       - Consider small-batch production or specialized manufacturing cells
       - Evaluate flexible manufacturing systems
    
    3. **Consider Outsourcing (High Volume, Low Margin)**
       - Products where economies of scale are critical
       - Evaluate contract manufacturing or strategic partnerships
       - Focus on supply chain optimization to reduce costs
    
    4. **Buy or Discontinue (Low Volume, Low Margin)**
       - Non-core products with poor economics
       - Consider discontinuation or complete outsourcing
       - Evaluate product redesign to improve margins
    """)

    # Display data table
    st.dataframe(make_buy_data)

