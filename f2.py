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
    log_x=True,  # Use logarithmic scale for market share
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

# Add quadrant annotations
fig.add_annotation(x=product_metrics["market_share"].max()*0.9, y=product_metrics["growth_rate"].max()*0.9,
                   text="STARS", showarrow=False, font=dict(size=14, color="black", family="Arial Black"))

fig.add_annotation(x=product_metrics["market_share"].min()*1.1, y=product_metrics["growth_rate"].max()*0.9,
                   text="QUESTION MARKS", showarrow=False, font=dict(size=14, color="black", family="Arial Black"))

fig.add_annotation(x=product_metrics["market_share"].max()*0.9, y=product_metrics["growth_rate"].min()*1.1,
                   text="CASH COWS", showarrow=False, font=dict(size=14, color="black", family="Arial Black"))

fig.add_annotation(x=product_metrics["market_share"].min()*1.1, y=product_metrics["growth_rate"].min()*1.1,
                   text="DOGS", showarrow=False, font=dict(size=14, color="black", family="Arial Black"))

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

# === PARETO ANALYSIS (80/20 RULE) ===
st.header("📊 Pareto Analysis: Revenue Concentration")

# Calculate cumulative percentage
product_revenue = filtered_df.groupby("product_name")["total_revenue"].sum().reset_index()
product_revenue = product_revenue.sort_values("total_revenue", ascending=False)
product_revenue["revenue_percentage"] = product_revenue["total_revenue"] / product_revenue["total_revenue"].sum() * 100
product_revenue["cumulative_percentage"] = product_revenue["revenue_percentage"].cumsum()

# Filter to show top products that make up 80% of revenue
top_products = product_revenue[product_revenue["cumulative_percentage"] <= 80]
product_count = len(product_revenue)
top_count = len(top_products)
concentration_ratio = top_count / product_count

# Create visualization
fig = go.Figure()

# Add bar chart for revenue percentage
fig.add_trace(go.Bar(
    x=product_revenue["product_name"],
    y=product_revenue["revenue_percentage"],
    name="Revenue Percentage",
    marker_color="#1F77B4"
))

# Add line chart for cumulative percentage
fig.add_trace(go.Scatter(
    x=product_revenue["product_name"],
    y=product_revenue["cumulative_percentage"],
    mode="lines+markers",
    name="Cumulative Percentage",
    marker=dict(color="#FF7F0E", size=8),
    line=dict(color="#FF7F0E", width=3)
))

# Add 80% threshold line
fig.add_shape(
    type="line",
    x0=0,
    y0=80,
    x1=len(product_revenue),
    y1=80,
    line=dict(color="Red", width=2, dash="dash")
)

fig.update_layout(
    title=f"Pareto Analysis: {top_count} out of {product_count} products make up 80% of revenue",
    xaxis_title="Products (Ordered by Revenue)",
    yaxis_title="Percentage (%)",
    legend=dict(x=0.01, y=0.99),
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

with st.expander("📊 Business Insight"):
    st.markdown(f"""
    **Revenue Concentration Analysis:**
    - **{top_count} products** ({(top_count/product_count*100):.1f}% of your portfolio) generate 80% of your revenue
    - **Concentration Ratio:** {concentration_ratio:.2f} (lower is more concentrated)
    
    **Strategic Recommendations:**
    - Focus marketing, production, and quality efforts on these top revenue-generating products
    - Consider streamlining the product portfolio by evaluating the bottom 20% performers
    - Implement product-specific strategies based on this concentration analysis
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

# === PRODUCT CLUSTERING ANALYSIS ===
st.header("🔍 Product Segmentation Analysis")

# Prepare data for clustering
clustering_data = filtered_df.groupby("product_name").agg({
    "total_revenue": "sum",
    "profit_margin": "mean",
    "order_quantity": "sum"
}).reset_index()

# Normalize data for clustering
from sklearn.preprocessing import StandardScaler
X = clustering_data[["total_revenue", "profit_margin", "order_quantity"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal number of clusters
wcss = []
max_clusters = min(8, len(clustering_data))
for i in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Apply K-means clustering
n_clusters = st.slider("Number of Product Segments", 2, max_clusters, 4)
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster info to dataframe
clustering_data["cluster"] = clusters

# Create 3D visualization
fig = px.scatter_3d(
    clustering_data,
    x="total_revenue",
    y="profit_margin",
    z="order_quantity",
    color="cluster",
    hover_name="product_name",
    labels={
        "total_revenue": "Revenue",
        "profit_margin": "Profit Margin (%)",
        "order_quantity": "Order Quantity"
    },
    title=f"Product Segmentation - {n_clusters} Clusters"
)

st.plotly_chart(fig, use_container_width=True)

# Display cluster characteristics
cluster_summary = clustering_data.groupby("cluster").agg({
    "product_name": "count",
    "total_revenue": "mean",
    "profit_margin": "mean",
    "order_quantity": "mean"
}).reset_index()

cluster_summary.columns = ["Cluster", "Number of Products", "Avg Revenue", "Avg Profit Margin", "Avg Order Quantity"]
st.dataframe(cluster_summary)

# Generate business insights for each cluster
for i in range(n_clusters):
    cluster_products = clustering_data[clustering_data["cluster"] == i]
    
    with st.expander(f"Cluster {i} - {len(cluster_products)} Products"):
        st.dataframe(cluster_products)
        
        # Generate insights based on cluster characteristics
        avg_revenue = cluster_products["total_revenue"].mean()
        avg_margin = cluster_products["profit_margin"].mean()
        avg_quantity = cluster_products["order_quantity"].mean()
        
        st.markdown("### Cluster Characteristics")
        
        if avg_revenue > clustering_data["total_revenue"].mean() and avg_margin > clustering_data["profit_margin"].mean():
            st.success("💎 **Premium Performers**: High revenue and high margin products")
            st.markdown("**Recommendation**: Invest in these products, potentially expand product line")
        elif avg_revenue > clustering_data["total_revenue"].mean() and avg_margin < clustering_data["profit_margin"].mean():
            st.warning("🏆 **Volume Leaders**: High revenue but lower margin products")
            st.markdown("**Recommendation**: Focus on cost reduction to improve margins")
        elif avg_revenue < clustering_data["total_revenue"].mean() and avg_margin > clustering_data["profit_margin"].mean():
            st.info("💰 **Profit Niches**: Lower revenue but high margin products")
            st.markdown("**Recommendation**: Increase marketing to drive higher volumes")
        else:
            st.error("⚠️ **Underperformers**: Lower revenue and lower margin products")
            st.markdown("**Recommendation**: Consider product redesign or discontinuation")

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

# === INVENTORY AND CASH FLOW IMPACT ===
st.header("💰 Inventory & Cash Flow Analysis")

# Calculate days of inventory for each product
if "inventory_level" in filtered_df.columns:
    inventory_data = filtered_df.groupby("product_name").agg({
        "inventory_level": "mean",
        "order_quantity": "sum",
        "total_cost": "sum"
    }).reset_index()
    
    # Calculate days of inventory and turnover
    time_period_days = (filtered_df["date_of_purchase"].max() - filtered_df["date_of_purchase"].min()).days
    inventory_data["avg_daily_demand"] = inventory_data["order_quantity"] / time_period_days
    inventory_data["days_of_inventory"] = inventory_data["inventory_level"] / inventory_data["avg_daily_demand"]
    inventory_data["inventory_turnover"] = 365 / inventory_data["days_of_inventory"]
    inventory_data["inventory_value"] = inventory_data["inventory_level"] * inventory_data["total_cost"] / inventory_data["order_quantity"]
    
    # Create visualization
    fig = px.bar(
        inventory_data.sort_values("days_of_inventory", ascending=False),
        x="product_name",
        y="days_of_inventory",
        color="inventory_value",
        title="Days of Inventory by Product",
        labels={
            "product_name": "Product",
            "days_of_inventory": "Days of Inventory",
            "inventory_value": "Inventory Value ($)"
        }
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate cash tied up in inventory
    total_inventory_value = inventory_data["inventory_value"].sum()
    
    st.metric("Total Cash Tied in Inventory", f"${total_inventory_value:,.2f}")
else:
    st.info("Inventory data not available in the dataset. Add 'inventory_level' column to enable this analysis.")

# === PAYMENT TERMS & RECEIVABLES ANALYSIS ===
if "payment_due" in filtered_df.columns and "date_of_purchase" in filtered_df.columns:
    st.header("💳 Payment Terms & Cash Flow Impact")
    
    # Calculate days to payment
    filtered_df["days_to_payment"] = (filtered_df["payment_due"] - filtered_df["date_of_purchase"]).dt.days
    
    # Calculate average payment terms by product
    payment_data = filtered_df.groupby("product_name").agg({
        "days_to_payment": "mean",
        "total_revenue": "sum"
    }).reset_index()
    
    # Sort by revenue impact
    payment_data["cash_flow_impact"] = payment_data["days_to_payment"] * payment_data["total_revenue"] / 365
    payment_data = payment_data.sort_values("cash_flow_impact", ascending=False)
    
    # Create visualization
    fig = px.bar(
        payment_data,
        x="product_name",
        y="days_to_payment",
        color="total_revenue",
        title="Average Payment Terms by Product",
        labels={
            "product_name": "Product",
            "days_to_payment": "Avg Days to Payment",
            "total_revenue": "Total Revenue ($)"
        }
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate total cash flow impact
    total_impact = payment_data["cash_flow_impact"].sum()
    
    st.metric("Cash Flow Impact of Current Payment Terms", f"${total_impact:,.2f}")
    
    # Provide recommendations
    with st.expander("💸 Payment Terms Optimization"):
        st.markdown("""
        ### Payment Terms Optimization Strategies
        
        1. **High-Value Quick Payment Incentives**
           - Offer discounts for early payment on high-value products
           - Target: Products with the highest cash flow impact
        
        2. **Standardize Payment Terms**
           - Reduce variations in payment terms across products
           - Target: Products with unusually long payment terms
        
        3. **Automate Receivables Management**
           - Implement automated reminders for upcoming payments
           - Target: All products to reduce administrative overhead
        """)
else:
    st.info("Payment data not available in the dataset. Add 'payment_due' column to enable this analysis.")

# === PRICE OPTIMIZATION & CANNIBALIZATION RISK ===
st.header("📊 Price Optimization & Cannibalization Analysis")

# Prepare data for cannibalization analysis
if "product_category" in filtered_df.columns:
    # Group products by category
    category_data = filtered_df.groupby(["product_category", "product_name"]).agg({
        "product_price": "mean",
        "total_profit": "sum",
        "order_quantity": "sum"
    }).reset_index()
    
    # Create price range by category
    category_price_range = category_data.groupby("product_category").agg({
        "product_price": ["min", "max", "mean", "std"]
    }).reset_index()
    
    category_price_range.columns = ["product_category", "min_price", "max_price", "avg_price", "price_std"]
    category_price_range["price_range"] = category_price_range["max_price"] - category_price_range["min_price"]
    category_price_range["cv"] = category_price_range["price_std"] / category_price_range["avg_price"]
    
    # Select category for analysis
    selected_category = st.selectbox("Select Product Category for Price Analysis", 
                                    filtered_df["product_category"].unique())
    
    # Filter data for selected category
    category_products = category_data[category_data["product_category"] == selected_category]
    
    # Create visualization
    fig = px.scatter(
        category_products,
        x="product_price",
        y="order_quantity",
        size="total_profit",
        hover_name="product_name",
        text="product_name",
        title=f"Price vs Demand Analysis - {selected_category}",
        labels={
            "product_price": "Product Price ($)",
            "order_quantity": "Order Quantity",
            "total_profit": "Total Profit ($)"
        }
    )
    
    # Add trend line
    fig.update_layout(showlegend=False)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate cannibalization risk
    category_stats = category_price_range[category_price_range["product_category"] == selected_category].iloc[0]
    price_range = category_stats["price_range"]
    price_cv = category_stats["cv"]
    
    # Define risk levels
    cannibalization_risk = "Low"
    if price_range / category_stats["avg_price"] < 0.2:  # If price range is less than 20% of average price
        cannibalization_risk = "High"
    elif price_cv < 0.15:  # If coefficient of variation is low
        cannibalization_risk = "Medium"
    
    # Display risk assessment
    st.subheader("Product Cannibalization Risk Assessment")
    st.markdown(f"""
    **Category:** {selected_category}
    
    **Price Range:** ${category_stats["min_price"]:.2f} - ${category_stats["max_price"]:.2f} (Range: ${price_range:.2f})
    
    **Price Variability:** {price_cv:.2%} coefficient of variation
    
    **Cannibalization Risk:** {cannibalization_risk}
    """)
    
    # Provide recommendations
    if cannibalization_risk == "High":
        st.warning("""
        **High Cannibalization Risk Detected**
        
        Products in this category have very similar price points, likely causing them to compete with each other.
        
        **Recommendations:**
        - Increase price differentiation between products
        - Ensure clear feature differentiation in marketing
        - Consider consolidating similar products
        """)
    elif cannibalization_risk == "Medium":
        st.info("""
        **Medium Cannibalization Risk Detected**
        
        Some price overlap may be occurring in this category.
        
        **Recommendations:**
        - Review product positioning to ensure clear differentiation
        - Consider adjusting prices to create clearer tiers
        """)
    else:
        st.success("""
        **Low Cannibalization Risk Detected**
        
        Good price differentiation between products in this category.
        
        **Recommendations:**
        - Maintain current price tiers
        - Continue monitoring for changes in purchasing patterns
        """)
else:
    st.info("Product category data not available. Add 'product_category' column to enable cannibalization analysis.")

# === FORECAST SECTION ===
st.header("🔮 Sales & Profit Forecast")

# Prepare data for forecasting
product_filter = st.selectbox("Choose a product to forecast:", filtered_df["product_name"].unique())
df_filtered = filtered_df[filtered_df["product_name"] == product_filter]
df_yearly = df_filtered.groupby("year").agg({
    "order_quantity": "sum",
    "total_profit": "sum",
    "total_revenue": "sum"
}).reset_index()

if len(df_yearly) >= 2:
    df_yearly["quantity_growth_%"] = df_yearly["order_quantity"].pct_change() * 100
    df_yearly["profit_growth_%"] = df_yearly["total_profit"].pct_change() * 100
    df_yearly["revenue_growth_%"] = df_yearly["total_revenue"].pct_change() * 100
    
    st.subheader("Historical Performance")
    st.dataframe(df_yearly)
    
    # Forecast settings
    forecast_years = st.slider("Forecast Horizon (Years)", 1, 5, 3)
    
    # Build forecast models
    X = df_yearly["year"].values.reshape(-1, 1)
    future_years = np.arange(df_yearly["year"].max() + 1, df_yearly["year"].max() + 1 + forecast_years).reshape(-1, 1)
    
    # Linear regression for quantity
    qty_model = LinearRegression().fit(X, df_yearly["order_quantity"])
    forecast_qty = qty_model.predict(future_years)
    
    # Linear regression for profit
    profit_model = LinearRegression().fit(X, df_yearly["total_profit"])
    forecast_profit = profit_model.predict(future_years)
    
    # Linear regression for revenue
    revenue_model = LinearRegression().fit(X, df_yearly["total_revenue"])
    forecast_revenue = revenue_model.predict(future_years)
    
    # Create forecast dataframe
    future_df = pd.DataFrame({
        "year": future_years.flatten(),
        "forecast_quantity": forecast_qty,
        "forecast_profit": forecast_profit,
        "forecast_revenue": forecast_revenue
    })
    
    # Calculate forecast accuracy metrics
    qty_r2 = qty_model.score(X, df_yearly["order_quantity"])
    profit_r2 = profit_model.score(X, df_yearly["total_profit"])
    revenue_r2 = revenue_model.score(X, df_yearly["total_revenue"])
    
    # Display forecast metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Quantity Forecast Accuracy", f"{qty_r2:.2%} R²")
    
    with col2:
        st.metric("Profit Forecast Accuracy", f"{profit_r2:.2%} R²")
    
    with col3:
        st.metric("Revenue Forecast Accuracy", f"{revenue_r2:.2%} R²")
    
    # Create visualization
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=df_yearly["year"],
        y=df_yearly["total_revenue"],
        mode="lines+markers",
        name="Historical Revenue",
        line=dict(color="#1F77B4", width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=df_yearly["year"],
        y=df_yearly["total_profit"],
        mode="lines+markers",
        name="Historical Profit",
        line=dict(color="#2CA02C", width=3)
    ))
    
    # Add forecast data
    fig.add_trace(go.Scatter(
        x=future_df["year"],
        y=future_df["forecast_revenue"],
        mode="lines+markers",
        name="Forecast Revenue",
        line=dict(color="#1F77B4", width=3, dash="dash")
    ))
    
    fig.add_trace(go.Scatter(
        x=future_df["year"],
        y=future_df["forecast_profit"],
        mode="lines+markers",
        name="Forecast Profit",
        line=dict(color="#2CA02C", width=3, dash="dash")
    ))
    
    fig.update_layout(
        title=f"{product_filter} - Revenue & Profit Forecast",
        xaxis=dict(title="Year"),
        yaxis=dict(title="Amount ($)"),
        legend=dict(x=0.01, y=0.99),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display forecast data table
    st.subheader("Forecast Results")
    st.dataframe(future_df)
    
    # Business insights
    last_historical_year = df_yearly["year"].max()
    last_historical_revenue = df_yearly[df_yearly["year"] == last_historical_year]["total_revenue"].values[0]
    last_historical_profit = df_yearly[df_yearly["year"] == last_historical_year]["total_profit"].values[0]
    
    final_forecast_year = future_df["year"].max()
    final_forecast_revenue = future_df[future_df["year"] == final_forecast_year]["forecast_revenue"].values[0]
    final_forecast_profit = future_df[future_df["year"] == final_forecast_year]["forecast_profit"].values[0]
    
    revenue_growth = (final_forecast_revenue - last_historical_revenue) / last_historical_revenue * 100
    profit_growth = (final_forecast_profit - last_historical_profit) / last_historical_profit * 100
    
    st.subheader("Business Impact")
    st.markdown(f"""
    By **{final_forecast_year}**, this product is projected to:
    
    - Generate **${final_forecast_revenue:,.2f}** in revenue (**{revenue_growth:.1f}%** growth)
    - Deliver **${final_forecast_profit:,.2f}** in profit (**{profit_growth:.1f}%** growth)
    
    **Strategic Recommendations:**
    """)
    
    if profit_growth > 20:
        st.success("🚀 **High Growth Product** - Invest in production capacity and marketing")
    elif profit_growth > 0:
        st.info("📈 **Steady Growth Product** - Maintain current strategy with incremental improvements")
    else:
        st.warning("📉 **Declining Product** - Consider product redesign or gradual phase-out")
else:
    st.warning("Not enough historical data points to forecast this product.")

# === FOOTER ===
st.markdown("---")
st.caption("Manufacturing & Profitability Intelligence Dashboard | Data Last Updated: " + 
          datetime.now().strftime("%Y-%m-%d"))