import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Set page config
st.set_page_config(
    page_title="Dynamic Pricing Dashboard",
    page_icon="📊",
    layout="wide"
)

# Add custom CSS for tier badges
st.markdown("""
<style>
    .tier-badge {
        display: inline-block;
        padding: 8px 12px;
        border-radius: 24px;
        font-weight: bold;
        text-align: center;
        width: 120px;
        margin: 0 auto;
    }
    .tier-gold-customer {
        background-color: #FFD700;
        color: black;
    }
    .tier-silver-customer {
        background-color: #C0C0C0;
        color: black;
    }
    .tier-bronze-customer {
        background-color: #CD7F32;
        color: white;
    }
    .tier-high-product {
        background-color: #746f91;
        color: white;
    }
    .tier-medium-product {
        background-color: #1E90FF;
        color: white;
    }
    .tier-low-product {
        background-color: #32CD32;
        color: black;
    }
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
    }
    .info-box {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 10px;
        border-left: 5px solid #4CAF50;
    }
    .metric-card {
        background-color: white;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 0 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        margin: 10px 0;
    }
    .metric-title {
        font-size: 16px;
        color: #555;
    }
    .table-wrapper {
        border-radius: 5px;
        overflow: hidden;
        box-shadow: 0 0 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("""
<div class="header-container">
    <h1>📊 Dynamic Pricing Dashboard</h1>
</div>
<div class="info-box">
    <p>This dashboard provides comprehensive pricing analytics across customer and product tiers. 
    Use the filters on the left to explore pricing recommendations based on customer payment behavior and product costs.</p>
</div>
""", unsafe_allow_html=True)

# Function to determine customer segment based on payment behavior
def assign_segment(row):
    if row["on_time_rate"] >= 90:
        return "Reliable Payers"
    elif row["on_time_rate"] >= 70:
        return "Mostly Reliable"
    elif row["on_time_rate"] >= 50:
        return "Inconsistent Payers"
    else:
        return "High Risk"

# Function to determine customer tier based on payment behavior
def get_customer_tier(actual_repayment_days, days_given_for_payment):
    # Early payment (before due date)
    if actual_repayment_days < days_given_for_payment:
        return "Gold"
    # On-time payment (exactly on due date)
    elif actual_repayment_days == days_given_for_payment:
        return "Silver"
    # Late payment (after due date)
    else:
        return "Bronze"

# Function to determine payment status
def get_payment_status(actual_repayment_days, days_given_for_payment):
    # Early payment
    if actual_repayment_days < days_given_for_payment:
        return "Early"
    # On-time payment
    elif actual_repayment_days == days_given_for_payment:
        return "On Time"
    # Late payment
    else:
        return "Late"

# Function to determine product tier based on cost
def get_product_tier(product_cost):
    if product_cost >= 1.30:
        return "High"
    elif product_cost < 1.10:
        return "Low"
    else:
        return "Medium"

# Function to calculate recommended price based on tiers
def calculate_recommended_price(row):
    # Base margin based on customer tier
    if row['customer_tier'] == 'Gold':
        margin = 0.15  # Lower margin for high-value customers (early payers)
    elif row['customer_tier'] == 'Bronze':
        margin = 0.45  # Higher margin for lower-tier customers (late payers)
    else:  # Silver
        margin = 0.30  # Medium margin (on-time payers)
    
    # Product tier adjustment
    if row['product_tier'] == 'High':
        margin += 0.05  # Premium products can carry higher margins
    elif row['product_tier'] == 'Low':
        margin -= 0.02  # Economy products need competitive pricing
    
    # Calculate final price
    return round(row['product_cost'] * (1 + margin), 2)

# Function to format tier badge HTML
def format_tier_badge(tier, tier_type):
    class_name = f"tier-badge tier-{tier.lower()}-{tier_type}"
    return f'<span class="{class_name}">{tier}</span>'

# Load and process data
@st.cache_data
def load_data():
    try:
        # Load data
        df = pd.read_csv('3000_Varied_Product_Transactions.csv')
        
        # Convert date columns to datetime
        df['date_of_purchase'] = pd.to_datetime(df['date_of_purchase'])
        df['payment_due'] = pd.to_datetime(df['payment_due'])
        
        # Add year-month column
        df['year_month'] = df['date_of_purchase'].dt.strftime('%Y-%m')
        
        # Calculate payment status based on actual_repayment_days and days_given_for_payment
        df['payment_status'] = df.apply(
            lambda x: get_payment_status(x['actual_repayment_days'], x['days_given_for_payment']), 
            axis=1
        )
        
        # Calculate if payment was on time (1) or late (0)
        df['on_time'] = (df['actual_repayment_days'] <= df['days_given_for_payment']).astype(int)
        
        # Calculate days overdue/early
        df['days_overdue'] = np.maximum(0, df['actual_repayment_days'] - df['days_given_for_payment'])
        df['days_early'] = np.maximum(0, df['days_given_for_payment'] - df['actual_repayment_days'])
        
        # Add customer and product tiers
        df['customer_tier'] = df.apply(
            lambda x: get_customer_tier(x['actual_repayment_days'], x['days_given_for_payment']), 
            axis=1
        )
        df['product_tier'] = df['product_cost'].apply(get_product_tier)
        
        # Calculate recommended prices
        df['recommended_price'] = df.apply(calculate_recommended_price, axis=1)
        df['recommended_margin_pct'] = ((df['recommended_price'] - df['product_cost']) / df['recommended_price'] * 100).round(2)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Main function
def main():
    # Load data
    df = load_data()
    
    if df is None:
        st.error("Failed to load data. Please ensure '3000_Varied_Product_Transactions.csv' is in the current directory.")
        return
    
    # Create sidebar filters
    st.sidebar.header("Filters")
    
    # Time period filter
    st.sidebar.subheader("Time Period")
    all_periods = sorted(df['year_month'].unique(), reverse=True)
    selected_periods = st.sidebar.multiselect(
        "Select Months", 
        options=all_periods,
        default=all_periods[:3]  # Default to last 3 months
    )
    
    # Payment status filter
    st.sidebar.subheader("Payment Status")
    all_payment_statuses = sorted(df['payment_status'].unique())
    selected_payment_statuses = st.sidebar.multiselect(
        "Select Payment Status", 
        options=all_payment_statuses,
        default=all_payment_statuses
    )
    
    # Customer tier filter
    st.sidebar.subheader("Customer Tiers")
    all_customer_tiers = sorted(df['customer_tier'].unique())
    selected_customer_tiers = st.sidebar.multiselect(
        "Select Customer Tiers", 
        options=all_customer_tiers,
        default=all_customer_tiers
    )
    
    # Product tier filter
    st.sidebar.subheader("Product Tiers")
    all_product_tiers = sorted(df['product_tier'].unique())
    selected_product_tiers = st.sidebar.multiselect(
        "Select Product Tiers", 
        options=all_product_tiers,
        default=all_product_tiers
    )
    
    # Customer filter
    st.sidebar.subheader("Customers")
    all_customers = sorted(df['customer_name'].unique())
    selected_customers = st.sidebar.multiselect(
        "Select Customers", 
        options=all_customers,
        default=[]
    )
    
    # Product filter
    st.sidebar.subheader("Products")
    all_products = sorted(df['product_name'].unique())
    selected_products = st.sidebar.multiselect(
        "Select Products", 
        options=all_products,
        default=[]
    )
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_periods:
        filtered_df = filtered_df[filtered_df['year_month'].isin(selected_periods)]
    
    if selected_payment_statuses:
        filtered_df = filtered_df[filtered_df['payment_status'].isin(selected_payment_statuses)]
    
    if selected_customer_tiers:
        filtered_df = filtered_df[filtered_df['customer_tier'].isin(selected_customer_tiers)]
    
    if selected_product_tiers:
        filtered_df = filtered_df[filtered_df['product_tier'].isin(selected_product_tiers)]
    
    if selected_customers:
        filtered_df = filtered_df[filtered_df['customer_name'].isin(selected_customers)]
    
    if selected_products:
        filtered_df = filtered_df[filtered_df['product_name'].isin(selected_products)]
    
    # Main content
    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
        return
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Products", "Customers", "Pricing Table"])
    
    with tab1:
        display_overview(filtered_df)
    
    with tab2:
        display_products(filtered_df)
    
    with tab3:
        display_customers(filtered_df, df)
    
    with tab4:
        display_pricing_table(filtered_df)

# Function to display overview tab
def display_overview(df):
    st.header("Dashboard Overview")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Avg Recommended Price</div>
            <div class="metric-value" style="color: #1E88E5;">$%.2f</div>
        </div>
        """ % df['recommended_price'].mean(), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Total Revenue</div>
            <div class="metric-value" style="color: #43A047;">$%.2f</div>
        </div>
        """ % (df['product_price'] * df['order_quantity']).sum(), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Avg Margin %%</div>
            <div class="metric-value" style="color: #FF5722;">%.1f%%</div>
        </div>
        """ % df['recommended_margin_pct'].mean(), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">On-Time Payment Rate</div>
            <div class="metric-value" style="color: #9C27B0;">%.1f%%</div>
        </div>
        """ % (df['on_time'].mean() * 100), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Monthly trend chart
    st.subheader("Monthly Revenue & Profit Trends")
    
    monthly_data = df.groupby('year_month').agg(
        Revenue=('product_price', lambda x: (x * df.loc[x.index, 'order_quantity']).sum()),
        Profit=('profit', 'sum'),
        Orders=('transact_id', 'count')
    ).reset_index()
    
    # Sort by year_month
    monthly_data['year_month'] = pd.to_datetime(monthly_data['year_month'], format='%Y-%m')
    monthly_data = monthly_data.sort_values('year_month')
    monthly_data['year_month'] = monthly_data['year_month'].dt.strftime('%Y-%m')
    
    # Create the chart
    fig = px.line(
        monthly_data,
        x='year_month',
        y=['Revenue', 'Profit'],
        title='Revenue and Profit Trends',
        markers=True,
        labels={'value': 'Amount ($)', 'year_month': 'Month', 'variable': 'Metric'}
    )
    
    fig.update_layout(
        xaxis_title='Month',
        yaxis_title='Amount ($)',
        legend_title='Metric',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Payment Status and Customer Tier Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Payment Status Distribution")
        payment_status_counts = df['payment_status'].value_counts().reset_index()
        payment_status_counts.columns = ['Status', 'Count']
        
        fig = px.pie(
            payment_status_counts,
            values='Count',
            names='Status',
            color='Status',
            color_discrete_map={
                'Early': '#22c55e',
                'On Time': '#3b82f6',
                'Late': '#ef4444'
            },
            title='Payment Status Distribution'
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Customer Tier Distribution")
        customer_tier_counts = df['customer_tier'].value_counts().reset_index()
        customer_tier_counts.columns = ['Tier', 'Count']
        
        fig = px.pie(
            customer_tier_counts,
            values='Count',
            names='Tier',
            color='Tier',
            color_discrete_map={
                'Gold': '#FFD700',
                'Silver': '#C0C0C0',
                'Bronze': '#CD7F32'
            },
            title='Customer Tier Distribution'
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    # Product Tier Distribution
    st.subheader("Product Tier Distribution")
    product_tier_counts = df['product_tier'].value_counts().reset_index()
    product_tier_counts.columns = ['Tier', 'Count']
    
    fig = px.pie(
        product_tier_counts,
        values='Count',
        names='Tier',
        color='Tier',
        color_discrete_map={
            'High': '#4B0082',   # Purple
            'Medium': '#1E90FF',  # Blue
            'Low': '#32CD32'     # Green
        },
        title='Product Tier Distribution'
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

# Function to display products tab
def display_products(df):
    st.header("Product Analysis")
    
    # Group by product
    product_data = df.groupby(['product_name', 'product_tier']).agg(
        AvgCost=('product_cost', 'mean'),
        AvgPrice=('product_price', 'mean'),
        AvgRecommendedPrice=('recommended_price', 'mean'),
        AvgMargin=('recommended_margin_pct', 'mean'),
        TotalUnits=('order_quantity', 'sum'),
        TotalRevenue=('product_price', lambda x: (x * df.loc[x.index, 'order_quantity']).sum()),
        TotalProfit=('profit', 'sum')
    ).reset_index()
    
    # Top products by volume
    st.subheader("Top Products by Sales Volume")
    top_volume = product_data.sort_values('TotalUnits', ascending=False).head(5)
    
    fig = px.bar(
        top_volume,
        x='product_name',
        y='TotalUnits',
        color='product_tier',
        color_discrete_map={
            'High': '#4B0082',
            'Medium': '#1E90FF',
            'Low': '#32CD32'
        },
        title='Top 5 Products by Sales Volume',
        labels={'product_name': 'Product', 'TotalUnits': 'Units Sold', 'product_tier': 'Product Tier'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Top products by profit
    st.subheader("Top Products by Profit")
    top_profit = product_data.sort_values('TotalProfit', ascending=False).head(5)
    
    fig = px.bar(
        top_profit,
        x='product_name',
        y='TotalProfit',
        color='product_tier',
        color_discrete_map={
            'High': '#4B0082',
            'Medium': '#1E90FF',
            'Low': '#32CD32'
        },
        title='Top 5 Products by Total Profit',
        labels={'product_name': 'Product', 'TotalProfit': 'Total Profit ($)', 'product_tier': 'Product Tier'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Product pricing scatter plot
    st.subheader("Product Pricing Analysis")
    
    fig = px.scatter(
        product_data,
        x='AvgCost',
        y='AvgRecommendedPrice',
        size='TotalUnits',
        color='product_tier',
        hover_name='product_name',
        color_discrete_map={
            'High': '#4B0082',
            'Medium': '#1E90FF',
            'Low': '#32CD32'
        },
        title='Price vs Cost (bubble size = sales volume)',
        labels={
            'AvgCost': 'Average Cost ($)',
            'AvgRecommendedPrice': 'Recommended Price ($)',
            'product_tier': 'Product Tier'
        }
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Product table
    st.subheader("Product Details")
    
    # Format the dataframe for display
    display_df = product_data.copy()
    display_df['AvgCost'] = display_df['AvgCost'].map('${:.2f}'.format)
    display_df['AvgPrice'] = display_df['AvgPrice'].map('${:.2f}'.format)
    display_df['AvgRecommendedPrice'] = display_df['AvgRecommendedPrice'].map('${:.2f}'.format)
    display_df['AvgMargin'] = display_df['AvgMargin'].map('{:.1f}%'.format)
    display_df['TotalRevenue'] = display_df['TotalRevenue'].map('${:.2f}'.format)
    display_df['TotalProfit'] = display_df['TotalProfit'].map('${:.2f}'.format)
    
    # Create HTML for product tier badges
    display_df['ProductTier'] = display_df['product_tier'].apply(
        lambda x: format_tier_badge(x, 'product')
    )
    
    # Select and rename columns for display
    display_cols = {
        'product_name': 'Product',
        'ProductTier': 'Product Tier',
        'AvgCost': 'Avg Cost',
        'AvgRecommendedPrice': 'Recommended Price',
        'AvgMargin': 'Margin %',
        'TotalUnits': 'Units Sold',
        'TotalRevenue': 'Total Revenue',
        'TotalProfit': 'Total Profit'
    }
    
    final_df = display_df[display_cols.keys()].rename(columns=display_cols)
    
    # Display the table
    st.markdown('<div class="table-wrapper">', unsafe_allow_html=True)
    st.write(final_df.to_html(escape=False, index=False), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Function to display customers tab
def display_customers(df, full_df):
    st.header("Customer Analysis")
    
    # Group by customer
    customer_data = df.groupby(['customer_name', 'customer_id', 'customer_tier']).agg(
        AvgRepaymentDays=('actual_repayment_days', 'mean'),
        AvgPaymentTerms=('days_given_for_payment', 'mean'),
        OnTimeRate=('on_time', 'mean'),
        DaysOverdue=('days_overdue', 'mean'),
        DaysEarly=('days_early', 'mean'),
        TotalSpend=('product_price', lambda x: (x * df.loc[x.index, 'order_quantity']).sum()),
        TotalOrders=('transact_id', 'count'),
        TotalProfit=('profit', 'sum')
    ).reset_index()
    
    # Calculate on_time_rate as percentage
    customer_data['on_time_rate'] = customer_data['OnTimeRate'] * 100
    
    # Assign payment behavior segments
    customer_data['payment_segment'] = customer_data.apply(assign_segment, axis=1)
    
    # Top customers by spend
    st.subheader("Top Customers by Total Spend")
    top_spend = customer_data.sort_values('TotalSpend', ascending=False).head(5)
    
    fig = px.bar(
        top_spend,
        x='customer_name',
        y='TotalSpend',
        color='customer_tier',
        color_discrete_map={
            'Gold': '#FFD700',  # Gold color
            'Silver': '#C0C0C0', # Silver color
            'Bronze': '#CD7F32'  # Bronze color
        },
        title='Top 5 Customers by Total Spend',
        labels={'customer_name': 'Customer', 'TotalSpend': 'Total Spend ($)', 'customer_tier': 'Customer Tier'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # === CUSTOMER SEGMENTATION BY PAYMENT BEHAVIOR ===
    st.markdown("<div style='background-color: #f0f2fa; padding: 15px; border-radius: 10px; border-left: 5px solid #4b70e2;'><h3>👥 Customer Payment Behavior Segmentation</h3></div>", unsafe_allow_html=True)
    
    # Create a scatter plot of customer payment behavior
    fig = px.scatter(
        customer_data,
        x='DaysOverdue',
        y='on_time_rate',
        color='payment_segment',
        size='TotalSpend',
        hover_name='customer_name',
        text='customer_name',
        title='Customer Segmentation by Payment Behavior',
        color_discrete_map={
            'Reliable Payers': '#22c55e',
            'Mostly Reliable': '#3b82f6',
            'Inconsistent Payers': '#eab308',
            'High Risk': '#ef4444'
        },
        size_max=25
    )

    fig.update_layout(
        xaxis_title='Average Days Overdue',
        yaxis_title='On-Time Payment Rate (%)',
        height=500
    )

    fig.update_traces(
        textposition='top center',
        marker=dict(line=dict(width=1, color='DarkSlateGrey'))
    )

    # Add quadrant lines
    fig.add_hline(
        y=70,
        line_dash='dash',
        line_color='gray'
    )

    fig.add_vline(
        x=5,
        line_dash='dash',
        line_color='gray'
    )

    # Add annotations for quadrants
    fig.add_annotation(
        x=2.5, y=85,
        text='RELIABLE',
        showarrow=False,
        font=dict(size=14, color='#22c55e', family='Arial Black')
    )

    fig.add_annotation(
        x=10, y=85,
        text='DELAYED BUT PREDICTABLE',
        showarrow=False,
        font=dict(size=14, color='#3b82f6', family='Arial Black')
    )

    fig.add_annotation(
        x=2.5, y=35,
        text='INCONSISTENT',
        showarrow=False,
        font=dict(size=14, color='#eab308', family='Arial Black')
    )

    fig.add_annotation(
        x=10, y=35,
        text='HIGH RISK',
        showarrow=False,
        font=dict(size=14, color='#ef4444', family='Arial Black')
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # Display customer segment summary
    segment_summary = customer_data.groupby('payment_segment').agg({
        'customer_id': 'count',
        'on_time_rate': 'mean',
        'DaysOverdue': 'mean',
        'DaysEarly': 'mean',
        'TotalSpend': 'mean',
        'TotalOrders': 'mean'
    }).reset_index()

    segment_summary.columns = [
        'Segment', 'Customer Count', 'Avg On-Time Rate (%)', 
        'Avg Days Overdue', 'Avg Days Early', 'Avg Order Value ($)',
        'Avg Transaction Count'
    ]
    
    st.markdown("<div style='background-color: white; padding: 15px; border-radius: 5px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);'>", unsafe_allow_html=True)
    st.markdown("#### Customer Segment Summary")
    st.dataframe(segment_summary.style.format({
        'Avg On-Time Rate (%)': '{:.1f}',
        'Avg Days Overdue': '{:.1f}',
        'Avg Days Early': '{:.1f}',
        'Avg Order Value ($)': '${:.2f}',
        'Avg Transaction Count': '{:.1f}'
    }), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Business recommendations based on segments
    st.markdown("<div style='background-color: #f0f2ff; border-radius: 10px; padding: 15px; margin-top: 20px; border-left: 5px solid #4f46e5;'>", unsafe_allow_html=True)
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
    
    # Payment behavior analysis
    st.markdown("<div style='background-color: #f0f2fa; padding: 15px; border-radius: 10px; border-left: 5px solid #4b70e2; margin-top: 30px;'><h3>🔍 Payment Behavior Analysis</h3></div>", unsafe_allow_html=True)
    
    # === PAYMENT BEHAVIOR OVERVIEW ===
    tab1, tab2 = st.tabs(["Payment Status Distribution", "Timeline Analysis"])

    with tab1:
        # Create columns for side-by-side charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Payment status distribution pie chart
            payment_status_counts = df["payment_status"].value_counts().reset_index()
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
            # Prepare data for days until payment histogram
            # Calculate days from purchase to payment
            df_with_time = df.copy()
            df_with_time['days_until_payment'] = df_with_time['actual_repayment_days']
            df_with_time['days_to_payment'] = df_with_time['days_given_for_payment']
            
            # Days to payment distribution
            fig = px.histogram(
                df_with_time,
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
                x=df_with_time["days_to_payment"].median(),
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
        # Timeline analysis using actual_repayment_days as days_until_payment
        df_for_timeline = df.copy()
        df_for_timeline['days_until_payment'] = df_for_timeline['actual_repayment_days']
        
        fig = px.scatter(
            df_for_timeline,
            x="date_of_purchase",
            y="days_until_payment",
            color="payment_status",
            size="product_price",
            hover_name="customer_name",
            hover_data=["customer_id", "days_until_payment", "payment_status", "product_price"],
            title="Payment Timeline Analysis",
            color_discrete_map={
                "Early": "#22c55e",
                "On Time": "#3b82f6",
                "Late": "#ef4444"
            },
            size_max=20
        )
        
        fig.add_hline(
            y=df_for_timeline["days_given_for_payment"].median(),
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
    
    # Customer table
    st.subheader("Customer Details")
    
    # Format the dataframe for display
    display_df = customer_data.copy()
    display_df['AvgRepaymentDays'] = display_df['AvgRepaymentDays'].map('{:.1f}'.format)
    display_df['AvgPaymentTerms'] = display_df['AvgPaymentTerms'].map('{:.1f}'.format)
    display_df['OnTimeRate'] = display_df['OnTimeRate'].map('{:.1%}'.format)
    display_df['DaysOverdue'] = display_df['DaysOverdue'].map('{:.1f}'.format)
    display_df['DaysEarly'] = display_df['DaysEarly'].map('{:.1f}'.format)
    display_df['TotalSpend'] = display_df['TotalSpend'].map('${:.2f}'.format)
    display_df['TotalProfit'] = display_df['TotalProfit'].map('${:.2f}'.format)
    
    # Create HTML for customer tier badges
    display_df['CustomerTier'] = display_df['customer_tier'].apply(
        lambda x: format_tier_badge(x, 'customer')
    )
    
    # Create HTML for payment segment badges
    display_df['PaymentSegment'] = display_df['payment_segment'].apply(
        lambda x: f'<span style="display: inline-block; padding: 4px 8px; border-radius: 12px; font-weight: bold; text-align: center; background-color: {
            "#22c55e" if x == "Reliable Payers" else 
            "#3b82f6" if x == "Mostly Reliable" else 
            "#eab308" if x == "Inconsistent Payers" else "#ef4444"
        }; color: white;">{x}</span>'
    )
    
    # Select and rename columns for display
    display_cols = {
        'customer_name': 'Customer',
        'CustomerTier': 'Customer Tier',
        'PaymentSegment': 'Payment Segment',
        'AvgRepaymentDays': 'Avg Repayment Days',
        'AvgPaymentTerms': 'Avg Payment Terms',
        'OnTimeRate': 'On-Time Rate',
        'DaysEarly': 'Days Early',
        'DaysOverdue': 'Days Overdue',
        'TotalOrders': 'Total Orders',
        'TotalSpend': 'Total Spend',
        'TotalProfit': 'Total Profit'
    }
    
    final_df = display_df[display_cols.keys()].rename(columns=display_cols)
    
    # Display the table
    st.markdown('<div class="table-wrapper">', unsafe_allow_html=True)
    st.write(final_df.to_html(escape=False, index=False), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Function to display pricing table
def display_pricing_table(df):
    st.header("Dynamic Pricing Table")
    
    # Sort the dataframe by year_month (descending) and customer_name
    sorted_df = df.sort_values(['year_month', 'customer_name'], ascending=[False, True])
    
    # Select most recent entries for display (limit to 20)
    display_df = sorted_df.head(20).copy()
    
    # Format the columns
    display_df['product_cost'] = display_df['product_cost'].map('${:.2f}'.format)
    display_df['recommended_price'] = display_df['recommended_price'].map('${:.2f}'.format)
    display_df['recommended_margin_pct'] = display_df['recommended_margin_pct'].map('{:.1f}%'.format)
    display_df['actual_repayment_days'] = display_df['actual_repayment_days'].map('{:.0f}'.format)
    display_df['days_given_for_payment'] = display_df['days_given_for_payment'].map('{:.0f}'.format)
    display_df['days_overdue'] = display_df['days_overdue'].map('{:.0f}'.format)
    display_df['days_early'] = display_df['days_early'].map('{:.0f}'.format)
    
    # Create HTML for tier badges
    display_df['CustomerTier'] = display_df['customer_tier'].apply(
        lambda x: format_tier_badge(x, 'customer')
    )
    display_df['ProductTier'] = display_df['product_tier'].apply(
        lambda x: format_tier_badge(x, 'product')
    )
    display_df['PaymentStatus'] = display_df['payment_status'].apply(
        lambda x: f'<span style="padding: 4px 8px; border-radius: 12px; font-weight: bold; background-color: {
            "#22c55e" if x == "Early" else "#3b82f6" if x == "On Time" else "#ef4444"
        }; color: white;">{x}</span>'
    )
    
    # Select and rename columns for display
    display_cols = {
        'year_month': 'Year/Month',
        'customer_name': 'Customer Name',
        'CustomerTier': 'Customer Tier',
        'product_name': 'Product Name',
        'ProductTier': 'Product Tier',
        'product_cost': 'Product Cost',
        'recommended_price': 'Recommended Price',
        'recommended_margin_pct': 'Margin %',
        'actual_repayment_days': 'Repayment Days',
        'days_given_for_payment': 'Payment Terms',
        'PaymentStatus': 'Payment Status'
    }
    
    final_df = display_df[display_cols.keys()].rename(columns=display_cols)
    
    # Display the table
    st.markdown('<div class="table-wrapper">', unsafe_allow_html=True)
    st.write(final_df.to_html(escape=False, index=False), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show the tier and status color code explanation
    st.markdown("### Color Legend")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <span style="background-color: #FFD700; width: 20px; height: 20px; display: inline-block; margin-right: 5px; border-radius: 3px;"></span>
            <span>Gold Customer (Early Payment)</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <span style="background-color: #C0C0C0; width: 20px; height: 20px; display: inline-block; margin-right: 5px; border-radius: 3px;"></span>
            <span>Silver Customer (On-Time Payment)</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <span style="background-color: #CD7F32; width: 20px; height: 20px; display: inline-block; margin-right: 5px; border-radius: 3px;"></span>
            <span>Bronze Customer (Late Payment)</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <span style="background-color: #746f91; width: 20px; height: 20px; display: inline-block; margin-right: 5px; border-radius: 3px;"></span>
            <span>High Product</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <span style="background-color: #1E90FF; width: 20px; height: 20px; display: inline-block; margin-right: 5px; border-radius: 3px;"></span>
            <span>Medium Product</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <span style="background-color: #32CD32; width: 20px; height: 20px; display: inline-block; margin-right: 5px; border-radius: 3px;"></span>
            <span>Low Product</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Pricing explanation
    st.markdown("### Pricing Logic")
    st.markdown("""
    The recommended prices are calculated based on the following rules:
    
    1. **Base margin by customer tier (based on payment behavior):**
       - Gold customers (Early Payers): 15% (lower margin for high-value customers)
       - Silver customers (On-Time Payers): 30% (medium margin)
       - Bronze customers (Late Payers): 45% (higher margin for higher-risk customers)
    
    2. **Product tier adjustments:**
       - High products: +5% (premium products can carry higher margins)
       - Medium products: No adjustment
       - Low products: -2% (economy products need competitive pricing)
    
    3. **Customer Segmentation by Payment Reliability:**
       - Reliable Payers (≥90% on-time rate): Eligible for additional loyalty discounts
       - Mostly Reliable (70-89% on-time rate): Standard pricing
       - Inconsistent Payers (50-69% on-time rate): Monitored pricing
       - High Risk (<50% on-time rate): Consider additional risk premium
    
    This dynamic pricing strategy aims to reward reliable customers while managing risk with tiered pricing based on payment behavior.
    """)
    
    # Download section
    st.header("Download Data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name=f"dynamic_pricing_export_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

if __name__ == "__main__":
    main()