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

# Add custom CSS
st.markdown("""
<style>
    .tier-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
    }
    .tier-gold {
        background-color: gold;
        color: black;
    }
    .tier-silver {
        background-color: silver;
        color: black;
    }
    .tier-bronze {
        background-color: #CD7F32;
        color: white;
    }
    .tier-premium {
        background-color: #4B0082;
        color: white;
    }
    .tier-standard {
        background-color: #1E90FF;
        color: white;
    }
    .tier-economy {
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
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("""
<div class="header-container">
    <h1>📊 5-Year Dynamic Pricing Dashboard</h1>
</div>
<div class="info-box">
    <p>This dashboard provides comprehensive pricing analytics across customer and product tiers. 
    Use the filters on the left to explore by time period, tiers, customers, or products.</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# PRICING MODEL FUNCTIONS
# -------------------------------------------------

def process_dynamic_pricing(input_file='3000_Varied_Product_Transactions.csv'):
    """
    Process customer transaction data to generate dynamic pricing recommendations
    with enhanced tier classification for both customers and products
    
    Parameters:
    input_file (str): Path to input CSV file (default: 3000_Varied_Product_Transactions.csv)
    
    Returns:
    pandas.DataFrame: Processed pricing data
    """
    st.write(f"Processing file: {input_file}")
    
    # --- 1. Load dataset ---
    try:
        df = pd.read_csv(input_file, parse_dates=['date_of_purchase', 'payment_due'])
        st.write(f"Successfully loaded {len(df)} rows from {input_file}")
    except Exception as e:
        st.error(f"Error loading {input_file}: {e}")
        raise Exception(f"Failed to load the dataset: {e}")
    
    # --- 2. Add year-month period for easier filtering ---
    df['year_month'] = df['date_of_purchase'].dt.to_period('M')
    
    # --- 3. Ensure we have 5 years of data to analyze ---
    # If we don't have 5 years of data in the current file, we'll work with what we have
    all_periods = sorted(df['year_month'].unique())
    st.write(f"Data spans {len(all_periods)} months from {all_periods[0]} to {all_periods[-1]}")
    
    # --- 4. Create customer tiers based on credit score ---
    try:
        df['customer_tier'] = pd.qcut(
            df['credit_score'], 
            q=3, 
            labels=["Bronze", "Silver", "Gold"],
            duplicates='drop'  # Handle duplicate bin edges
        )
    except ValueError:
        # Fallback if there are too few unique values
        st.warning("Not enough unique credit score values. Using manual tier assignment.")
        # Manual assignment based on value ranges
        conditions = [
            (df['credit_score'] <= df['credit_score'].quantile(0.33)),
            (df['credit_score'] > df['credit_score'].quantile(0.33)) & 
            (df['credit_score'] <= df['credit_score'].quantile(0.66)),
            (df['credit_score'] > df['credit_score'].quantile(0.66))
        ]
        choices = ["Bronze", "Silver", "Gold"]
        df['customer_tier'] = np.select(conditions, choices, default="Silver")
    
    # --- 5. Aggregate data per customer-product pair for each period ---
    results = []
    
    for period in all_periods:
        df_period = df[df['year_month'] == period].copy()
        
        grouped = df_period.groupby(['customer_id', 'customer_name', 'product_id', 'product_name'], observed=False).agg({
            'product_cost': 'mean',
            'product_price': 'mean',
            'order_quantity': 'sum',
            'credit_score': 'first',
            'customer_tier': 'first'
        }).reset_index()
        
        # Calculate transaction value
        grouped['total_transaction_value'] = grouped['product_price'] * grouped['order_quantity']
        
        # Create product tiers based on product cost
        # Handle duplicate values by adding the duplicates parameter
        try:
            grouped['product_tier'] = pd.qcut(
                grouped['product_cost'], 
                q=3, 
                labels=["Economy", "Standard", "Premium"],
                duplicates='drop'  # Add this to handle duplicate bin edges
            )
        except ValueError:
            # Fallback if there are too few unique values
            st.warning(f"Not enough unique values for product costs in period {period}. Using manual tier assignment.")
            # Manual assignment based on value ranges
            conditions = [
                (grouped['product_cost'] <= grouped['product_cost'].quantile(0.33)),
                (grouped['product_cost'] > grouped['product_cost'].quantile(0.33)) & 
                (grouped['product_cost'] <= grouped['product_cost'].quantile(0.66)),
                (grouped['product_cost'] > grouped['product_cost'].quantile(0.66))
            ]
            choices = ["Economy", "Standard", "Premium"]
            grouped['product_tier'] = np.select(conditions, choices, default="Standard")
        
        # Dynamic pricing rule with enhanced factors
        def calculate_dynamic_price(row):
            base_cost = row['product_cost']
            customer_tier = row['customer_tier']
            product_tier = row['product_tier']
            volume = row['order_quantity']
            credit_score = row['credit_score']
            
            # Base margin starts based on customer tier
            if customer_tier == 'Gold':
                margin = 0.15  # Lower margin for high-value customers
            elif customer_tier == 'Silver':
                margin = 0.25  # Medium margin
            else:  # Bronze
                margin = 0.35  # Higher margin for lower-tier customers
            
            # Adjust based on order volume
            volume_adjustment = min(volume / 1000, 0.05)
            
            # Apply volume discount for better customers, increase for others
            if customer_tier == 'Gold':
                margin -= volume_adjustment
            elif customer_tier == 'Bronze':
                margin += volume_adjustment
            
            # Product tier adjustment
            if product_tier == 'Premium':
                margin += 0.05  # Premium products can carry higher margins
            elif product_tier == 'Economy':
                margin -= 0.02  # Economy products need competitive pricing
            
            # Credit risk adjustment
            if credit_score < 600:
                margin += 0.03  # Higher margin for riskier customers
                
            return round(base_cost * (1 + margin), 2)
        
        # Apply pricing logic
        grouped['recommended_price'] = grouped.apply(calculate_dynamic_price, axis=1)
        grouped['total_revenue'] = grouped['recommended_price'] * grouped['order_quantity']
        grouped['total_cost'] = grouped['product_cost'] * grouped['order_quantity']
        grouped['gross_margin'] = grouped['total_revenue'] - grouped['total_cost']
        grouped['margin_percentage'] = (grouped['gross_margin'] / grouped['total_revenue'] * 100).round(2)
        grouped['year_month'] = str(period)
        
        results.append(grouped)
    
    # Combine all periods
    final_df = pd.concat(results, ignore_index=True)
    
    # --- 6. Return the final processed dataframe ---
    st.write(f"Processing complete. Generated recommendations for {len(final_df)} customer-product combinations")
    
    return final_df

# -------------------------------------------------
# DASHBOARD FUNCTIONS
# -------------------------------------------------

@st.cache_data
def load_data():
    try:
        # Process data directly from 3000_Varied_Product_Transactions.csv
        with st.spinner("Processing data from 3000_Varied_Product_Transactions.csv... This may take a moment."):
            df = process_dynamic_pricing('3000_Varied_Product_Transactions.csv')
            # Ensure year_month is properly formatted for display
            if 'year_month' in df.columns:
                df['year_month'] = pd.to_datetime(df['year_month']).dt.strftime('%Y-%m')
            return df
    except FileNotFoundError:
        st.error("3000_Varied_Product_Transactions.csv not found. Please ensure this file is in the current directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.exception(e)  # Show full exception details for better debugging
        st.stop()

try:
    df = load_data()
    
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
    else:
        # Key metrics row
        st.header("Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric(
            "Avg Recommended Price", 
            f"${filtered_df['recommended_price'].mean():.2f}",
            f"{((filtered_df['recommended_price'].mean() / filtered_df['product_cost'].mean()) - 1) * 100:.1f}%"
        )
        
        col2.metric(
            "Total Gross Margin", 
            f"${filtered_df['gross_margin'].sum():,.2f}"
        )
        
        col3.metric(
            "Avg Margin %", 
            f"{filtered_df['margin_percentage'].mean():.1f}%"
        )
        
        col4.metric(
            "Avg Credit Score", 
            f"{filtered_df['credit_score'].mean():.0f}"
        )
        
        # Trend analysis by product
        st.header("Product Pricing Trends Over Time")
        
        # Get top products by volume for trend analysis
        top_products = filtered_df.groupby('product_name', observed=False)['order_quantity'].sum().nlargest(10).index.tolist()
        
        # Allow users to select products to view
        selected_trend_products = st.multiselect(
            "Select products to view trends (top 10 by volume shown by default)",
            options=sorted(filtered_df['product_name'].unique()),
            default=top_products[:5] if len(top_products) >= 5 else top_products
        )
        
        if selected_trend_products:
            # Filter for selected products
            product_trend_df = filtered_df[filtered_df['product_name'].isin(selected_trend_products)]
            
            # Group by product and month
            product_trends = product_trend_df.groupby(['year_month', 'product_name'], observed=False).agg({
                'recommended_price': 'mean',
                'margin_percentage': 'mean',
                'order_quantity': 'sum'
            }).reset_index()
            
            # Plot product price trends
            st.subheader("Product Price Trends")
            fig = px.line(
                product_trends,
                x='year_month',
                y='recommended_price',
                color='product_name',
                title='Recommended Price by Product Over Time',
                markers=True,
                hover_data=['order_quantity']
            )
            fig.update_layout(
                xaxis_title='Month',
                yaxis_title='Price ($)',
                legend_title='Product',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Plot product margin trends
            st.subheader("Product Margin Trends")
            fig = px.line(
                product_trends,
                x='year_month',
                y='margin_percentage',
                color='product_name',
                title='Margin Percentage by Product Over Time',
                markers=True,
                hover_data=['order_quantity']
            )
            fig.update_layout(
                xaxis_title='Month',
                yaxis_title='Margin %',
                legend_title='Product',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Add volume bubble chart for price vs margin
            st.subheader("Price vs. Margin by Sales Volume")
            
            bubble_df = product_trend_df.groupby('product_name', observed=False).agg({
                'recommended_price': 'mean',
                'margin_percentage': 'mean',
                'order_quantity': 'sum',
                'product_tier': 'first'
            }).reset_index()
            
            fig = px.scatter(
                bubble_df,
                x='recommended_price',
                y='margin_percentage',
                size='order_quantity',
                color='product_tier',
                hover_name='product_name',
                size_max=50,
                color_discrete_map={
                    'Premium': '#4B0082',
                    'Standard': '#1E90FF',
                    'Economy': '#32CD32'
                },
                title='Price vs Margin (bubble size = sales volume)'
            )
            fig.update_layout(
                xaxis_title='Average Recommended Price ($)',
                yaxis_title='Average Margin %',
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select at least one product to view trends")
        
        # Tier Analysis
        st.header("Tier Analysis")
        
        tab1, tab2, tab3 = st.tabs(["Customer Tier", "Product Tier", "Combined"])
        
        with tab1:
            # Customer tier analysis
            col1, col2 = st.columns(2)
            
            # Gross margin by customer tier
            with col1:
                customer_margin = filtered_df.groupby('customer_tier', observed=False)['gross_margin'].sum().reset_index()
                fig = px.bar(
                    customer_margin,
                    x='customer_tier',
                    y='gross_margin',
                    title='Gross Margin by Customer Tier',
                    color='customer_tier',
                    color_discrete_map={
                        'Gold': 'gold',
                        'Silver': 'silver',
                        'Bronze': '#CD7F32'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Average price by customer tier
            with col2:
                customer_price = filtered_df.groupby('customer_tier', observed=False)['recommended_price'].mean().reset_index()
                fig = px.bar(
                    customer_price,
                    x='customer_tier',
                    y='recommended_price',
                    title='Average Recommended Price by Customer Tier',
                    color='customer_tier',
                    color_discrete_map={
                        'Gold': 'gold',
                        'Silver': 'silver',
                        'Bronze': '#CD7F32'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Product tier analysis
            col1, col2 = st.columns(2)
            
            # Gross margin by product tier
            with col1:
                product_margin = filtered_df.groupby('product_tier', observed=False)['gross_margin'].sum().reset_index()
                fig = px.bar(
                    product_margin,
                    x='product_tier',
                    y='gross_margin',
                    title='Gross Margin by Product Tier',
                    color='product_tier',
                    color_discrete_map={
                        'Premium': '#4B0082',
                        'Standard': '#1E90FF',
                        'Economy': '#32CD32'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Average price by product tier
            with col2:
                product_price = filtered_df.groupby('product_tier', observed=False)['recommended_price'].mean().reset_index()
                fig = px.bar(
                    product_price,
                    x='product_tier',
                    y='recommended_price',
                    title='Average Recommended Price by Product Tier',
                    color='product_tier',
                    color_discrete_map={
                        'Premium': '#4B0082',
                        'Standard': '#1E90FF',
                        'Economy': '#32CD32'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Combined tier analysis (heatmap)
            combined_margin = filtered_df.groupby(['customer_tier', 'product_tier'], observed=False)['margin_percentage'].mean().reset_index().pivot(
                index='customer_tier',
                columns='product_tier',
                values='margin_percentage'
            )
            
            fig = px.imshow(
                combined_margin,
                title='Average Margin % by Customer & Product Tier',
                color_continuous_scale='Blues',
                text_auto='.1f',
                labels=dict(x="Product Tier", y="Customer Tier", color="Margin %")
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Pricing Table
        st.header("Detailed Pricing Table")
        
        # Add tier information without using complex HTML styling
        # Instead, we'll use simpler text indicators that work with pandas
        
        # Select columns to display
        display_cols = [
            'year_month', 'customer_name', 'customer_tier', 'product_name', 'product_tier',
            'product_cost', 'recommended_price', 'margin_percentage', 'credit_score'
        ]
        
        display_df = filtered_df[display_cols].sort_values(by=['year_month', 'customer_name'], ascending=[False, True])
        
        # Format the display dataframe
        display_df['product_cost'] = display_df['product_cost'].map('${:.2f}'.format)
        display_df['recommended_price'] = display_df['recommended_price'].map('${:.2f}'.format)
        display_df['margin_percentage'] = display_df['margin_percentage'].map('{:.1f}%'.format)
        
        # Display the table without complex styling that can cause errors
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Show the tier color code explanation beneath the table
        st.markdown("""
        <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; margin-bottom: 20px;">
            <div style="display: flex; align-items: center;">
                <span style="background-color: gold; width: 20px; height: 20px; display: inline-block; margin-right: 5px; border-radius: 3px;"></span>
                <span>Gold Customer</span>
            </div>
            <div style="display: flex; align-items: center;">
                <span style="background-color: silver; width: 20px; height: 20px; display: inline-block; margin-right: 5px; border-radius: 3px;"></span>
                <span>Silver Customer</span>
            </div>
            <div style="display: flex; align-items: center;">
                <span style="background-color: #CD7F32; width: 20px; height: 20px; display: inline-block; margin-right: 5px; border-radius: 3px;"></span>
                <span>Bronze Customer</span>
            </div>
            <div style="display: flex; align-items: center;">
                <span style="background-color: #4B0082; width: 20px; height: 20px; display: inline-block; margin-right: 5px; border-radius: 3px; color: white;"></span>
                <span>Premium Product</span>
            </div>
            <div style="display: flex; align-items: center;">
                <span style="background-color: #1E90FF; width: 20px; height: 20px; display: inline-block; margin-right: 5px; border-radius: 3px;"></span>
                <span>Standard Product</span>
            </div>
            <div style="display: flex; align-items: center;">
                <span style="background-color: #32CD32; width: 20px; height: 20px; display: inline-block; margin-right: 5px; border-radius: 3px;"></span>
                <span>Economy Product</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk Assessment Section
        st.header("⚠️ Risk Assessment")
        
        # Convert back to numeric for filtering
        risk_df = filtered_df.copy()
        
        # Identify high-risk customers (low credit score, high margin)
        high_risk = risk_df[
            (risk_df['credit_score'] < 600) & 
            (risk_df['margin_percentage'] > 30)
        ].sort_values('credit_score')
        
        if not high_risk.empty:
            st.warning(f"Found {len(high_risk)} high-risk customer-product combinations")
            
            # Display risk table
            risk_cols = [
                'year_month', 'customer_name', 'customer_tier', 'product_name', 
                'credit_score', 'margin_percentage', 'recommended_price'
            ]
            
            risk_display = high_risk[risk_cols].copy()
            risk_display['recommended_price'] = risk_display['recommended_price'].map('${:.2f}'.format)
            risk_display['margin_percentage'] = risk_display['margin_percentage'].map('{:.1f}%'.format)
            
            st.dataframe(risk_display, use_container_width=True)
        else:
            st.success("No high-risk customers in the current selection")
        
        # Download section
        st.header("Download Data")
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name=f"dynamic_pricing_export_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

except Exception as e:
    st.error(f"Error loading data: {e}")
    
    # Show troubleshooting instructions
    st.info("""
    ### Troubleshooting Steps:
    
    1. Verify the file `3000_Varied_Product_Transactions.csv` exists in the current directory
    2. Check that the CSV file has the expected columns (date_of_purchase, payment_due, etc.)
    3. Restart this dashboard
    """)

# Instructions for running the app
if st.sidebar.checkbox("Show instructions"):
    st.sidebar.markdown("""
    ### How to use this dashboard:
    
    1. Place the `3000_Varied_Product_Transactions.csv` file in the same directory as this script
    2. Run the app with `streamlit run dynamic_pricing_dashboard_all_in_one.py`
    3. Use the filters on the left to analyze different segments of your data
    """)