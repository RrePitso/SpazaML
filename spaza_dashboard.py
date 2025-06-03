
# spaza_dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
import holidays # For South African holidays
import kagglehub
from kagglehub import KaggleDatasetAdapter
import warnings

# Suppress the DeprecationWarning for load_dataset
warnings.filterwarnings("ignore", category=DeprecationWarning, module="kagglehub")


# --- 1. Data Loading and Preprocessing (Cached for performance) ---
@st.cache_data # Cache the data loading and initial processing
def load_and_preprocess_data():
    """Loads the dataset and performs initial preprocessing."""
    try:
        # --- IMPORTANT CHANGE: New file path ---
        file_path = "point_of_sale_dataset.csv"
        # --- IMPORTANT CHANGE: New dataset identifier ---
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "smmmmmmmmmmmm/point-of-sales", # This matches the dataset you downloaded
            file_path,
        )
        st.success("Dataset loaded successfully!")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop() # Stop the app if data can't be loaded

    # --- IMPORTANT CHANGE: New date column name and format ---
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d') # Assuming YYYY-MM-DD format from your output

    # Aggregate to daily sales per Product_Name (instead of category)
    # --- IMPORTANT CHANGE: Group by 'Date' and 'Product_Name' ---
    daily_product_sales = df.groupby([df['Date'].dt.date, 'Product_Name'])['Quantity'].sum().reset_index()
    daily_product_sales['Date'] = pd.to_datetime(daily_product_sales['Date']) # Convert back to datetime for Prophet

    # Rename columns to match expected for Prophet's 'category' equivalent
    daily_product_sales = daily_product_sales.rename(columns={'Date': 'invoice_date', 'Product_Name': 'category', 'Quantity': 'quantity'})

    return daily_product_sales

# --- 2. Prophet Model Training (Cached for performance) ---
@st.cache_resource # Cache the trained model
def train_prophet_model(df_sales, selected_product_name): # Renamed selected_category to selected_product_name for clarity
    """Trains a Prophet model for a given product name."""
    # --- IMPORTANT CHANGE: Filter by 'category' (which is now Product_Name) ---
    product_df = df_sales[df_sales['category'] == selected_product_name].copy()

    if product_df.empty:
        return None, None # No data for this product

    # Ensure all dates are present in the time series (fill missing with 0)
    start_date = product_df['invoice_date'].min()
    end_date = product_df['invoice_date'].max()
    full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    full_product_df = pd.DataFrame({'invoice_date': full_date_range})
    # --- IMPORTANT CHANGE: Set 'category' column to selected_product_name ---
    full_product_df['category'] = selected_product_name

    # --- IMPORTANT CHANGE: Merge on 'invoice_date' and 'category' (Product_Name) ---
    product_df_filled = pd.merge(full_product_df, product_df, on=['invoice_date', 'category'], how='left')
    product_df_filled['quantity'] = product_df_filled['quantity'].fillna(0)

    # Prepare data for Prophet
    prophet_df = product_df_filled.rename(columns={'invoice_date': 'ds', 'quantity': 'y'})

    # Get SA holidays (for demonstration of capability)
    # Adjust years based on your dataset's date range and current year
    # Data is from Jan 2024, so include 2024 and 2025 for future forecast
    sa_holidays = holidays.country_holidays(
        'ZA',
        years=range(prophet_df['ds'].min().year, prophet_df['ds'].max().year + 2)
    )
    sa_holidays_df = pd.DataFrame({
        'holiday': 'ZA_Holiday',
        'ds': pd.to_datetime(list(sa_holidays.keys())),
        'lower_window': 0,
        'upper_window': 0,
    })

    model = Prophet(
        yearly_seasonality=True, # May not be strong with only 20 days of data
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=sa_holidays_df
    )
    model.fit(prophet_df)

    # Make future dataframe for 30 days (for next month's forecast)
    future = model.make_future_dataframe(periods=30, freq='D')

    forecast = model.predict(future)

    return model, forecast

# --- Streamlit App Layout ---
st.set_page_config(
    page_title="Spaza Shop Insights Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Spaza Shop Business Intelligence Demo 🇿🇦")
st.markdown("### Powered by Data Science & Machine Learning")

# Sidebar for navigation
st.sidebar.header("Navigation")
page_selection = st.sidebar.radio(
    "Choose a section:",
    ["📈 Demand Forecasting", "📦 Inventory Management", "🤝 Bulk Buying Advantage"]
)

# Load data once at the start of the app run
daily_product_sales_df = load_and_preprocess_data()
# --- IMPORTANT CHANGE: Use 'category' column, which is now Product_Name ---
available_products = sorted(daily_product_sales_df['category'].unique().tolist())

# --- Page 1: Demand Forecasting ---
if page_selection == "📈 Demand Forecasting":
    st.header("Predicting What Your Customers Will Buy")
    st.write(
        "Never run out of popular items or have too much unsold stock. "
        "This tool predicts how much of each product you'll sell in the coming weeks."
    )

    st.markdown("---")

    # --- IMPORTANT CHANGE: Select a Product (not category) ---
    selected_product = st.selectbox(
        "Select a Product to Forecast:",
        available_products,
        index=0 # Default to the first product
    )

    if selected_product:
        with st.spinner(f"Generating forecast for {selected_product}..."):
            # --- IMPORTANT CHANGE: Pass selected_product to the model training function ---
            model, forecast_df = train_prophet_model(daily_product_sales_df, selected_product)

            if model and forecast_df is not None:
                st.subheader(f"Sales Forecast for '{selected_product}'")

                # Plotting with Plotly for interactivity
                fig = px.line(
                    forecast_df,
                    x='ds',
                    y='yhat',
                    title=f'Predicted Daily Sales for {selected_product}',
                    labels={'ds': 'Date', 'yhat': 'Predicted Quantity Sold'}
                )
                fig.add_scatter(x=model.history['ds'], y=model.history['y'], mode='markers', name='Actual Sales',
                                marker=dict(size=4, opacity=0.6))
                fig.update_layout(hovermode="x unified", legend_title_text="Legend")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### Forecasted Quantities for the Next 30 Days:")
                # Filter forecast to show only future dates
                future_forecast = forecast_df[forecast_df['ds'] > daily_product_sales_df['invoice_date'].max()].copy()
                future_forecast['ds'] = future_forecast['ds'].dt.date # Display date only
                st.dataframe(future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                             .rename(columns={'ds': 'Date', 'yhat': 'Predicted Sales',
                                              'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound'})
                             .set_index('Date')
                )
                st.info(
                    "The `Predicted Sales` is the most likely quantity. "
                    "The `Lower Bound` and `Upper Bound` give you a range of possible sales."
                )
            else:
                st.warning(f"No sufficient data to generate forecast for '{selected_product}'. Please select another product.")
    else:
        st.info("Please select a product to view its demand forecast.")


# --- Page 2: Inventory Management (Simulated) ---
elif page_selection == "📦 Inventory Management":
    st.header("Smart Inventory Management for Your Spaza Shop")
    st.write(
        "Stop guessing! Our system helps you visualize your stock levels and provides smart reorder suggestions "
        "to ensure you always have what your customers want, when they want it."
    )

    st.markdown("---")

    st.subheader("Simulated Stock Status & Reorder Alerts")
    st.info(
        "This section shows how the system would work. "
        "With your actual sales data, we can give you precise, real-time alerts."
    )

    # --- IMPORTANT CHANGE: Use Product names from the new dataset or common grocery items ---
    # We'll use a mix of product names from your new dataset for better demo relevance
    # Your new dataset has 'Product_1', 'Product_2', etc. Let's make it more realistic.
    # Replace these with actual common grocery items for a better demo
    simulated_products_data = {
        'Product': ['Milk 1L', 'White Bread', 'Cola 330ml Can', 'Chips (Large)', 'Bottled Water 500ml'],
        'Current Stock (Units)': [50, 10, 100, 150, 75],
        'Daily Predicted Sales': [30, 25, 40, 35, 20], # Adjusted to be higher to reflect real shop sales
        'Reorder Threshold (Days of Sales)': [2, 1, 3, 2, 4] # Example threshold
    }
    simulated_df = pd.DataFrame(simulated_products_data)

    simulated_df['Reorder Point (Units)'] = simulated_df['Daily Predicted Sales'] * simulated_df['Reorder Threshold (Days of Sales)']
    simulated_df['Status'] = '🟢 Good Stock'
    simulated_df['Reorder Suggestion'] = 'None'

    for index, row in simulated_df.iterrows():
        if row['Current Stock (Units)'] < row['Reorder Point (Units)']:
            if row['Current Stock (Units)'] <= row['Daily Predicted Sales'] * 1.5: # Very low, urgent
                simulated_df.loc[index, 'Status'] = '🔴 Critical Stock - Reorder ASAP!'
                simulated_df.loc[index, 'Reorder Suggestion'] = f"Order {row['Daily Predicted Sales'] * 7} units (approx. 1 week supply)"
            else: # Low, but not critical
                simulated_df.loc[index, 'Status'] = '🟡 Low Stock - Consider Reordering'
                simulated_df.loc[index, 'Reorder Suggestion'] = f"Order {row['Daily Predicted Sales'] * 5} units (approx. 5 days supply)"

    st.dataframe(simulated_df[['Product', 'Current Stock (Units)', 'Daily Predicted Sales', 'Status', 'Reorder Suggestion']]
                 .set_index('Product'))

    st.markdown("""
    **How this helps your business:**
    * **Reduce Stockouts:** Never miss a sale because an item isn't on the shelf.
    * **Minimize Waste:** Avoid holding too much stock, especially for perishable items, reducing losses.
    * **Improve Cash Flow:** Don't tie up your money in inventory that isn't selling quickly.
    """)

# --- Page 3: Bulk Buying Advantage (Conceptual) ---
elif page_selection == "🤝 Bulk Buying Advantage":
    st.header("Unlock Bigger Savings with Collective Buying Power")
    st.write(
        "Imagine uniting with other spaza shops in your area. By combining your predicted demands, "
        "you can place larger orders with wholesalers and unlock significant discounts, just like big supermarkets!"
    )

    st.markdown("---")

    st.subheader("The Power of Aggregation (Conceptual Example)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Scenario A: Individual Ordering")
        st.image("https://via.placeholder.com/300x150?text=Shop+A+Order+-+Expensive", caption="Spaza Shop A Orders Individually")
        st.image("https://via.placeholder.com="https://via.placeholder.com/300x150?text=Shop+B+Order+-+Expensive", caption="Spaza Shop B Orders Individually")
        st.markdown("""
        * Each shop orders small quantities.
        * **Wholesaler Price:** R100 per bag of Maize Meal.
        * **Result:** Higher unit cost, less profit per item.
        """)

    with col2:
        st.markdown("#### Scenario B: Collective Ordering (with our platform)")
        st.image("https://via.placeholder.com/300x150?text=Combined+Order+-+Cheaper", caption="Multiple Spaza Shops Order Together")
        st.markdown("""
        * Our system aggregates predicted demand from multiple shops.
        * **Combined Order Volume:** Enough to qualify for bulk discount.
        * **Wholesaler Price:** R80 per bag of Maize Meal (example discount).
        * **Result:** **Significant savings for *each* participating shop!**
        """)

    st.markdown("---")
    st.markdown("""
    **How this can boost your profits:**
    * **Lower Procurement Costs:** Buy at wholesale prices usually reserved for large retailers.
    * **Increased Margins:** Sell items at competitive prices while still making more profit.
    * **Reduced Transport Costs:** Potentially optimized delivery for bulk orders.

    This feature would thrive as more shops join our platform, creating a powerful collective buying network.
    """)
