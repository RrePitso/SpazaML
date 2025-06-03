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
        file_path = "Different_stores_data_V2.csv"
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "kzmontage/sales-from-different-stores",
            file_path,
        )
        st.success("Dataset loaded successfully!")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop() # Stop the app if data can't be loaded

    df['invoice_date'] = pd.to_datetime(df['invoice_date'], format='%m/%d/%Y %H:%M')

    # Aggregate to daily sales per category
    daily_category_sales = df.groupby([df['invoice_date'].dt.date, 'category'])['quantity'].sum().reset_index()
    daily_category_sales['invoice_date'] = pd.to_datetime(daily_category_sales['invoice_date'])

    return daily_category_sales

# --- 2. Prophet Model Training (Cached for performance) ---
@st.cache_resource # Cache the trained model
def train_prophet_model(df_sales, selected_category):
    """Trains a Prophet model for a given category."""
    category_df = df_sales[df_sales['category'] == selected_category].copy()

    if category_df.empty:
        return None, None # No data for this category

    # Ensure all dates are present in the time series (fill missing with 0)
    start_date = category_df['invoice_date'].min()
    end_date = category_df['invoice_date'].max()
    full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    full_category_df = pd.DataFrame({'invoice_date': full_date_range})
    full_category_df['category'] = selected_category

    category_df_filled = pd.merge(full_category_df, category_df, on=['invoice_date', 'category'], how='left')
    category_df_filled['quantity'] = category_df_filled['quantity'].fillna(0)

    # Prepare data for Prophet
    prophet_df = category_df_filled.rename(columns={'invoice_date': 'ds', 'quantity': 'y'})

    # Get SA holidays (for demonstration of capability)
    # Adjust years based on your dataset's date range and current year
    sa_holidays = holidays.country_holidays(
        'ZA',
        years=range(prophet_df['ds'].min().year, prophet_df['ds'].max().year + 2) # Include years for future forecast
    )
    sa_holidays_df = pd.DataFrame({
        'holiday': 'ZA_Holiday',
        'ds': pd.to_datetime(list(sa_holidays.keys())),
        'lower_window': 0,
        'upper_window': 0,
    })

    model = Prophet(
        yearly_seasonality=True,
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
daily_category_sales_df = load_and_preprocess_data()
available_categories = sorted(daily_category_sales_df['category'].unique().tolist())

# --- Page 1: Demand Forecasting ---
if page_selection == "📈 Demand Forecasting":
    st.header("Predicting What Your Customers Will Buy")
    st.write(
        "Never run out of popular items or have too much unsold stock. "
        "This tool predicts how much of each product category you'll sell in the coming weeks."
    )

    st.markdown("---")

    selected_category = st.selectbox(
        "Select a Product Category to Forecast:",
        available_categories,
        index=available_categories.index('Food & Beverage') if 'Food & Beverage' in available_categories else 0
    )

    if selected_category:
        with st.spinner(f"Generating forecast for {selected_category}..."):
            model, forecast_df = train_prophet_model(daily_category_sales_df, selected_category)

            if model and forecast_df is not None:
                st.subheader(f"Sales Forecast for '{selected_category}'")

                # Plotting with Plotly for interactivity
                fig = px.line(
                    forecast_df,
                    x='ds',
                    y='yhat',
                    title=f'Predicted Daily Sales for {selected_category}',
                    labels={'ds': 'Date', 'yhat': 'Predicted Quantity Sold'}
                )
                fig.add_scatter(x=model.history['ds'], y=model.history['y'], mode='markers', name='Actual Sales',
                                marker=dict(size=4, opacity=0.6))
                fig.update_layout(hovermode="x unified", legend_title_text="Legend")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### Forecasted Quantities for the Next 30 Days:")
                # Filter forecast to show only future dates
                future_forecast = forecast_df[forecast_df['ds'] > daily_category_sales_df['invoice_date'].max()].copy()
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
                st.warning(f"No sufficient data to generate forecast for '{selected_category}'. Please select another category.")
    else:
        st.info("Please select a category to view its demand forecast.")


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

    # Simulate some product data and apply reorder logic based on a hypothetical forecast
    simulated_products_data = {
        'Product': ['Cooking Oil 2L', 'Maize Meal 10kg', 'Soap Bar', 'Soft Drink 330ml', 'Bread Loaf'],
        'Current Stock (Units)': [50, 10, 100, 150, 5],
        'Daily Predicted Sales': [3, 2, 5, 10, 4], # Hypothetical daily forecast from the model
        'Reorder Threshold (Days of Sales)': [7, 5, 10, 7, 2] # Example threshold
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
        st.image("https://via.placeholder.com/300x150?text=Shop+B+Order+-+Expensive", caption="Spaza Shop B Orders Individually")
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
