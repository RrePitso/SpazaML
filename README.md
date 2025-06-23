📊 Spaza Shop Demand Forecasting Dashboard
This interactive Streamlit dashboard uses machine learning (Prophet) to help spaza shop owners make smarter inventory decisions. It predicts product demand, simulates stock management, and demonstrates how collective purchasing power can increase profitability.

🚀 Live Demo
https://spazaml-6rnex2xkxidhmv2xrqjxnn.streamlit.app/

🧠 Project Overview
This dashboard was built as part of a broader mission to bring data science into the informal economy. South African spaza shops often rely on intuition for ordering stock. This app demonstrates how machine learning can empower small businesses with predictive analytics typically used by large retailers.

Key functionalities:

✅ Forecast demand for individual products using Facebook Prophet

📦 Simulate inventory levels and suggest reorder quantities

🤝 Show how cooperative bulk buying can lower procurement costs

📅 Integrate local (South African) public holidays into demand forecasting

📈 Visualize actual vs. predicted sales with Plotly

📂 Project Structure
text
Copy
Edit
spaza_dashboard.py          # Main Streamlit dashboard code
point_of_sale_dataset.csv   # Source dataset (via KaggleHub)
🛠️ Technologies Used
Streamlit

Prophet

Pandas

Plotly

Holidays

KaggleHub (for loading the dataset)

📉 Data Source
The dataset was sourced from Kaggle:
Dataset: smmmmmmmmmmmm/point-of-sales
Contains simulated point-of-sale transactions including Product_Name, Date, and Quantity.

🧩 Features
1. 📈 Demand Forecasting
Forecasts daily product sales for the next 30 days

Based on historical quantity sold

Includes South African public holidays to account for sales dips/spikes

2. 📦 Inventory Management Simulation
Compares current stock to predicted demand

Highlights low stock and critical reorder situations

Suggests how much to reorder to maintain supply

3. 🤝 Bulk Buying Advantage (Conceptual)
Demonstrates how collective orders can reduce unit costs

Encourages cooperative procurement among small shops

📍 Challenges & How I Solved Them
🔻 1. Limited Historical Data
Challenge: The dataset had very few rows per product — too little for stable time series modeling.
Solution:

Filled missing dates with 0 to ensure continuity

Skipped training for products with <2 non-zero sales days (added validation)

Simulated extended forecasts to maintain functionality

🛑 2. Inconsistent Product Names
Challenge: Different spelling or formats of product names across rows
Solution:

Normalized Product_Name as category

Aggregated data at daily level to smooth inconsistencies

🏷️ 3. No Categories or Hierarchies
Challenge: The dataset lacked product categories (e.g., food, drinks)
Solution:

Focused forecasting at the Product_Name level

Removed need for category-level breakdowns for simplicity

📊 Insights
Even a simple model can provide valuable foresight for small businesses.

Inventory management benefits greatly from predictive alerts.

Collective buying isn't just about logistics — data sharing enables cost savings.

✅ How to Run the App Locally
1. Clone the repo
bash
Copy
Edit
git clone https://github.com/yourusername/spaza-dashboard.git
cd spaza-dashboard
2. Install dependencies
bash
Copy
Edit
pip install streamlit prophet pandas plotly holidays kagglehub
3. Run the app
bash
Copy
Edit
streamlit run spaza_dashboard.py
🔑 Make sure your Kaggle API key is configured if you're pulling directly from Kaggle using kagglehub.

📌 Future Improvements
Upload custom sales CSVs directly from the app

Add support for shop-specific inventory tracking

Integrate actual cooperative ordering logic and user authentication

📬 Contact
Ofentse Pitso
📧 ofentsepitsopop@gmail.com
🌐 LinkedIn
📍 South Africa

📝 License
This project is open-source and free to use under the MIT License.

