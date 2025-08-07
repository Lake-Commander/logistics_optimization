## ⚠️ Disclaimer
> This application is strictly for educational and research purposes.

# 🚚 SwiftChain Delivery Delay Prediction App

A user-friendly Streamlit application designed to predict delivery delays for SwiftChain Analytics using machine learning. The app accepts detailed order and shipping information and returns whether a delivery is expected to be delayed.

---

## 📌 Features

- ✅ **Delivery Delay Prediction** using a trained ML model.
- 📊 **Interactive Dashboard** with visual insights from EDA:
  - Delivery Status Distribution
  - Discount vs Delivery Status
  - Missing Values Matrix
  - Shipping Mode vs Delivery Status
  - Top 10 Regions vs Delivery Status
- 📋 **Input Schema Tab** to guide users on required fields.

---

## 🧠 Model Information

The model was trained on historical shipping data using a classification algorithm. Key features used include shipping delays, order dates, discounts, regions, and customer segments.

---

## 📥 Expected Input Schema

All fields below are required for prediction. Ensure correct formatting (especially for dates and categorical fields).

| Column Name                 | Description                                                   |
|----------------------------|---------------------------------------------------------------|
| payment_type               | Mode of payment (e.g., Credit Card, Cash)                     |
| profit_per_order           | Profit made per order                                         |
| sales_per_customer         | Total sales value per customer                                |
| category_id                | ID of the product category                                    |
| category_name              | Name of the product category                                  |
| customer_city              | Customer's city                                               |
| customer_country           | Customer's country                                            |
| customer_id                | Unique customer ID                                            |
| customer_segment           | Customer segmentation (e.g., Consumer, Corporate)             |
| customer_state             | Customer's state                                              |
| customer_zipcode           | Customer's ZIP code                                           |
| department_id              | ID of the department selling the product                      |
| department_name            | Name of the department                                        |
| latitude                   | Shipping location latitude                                    |
| longitude                  | Shipping location longitude                                   |
| market                     | Geographic market segment (e.g., Africa, EU)                 |
| order_city                 | City where order was placed                                   |
| order_country              | Country where order was placed                                |
| order_customer_id          | Customer ID used in the order                                 |
| order_date                 | Date when the order was placed (YYYY-MM-DD)                   |
| order_id                   | Unique identifier of the order                                |
| order_item_cardprod_id     | Product ID in the order                                       |
| order_item_discount        | Discount amount on the item                                   |
| order_item_discount_rate   | Discount as a percentage                                      |
| order_item_id              | ID of the order item                                          |
| order_item_product_price   | Price of the product                                          |
| order_item_profit_ratio    | Profit ratio on the product                                   |
| order_item_quantity        | Quantity of the item ordered                                  |
| sales                      | Total sales amount                                            |
| order_item_total_amount    | Total amount for the item                                     |
| order_profit_per_order     | Profit for this entire order                                  |
| order_region               | Region of the order                                           |
| order_state                | State of the order                                            |
| order_status               | Status (e.g., Completed, Cancelled)                           |
| product_card_id            | ID of the product card                                        |
| product_category_id        | Category ID of the product                                    |
| product_name               | Name of the product                                           |
| product_price              | Listed price of the product                                   |
| shipping_date              | Date when the item was shipped (YYYY-MM-DD)                   |
| shipping_mode              | Mode of shipping (e.g., First Class, Standard)                |
| is_late_shipping           | Boolean: Was the shipping late? *(auto-generated)*            |
| is_weekend_shipping        | Boolean: Did shipping occur on a weekend? *(auto-generated)*  |
| order_weekday              | Day of week order was placed (0=Monday, 6=Sunday) *(auto)*    |
| shipping_delay_days        | Days between order and shipping *(auto-generated)*            |

---

## 📁 Project Structure

```
📦logistics_optimization/
┣ 📂eda/
┃ ┗ 📂plots/
┃ ┣ 📜delivery_status_distribution.png
┃ ┣ 📜discount_vs_label.png
┃ ┣ 📜missing_values_matrix.png
┃ ┣ 📜shipping_mode_vs_label.png
┃ ┗ 📜top10_regions_vs_label.png
┣ 📜model.pkl
┣ 📜scaler.pkl
┣ 📜app.py
┣ 📜logistics.csv
┗ 📜README.md
```

---

## ▶️ Getting Started

### ✅ Requirements

- Python 3.8+
- Streamlit
- Pandas
- Scikit-learn
- Joblib
- PIL

### 🚀 Run the App

```bash
streamlit run app.py


## 💻 Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/Lake-Commander/logistics_optimization.git
cd LLM_fake_news
```

###2. Install dependencies
Make sure you have Python 3.13 installed.
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

## 🧪 Train Your Own Model
**To train a new model with your dataset:**

1. Prepare your dataset in CSV format. Use the sample datasets here for schema reference.

2. Open and run logistics_optimization.ipynb in Jupyter Notebook.

3. Follow the steps to preprocess, train, and save the model.

4. Ensure your app loads the new model path.

## 🌐 Deployed App
Access the live app:
👉 [Click here to open the app]([https://llm-fake-news.streamlit.app/](https://logisticsoptimization.streamlit.app/)).

## 🙏 Acknowledgments
This project was built under the guidance and mentorship of the 3MTT (Three Million Technical Talent) program by the National Information Technology Development Agency (NITDA), Nigeria.

We sincerely appreciate NITDA and the Federal Ministry of Communications, Innovation and Digital Economy for the opportunity to learn, grow, and contribute to Nigeria’s digital transformation journey.

Thank you for empowering Nigerian youths with the skills to build real-world solutions.



