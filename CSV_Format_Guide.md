# 📊 CLV Model CSV Format Guide

## 🎯 Sample File: `sample_customer_data.csv`

This file contains 100 sample customers with realistic data that will work perfectly with your CLV prediction model.

## 📋 Required Columns

Your CSV file must include these exact column names:

| Column Name | Data Type | Description | Example Values |
|-------------|-----------|-------------|----------------|
| `customer_id` | Text | Unique identifier for each customer | CUST_0001, CUST_0002, etc. |
| `age` | Integer | Customer age (18-120) | 25, 35, 42, 55 |
| `total_purchases` | Integer | Total number of purchases made | 5, 12, 25, 35 |
| `avg_order_value` | Decimal | Average order value in dollars | 55.80, 120.75, 250.00 |
| `days_since_first_purchase` | Integer | Days since first purchase (≥180) | 180, 450, 720, 850 |
| `days_since_last_purchase` | Integer | Days since last purchase (≤first_purchase) | 0, 15, 45, 120 |
| `acquisition_channel` | Text | How customer was acquired | online, store, social, referral |
| `location` | Text | Customer location | urban, suburban, rural |
| `subscription_status` | Text | Current subscription status | active, inactive, none |

## ✅ Data Validation Rules

The model will validate your data and show errors/warnings for:

### ❌ **Errors (Will Block Predictions):**
- Missing required columns
- Duplicate customer IDs
- Negative values in numeric fields
- `days_since_last_purchase` > `days_since_first_purchase`

### ⚠️ **Warnings (Will Allow Predictions):**
- Age values < 13 or > 120
- Unrealistic data patterns

## 🚀 How to Use the Sample File

1. **Download the file:** `sample_customer_data.csv` is now in your project directory
2. **Upload in the app:** 
   - Go to your CLV Predictor app
   - Select "Upload CSV File" in the sidebar
   - Choose `sample_customer_data.csv`
   - Click "Generate CLV Predictions"

## 📊 Sample Data Characteristics

The sample file includes:

- **100 customers** with realistic data distributions
- **Age range:** 24-55 years (realistic customer base)
- **Purchase patterns:** 5-35 total purchases per customer
- **Order values:** $52-$250 average order value
- **Customer lifetime:** 150-850 days since first purchase
- **Recency:** 0-150 days since last purchase
- **Diverse segments:** Mix of acquisition channels, locations, and subscription statuses

## 🔧 Creating Your Own CSV

To create your own customer data file:

1. **Use the exact column names** shown above
2. **Follow the data types** (integers for age/purchases/days, decimals for money)
3. **Ensure data quality:**
   - No duplicate customer IDs
   - Realistic age values (18-120)
   - Positive values only
   - Logical date relationships

## 📈 Expected Results

When you upload this sample file, you should see:

- ✅ **Data validation passed**
- 📊 **100 customers loaded**
- 🎯 **CLV predictions generated**
- 📈 **18+ comprehensive visualizations**
- 💰 **Customer segmentation analysis**
- ⚠️ **Risk assessment**
- 🎯 **Actionable insights**

## 🛠️ Troubleshooting

If you encounter issues:

1. **Check column names:** Must match exactly (case-sensitive)
2. **Verify data types:** Age, purchases, and days should be numbers
3. **Check for duplicates:** Each customer_id must be unique
4. **Validate relationships:** Last purchase ≤ First purchase

## 📝 Example Row

```csv
customer_id,age,total_purchases,avg_order_value,days_since_first_purchase,days_since_last_purchase,acquisition_channel,location,subscription_status
CUST_0001,35,12,85.50,450,15,online,urban,active
```

This customer is:
- 35 years old
- Made 12 purchases
- Average order value: $85.50
- First purchase: 450 days ago
- Last purchase: 15 days ago
- Acquired online, lives in urban area, has active subscription

---

**Ready to test?** Upload `sample_customer_data.csv` and see your CLV model in action! 🚀 