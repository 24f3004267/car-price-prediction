# 🚗 Used Car Price Prediction

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Library-Pandas-150458.svg)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)]()

An end-to-end Machine Learning pipeline that predicts used car prices using historical listings from Quikr. This project includes extensive Exploratory Data Analysis (EDA), robust text and numerical data cleaning, outlier mitigation, and automated model training with Scikit-Learn Pipelines.

---

## 📌 Project Overview

Predicting fair market value for second-hand vehicles is challenging due to the variance in car condition, age, usage (kilometers driven), brand equity, and fuel type.

The raw dataset contains messy web-scraped records with inconsistent string formats, non-numeric values, and extreme outliers. This repository establishes a reproducible, production-ready pipeline to clean raw records, encode categorical features, and train regression models.

---

## 📊 Dataset Description

The dataset consists of **892 original raw records** with the following 6 attributes:

| Feature | Type | Description | Raw Data Issues Found |
| :--- | :--- | :--- | :--- |
| `name` | Object | Full commercial vehicle name | Contains long descriptions, variants, trim levels |
| `company` | Object | Manufacturing brand / company | Corrupted strings in non-car entries |
| `year` | Object / Int | Manufacturing year | Contains invalid non-year string characters |
| `Price` | Object / Int | Selling price in INR (Target) | Formatted with commas and text (`"Ask For Price"`) |
| `kms_driven` | Object / Int | Distance driven in kilometers | Suffix `"kms"`, commas, and corrupted entries |
| `fuel_type` | Object | Fuel type (`Petrol`, `Diesel`, `LPG`) | Missing / NaN values on invalid rows |

---

## 🛠️ Data Cleaning & Preprocessing Workflow

1. **Price Standardization:**
   - Replaced `"Ask For Price"` text entries with `NaN`.
   - Stripped commas (`,`) from price values.
   - Converted to numeric integers and dropped null entries.

2. **Odometer Reading (`kms_driven`):**
   - Stripped commas and `"kms"` / `"Kms"` unit suffixes.
   - Converted string entries to integer and eliminated non-numeric entries.

3. **Year Cleaning:**
   - Coerced non-numeric string values to numeric `Int64`.
   - Filtered out records with missing valid registration years.

4. **Car Name Normalization:**
   - Raw names contained up to 10 words (e.g., `Hyundai Santro Xing XO eRLX Euro III`).
   - Kept the **first 3 words** (`Brand + Model Family`, e.g., `Hyundai Santro Xing`, `Maruti Suzuki Swift`) to reduce high cardinality while preserving essential vehicle identity.

5. **Missing Value Removal:**
   - Dropped null records in `fuel_type`, leaving a clean dataset of **816 valid listings**.

---

## 📈 Outlier Handling

During Exploratory Data Analysis (EDA), an extreme anomaly was detected:
- A listing with a price of **₹8,500,003** (~₹85 Lakhs / 8.5 Crore) was heavily distorting model residuals.
- **Remedy:** Filtered the dataset to retain vehicles with `Price < ₹6,000,000` (60 Lakhs), ensuring robust regression weights.

---

## 🤖 Model Pipeline Architecture

The end-to-end preprocessing and prediction are bundled into a Scikit-Learn `Pipeline` using `ColumnTransformer`:

```text
Raw Features [name, company, year, kms_driven, fuel_type]
                        │
                        ▼
             ┌─────────────────────┐
             │  ColumnTransformer  │
             └──────────┬──────────┘
                        │
       ┌────────────────┴────────────────┐
       ▼                                 ▼
Categorical Features             Numerical Features
['name', 'company', 'fuel_type'] ['year', 'kms_driven']
       │                                 │
       ▼                                 ▼
 [OneHotEncoder]                   [Passthrough]
 (handle_unknown='ignore')               │
       │                                 │
       └────────────────┬────────────────┘
                        │
                        ▼
                 Concatenated Array
                        │
                        ▼
          [Regressor (Linear / Random Forest)]
                        │
                        ▼
                 Predicted Price (₹)
```

---

## 🏆 Model Performance & Evaluation

Both models were trained on 80% of data and evaluated on an unseen 20% test holdout:

| Model | MAE (Mean Absolute Error) | RMSE (Root Mean Squared Error) | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Linear Regression (Baseline)** | ₹1,20,824 | ₹1,89,166 | 0.6667 (66.67%) |
| **Random Forest Regressor (200 Trees)** | **₹92,012** | **₹1,85,122** | **0.6808 (68.08%)** |

> **Key Takeaway:** The Random Forest Regressor reduced Mean Absolute Error by nearly **₹28,000**, capturing nonlinear brand and age interactions more effectively.

---

## 📁 Repository Structure

```text
car-price-prediction/
│
├── Data-set/
│   └── quikr_car (2).csv          # Raw dataset
├── cleaned_car.csv                # Processed and cleaned dataset (auto-generated)
├── car_price_prediction.py        # Complete modular Python execution script
├── quikr_car.ipynb                # Interactive Jupyter analysis notebook
├── CarPricePredictorModel.pkl     # Serialized best model pipeline (auto-generated)
├── requirements.txt               # Project dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/24f3004267/car-price-prediction.git
cd car-price-prediction
```

### 2. Set Up a Virtual Environment (Optional but Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Prediction Pipeline
```bash
python car_price_prediction.py
```

---

## 💡 Quick Inference Example

You can use the saved model (`CarPricePredictorModel.pkl`) to predict prices directly in Python:

```python
import pickle
import pandas as pd

# Load the trained pipeline
with open("CarPricePredictorModel.pkl", "rb") as f:
    model = pickle.load(f)

# Input test car data
new_car = pd.DataFrame({
    'name': ['Maruti Suzuki Swift'],
    'company': ['Maruti'],
    'year': [2016],
    'kms_driven': [45000],
    'fuel_type': ['Petrol']
})

# Predict price
estimated_price = model.predict(new_car)[0]
print(f"Predicted Price: ₹{estimated_price:,.2f}")
```

---

## 🔮 Future Roadmap

- [ ] Add interactive web dashboard using **Streamlit**.
- [ ] Implement hyperparameter optimization with `GridSearchCV` / `Optuna`.
- [ ] Experiment with **XGBoost** and **LightGBM** gradient boosted decision trees.
- [ ] Containerize application with **Docker** for cloud deployment.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
