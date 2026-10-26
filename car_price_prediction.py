"""
================================================================================
CAR PRICE PREDICTION - END-TO-END MACHINE LEARNING PIPELINE
================================================================================
Description:
    This script performs comprehensive Exploratory Data Analysis (EDA),
    data cleaning, outlier handling, feature engineering, and model training
    for predicting used car prices based on features like name, company,
    manufacturing year, kilometers driven, and fuel type.

Models Implemented:
    1. Linear Regression Pipeline (Baseline)
    2. Random Forest Regressor Pipeline (Ensemble)

Author: Aditya
Repository: car-price-prediction
================================================================================
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


# ==============================================================================
# SECTION 1: DATA LOADING & PATH RESOLUTION
# ==============================================================================

def load_dataset():
    """
    Locates and loads the Quikr car dataset from possible relative paths.
    
    Returns:
        pd.DataFrame: Raw car sales dataset.
    """
    candidate_paths = [
        os.path.join("Data-set", "quikr_car (2).csv"),
        "quikr_car (2).csv",
        os.path.join("Data-set", "quikr_car.csv"),
        "quikr_car.csv"
    ]
    
    dataset_path = None
    for path in candidate_paths:
        if os.path.exists(path):
            dataset_path = path
            break
            
    if dataset_path is None:
        raise FileNotFoundError(
            "Dataset file not found! Checked locations: " + ", ".join(candidate_paths)
        )
        
    print(f"[INFO] Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    return df


# ==============================================================================
# SECTION 2: EXPLORATORY DATA ANALYSIS (EDA) - INITIAL INSPECTION
# ==============================================================================

def perform_initial_eda(df):
    """
    Performs initial exploratory data analysis to understand data structure,
    identifying non-null counts, data types, missing values, and anomalies.
    """
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS (EDA) - INITIAL OVERVIEW")
    print("=" * 60)
    
    print("\n--- First 5 Rows ---")
    print(df.head())
    
    print(f"\n--- Dataset Dimensions ---")
    print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")
    
    print("\n--- DataFrame Information & Schema ---")
    df.info()
    
    print("\n--- Missing Values Count Per Column ---")
    print(df.isnull().sum())
    
    print("\n--- Duplicate Rows Count ---")
    print(f"Duplicate records: {df.duplicated().sum()}")
    
    print("\n--- Summary of Unique Values per Column ---")
    for col in df.columns:
        print(f"Column '{col}' has {df[col].nunique()} unique entries.")


# ==============================================================================
# SECTION 3: DATA PREPROCESSING & DATA CLEANING
# ==============================================================================

def clean_data(df_raw):
    """
    Cleans raw dataset by fixing data quality issues:
      - Price: Removes 'Ask For Price', strips commas, converts to integer, drops NaNs.
      - kms_driven: Strips 'kms' string & commas, converts to numeric, drops NaNs.
      - year: Converts messy year strings to numeric integers.
      - name: Truncates long car titles to first 3 words for better categorization.
      - fuel_type: Filters out records where fuel_type is missing or corrupted.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING & CLEANING")
    print("=" * 60)
    
    df = df_raw.copy()
    
    # -------------------------------------------------------------------------
    # 3.1 Cleaning 'Price' Column
    # -------------------------------------------------------------------------
    # Issue: Non-numeric entries ('Ask For Price') and formatting with commas
    print("[STEP 1/5] Cleaning 'Price' column...")
    df["Price"] = df["Price"].replace("Ask For Price", np.nan)
    df["Price"] = df["Price"].astype(str).str.replace(",", "", regex=False)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["Price"])
    df["Price"] = df["Price"].astype(int)
    print(f"  -> Valid price entries remaining: {len(df)}")
    
    # -------------------------------------------------------------------------
    # 3.2 Cleaning 'kms_driven' Column
    # -------------------------------------------------------------------------
    # Issue: Suffix 'kms', commas, and occasional corrupted text values
    print("[STEP 2/5] Cleaning 'kms_driven' column...")
    df["kms_driven"] = (
        df["kms_driven"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("kms", "", case=False, regex=False)
        .str.strip()
    )
    df["kms_driven"] = pd.to_numeric(df["kms_driven"], errors="coerce")
    df = df.dropna(subset=["kms_driven"])
    df["kms_driven"] = df["kms_driven"].astype(int)
    print(f"  -> Valid kms_driven entries remaining: {len(df)}")
    
    # -------------------------------------------------------------------------
    # 3.3 Cleaning 'year' Column
    # -------------------------------------------------------------------------
    # Issue: Non-numeric text values present in 'year'
    print("[STEP 3/5] Cleaning 'year' column...")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    print(f"  -> Valid year entries remaining: {len(df)}")
    
    # -------------------------------------------------------------------------
    # 3.4 Cleaning 'fuel_type' Column
    # -------------------------------------------------------------------------
    # Filter out any records missing valid fuel type
    print("[STEP 4/5] Cleaning 'fuel_type' column...")
    df = df[~df["fuel_type"].isna()]
    print(f"  -> Valid fuel_type entries remaining: {len(df)}")
    
    # -------------------------------------------------------------------------
    # 3.5 Cleaning 'name' Column
    # -------------------------------------------------------------------------
    # Issue: Car names are overly specific full model descriptions with trims.
    # Keep the first 3 words to capture [Brand + Model Family] (e.g., 'Hyundai Santro Xing')
    print("[STEP 5/5] Normalizing car names (keeping first 3 words)...")
    df["name"] = df["name"].str.split(" ").str.slice(0, 3).str.join(" ")
    
    # Reset index after filtering
    df = df.reset_index(drop=True)
    return df


# ==============================================================================
# SECTION 4: OUTLIER DETECTION & TREATMENT
# ==============================================================================

def handle_outliers(df):
    """
    Detects and filters extreme outliers in target variable ('Price').
    A car priced at 8.5 Crore (85,00,003) is an extreme unrealistic outlier.
    Filters dataset to cars with Price < 6,000,000 (6e6).
    """
    print("\n" + "=" * 60)
    print("OUTLIER DETECTION & REMOVAL")
    print("=" * 60)
    
    before_count = len(df)
    max_price_before = df["Price"].max()
    print(f"[OUTLIER CHECK] Max car price before filtering: INR {max_price_before:,.2f}")
    
    # Filter out extreme price outlier (Price < 60 Lakhs)
    df = df[df["Price"] < 6e6].reset_index(drop=True)
    
    after_count = len(df)
    max_price_after = df["Price"].max()
    print(f"[OUTLIER CHECK] Outliers removed: {before_count - after_count}")
    print(f"[OUTLIER CHECK] Max car price after filtering:  INR {max_price_after:,.2f}")
    print(f"[INFO] Final cleaned dataset shape: {df.shape}")
    
    print("\n--- Cleaned Dataset Statistical Summary ---")
    print(df.describe())
    
    # Save cleaned data to CSV
    output_cleaned_path = "cleaned_car.csv"
    df.to_csv(output_cleaned_path, index=False)
    print(f"[INFO] Cleaned dataset saved to: {output_cleaned_path}")
    
    return df


# ==============================================================================
# SECTION 5: MODEL TRAINING & PIPELINE ARCHITECTURE
# ==============================================================================

def build_and_evaluate_models(df):
    """
    Constructs scikit-learn pipelines with ColumnTransformer for one-hot
    encoding categorical features, trains Linear Regression and Random Forest
    regressors, and evaluates on test data.
    """
    print("\n" + "=" * 60)
    print("MODEL TRAINING & EVALUATION PIPELINE")
    print("=" * 60)
    
    # 5.1 Train-Test Split
    X = df.drop("Price", axis=1)
    y = df["Price"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    print(f"[SPLIT] Training Set: {X_train.shape[0]} samples")
    print(f"[SPLIT] Testing Set:  {X_test.shape[0]} samples")
    
    # 5.2 Preprocessor with ColumnTransformer
    # Categorical features: 'name', 'company', 'fuel_type' -> OneHotEncoder
    # Numerical features: 'year', 'kms_driven' -> Passthrough
    categorical_cols = ["name", "company", "fuel_type"]
    column_trans = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        remainder="passthrough"
    )
    
    # -------------------------------------------------------------------------
    # 5.3 Baseline Model: Linear Regression
    # -------------------------------------------------------------------------
    print("\n--> Training Linear Regression Pipeline...")
    lr_model = LinearRegression()
    lr_pipeline = make_pipeline(column_trans, lr_model)
    lr_pipeline.fit(X_train, y_train)
    
    lr_preds = lr_pipeline.predict(X_test)
    lr_mae = mean_absolute_error(y_test, lr_preds)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
    lr_r2 = r2_score(y_test, lr_preds)
    
    print("\n[METRICS] Linear Regression Results:")
    print(f"  - Mean Absolute Error (MAE): INR {lr_mae:,.2f}")
    print(f"  - Root Mean Squared Error (RMSE): INR {lr_rmse:,.2f}")
    print(f"  - R2 Score: {lr_r2:.4f} ({lr_r2 * 100:.2f}%)")
    
    # -------------------------------------------------------------------------
    # 5.4 Advanced Model: Random Forest Regressor
    # -------------------------------------------------------------------------
    print("\n--> Training Random Forest Regressor Pipeline...")
    rf_model = RandomForestRegressor(n_estimators=200, random_state=0)
    rf_pipeline = make_pipeline(column_trans, rf_model)
    rf_pipeline.fit(X_train, y_train)
    
    rf_preds = rf_pipeline.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_preds)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    rf_r2 = r2_score(y_test, rf_preds)
    
    print("\n[METRICS] Random Forest Regressor Results:")
    print(f"  - Mean Absolute Error (MAE): INR {rf_mae:,.2f}")
    print(f"  - Root Mean Squared Error (RMSE): INR {rf_rmse:,.2f}")
    print(f"  - R2 Score: {rf_r2:.4f} ({rf_r2 * 100:.2f}%)")
    
    # -------------------------------------------------------------------------
    # 5.5 Model Selection & Persistence
    # -------------------------------------------------------------------------
    best_pipeline = rf_pipeline if rf_r2 > lr_r2 else lr_pipeline
    best_name = "RandomForest" if rf_r2 > lr_r2 else "LinearRegression"
    
    model_filename = "CarPricePredictorModel.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(best_pipeline, f)
    print(f"\n[INFO] Best performing model ({best_name}) saved to: {model_filename}")
    
    return best_pipeline, X_test, y_test


# ==============================================================================
# SECTION 6: SAMPLE INFERENCE DEMONSTRATION
# ==============================================================================

def demonstrate_inference(pipeline):
    """
    Demonstrates model prediction on realistic sample car inputs.
    """
    print("\n" + "=" * 60)
    print("SAMPLE INFERENCE & PREDICTION DEMO")
    print("=" * 60)
    
    sample_cars = pd.DataFrame([
        {
            "name": "Maruti Suzuki Swift",
            "company": "Maruti",
            "year": 2016,
            "kms_driven": 45000,
            "fuel_type": "Petrol"
        },
        {
            "name": "Hyundai Grand i10",
            "company": "Hyundai",
            "year": 2015,
            "kms_driven": 35000,
            "fuel_type": "Diesel"
        },
        {
            "name": "Toyota Innova 2.5",
            "company": "Toyota",
            "year": 2014,
            "kms_driven": 80000,
            "fuel_type": "Diesel"
        }
    ])
    
    predictions = pipeline.predict(sample_cars)
    
    for i, pred in enumerate(predictions):
        car = sample_cars.iloc[i]
        print(f"\nCar: {car['name']} ({car['year']}) | {car['kms_driven']:,} kms | {car['fuel_type']}")
        print(f"  -> Predicted Estimated Price: INR {pred:,.2f}")


# ==============================================================================
# MAIN EXECUTION ENTRY POINT
# ==============================================================================

def main():
    print("*" * 60)
    print("       USED CAR PRICE PREDICTION PIPELINE - EXECUTION")
    print("*" * 60)
    
    # 1. Load data
    df_raw = load_dataset()
    
    # 2. Perform initial Exploratory Data Analysis
    perform_initial_eda(df_raw)
    
    # 3. Clean raw data
    df_cleaned = clean_data(df_raw)
    
    # 4. Handle outliers
    df_final = handle_outliers(df_cleaned)
    
    # 5. Train & evaluate models
    trained_pipeline, _, _ = build_and_evaluate_models(df_final)
    
    # 6. Run inference test
    demonstrate_inference(trained_pipeline)
    
    print("\n" + "=" * 60)
    print("[SUCCESS] All pipeline stages executed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
