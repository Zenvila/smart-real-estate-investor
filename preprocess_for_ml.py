import pandas as pd
import os
import numpy as np
import ast
import re
from datetime import datetime

# Path to the merged CSV
input_path = os.path.join(os.getcwd(), 'final', 'merged_final.csv')
output_path = os.path.join(os.getcwd(), 'final', 'ml_ready_data.csv')

print("=== REAL ESTATE ML PREPROCESSING ===")
print("=" * 50)

# Read the merged data
df = pd.read_csv(input_path, encoding='latin1')
print(f"✓ Loaded {len(df)} rows from merged_final.csv")

# Step 1: Clean and standardize data
print("\n1. Cleaning and standardizing data...")

# Remove duplicate columns (ending with .1, .2, etc.)
df = df.loc[:, ~df.columns.str.contains(r'\.\d+$')]

# Remove unnamed columns
df = df.loc[:, ~df.columns.str.startswith('Unnamed:')]

# Remove fully empty rows and columns
df = df.dropna(how='all')
df = df.dropna(axis=1, how='all')

print(f"  ✓ Removed duplicate and empty columns")
print(f"  ✓ Remaining columns: {len(df.columns)}")

# Step 2: Handle numeric features for ML models
print("\n2. Processing numeric features...")

# Convert price columns to numeric
price_cols = ['price', 'area_marla', 'price_per_marla']
for col in price_cols:
    if col in df.columns:
        # Clean price strings (remove PKR, commas, etc.)
        df[col] = df[col].astype(str).str.replace('PKR', '', regex=False)
        df[col] = df[col].str.replace(',', '', regex=False)
        df[col] = df[col].str.replace(' ', '', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Handle price indexing columns - IMPROVED LOGIC
price_index_cols = ['price_index_now', 'price_index_6mo', 'price_index_12mo', 'price_index_24mo']
for col in price_index_cols:
    if col in df.columns:
        print(f"  Processing {col}...")
        
        # First, clean the price index values
        df[col] = df[col].astype(str).str.replace('PKR ', '', regex=False)
        df[col] = df[col].str.replace(',', '', regex=False)
        df[col] = df[col].str.replace(' ', '', regex=False)
        
        # Store original non-empty values
        original_mask = (df[col] != '') & (df[col] != 'nan') & (df[col] != 'None')
        
        # Convert to numeric, but be more careful
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Only fill values that were originally empty/NaN
        empty_mask = ~original_mask
        df.loc[empty_mask, col] = -1
        
        # Print some stats
        valid_count = df[col].notna().sum() - (df[col] == -1).sum()
        print(f"    Valid values: {valid_count}, Missing (filled with -1): {(df[col] == -1).sum()}")

print(f"  ✓ Processed {len(price_cols)} price columns")
print(f"  ✓ Processed {len(price_index_cols)} price indexing columns")

# Step 3: Standardize categorical features for Filter Agent
print("\n3. Standardizing categorical features...")

# Clean and standardize categorical columns
cat_cols = ['city', 'type', 'purpose']
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower()
        # Remove special characters and standardize
        df[col] = df[col].str.replace('[^\w\s]', '', regex=True)
        df[col] = df[col].str.replace(' ', '_', regex=False)

# Create location-based features
if 'location' in df.columns:
    df['location_clean'] = df['location'].astype(str).str.strip().str.lower()
    df['location_clean'] = df['location_clean'].str.replace('[^\w\s]', '', regex=True)

print(f"  ✓ Standardized {len(cat_cols)} categorical columns")

# Step 4: Parse feature lists for ML features
print("\n4. Parsing feature lists...")

def parse_list_safe(col):
    def try_parse(x):
        if isinstance(x, str) and x.startswith('['):
            try:
                return ast.literal_eval(x)
            except Exception:
                return []
        return []
    return col.apply(try_parse)

# Parse feature lists
feature_cols = ['main_features', 'room_features']
for col in feature_cols:
    if col in df.columns:
        df[col] = parse_list_safe(df[col])
        # Create feature count for ML
        df[f'{col}_count'] = df[col].apply(len)

print(f"  ✓ Parsed {len(feature_cols)} feature list columns")

# Step 5: Create derived features for ML models
print("\n5. Creating derived features for ML...")

# Price per square foot/marla ratio
if 'price' in df.columns and 'area_marla' in df.columns:
    df['price_per_marla_ratio'] = df['price'] / df['area_marla'].replace(0, np.nan)
    df['price_per_marla_ratio'] = df['price_per_marla_ratio'].fillna(0)

# Price trend features (for Risk Analysis) - IMPROVED
price_index_cols_clean = [col for col in price_index_cols if col in df.columns]
if len(price_index_cols_clean) >= 2:
    # Only use valid price index values (not -1) for calculations
    valid_price_data = df[price_index_cols_clean].copy()
    valid_price_data = valid_price_data.replace(-1, np.nan)
    
    # Calculate price volatility only from valid data
    df['price_volatility'] = valid_price_data.std(axis=1)
    # Calculate price trend only from valid data
    df['price_trend'] = valid_price_data.mean(axis=1)
    
    # Fill NaN with 0 for rows with no valid price index data
    df['price_volatility'] = df['price_volatility'].fillna(0)
    df['price_trend'] = df['price_trend'].fillna(0)

# Location-based features
if 'city' in df.columns:
    # Create city encoding for ML
    city_dummies = pd.get_dummies(df['city'], prefix='city')
    df = pd.concat([df, city_dummies], axis=1)

# Property type encoding
if 'type' in df.columns:
    type_dummies = pd.get_dummies(df['type'], prefix='type')
    df = pd.concat([df, type_dummies], axis=1)

print(f"  ✓ Created derived features for ML models")

# Step 6: Handle missing values
print("\n6. Handling missing values...")

# Fill missing values appropriately
df = df.fillna({
    'price': 0,
    'area_marla': 0,
    'price_per_marla': 0,
    'bedrooms': 'unknown',
    'bathrooms': 'unknown',
    'description': '',
    'title': ''
})

# For categorical columns, fill with 'unknown'
cat_cols_all = df.select_dtypes(include=['object']).columns
for col in cat_cols_all:
    if col not in ['main_features', 'room_features']:  # Skip list columns
        df[col] = df[col].fillna('unknown')

print(f"  ✓ Handled missing values")

# Step 7: Remove unnecessary columns
print("\n7. Removing unnecessary columns...")

# Remove columns that are mostly 'unknown' or empty
columns_to_remove = []
for col in df.columns:
    if col in df.columns:
        # Check if column is mostly 'unknown' values
        if df[col].dtype == 'object':
            unknown_ratio = (df[col] == 'unknown').sum() / len(df)
            if unknown_ratio > 0.8:  # If more than 80% are 'unknown'
                columns_to_remove.append(col)
                print(f"  Removing {col} (mostly 'unknown' values)")

# Remove columns with all zeros or very low variance
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if col in df.columns:
        # Check if column has very low variance (mostly same value)
        if df[col].std() < 0.01:  # Very low standard deviation
            columns_to_remove.append(col)
            print(f"  Removing {col} (very low variance)")

# Remove the identified columns
df = df.drop(columns=columns_to_remove)

print(f"  ✓ Removed {len(columns_to_remove)} unnecessary columns")

# Step 8: Final cleaning and validation
print("\n8. Final cleaning and validation...")

# Remove any remaining rows with all NaN
df = df.dropna(how='all')

# Ensure all numeric columns are actually numeric
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Remove any infinite values
df = df.replace([np.inf, -np.inf], 0)

print(f"  ✓ Final validation complete")

# Step 9: Save the ML-ready data
print("\n9. Saving ML-ready data...")

df.to_csv(output_path, index=False, encoding='utf-8')

print(f"✓ ML-ready data saved to: {output_path}")
print(f"  Final dataset shape: {df.shape}")
print(f"  Columns: {len(df.columns)}")
print(f"  Rows: {len(df)}")

# Print summary for ML pipeline
print("\n=== ML PIPELINE READY FEATURES ===")
print("Filter Agent features:")
filter_features = ['city', 'type', 'purpose', 'price', 'area_marla', 'location_clean']
for feat in filter_features:
    if feat in df.columns:
        print(f"  ✓ {feat}")

print("\nRisk Analysis Agent features:")
risk_features = ['price_volatility', 'price_trend', 'price_per_marla_ratio'] + price_index_cols_clean
for feat in risk_features:
    if feat in df.columns:
        print(f"  ✓ {feat}")

print("\nROI Predictor Agent features:")
roi_features = ['price_per_marla_ratio', 'main_features_count', 'room_features_count'] + price_index_cols_clean
for feat in roi_features:
    if feat in df.columns:
        print(f"  ✓ {feat}")

print("\n✓ Data is now ready for your ML pipeline!") 