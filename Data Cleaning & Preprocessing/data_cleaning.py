import pandas as pd
import numpy as np
import os
from pathlib import Path

def clean_and_preprocess_data():
    print("Starting Data Cleaning & Preprocessing...")
    
    # Define paths
    base_dir = Path(r"c:\Users\deepu\OneDrive\Desktop\4th sem Academic Projects\Team Tech Titans")
    data_path = base_dir / "Project Work Flow" / "Data Acquisition" / "Kaggle Dataset" / "California Housing Prices" / "Dataset" / "housing.csv"
    output_dir = base_dir / "Project Work Flow" / "Data Cleaning & Preprocessing" / "Cleaned Data" / "Cleaned_Data Set"
    output_path = output_dir / "cleaned_housing.csv"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load the Dataset
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # 2. Handle Missing Values
    print("Handling missing values...")
    # 'total_bedrooms' has missing values. We will fill them with the median.
    median_bedrooms = df['total_bedrooms'].median()
    df['total_bedrooms'].fillna(median_bedrooms, inplace=True)
    
    # 3. Feature Engineering
    print("Performing feature engineering...")
    # Creating some useful ratio features
    df['rooms_per_household'] = df['total_rooms'] / df['households']
    df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
    df['population_per_household'] = df['population'] / df['households']
    
    # 4. Handle Categorical Data
    print("Encoding categorical variables...")
    # 'ocean_proximity' is categorical. We will use One-Hot Encoding.
    df_encoded = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=False)
    
    # Optional: If you want boolean columns as int (0 and 1)
    for col in df_encoded.columns:
        if df_encoded[col].dtype == bool:
            df_encoded[col] = df_encoded[col].astype(int)
            
    # 5. Save the Cleaned Data
    print(f"Saving cleaned data to: {output_path}")
    df_encoded.to_csv(output_path, index=False)
    print("Data Cleaning & Preprocessing Completed Successfully!")

if __name__ == "__main__":
    clean_and_preprocess_data()
