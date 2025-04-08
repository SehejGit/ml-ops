import pandas as pd
import numpy as np
import os
import yaml
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load parameters if params.yaml exists
try:
    with open("params.yaml", 'r') as file:
        params = yaml.safe_load(file)['preprocessing']
        input_path = params['input_path']
        output_train_path = params['output_train_path']
        output_test_path = params['output_test_path']
        output_pipeline_path = params['output_pipeline_path']
        test_size = params['test_size']
        random_state = params['random_state']
except:
    # Default parameters if no params.yaml file
    input_path = "data/Sport car price.csv"
    output_train_path = "data/processed_train_data.csv"
    output_test_path = "data/processed_test_data.csv"
    output_pipeline_path = "data/pipeline.pkl"
    test_size = 0.2
    random_state = 42

def load_data(filepath):
    """Load the CSV data and perform basic cleanup."""
    print(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    
    # Rename columns to remove spaces and special characters
    df.columns = [col.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_') for col in df.columns]
    
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df

def clean_data(df):
    """Clean the data by handling missing values and data types."""
    print("Cleaning data...")
    
    # Convert numeric columns to correct data types
    numeric_cols = ['Year', 'Engine_Size_L', 'Horsepower', 'Torque_lb_ft', '0_60_MPH_Time_seconds', 'Price_in_USD']
    
    for col in numeric_cols:
        if col in df.columns:
            # Remove any non-numeric characters and convert to float
            df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print("Data cleaning completed.")
    return df

def split_data(df, test_size=0.2, random_state=42):
    """Split data into training and testing sets."""
    print(f"Splitting data with test_size={test_size}, random_state={random_state}")
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Split into train/test
    test_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:test_idx].copy()
    test_df = df.iloc[test_idx:].copy()
    
    print(f"Train set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    
    return train_df, test_df

def create_preprocessing_pipeline(df):
    """Create a preprocessing pipeline for the data."""
    print("Creating preprocessing pipeline...")
    
    # Identify categorical and numerical columns
    categorical_cols = ['Car_Make', 'Car_Model']
    numerical_cols = ['Year', 'Engine_Size_L', 'Horsepower', 'Torque_lb_ft', '0_60_MPH_Time_seconds']
    
    # Create preprocessing pipelines for each column type
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_cols),
            ('num', numerical_transformer, numerical_cols)
        ],
        remainder='passthrough'
    )
    
    print("Preprocessing pipeline created.")
    return preprocessor

def main():
    """Main function to execute the preprocessing pipeline."""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_train_path), exist_ok=True)
    
    # Load and clean data
    df = load_data(input_path)
    df = clean_data(df)
    
    # Split data
    train_df, test_df = split_data(df, test_size, random_state)
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(df)
    
    # Save original training and test data (before transformation)
    train_df.to_csv(output_train_path, index=False)
    test_df.to_csv(output_test_path, index=False)
    
    # Save the preprocessing pipeline
    with open(output_pipeline_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    
    print(f"Preprocessing completed successfully.")
    print(f"Files saved:")
    print(f" - Training data: {output_train_path}")
    print(f" - Testing data: {output_test_path}")
    print(f" - Pipeline: {output_pipeline_path}")

if __name__ == "__main__":
    main()