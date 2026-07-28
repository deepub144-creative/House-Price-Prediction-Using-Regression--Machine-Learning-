import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import os

def train():
    data_path = r"C:\Users\deepu\OneDrive\Desktop\Team Tech Titans\Project Work Flow\Data Acquisition\Kaggle Dataset\California Housing Prices\Dataset\housing.csv"
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        return

    print("Loading dataset...")
    df = pd.read_csv(data_path)

    # Separate features and target
    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]

    # Numerical pipelines
    num_features = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipelines
    cat_features = ["ocean_proximity"]
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ])

    # Create the full pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    print("Training model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    print(f"Test R2 Score: {model.score(X_test, y_test):.4f}")

    # Save the model
    joblib.dump(model, 'model.pkl')
    joblib.dump(df, 'dataset_cache.pkl') # Cache for dashboard
    print("Model saved to model.pkl")

if __name__ == "__main__":
    train()
