# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, r2_score


# 2. LOAD FEATURE ENGINEERED DATA
df = pd.read_csv(r"C:\Users\deepu\Downloads\featured_housing.csv")

print("Data Loaded Successfully")
print(df.shape)


# 3. DEFINE FEATURES & TARGET
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]


# 4. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 5. PIPELINES 
# Linear Regression
pipeline_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

# Ridge Regression
pipeline_ridge = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge())
])

# Random Forest (BEST)
pipeline_rf = Pipeline([
    ("model", RandomForestRegressor(n_estimators=200, random_state=42))
])


# 6. TRAIN MODELS
pipeline_lr.fit(X_train, y_train)
pipeline_ridge.fit(X_train, y_train)
pipeline_rf.fit(X_train, y_train)


# 7. PREDICTIONS
pred_lr = pipeline_lr.predict(X_test)
pred_ridge = pipeline_ridge.predict(X_test)
pred_rf = pipeline_rf.predict(X_test)


# 8. EVALUATION FUNCTION
def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return rmse, r2


# 9. RESULTS
results = pd.DataFrame({
    "Model": ["Linear Regression", "Ridge Regression", "Random Forest"],
    "RMSE": [
        evaluate(y_test, pred_lr)[0],
        evaluate(y_test, pred_ridge)[0],
        evaluate(y_test, pred_rf)[0]
    ],
    "R2 Score": [
        evaluate(y_test, pred_lr)[1],
        evaluate(y_test, pred_ridge)[1],
        evaluate(y_test, pred_rf)[1]
    ]
})

print("\nMODEL PERFORMANCE:")
print(results)


# 10. BEST MODEL
best_model = results.loc[results["R2 Score"].idxmax()]
print("\nBest Model:")
print(best_model)


# 11. SAVE MODEL

import os
print(os.getcwd())
import joblib

joblib.dump(pipeline_rf, r"C:\Users\deepu\OneDrive\Desktop\Machine Learning - Project\house_price_model.pkl")
