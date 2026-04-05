import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


data_path = "../data/electricity_data.csv"
df = pd.read_csv(data_path)

print("✅ Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)
print("Columns:", df.columns.tolist())

# ----------------------------------------------------
# 2) Split features (X) and targets (y)
# Targets:
#   - bill
#   - unit_consume
# ----------------------------------------------------
X = df.drop(["bill", "unit_consume"], axis=1)
y_bill = df["bill"]
y_unit = df["unit_consume"]

# ----------------------------------------------------
# 3) Identify categorical + numeric columns
# ----------------------------------------------------
categorical_cols = ["house_size", "season"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# ----------------------------------------------------
# 4) Preprocessor:
# OneHotEncode categorical columns
# ----------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

# ----------------------------------------------------
# 5) Train-test split
# ----------------------------------------------------
X_train, X_test, y_bill_train, y_bill_test, y_unit_train, y_unit_test = train_test_split(
    X, y_bill, y_unit, test_size=0.2, random_state=42
)

# ----------------------------------------------------
# 6) Transform input features
# ----------------------------------------------------
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# ----------------------------------------------------
# 7) Train Bill Model (Linear Regression)
# ----------------------------------------------------
bill_model = LinearRegression()
bill_model.fit(X_train_transformed, y_bill_train)

# ----------------------------------------------------
# 8) Train Unit Consume Model (Linear Regression)
# ----------------------------------------------------
unit_model = LinearRegression()
unit_model.fit(X_train_transformed, y_unit_train)

# ----------------------------------------------------
# 9) Evaluate Bill Model
# ----------------------------------------------------
bill_preds = bill_model.predict(X_test_transformed)
bill_mae = mean_absolute_error(y_bill_test, bill_preds)
bill_r2 = r2_score(y_bill_test, bill_preds)

print("\n✅ Bill Model Evaluation:")
print("Bill MAE:", bill_mae)
print("Bill R2 Score:", bill_r2)

unit_preds = unit_model.predict(X_test_transformed)
unit_mae = mean_absolute_error(y_unit_test, unit_preds)
unit_r2 = r2_score(y_unit_test, unit_preds)

print("\n✅ Unit Consume Model Evaluation:")
print("Unit MAE:", unit_mae)
print("Unit R2 Score:", unit_r2)

joblib.dump(preprocessor, "../model/preprocessor.pkl")
joblib.dump(bill_model, "../model/bill_model.pkl")
joblib.dump(unit_model, "../model/unit_model.pkl")

print("\n✅ Models Saved Successfully!")
print("Saved: ../model/preprocessor.pkl")
print("Saved: ../model/bill_model.pkl")
print("Saved: ../model/unit_model.pkl")
