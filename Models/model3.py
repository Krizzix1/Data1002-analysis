import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score

# Load data
train_df = pd.read_csv("./CleanedDataset/train.csv")
validation_df = pd.read_csv("./CleanedDataset/validation.csv")
test_df = pd.read_csv("./CleanedDataset/test.csv")

# Log-transform target
train_y = np.log(train_df["price"])
val_y = np.log(validation_df["price"])
test_y = np.log(test_df["price"])

train_x = train_df.drop(columns=["price"])
val_x = validation_df.drop(columns=["price"])
test_x = test_df.drop(columns=["price"])

# Columns
categorical_cols = ['suburb', 'type']
numerical_cols = [c for c in train_x.columns if c not in categorical_cols]

# Preprocessing
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# SVR model (tuned)
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', SVR(kernel='rbf', C=100, epsilon=0.05, gamma='scale'))
])

# Train
model.fit(train_x, train_y)

# Predict (convert back from log scale)
y_val_pred = np.exp(model.predict(val_x))
y_test_pred = np.exp(model.predict(test_x))

val_y_actual = np.exp(val_y)
test_y_actual = np.exp(test_y)

# Evaluate
print("Validation MAE:", mean_absolute_error(val_y_actual, y_val_pred))
print("Validation R²:", r2_score(val_y_actual, y_val_pred))
print("Test MAE:", mean_absolute_error(test_y_actual, y_test_pred))
print("Test R²:", r2_score(test_y_actual, y_test_pred))
