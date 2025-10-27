import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score



train_df = pd.read_csv("./CleanedDataset/train.csv")
validation_df = pd.read_csv("./CleanedDataset/validation.csv")
test_df = pd.read_csv("./CleanedDataset/test.csv")


train_x = train_df.drop(columns=["price"])
train_y = train_df["price"]

val_x = validation_df.drop(columns=["price"])
val_y = validation_df["price"]

test_x = test_df.drop(columns=["price"])
test_y = test_df["price"]   



# Identify column types
categorical_cols = ['suburb', 'type']
numerical_cols = [col for col in train_x .columns if col not in categorical_cols]

# Preprocess categorical → one-hot encode
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# Build simple regression pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=200, random_state=64))       #THIS NEEDS TO BE TUNED FOR HYPERPARAMETERS TUNING
])

# Train
model.fit(train_x , train_y)

# Validate performance
y_val_pred = model.predict(val_x)
print("Validation MAE:", mean_absolute_error(val_y, y_val_pred))
print("Validation R2:", r2_score(val_y, y_val_pred))
#print("Validation RMSE:", root_mean_squared_error(val_y, y_val_pred, squared=False))

# Test performance
y_test_pred = model.predict(test_x)
print("Test MAE:", mean_absolute_error(test_y, y_test_pred))
print("Test R2:", r2_score(test_y, y_test_pred))
#print("Test RMSE:", root_mean_squared_error(test_y, y_test_pred, squared=False))