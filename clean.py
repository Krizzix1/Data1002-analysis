import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("OriginalDataset/domain_properties.csv") 


#split date into two floats so it can be interpreted by models.
df['date_sold'] = pd.to_datetime(df['date_sold'], format='%d/%m/%y')
df['sold_year'] = df['date_sold'].dt.year.astype(float)
df['sold_month'] = df['date_sold'].dt.month.astype(float)
df = df.drop(columns=['date_sold'])


# split percentages
train_pct = 0.7   
val_pct = 0.15    
test_pct = 0.15   

# Step 1: Split off the test set
train_val_df, test_df = train_test_split(df, test_size=test_pct, random_state=42)

#Splits remaining data into training and validation
val_adjusted = val_pct / (train_pct + val_pct)
train_df, val_df = train_test_split(train_val_df, test_size=val_adjusted, random_state=42)


#Save Splits into own csvs
train_df.to_csv("CleanedDataset/train.csv", index=False)
val_df.to_csv("CleanedDataset/validation.csv", index=False)
test_df.to_csv("CleanedDataset/test.csv", index=False)