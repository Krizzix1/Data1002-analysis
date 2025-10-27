import pandas as pd
from sklearn.model_selection import train_test_split


#Read excel file of original dataset, (the second sheet and skip first 9 rows as they are headers)
df = pd.read_excel("OriginalDataset/643202.xlsx", sheet_name=1, skiprows=9)

#Index by the first column which is date
df.set_index(df.columns[0], inplace=True)

#Get the Median house prices of sydney
MedianHousePrices = df.iloc[: , 0]

MedianHousePrices = MedianHousePrices.dropna()

#Set the new extracted & cleaned column to its own dataframe
cleanedDF = pd.DataFrame(MedianHousePrices)



#Rename Columns & Index
cleanedDF.columns = ["Median House Price (*1,000)"]
cleanedDF.rename_axis("Date", inplace=True)
cleanedDF.index = cleanedDF.index.strftime('%Y-%m')


test_split = 0.1
validation_split = 0.3

unsplit_training_data, test_data = train_test_split(cleanedDF, test_size=test_split)

validation_split *= validation_split*len(cleanedDF)/(validation_split*len(unsplit_training_data))

print(validation_split)

training_data, validation_data = train_test_split(unsplit_training_data, test_size=validation_split)


print(f"    DATA SPLIT (%)\nTraining Data = {len(training_data)/len(cleanedDF)}\n\
Validation Data = {len(validation_data)/len(cleanedDF)}\n\
Test Data = {len(test_data)/len(cleanedDF)}")


#Save all dataframes as excel files
cleanedDF.to_excel("CleanedDataset/Cleaned.xlsx")
training_data.to_excel("CleanedDataset/TrainSet.xlsx")
validation_data.to_excel("CleanedDataset/ValidationSet.xlsx")
test_data.to_excel("CleanedDataset/TestSet.xlsx")

