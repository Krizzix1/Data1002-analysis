import pandas as pd


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


#Partition data
TrainSet = cleanedDF.iloc[:70]
ValidationSet = cleanedDF.iloc[70:84]
TestSet = cleanedDF.iloc[84:]



#Save all dataframes as excel files
cleanedDF.to_excel("CleanedDataset/Cleaned.xlsx")
TrainSet.to_excel("CleanedDataset/TrainSet.xlsx")
ValidationSet.to_excel("CleanedDataset/ValidationSet.xlsx")
TestSet.to_excel("CleanedDataset/TestSet.xlsx")

