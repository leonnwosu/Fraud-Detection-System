#check the split in the dataset
import pandas as pd

df = pd.read_csv('transactions.csv')

# find the ration of fraud and non-fraud transactions

print(df['Class'].value_counts())