#read csv
import csv
import pandas as pd
data = pd.read_csv('symptoms.csv')
print(data.head())
print(data.columns)
print(data.shape)