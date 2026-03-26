import os
import pandas as pd

# Load the CSV data
def load_data():
    # Dynamically construct the absolute path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'Training_2.csv')
    print(f"Loading data from: {file_path}")
    data = pd.read_csv(file_path)
    return data

data = load_data()
print(data.head())
print(data.columns)
print(data.shape)