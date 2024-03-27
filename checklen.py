import pandas as pd

def count_columns(csv_file):
    df = pd.read_csv(csv_file)
    num_columns = len(df.columns)
    return num_columns

# Example usage:
csv_file = 'Dataset/filtered_file.csv'
num_columns = count_columns(csv_file)
print("Number of columns:", num_columns)
