import pandas as pd

# List of paths to the CSV files you want to merge
csv_files = [
    r'D:\AIS_ML\AIS_ML\Dataset\adc_1m_hard_surface.csv',
    r'D:\AIS_ML\AIS_ML\Dataset\adc_50cm_hard_surface.csv',
    r'D:\AIS_ML\AIS_ML\Dataset\adc_person_sitting.csv',
    r'D:\AIS_ML\AIS_ML\Dataset\adc_person_standing.csv',
]

# List to hold the data from each CSV file
data_frames = []

# Loop through each file path in the list
for file_path in csv_files:
    df = pd.read_csv(file_path, header=None)  # Read the CSV file into a DataFrame
    data_frames.append(df)  # Add the DataFrame to the list

# Concatenate all DataFrames in the list into one DataFrame
merged_df = pd.concat(data_frames, ignore_index=True)

# Optional: Save the merged DataFrame to a new CSV file
output_file_path = 'merged_dataset_1.csv'
merged_df.to_csv(output_file_path, index=False)
