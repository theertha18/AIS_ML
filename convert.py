
import pandas as pd

import os

# Load the CSV file
df = pd.read_csv('Dataset/adc_1m_hard_surface.csv')

# Delete 999 rows
df = df.iloc[999:]

# Save the filtered DataFrame back to CSV
output_file_path = 'Dataset/filtered_file.csv'
df.to_csv(output_file_path, index=False)

# Set permissions for the file
os.chmod(output_file_path, 0o644)  # This sets the file permissions to 644 (read/write for owner, read for others)

