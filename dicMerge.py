import os
import pandas as pd
# Set the folder path where CSV files are located
folder_path = '/home/ganesh/Desktop/Goat-for-Bli/goat-for-bli/omlProject/en-hi_acronym_dicts_2'

# Initialize an empty DataFrame to store the combined data
combined_data = pd.DataFrame()

# Loop through all CSV files in the folder and append them to the combined data
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        print(filename+',',end="")
        file_path = os.path.join(folder_path, filename)
        csv_data = pd.read_csv(file_path)
        combined_data = combined_data.append(csv_data)

# Write the combined data to a new CSV file
