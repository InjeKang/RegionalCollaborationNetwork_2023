import pandas as pd
import os

# Specify the folder path where your Excel files are located
folder_path = "D:/Analysis/2023_Park/data/patent"

# Initialize an empty list to store the dataframes
data = []

# Iterate over all files in the folder
for file_name in os.listdir(folder_path):
    # Check if the file is an Excel file
    if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
        # Construct the full file path
        file_path = os.path.join(folder_path, file_name)
        
        # Read the Excel file into a dataframe
        df = pd.read_excel(file_path)
        
        # Add a new column with the file name
        df['File Name'] = file_name

        # Append the dataframe to the list
        data.append(df)

# Concatenate all dataframes into a single dataframe
combined_df = pd.concat(data, ignore_index=True)

len(combined_df)

drop_duplicate = combined_df.drop_duplicates(subset='WIPS ON key', keep='first')
len(drop_duplicate)

# Identify duplicated values in the "WIPS ON key" column
duplicated_mask = combined_df.duplicated(subset='WIPS ON key', keep=False)

# Keep only the duplicated values in the dataframe
duplicated_df = combined_df[duplicated_mask]

duplicated_df2 = duplicated_df[['WIPS ON key', '출원일', 'File Name']]
duplicated_df2.sort_values("WIPS ON key")
duplicated_df2.to_excel(f'{folder_path}\\duplicate.xlsx')

duplicated_df2['File Name'].unique()

# Check for duplicate values in the "WIPS ON key" column
duplicates = combined_df.duplicated(subset='WIPS ON key')

# Filter the dataframe to show only the rows with duplicate values
duplicated_rows = combined_df[duplicates]
len(duplicated_rows)

duplicated_rows2 = duplicated_rows[['WIPS ON key', '출원일', 'File Name']]

# Get the unique duplicate values in the "WIPS ON key" column
duplicate_values = duplicated_rows['WIPS ON key'].unique()

