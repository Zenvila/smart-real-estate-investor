import pandas as pd
import os

# Path to the final folder
final_folder = os.path.join(os.getcwd(), 'final')

# List all CSV files in the folder
csv_files = [f for f in os.listdir(final_folder) if f.endswith('.csv')]
csv_files.sort()  # Sort alphabetically

print(f"Found {len(csv_files)} CSV files to merge:")
for i, file in enumerate(csv_files, 1):
    print(f"  {i:2d}. {file}")

print("\nStarting merge process...")

# Read and concatenate all CSVs
all_dfs = []
total_rows = 0

for i, file in enumerate(csv_files):
    file_path = os.path.join(final_folder, file)
    try:
        # Read CSV with latin1 encoding to avoid Unicode errors
        df = pd.read_csv(file_path, encoding='latin1')
        
        # Remove fully empty rows
        df = df.dropna(how='all')
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # For files after the first one, skip the header row if it exists
        if i > 0 and len(df) > 0:
            # Check if first row is a header (matches column names)
            first_row = df.iloc[0].astype(str)
            column_names = df.columns.astype(str)
            
            # If first row matches column names, remove it
            if first_row.equals(column_names):
                df = df.iloc[1:].reset_index(drop=True)
        
        all_dfs.append(df)
        total_rows += len(df)
        
        print(f"  ✓ {file}: {len(df)} rows")
        
    except Exception as e:
        print(f"  ✗ {file}: ERROR - {str(e)}")

# Concatenate all DataFrames
if all_dfs:
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    # Remove any remaining fully empty rows
    merged_df = merged_df.dropna(how='all')
    
    # Save the merged CSV
    output_path = os.path.join(final_folder, 'merged_final.csv')
    merged_df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\n✓ Merge completed successfully!")
    print(f"  Total files merged: {len(all_dfs)}")
    print(f"  Total rows: {len(merged_df)}")
    print(f"  Output file: {output_path}")
    
else:
    print("\n✗ No files were successfully read. Check for errors above.") 