import pandas as pd
import ast
import argparse

def has_valid_smiles(smiles_str):
    """
    Check if the smiles string has at least one non-empty entry.
    
    Valid examples:
    - ['', 'CC1=NC=C(N1CCO)[N+](=O)[O-]'] -> Valid (contains one non-empty string)
    
    Invalid examples:
    - ['', ''] -> Invalid (all strings are empty)
    - [''] -> Invalid (all strings are empty)
    - [] -> Invalid (empty list)
    """
    try:
        # Handle NaN values
        if pd.isna(smiles_str):
            return False
        
        # Convert the string representation of a list to an actual list
        smiles_list = ast.literal_eval(smiles_str)
        
        # Check if there's at least one non-empty string in the list
        return any(smile for smile in smiles_list if smile)
    except (ValueError, SyntaxError, TypeError):
        # If there's an issue parsing the string, consider it invalid
        return False

def merge_csv_files(file1_path, file2_path, output_path, smiles_column='smiless'):
    """
    Merge two CSV files, keeping only rows with valid SMILES entries.
    """
    # Read the CSV files
    print(f"Reading file 1: {file1_path}")
    df1 = pd.read_csv(file1_path)
    
    print(f"Reading file 2: {file2_path}")
    df2 = pd.read_csv(file2_path)
    
    # Check if the SMILES column exists
    if smiles_column not in df1.columns or smiles_column not in df2.columns:
        print(f"Error: Column '{smiles_column}' not found in one or both CSV files.")
        print(f"File 1 columns: {', '.join(df1.columns)}")
        print(f"File 2 columns: {', '.join(df2.columns)}")
        return
    
    print(f"File 1 initial size: {len(df1)} rows")
    print(f"File 2 initial size: {len(df2)} rows")
    
    # Filter rows with valid SMILES notation
    df1_filtered = df1[df1[smiles_column].apply(has_valid_smiles)]
    df2_filtered = df2[df2[smiles_column].apply(has_valid_smiles)]
    
    print(f"File 1 after filtering: {len(df1_filtered)} rows")
    print(f"File 2 after filtering: {len(df2_filtered)} rows")
    
    # Concatenate the filtered dataframes
    merged_df = pd.concat([df1_filtered, df2_filtered], ignore_index=True)
    
    print(f"Merged data: {len(merged_df)} rows")
    
    # Save the merged dataframe
    merged_df.to_csv(output_path, index=False)
    print(f"Merged data saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Merge two CSV files, keeping only rows with valid SMILES entries.')
    parser.add_argument('file1', help='Path to the first CSV file')
    parser.add_argument('file2', help='Path to the second CSV file')
    parser.add_argument('output', help='Path to save the merged CSV file')
    parser.add_argument('--column', default='smiless', help='Name of the column containing SMILES notations (default: smiless)')
    
    args = parser.parse_args()
    
    merge_csv_files(args.file1, args.file2, args.output, args.column)

if __name__ == "__main__":
    main()