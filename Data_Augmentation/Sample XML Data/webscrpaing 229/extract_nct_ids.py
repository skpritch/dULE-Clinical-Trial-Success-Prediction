import pickle
import argparse
import os
import pandas as pd

def extract_nct_ids_from_pickle(pickle_file, output_file):
    """
    Extract NCT IDs from a pickle file and save them to a text file.
    
    Args:
        pickle_file (str): Path to the pickle file
        output_file (str): Path to save the text file with NCT IDs
    """
    print(f"Loading pickle file: {pickle_file}")
    
    # Load the pickle file
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    # Extract NCT IDs - handling different possible formats
    nct_ids = []
    
    # Case 1: If data is a list/array of strings (directly NCT IDs)
    if isinstance(data, (list, tuple)) and all(isinstance(item, str) for item in data):
        nct_ids = data
    
    # Case 2: If data is a dictionary with NCT IDs as keys
    elif isinstance(data, dict) and all(isinstance(key, str) for key in data.keys()):
        nct_ids = list(data.keys())
    
    # Case 3: If data is a pandas DataFrame
    elif isinstance(data, pd.DataFrame):
        # Try to find a column that might contain NCT IDs
        for col in data.columns:
            if any(str(val).startswith('NCT') for val in data[col].dropna()):
                nct_ids = [str(val) for val in data[col].dropna() if str(val).startswith('NCT')]
                break
        
        # If no column was found with NCT IDs
        if not nct_ids and 'nct_id' in data.columns:
            nct_ids = [str(val) for val in data['nct_id'].dropna()]
        elif not nct_ids and 'nctid' in data.columns:
            nct_ids = [str(val) for val in data['nctid'].dropna()]
    
    # Case 4: If data is a more complex structure, print it and exit
    else:
        print(f"Unable to automatically extract NCT IDs. Your pickle file contains data of type: {type(data)}")
        
        # If it's a list or tuple, show the type of the first item
        if isinstance(data, (list, tuple)) and len(data) > 0:
            print(f"The first item in the list is of type: {type(data[0])}")
            
            # If the first item is a dict or has an attribute that might contain an NCT ID
            if isinstance(data[0], dict):
                print(f"First item keys: {list(data[0].keys())}")
                
                # Look for possible NCT ID keys
                for item in data[:5]:  # Check first 5 items
                    for key, value in item.items():
                        if isinstance(value, str) and value.startswith('NCT'):
                            nct_ids = [item[key] for item in data if key in item and isinstance(item[key], str)]
                            print(f"Found NCT IDs in key: {key}")
                            break
                    if nct_ids:
                        break
        
        if not nct_ids:
            print("Please examine your pickle file structure and modify this script accordingly.")
            return
    
    # Ensure all IDs start with NCT
    nct_ids = [id for id in nct_ids if str(id).startswith('NCT')]
    
    # Remove duplicates while preserving order
    unique_nct_ids = []
    for id in nct_ids:
        if id not in unique_nct_ids:
            unique_nct_ids.append(id)
    
    print(f"Found {len(unique_nct_ids)} unique NCT IDs")
    
    # Write NCT IDs to text file
    with open(output_file, 'w') as f:
        for nct_id in unique_nct_ids:
            f.write(f"{nct_id}\n")
    
    print(f"Saved NCT IDs to: {output_file}")
    print(f"You can now use this file with the ClinicalTrialsDownloader:")
    print(f"python clinical_trials_downloader.py --input {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Extract NCT IDs from a pickle file")
    parser.add_argument("--pickle", required=True, help="Path to the pickle file")
    parser.add_argument("--output", default="nct_ids.txt", help="Output text file path")
    
    args = parser.parse_args()
    
    extract_nct_ids_from_pickle(args.pickle, args.output)


if __name__ == "__main__":
    main()