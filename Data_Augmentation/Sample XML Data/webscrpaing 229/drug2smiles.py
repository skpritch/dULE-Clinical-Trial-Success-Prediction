import pandas as pd
import pubchempy as pcp
import ast
import time
import re
import os
import argparse

def clean_drug_name(drug_name):
    """Clean drug name to improve matching with PubChem"""
    if not drug_name:
        return None
        
    # Remove dosage information (e.g., "100mg", "10 mg/ml", "0.6 mg/kg/day")
    drug_name = re.sub(r'\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|µg)(?:/\w+)?(?:/\w+)?', '', drug_name)
    drug_name = re.sub(r'\d+\s*%', '', drug_name)  # Remove percentage (e.g., "3%")
    
    # Remove formulation information
    formulations = [
        r'\b(?:tablet|capsule|injection|solution|suspension|gel|cream|ointment)\b',
        r'\bdose[s]?\b', r'\bvehicle\b', r'\bplacebo\b'
    ]
    for pattern in formulations:
        drug_name = re.sub(pattern, '', drug_name, flags=re.IGNORECASE)
        
    # Skip certain terms that aren't drug names
    skip_terms = [
        'unknown', 'investigational', 'experimental', 'comparator', 'standard of care',
        'therapy', 'group', 'background', 'vehicle', 'placebo', 'part', 'escalation', 'expansion'
    ]
    if any(term in drug_name.lower() for term in skip_terms):
        return None
        
    # Remove parenthetical content
    drug_name = re.sub(r'\([^)]*\)', '', drug_name)
    
    # Remove trademark/registered symbols
    drug_name = re.sub(r'[®™©]', '', drug_name)
    
    # Replace common separators with spaces
    drug_name = re.sub(r'[-/+]', ' ', drug_name)
    
    # Remove any remaining symbols and extra whitespace
    drug_name = re.sub(r'[^\w\s]', ' ', drug_name)
    drug_name = ' '.join(drug_name.split())
    
    # Remove numbers that appear at the beginning or end of the string
    drug_name = re.sub(r'^\d+\s+', '', drug_name)
    drug_name = re.sub(r'\s+\d+$', '', drug_name)
    
    return drug_name.strip()

def split_compound_drug_name(drug_name):
    """Split a compound drug name into individual components"""
    if not drug_name:
        return []
        
    # Split by common separators
    components = re.split(r'\s*[,/+]\s*', drug_name)
    return [c.strip() for c in components if c.strip()]

def get_smiles(drug_name):
    """Get SMILES code for a drug name using PubChem"""
    if not drug_name or len(drug_name) < 3:
        print(f"    ↳ Skipping drug name (too short or empty): {drug_name}")
        return None
    
    # Map common drug names that might be hard to find in PubChem
    drug_mapping = {
        'cis platinum': 'cisplatin',
        'sterile water': 'water',
        'saline': 'sodium chloride',
        'levodopa carbidopa': 'levodopa carbidopa',
        'piperacillin tazobactam': 'piperacillin tazobactam',
        'doxycycline': 'doxycycline'
    }
    
    original_name = drug_name
    drug_name = clean_drug_name(drug_name)
    if not drug_name:
        print(f"    ↳ Skipping after cleaning: {original_name} -> None")
        return None
    
    if original_name != drug_name:
        print(f"    ↳ Cleaned drug name: {original_name} -> {drug_name}")
    
    # Check if it's a known drug with a mapping
    for key, value in drug_mapping.items():
        if key in drug_name.lower():
            drug_name = value
            print(f"    ↳ Mapped to known drug: {drug_name}")
            break
            
    try:
        # Try to get compound information from PubChem
        print(f"    ↳ Querying PubChem for: {drug_name}")
        compounds = pcp.get_compounds(drug_name, 'name')
        if compounds:
            # Return the SMILES code of the first match
            return compounds[0].canonical_smiles
            
        # If not found, try splitting into components for compound drugs
        components = split_compound_drug_name(original_name)
        if len(components) > 1:
            print(f"    ↳ Trying individual components: {components}")
            for component in components:
                component_clean = clean_drug_name(component)
                if component_clean and len(component_clean) >= 3:
                    print(f"    ↳ Querying PubChem for component: {component_clean}")
                    try:
                        compounds = pcp.get_compounds(component_clean, 'name')
                        if compounds:
                            return compounds[0].canonical_smiles
                    except Exception:
                        continue
                        
        print(f"    ↳ No compounds found in PubChem for: {drug_name}")
        return None
    except Exception as e:
        print(f"    ↳ Error getting SMILES for {drug_name}: {str(e)}")
        # Sleep briefly to avoid rate limiting
        time.sleep(1)
        return None

def convert_list_string_to_list(list_str):
    """Convert string representation of list to actual list"""
    if not list_str or list_str == '[]':
        return []
    
    try:
        # Try to parse as a Python list
        return ast.literal_eval(list_str)
    except (SyntaxError, ValueError):
        # Handle common format issues
        if list_str.startswith("['") and list_str.endswith("']"):
            items = list_str[2:-2].split("', '")
            return items
        return []

def process_csv_file(input_file, output_file, manual_start_row=None):
    """
    Process CSV file and update SMILES codes with resume capability
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        manual_start_row: If provided, start processing from this row (0-indexed)
    """
    # Check if output file exists to enable resume functionality
    start_index = 0
    drug_smiles_cache = {}
    
    # Load the input file regardless
    df = pd.read_csv(input_file)
    total_rows = len(df)
    
    # If manual start row is provided, use it
    if manual_start_row is not None:
        start_index = manual_start_row
        print(f"Manually starting from row {start_index+1} as specified.")
        
        # Check if the specified row is valid
        if start_index < 0 or start_index >= total_rows:
            print(f"Error: Specified start row {start_index+1} is out of range (1-{total_rows}).")
            return
    # Otherwise use automatic resume functionality
    elif os.path.exists(output_file):
        try:
            # Load the existing output file
            existing_df = pd.read_csv(output_file)
            existing_rows = len(existing_df)
            
            if existing_rows >= total_rows:
                print(f"Output file already contains all {existing_rows} rows from input file. Nothing to do.")
                return
                
            start_index = existing_rows
            print(f"Found existing output file with {start_index} processed rows.")
            print(f"Will resume processing from row {start_index+1}.")
            
            # Build cache from existing data
            for _, row in existing_df.iterrows():
                drugs_list = convert_list_string_to_list(row['drugs'])
                smiles_list = convert_list_string_to_list(row['smiless'])
                
                # Only add to cache if we have valid drug-SMILES pairs
                if len(drugs_list) == len(smiles_list):
                    for drug, smiles in zip(drugs_list, smiles_list):
                        if drug and drug not in drug_smiles_cache:
                            drug_smiles_cache[drug] = smiles if smiles else None
            
            print(f"Loaded {len(drug_smiles_cache)} drug-SMILES mappings from existing file.")
            
            # Copy already processed rows to our dataframe
            for i in range(start_index):
                df.iloc[i] = existing_df.iloc[i]
                
        except Exception as e:
            print(f"Error reading existing output file: {str(e)}")
            print("Starting from the beginning.")
            start_index = 0
    else:
        # No existing output file, start from scratch
        print("No existing output file found. Starting from the beginning.")
    
    # Process each row, starting from where we left off
    for index in range(start_index, total_rows):
        row = df.iloc[index]
        print(f"Processing row {index+1}/{total_rows}, NCT ID: {row['nctid']}")
        
        # Convert the drugs string to a list
        drugs_list = convert_list_string_to_list(row['drugs'])
        
        # Get SMILES for each drug
        smiles_list = []
        for drug in drugs_list:
            print(f"  Processing drug: {drug}")
            if drug in drug_smiles_cache:
                smiles = drug_smiles_cache[drug]
                print(f"    ↳ Using cached SMILES: {smiles if smiles else 'Not found'}")
            else:
                smiles = get_smiles(drug)
                drug_smiles_cache[drug] = smiles
                if smiles:
                    print(f"    ↳ Found SMILES: {smiles}")
                else:
                    print(f"    ↳ No SMILES found for drug: {drug}")
            
            smiles_list.append(smiles if smiles else "")
        
        # Update the smiless column
        df.at[index, 'smiless'] = str(smiles_list)
        
        # Every 10 rows, save progress to avoid losing everything if the script crashes
        if (index + 1) % 10 == 0 or index == total_rows - 1:
            df.to_csv(output_file, index=False)
            print(f"Progress saved at row {index+1}")
    
    # Save the final result if we haven't just saved
    if total_rows % 10 != 0 and (total_rows % 10) != (start_index % 10):
        df.to_csv(output_file, index=False)
    
    print(f"Processing complete. Results saved to {output_file}")
    
    # Print cache statistics
    print(f"Total unique drugs processed: {len(drug_smiles_cache)}")
    successful_lookups = sum(1 for s in drug_smiles_cache.values() if s)
    success_percent = (successful_lookups/len(drug_smiles_cache)*100) if drug_smiles_cache else 0
    print(f"Drugs with successful SMILES lookup: {successful_lookups} ({success_percent:.1f}% success rate)")
    
    # Print some examples of successful lookups
    print("\nExamples of successful SMILES lookups:")
    successful_items = [(drug, smiles) for drug, smiles in drug_smiles_cache.items() if smiles]
    for drug, smiles in successful_items[:5]:  # Show first 5 successful lookups
        print(f"  {drug} -> {smiles}")

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Convert drug names to SMILES codes')
    parser.add_argument('--input', type=str, default="clinical_trials_output.csv",
                        help='Input CSV file path (default: clinical_trials_output.csv)')
    parser.add_argument('--output', type=str, default="clinical_trials_with_smiles.csv",
                        help='Output CSV file path (default: clinical_trials_with_smiles.csv)')
    parser.add_argument('--start-row', type=int, 
                        help='Manually specify which row to start processing from (0-indexed)')
                        
    # Parse arguments
    args = parser.parse_args()
    
    # Process the CSV file
    process_csv_file(args.input, args.output, args.start_row)