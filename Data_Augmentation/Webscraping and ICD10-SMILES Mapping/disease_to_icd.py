#!/usr/bin/env python3
"""
Clinical Trials ICD-10 Mapper using a local ICD-10 CSV file
"""

import os
import json
import csv
import time
import sys
import re
from collections import defaultdict

def load_icd10_codes(file_path):
    """
    Load ICD-10 codes from a CSV file with the specific format from paste.txt
    
    Args:
        file_path: Path to the CSV file containing ICD-10 codes
        
    Returns:
        dict: Mapping from disease descriptions to ICD-10 codes
        dict: Mapping from ICD-10 codes to disease descriptions
    """
    print(f"Loading ICD-10 codes from {file_path}...")
    
    # Create dictionaries for lookups
    description_to_code = {}
    code_to_description = {}
    keywords_to_code = defaultdict(list)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            
            # Based on your CSV example, these are the correct column indices:
            # Column 2 (index 2) contains the full ICD-10 code (e.g., "A000")
            # Column 3 (index 3) contains the detailed description
            # Column 5 (index 5) contains the short description
            
            code_index = 2  # The full code like "A000" is in column 3 (index 2)
            desc_index = 3  # The long description is in column 4 (index 3)
            short_desc_index = 5  # The short description is in column 6 (index 5)
            
            # Load data from CSV
            row_count = 0
            for row in reader:
                row_count += 1
                if len(row) > max(code_index, desc_index, short_desc_index):
                    code = row[code_index].strip()
                    long_description = row[desc_index].strip()
                    short_description = row[short_desc_index].strip()
                    
                    if code and (long_description or short_description):
                        # Store both long and short descriptions
                        if long_description:
                            description_to_code[long_description.lower()] = code
                        if short_description:
                            description_to_code[short_description.lower()] = code
                        
                        # Store the code to description mapping (using the longer description)
                        code_to_description[code] = long_description if long_description else short_description
                        
                        # Create keyword mappings for both descriptions
                        for description in [long_description, short_description]:
                            if description:
                                words = description.lower().split()
                                for word in words:
                                    # Only use meaningful words as keywords (not short prepositions, etc.)
                                    if len(word) > 3 and word not in ['with', 'without', 'from', 'this', 'that', 'have', 'been']:
                                        keywords_to_code[word].append(code)
            
            print(f"Loaded {row_count} rows from CSV.")
            print(f"Created {len(description_to_code)} description-to-code mappings.")
            print(f"Created {len(keywords_to_code)} keyword-to-code mappings.")
        
        return description_to_code, code_to_description, keywords_to_code
        
    except Exception as e:
        print(f"Error loading ICD-10 codes: {str(e)}")
        return {}, {}, defaultdict(list)

def get_icd10_for_disease(disease_name, description_to_code, keywords_to_code):
    """
    Find ICD-10 code for a disease using loaded dictionaries
    
    Args:
        disease_name: Name of the disease
        description_to_code: Dictionary mapping descriptions to ICD-10 codes
        keywords_to_code: Dictionary mapping keywords to lists of ICD-10 codes
        
    Returns:
        list: List of dictionaries with code and description
    """
    results = []
    disease_key = disease_name.lower()
    
    # Strategy 1: Direct match with description
    if disease_key in description_to_code:
        code = description_to_code[disease_key]
        results.append({
            "code": code,
            "description": disease_name
        })
        return results
    
    # Strategy 2: Normalize and check again
    # Remove punctuation and extra spaces
    normalized_disease = re.sub(r'[^\w\s]', ' ', disease_key)
    normalized_disease = re.sub(r'\s+', ' ', normalized_disease).strip()
    
    if normalized_disease in description_to_code:
        code = description_to_code[normalized_disease]
        results.append({
            "code": code,
            "description": disease_name
        })
        return results
    
    # Strategy 3: Split by commas and check each part
    if ',' in disease_key:
        parts = [part.strip() for part in disease_key.split(',')]
        for part in parts:
            if part in description_to_code:
                code = description_to_code[part]
                results.append({
                    "code": code,
                    "description": part
                })
                return results
    
    # Strategy 4: Keyword matching
    # Use the most specific/longest word that gets a match
    words = normalized_disease.split()
    if len(words) > 0:
        # Sort words by length (longest first) to prioritize more specific terms
        words.sort(key=len, reverse=True)
        
        for word in words:
            if len(word) > 3 and word in keywords_to_code:
                # Take the first code associated with this keyword
                code = keywords_to_code[word][0]
                results.append({
                    "code": code,
                    "description": word
                })
                return results
    
    return results

def extract_diseases_from_trial(json_data):
    """
    Extract disease information from a clinical trial
    
    Args:
        json_data: Clinical trial data in JSON format
        
    Returns:
        list: List of disease names
    """
    diseases = []
    
    # Convert string to JSON if necessary
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data
    
    # Extract conditions from the conditionsModule
    conditions_module = data.get('protocolSection', {}).get('conditionsModule', {})
    if 'conditions' in conditions_module:
        diseases.extend(conditions_module['conditions'])
        
    # Also try to get keywords which may be better for ICD-10 matching
    if 'keywords' in conditions_module:
        for keyword in conditions_module['keywords']:
            if keyword not in diseases:
                diseases.append(keyword)
    
    return diseases

def get_multiple_codes_for_disease(disease_name, description_to_code, keywords_to_code, max_codes=7):
    """
    Get multiple ICD-10 codes for a disease
    
    Args:
        disease_name: Name of the disease
        description_to_code: Dictionary mapping descriptions to ICD-10 codes
        keywords_to_code: Dictionary mapping keywords to lists of ICD-10 codes
        max_codes: Maximum number of codes to return
        
    Returns:
        list: List of ICD-10 codes
    """
    all_codes = set()
    disease_key = disease_name.lower()
    
    # Direct match
    if disease_key in description_to_code:
        all_codes.add(description_to_code[disease_key])
    
    # Keyword matching for more codes
    normalized_disease = re.sub(r'[^\w\s]', ' ', disease_key)
    normalized_disease = re.sub(r'\s+', ' ', normalized_disease).strip()
    
    words = normalized_disease.split()
    for word in words:
        if len(word) > 3 and word in keywords_to_code:
            # Add all codes for this keyword, up to a limit
            for code in keywords_to_code[word][:max_codes - len(all_codes)]:
                if len(all_codes) < max_codes:
                    all_codes.add(code)
    
    # Convert to list and sort
    codes_list = sorted(list(all_codes))
    
    # If we don't have enough codes, pad with similar codes (by prefix)
    if codes_list and len(codes_list) < max_codes:
        prefix = codes_list[0][:1]  # Get the letter prefix
        
        # Find codes with same prefix
        for code in code_to_description:
            if code.startswith(prefix) and code not in codes_list:
                codes_list.append(code)
                if len(codes_list) >= max_codes:
                    break
    
    return codes_list

def process_trials(directory_path, output_csv, description_to_code, code_to_description, keywords_to_code):
    """
    Process all clinical trial files in a directory
    
    Args:
        directory_path: Path to directory containing clinical trial files
        output_csv: Path to save the CSV output file
        description_to_code: Dictionary mapping descriptions to ICD-10 codes
        code_to_description: Dictionary mapping ICD-10 codes to descriptions
        keywords_to_code: Dictionary mapping keywords to lists of ICD-10 codes
    """
    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' not found.")
        return
    
    # Get all JSON files in the directory
    files = [f for f in os.listdir(directory_path) 
             if f.endswith('.json') or f.endswith('.txt')]
    
    if not files:
        print(f"No JSON or TXT files found in '{directory_path}'.")
        return
    
    print(f"Found {len(files)} files to process.")
    
    # Create/open the CSV file
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['NCT ID', 'Disease', 'ICD-10 Codes'])
    
    # Track progress
    total_diseases = 0
    total_with_codes = 0
    matched_diseases = set()
    unmatched_diseases = set()
    
    # Process each file
    for i, filename in enumerate(files):
        filepath = os.path.join(directory_path, filename)
        
        # Print progress every 100 files or for the first few
        if i < 5 or i % 100 == 0 or i == len(files) - 1:
            print(f"Processing file {i+1}/{len(files)}: {filename}")
        
        try:
            # Load trial data
            with open(filepath, 'r', encoding='utf-8') as file:
                trial_data = json.load(file)
            
            # Extract NCT ID
            nct_id = trial_data.get('protocolSection', {}).get('identificationModule', {}).get('nctId', filename)
            
            # Extract diseases
            diseases = extract_diseases_from_trial(trial_data)
            
            if i < 5:  # Show detailed output for first few files
                print(f"Found {len(diseases)} diseases in {nct_id}: {', '.join(diseases)}")
            
            total_diseases += len(diseases)
            
            # Create a list to store arrays of ICD codes for this trial
            trial_icd_codes = []
            
            # Process each disease
            for disease in diseases:
                # Get multiple ICD-10 codes (up to 7)
                icd_codes = get_multiple_codes_for_disease(disease, description_to_code, keywords_to_code)
                
                if icd_codes:
                    total_with_codes += 1
                    matched_diseases.add(disease.lower())
                    
                    # Format codes in the required nested array format
                    # ["['E11.65', 'E11.9', 'E11.21', 'E11.36', 'E11.41', 'E11.42', 'E11.44']"]
                    formatted_codes = str(icd_codes).replace('"', "'")
                    trial_icd_codes.append([formatted_codes])
                else:
                    unmatched_diseases.add(disease.lower())
                    # Add empty array if no codes found
                    trial_icd_codes.append(["[]"])
                
                # Append to CSV - use the nested array format
                with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([nct_id, disease, json.dumps(trial_icd_codes[-1])])
                
        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
    
    print(f"\nProcessing complete. Results saved to {output_csv}")
    percentage = (total_with_codes/total_diseases*100) if total_diseases > 0 else 0.0
    print(f"Processed {total_diseases} diseases, found ICD-10 codes for {total_with_codes} ({percentage:.1f}%)")
    
    # Save lists of matched and unmatched diseases
    try:
        os.makedirs("results", exist_ok=True)
        
        with open("results/matched_diseases.txt", 'w', encoding='utf-8') as f:
            for disease in sorted(matched_diseases):
                f.write(f"{disease}\n")
        print(f"Saved list of {len(matched_diseases)} matched diseases to results/matched_diseases.txt")
        
        with open("results/unmatched_diseases.txt", 'w', encoding='utf-8') as f:
            for disease in sorted(unmatched_diseases):
                f.write(f"{disease}\n")
        print(f"Saved list of {len(unmatched_diseases)} unmatched diseases to results/unmatched_diseases.txt")
    except Exception as e:
        print(f"Error saving disease lists: {str(e)}")

def main():
    """
    Main function to run the script
    """
    # Hard-coded paths - edit these to match your file locations
    input_dir = "clinical_trials_json"  # Directory with your JSON files
    output_file = "results/icd10_mapping.csv"  # Where to save results
    icd10_csv = "codes_formatted.csv"  # Path to your ICD-10 CSV file
    
    print("Clinical Trials ICD-10 Mapper")
    print("============================")
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file}")
    print(f"ICD-10 CSV file: {icd10_csv}")
    print()  # Add a blank line for readability
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    # Check if the ICD-10 CSV file exists
    if not os.path.exists(icd10_csv):
        print(f"Error: ICD-10 CSV file '{icd10_csv}' not found.")
        print("Please ensure the ICD-10 CSV file is in your current directory.")
        return
    
    # Check if the input directory exists
    if not os.path.exists(input_dir):
        # Try to create it
        try:
            os.makedirs(input_dir, exist_ok=True)
            print(f"Created input directory: {input_dir}")
            print(f"Please place your clinical trial JSON files in {input_dir} and run again.")
            return
        except Exception as e:
            print(f"Error creating input directory: {str(e)}")
            return
    
    # Load ICD-10 codes from CSV
    description_to_code, code_to_description, keywords_to_code = load_icd10_codes(icd10_csv)
    
    if not description_to_code:
        print("Failed to load ICD-10 codes. Exiting.")
        return
    
    # Process all trials
    process_trials(input_dir, output_file, description_to_code, code_to_description, keywords_to_code)

if __name__ == "__main__":
    main()