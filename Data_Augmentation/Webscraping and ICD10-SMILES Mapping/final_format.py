import os
import json
import pandas as pd
import csv
import glob

# Step 1: Read the existing ICD-10 code CSV
icd_df = pd.read_csv('resulticd10.csv')
icd_dict = dict(zip(icd_df['NCT ID'], icd_df['ICD-10 Codes']))

labels_df = pd.read_csv('trial_classifications.csv', header=None)
labels_df.columns = ['nctid', 'label', 'status']
labels_dict = dict(zip(labels_df['nctid'], labels_df['label']))
# Step 2: Create a list to store all the extracted data
data_rows = []

# Step 3: Process each JSON file in the clinical_trials_json folder
json_files = glob.glob(os.path.join('clinical_trials_json', '*.json'))

for i, file_path in enumerate(json_files):
    # Print progress indicator
    print(f"Processing file {i+1}/{len(json_files)}: {os.path.basename(file_path)}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract the basic information
        protocol_section = data.get('protocolSection', {})
        
        # Extract nctid
        identification_module = protocol_section.get('identificationModule', {})
        nctid = identification_module.get('nctId', '')
        
        # Skip if no nctid found
        if not nctid:
            continue
        
        # Extract status and why_stop
        status_module = protocol_section.get('statusModule', {})
        status = status_module.get('overallStatus', '').lower()
        why_stop = status_module.get('whyStopped', '')
        
        # Extract phase information
        design_module = protocol_section.get('designModule', {})
        phases = design_module.get('phases', [])
        phase = phases[0].lower() if phases else ''
        
        # Extract diseases
        conditions_module = protocol_section.get('conditionsModule', {})
        diseases = conditions_module.get('conditions', [])
        
        # Get ICD-10 codes from the dictionary
        icdcodes = icd_dict.get(nctid, '')
        
        # Extract drugs from interventions
        arms_interventions_module = protocol_section.get('armsInterventionsModule', {})
        interventions = arms_interventions_module.get('interventions', [])
        arm_groups = arms_interventions_module.get('armGroups', [])
        
        # Collect drugs from interventions
        drugs = []
        for intervention in interventions:
            if intervention.get('type') == 'DRUG':
                drug_name = intervention.get('name', '')
                if drug_name:
                    drugs.append(drug_name)
        
        # If no drugs found in interventions, try arm groups descriptions
        if not drugs and arm_groups:
            for arm in arm_groups:
                desc = arm.get('description', '')
                if desc and ('drug' in desc.lower() or 'dose' in desc.lower()):
                    drugs.append(arm.get('label', ''))
        
        # Extract criteria
        eligibility_module = protocol_section.get('eligibilityModule', {})
        criteria = eligibility_module.get('eligibilityCriteria', '')
        
        # For SMILES strings, we would need another data source
        # For now, create an empty list with the same length as drugs
        smiless = [""] * len(drugs)
        
        # Set label based on status (0 for terminated, 1 for others)
        if nctid not in labels_dict:
            continue  # Skip this trial if not in our predefined list
            
        # Use the predefined label
        label = str(labels_dict[nctid])
        
        # Add the row to our data
        data_rows.append({
            'nctid': nctid,
            'status': status,
            'why_stop': why_stop,
            'label': label,
            'phase': phase,
            'diseases': str(diseases),
            'icdcodes': icdcodes,
            'drugs': str(drugs),
            'smiless': str(smiless),
            'criteria': criteria
        })
        
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")

# Step 4: Convert to DataFrame and save as CSV
if data_rows:
    df = pd.DataFrame(data_rows)
    
    # Clean up the format to match the example
    # Ensure the list representations are consistent
    df['diseases'] = df['diseases'].apply(lambda x: x.replace('"', "'"))
    df['drugs'] = df['drugs'].apply(lambda x: x.replace('"', "'"))
    df['smiless'] = df['smiless'].apply(lambda x: x.replace('"', "'"))
    
    # Save as CSV with minimal quoting to preserve formatting
    df.to_csv('clinical_trials_output.csv', index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Successfully processed {len(data_rows)} JSON files. Output saved to clinical_trials_output.csv")
else:
    print("No data was extracted. Check the JSON files path and content.")