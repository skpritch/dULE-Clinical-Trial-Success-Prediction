import pandas as pd
import os
import sys

def check_nctid_overlap(file1, file2, id_column='nctid', case_sensitive=False):
    """
    Check the overlap in NCTIDs between two CSV files.
    
    Args:
        file1 (str): Path to first CSV file
        file2 (str): Path to second CSV file
        id_column (str): Name of the column containing NCTIDs (default: 'nctid')
        case_sensitive (bool): Whether to treat NCTIDs as case-sensitive
    """
    try:
        # Load the datasets
        print(f"Loading {file1}...")
        df1 = pd.read_csv(file1)
        print(f"Loading {file2}...")
        df2 = pd.read_csv(file2)
        
        # Check if the ID column exists in both files
        if id_column not in df1.columns:
            raise ValueError(f"Column '{id_column}' not found in {file1}")
        if id_column not in df2.columns:
            raise ValueError(f"Column '{id_column}' not found in {file2}")
            
        # Extract the IDs
        ids1 = set(df1[id_column].astype(str))
        ids2 = set(df2[id_column].astype(str))
        
        # Handle case-sensitivity
        if not case_sensitive:
            ids1 = {id.upper() for id in ids1}
            ids2 = {id.upper() for id in ids2}
        
        # Calculate overlaps
        intersection = ids1.intersection(ids2)
        only_in_file1 = ids1 - ids2
        only_in_file2 = ids2 - ids1
        
        # Calculate percentages
        pct_overlap_of_file1 = 100 * len(intersection) / len(ids1) if len(ids1) > 0 else 0
        pct_overlap_of_file2 = 100 * len(intersection) / len(ids2) if len(ids2) > 0 else 0
        
        # Print results
        print("\nOverlap Analysis Results:")
        print(f"Total NCTIDs in {os.path.basename(file1)}: {len(ids1)}")
        print(f"Total NCTIDs in {os.path.basename(file2)}: {len(ids2)}")
        print(f"NCTIDs in both files: {len(intersection)} ({pct_overlap_of_file1:.1f}% of file 1, {pct_overlap_of_file2:.1f}% of file 2)")
        print(f"NCTIDs only in {os.path.basename(file1)}: {len(only_in_file1)} ({100 * len(only_in_file1) / len(ids1):.1f}% of file 1)")
        print(f"NCTIDs only in {os.path.basename(file2)}: {len(only_in_file2)} ({100 * len(only_in_file2) / len(ids2):.1f}% of file 2)")
        
        # Check if there's any overlap
        has_overlap = len(intersection) > 0
        print(f"\nDo the files have overlapping NCTIDs? {'YES' if has_overlap else 'NO'}")
        
        # Show a few examples if there's overlap
        if has_overlap:
            examples = sorted(list(intersection))[:10]  # Show up to 10 examples
            print(f"\nExample overlapping NCTIDs: {', '.join(examples)}")
            if len(intersection) > 10:
                print(f"...and {len(intersection) - 10} more")
        
        return {
            "has_overlap": has_overlap,
            "overlap_count": len(intersection),
            "file1_total": len(ids1),
            "file2_total": len(ids2),
            "only_in_file1": len(only_in_file1),
            "only_in_file2": len(only_in_file2)
        }
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python simple-nctid-checker.py file1.csv file2.csv [id_column]")
        print("  id_column: Optional name of the column containing NCTIDs (default: nctid)")
        sys.exit(1)
    
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    id_column = sys.argv[3] if len(sys.argv) > 3 else 'nctid'
    
    if not os.path.exists(file1):
        print(f"Error: File not found: {file1}")
        sys.exit(1)
    elif not os.path.exists(file2):
        print(f"Error: File not found: {file2}")
        sys.exit(1)
    else:
        check_nctid_overlap(file1, file2, id_column=id_column)