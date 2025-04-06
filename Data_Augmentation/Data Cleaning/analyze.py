import pandas as pd
import ast
import re
import os

def check_and_clean_matrix_formats(csv_file, auto_remove=False, output_file=None):
    """
    Check if any values in the 'criteria' column or other columns
    that should contain matrices have inconsistent formats.
    Optionally remove problematic rows and save the cleaned data.
    
    Args:
        csv_file (str): Path to the CSV file
        auto_remove (bool): If True, automatically remove problematic rows without prompting
        output_file (str): Path to save cleaned data. If None, will modify original file
    """
    try:
        # Load the data
        df = pd.read_csv(csv_file)
        print(f"Successfully loaded data with {len(df)} rows and {len(df.columns)} columns.")
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        return
    
    # Function to check for indentation issues in a string
    def has_indentation_issues(val):
        if not isinstance(val, str):
            return False
        
        # Check for inconsistent spacing at the beginning of lines
        lines = val.strip().split('\n')
        if len(lines) > 1:
            indent_patterns = [re.match(r'^\s*', line).group() for line in lines if line.strip()]
            return len(set(indent_patterns)) > 1
        return False
    
    # Function to check if a value is a numeric matrix
    def is_numeric_matrix(val):
        if not isinstance(val, str):
            return False
        
        try:
            parsed = ast.literal_eval(val)
            return isinstance(parsed, list) and all(isinstance(x, (int, float)) for x in parsed)
        except (ValueError, SyntaxError):
            return False
    
    # Function to analyze matrix issues
    def analyze_matrix_issues(val):
        issues = []
        if has_indentation_issues(val):
            issues.append("indentation issues")
        
        try:
            parsed = ast.literal_eval(val)
            if not isinstance(parsed, list):
                issues.append("not a list")
            elif not all(isinstance(x, (int, float)) for x in parsed):
                issues.append("contains non-numeric elements")
        except (ValueError, SyntaxError) as e:
            issues.append(f"parsing error: {str(e)}")
        
        return issues if issues else None
    
    # Get a key identifier column for better row identification
    id_column = 'nctid' if 'nctid' in df.columns else df.columns[0]
    
    # To store all problematic row indices
    all_problematic_rows = set()
    
    # Check the criteria column
    print("\nChecking 'criteria' column...")
    if 'criteria' in df.columns:
        non_matrix_rows = []
        
        for idx, row in df.iterrows():
            val = row['criteria']
            if pd.notna(val) and not is_numeric_matrix(val):
                issues = analyze_matrix_issues(val) if isinstance(val, str) else ["not a string"]
                id_value = row[id_column]
                non_matrix_rows.append((idx, id_value, val, issues))
                all_problematic_rows.add(idx)
        
        if non_matrix_rows:
            print(f"Found {len(non_matrix_rows)} rows where 'criteria' is not a numeric matrix:")
            for df_idx, id_value, val, issues in non_matrix_rows:  # Show all problematic rows
                issue_str = ", ".join(issues) if issues else "unknown issue"
                val_preview = str(val)[:100] + "..." if len(str(val)) > 100 else str(val)
                print(f"  DataFrame Row {df_idx}, {id_column}={id_value}: [Issues: {issue_str}]")
                print(f"    Value preview: {val_preview}")
        else:
            print("All values in 'criteria' column are properly formatted numeric matrices.")
    else:
        print("'criteria' column not found in the data.")
    
    # Identify other columns that might contain matrices
    print("\nChecking other columns that might contain matrices...")
    potential_matrix_cols = []
    
    for col in df.columns:
        if col in ['nctid', 'status', 'why_stop', 'label', 'phase']:
            continue  # Skip columns that clearly shouldn't be matrices
            
        # Sample some values to see if this column might contain matrices
        sample = df[col].dropna().head(5).tolist()
        matrix_count = sum(1 for val in sample if isinstance(val, str) and is_numeric_matrix(val))
        
        # If a significant portion of samples are matrices, consider it a matrix column
        if matrix_count >= 2:
            potential_matrix_cols.append(col)
    
    print(f"Identified {len(potential_matrix_cols)} columns that may contain matrices: {', '.join(potential_matrix_cols)}")
    
    # Check each potential matrix column for inconsistencies
    for col in potential_matrix_cols:
        if col == 'criteria':  # Already checked
            continue
            
        print(f"\nAnalyzing column '{col}'...")
        
        non_matrix_rows = []
        for idx, row in df.iterrows():
            val = row[col]
            if pd.notna(val) and not is_numeric_matrix(val):
                issues = analyze_matrix_issues(val) if isinstance(val, str) else ["not a string"]
                id_value = row[id_column]
                non_matrix_rows.append((idx, id_value, val, issues))
                all_problematic_rows.add(idx)
        
        total_non_na = df[col].count()
        matrix_rows = total_non_na - len(non_matrix_rows)
        
        print(f"  {matrix_rows} out of {total_non_na} non-NA values are numeric matrices ({matrix_rows/total_non_na*100:.1f}%)")
        
        if non_matrix_rows:
            print(f"  Found {len(non_matrix_rows)} rows with non-matrix format")
            
            # Count specific issues
            issue_counts = {}
            for _, _, _, issues in non_matrix_rows:
                if issues:
                    for issue in issues:
                        issue_counts[issue] = issue_counts.get(issue, 0) + 1
            
            # Print issue statistics
            if issue_counts:
                print("  Issue breakdown:")
                for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"    - {issue}: {count} occurrences ({count/len(non_matrix_rows)*100:.1f}%)")
            
            # Print all problematic rows
            print("  Problem rows:")
            for df_idx, id_value, val, issues in non_matrix_rows:
                issue_str = ", ".join(issues) if issues else "unknown issue"
                val_preview = str(val)[:100] + "..." if len(str(val)) > 100 else str(val)
                print(f"    DataFrame Row {df_idx}, {id_column}={id_value}: [Issues: {issue_str}]")
                print(f"      Value preview: {val_preview}")
        else:
            print("  All values are properly formatted numeric matrices.")
    
    # Print summary
    print("\nSummary:")
    inconsistent_cols = []
    
    if 'criteria' in df.columns and len(non_matrix_rows) > 0:
        inconsistent_cols.append('criteria')
    
    for col in potential_matrix_cols:
        if col == 'criteria':  # Already counted
            continue
        non_matrix_count = sum(1 for val in df[col] if pd.notna(val) and not is_numeric_matrix(val))
        if non_matrix_count > 0:
            inconsistent_cols.append(col)
    
    if inconsistent_cols:
        print(f"Found {len(inconsistent_cols)} columns with inconsistent matrix formatting: {', '.join(inconsistent_cols)}")
    else:
        print("All identified matrix columns have consistent formatting.")
    
    # Handle removal of problematic rows
    if all_problematic_rows:
        problematic_rows_list = sorted(list(all_problematic_rows))
        if auto_remove or input(f"\nDo you want to remove {len(problematic_rows_list)} problematic rows? (y/n): ").lower() == 'y':
            # Create a backup of the original file
            if output_file is None:  # In-place modification
                backup_file = f"{csv_file}.bak"
                print(f"Creating backup of original file as {backup_file}")
                try:
                    # Copy the file
                    with open(csv_file, 'r') as src, open(backup_file, 'w') as dst:
                        dst.write(src.read())
                except Exception as e:
                    print(f"Error creating backup file: {str(e)}")
                    return
            
            # Remove problematic rows
            print(f"Removing {len(problematic_rows_list)} problematic rows...")
            cleaned_df = df.drop(index=problematic_rows_list)
            print(f"Rows removed. New data has {len(cleaned_df)} rows.")
            
            # Save the cleaned data
            save_path = output_file if output_file else csv_file
            try:
                cleaned_df.to_csv(save_path, index=False)
                print(f"Cleaned data saved to {save_path}")
            except Exception as e:
                print(f"Error saving cleaned data: {str(e)}")
                return
            
            # Print removed row details
            print("\nRemoved rows details:")
            for idx in problematic_rows_list[:10]:  # Show first 10 removed rows
                print(f"  Row {idx}, {id_column}={df.loc[idx, id_column]}")
            if len(problematic_rows_list) > 10:
                print(f"  ... and {len(problematic_rows_list) - 10} more rows")
            
            return cleaned_df
        else:
            print("No rows were removed.")
    else:
        print("No problematic rows found. No changes needed.")
    
    return df

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Check and clean matrix formats in CSV file')
    parser.add_argument('file', nargs='?', help='Path to the CSV file')
    parser.add_argument('--auto-remove', action='store_true', help='Automatically remove problematic rows without prompting')
    parser.add_argument('--output', help='Output file path for cleaned data. If not provided, will modify original file')
    args = parser.parse_args()
    
    if args.file:
        file_path = args.file
    else:
        file_path = input("Enter the path to your CSV file: ")
    
    if os.path.exists(file_path):
        check_and_clean_matrix_formats(file_path, auto_remove=args.auto_remove, output_file=args.output)
    else:
        print(f"File not found: {file_path}")
        print("Please provide a valid file path.")
        # Default to paste.txt if available
        if os.path.exists("paste.txt"):
            print("Trying default file 'paste.txt'...")
            check_and_clean_matrix_formats("paste.txt", auto_remove=args.auto_remove, output_file=args.output)
        else:
            print("No valid file path provided. Exiting.")