#!/usr/bin/env python3
"""
Script to load TF-IDF vectors from a CSV file (where they're stored as strings),
convert them back to vectors, apply PCA to reduce them to 50 dimensions,
and keep the original TF-IDF column in the output.
"""

import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA
import sys
import os
import json

def parse_tfidf_string(tfidf_str, num_features):
    """
    Convert a string representation of a sparse TF-IDF vector back to a sparse vector.
    
    Args:
        tfidf_str (str): String representation of sparse vector in format "idx1:val1,idx2:val2,..."
        num_features (int): Total number of features in the TF-IDF vector space
        
    Returns:
        scipy.sparse.csr_matrix: Sparse vector representation
    """
    if not tfidf_str or pd.isna(tfidf_str):
        return sp.csr_matrix((1, num_features))
    
    indices = []
    data = []
    
    # Parse the string format "idx1:val1,idx2:val2,..."
    pairs = tfidf_str.split(',')
    for pair in pairs:
        if ':' in pair:
            idx_str, val_str = pair.split(':')
            try:
                idx = int(idx_str)
                val = float(val_str)
                indices.append(idx)
                data.append(val)
            except ValueError:
                print(f"Warning: Could not parse pair '{pair}', skipping")
    
    # Create a sparse vector
    row_indices = [0] * len(indices)  # All in the same row (just one vector)
    shape = (1, num_features)
    
    return sp.csr_matrix((data, (row_indices, indices)), shape=shape)

def tfidf_to_pca(input_file, output_file, tfidf_col='tfidf_vector_str', n_components=50, delimiter=',', matrix_column='pca_vector'):
    """
    Load TF-IDF vectors from a CSV file, convert them to dense vectors, and apply PCA.
    
    Args:
        input_file (str): Path to input CSV file with TF-IDF vectors as strings
        output_file (str): Path to output CSV file to save PCA results
        tfidf_col (str): Column name containing the TF-IDF string vectors
        n_components (int): Number of PCA components to generate
        delimiter (str): Delimiter used in the CSV file
        matrix_column (str): Name of the column to store the PCA matrix
    """
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return False
    
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file, sep=delimiter)
    
    if tfidf_col not in df.columns:
        print(f"Error: Column '{tfidf_col}' not found in the input file")
        return False
    
    print(f"Found {len(df)} rows in the input file")
    
    # Determine the number of features by examining the TF-IDF strings
    # We'll do this by finding the maximum index used in any vector
    max_feature_idx = 0
    for tfidf_str in df[tfidf_col].dropna():
        for pair in tfidf_str.split(','):
            if ':' in pair:
                idx_str = pair.split(':')[0]
                try:
                    idx = int(idx_str)
                    max_feature_idx = max(max_feature_idx, idx)
                except ValueError:
                    continue
    
    num_features = max_feature_idx + 1
    print(f"Detected {num_features} features in the TF-IDF vectors")
    
    # Convert TF-IDF strings back to sparse vectors
    print("Converting TF-IDF strings to sparse vectors...")
    sparse_matrices = []
    for idx, tfidf_str in enumerate(df[tfidf_col]):
        if idx % 1000 == 0 and idx > 0:
            print(f"  Processed {idx} rows...")
        sparse_matrices.append(parse_tfidf_string(tfidf_str, num_features))
    
    # Combine all sparse matrices into one large sparse matrix
    print("Combining sparse matrices...")
    X_sparse = sp.vstack(sparse_matrices)
    
    # Convert to dense matrix for PCA
    print("Converting to dense matrix for PCA...")
    X_dense = X_sparse.toarray()
    
    # Apply PCA
    print(f"Applying PCA to reduce to {n_components} dimensions...")
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_dense)
    
    print(f"PCA explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # Convert PCA results to a list of lists (matrix format)
    print("Converting PCA results to matrix format...")
    
    # Store as JSON strings
    pca_matrix_json = [json.dumps(vector.tolist()) for vector in X_pca]
    
    # Create a new DataFrame (keep all original columns)
    result_df = df.copy()
    
    # Add the PCA matrix as a single column
    result_df[matrix_column] = pca_matrix_json
    
    # Save to output file
    print(f"Saving results to {output_file}...")
    result_df.to_csv(output_file, index=False, sep=delimiter)
    
    # Also save the PCA object for future use
    print("Saving PCA model...")
    from joblib import dump
    dump(pca, 'pca_model.joblib')
    
    # Save a small metadata file with the explained variance
    explained_variance = {
        'total_explained_variance': float(np.sum(pca.explained_variance_ratio_)),
        'individual_explained_variance': pca.explained_variance_ratio_.tolist(),
        'n_components': n_components,
        'original_features': num_features
    }
    with open('pca_metadata.json', 'w') as f:
        json.dump(explained_variance, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"TF-IDF to PCA Conversion Summary")
    print(f"{'='*50}")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Rows processed: {len(df)}")
    print(f"Original features: {num_features}")
    print(f"PCA dimensions: {n_components}")
    print(f"Explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")
    print(f"Original TF-IDF column preserved: {tfidf_col}")
    print(f"PCA matrix stored in column: {matrix_column}")
    print(f"PCA model saved to: pca_model.joblib")
    print(f"PCA metadata saved to: pca_metadata.json")
    print(f"\nConversion complete!")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python tfidf_to_pca_keep_tfidf.py <input.csv> <output.csv> [--column COLUMN_NAME] [--dims DIMENSIONS] [--matrix MATRIX_COLUMN] [--tab]")
        print("\nThis script converts TF-IDF string vectors to PCA vectors and stores them in a matrix format,")
        print("while keeping the original TF-IDF column.")
        print("Parameters:")
        print("  input.csv: Input CSV file with TF-IDF string vectors")
        print("  output.csv: Output CSV file to save with PCA vectors")
        print("  --column COLUMN_NAME: Optional column name containing TF-IDF strings (default: tfidf_vector_str)")
        print("  --dims DIMENSIONS: Optional number of PCA dimensions (default: 50)")
        print("  --matrix MATRIX_COLUMN: Optional name for the column to store the PCA matrix (default: pca_vector)")
        print("  --tab: Optional flag for tab-delimited files\n")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Parse optional arguments
    tfidf_col = 'tfidf_vector_str'  # Default column name
    n_components = 50  # Default number of PCA dimensions
    delimiter = ','  # Default delimiter
    matrix_column = 'pca_vector'  # Default matrix column name
    
    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == '--column' and i + 1 < len(sys.argv):
            tfidf_col = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--dims' and i + 1 < len(sys.argv):
            try:
                n_components = int(sys.argv[i + 1])
                i += 2
            except ValueError:
                print(f"Error: Invalid number of dimensions: {sys.argv[i + 1]}")
                sys.exit(1)
        elif sys.argv[i] == '--matrix' and i + 1 < len(sys.argv):
            matrix_column = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--tab':
            delimiter = '\t'
            i += 1
        else:
            print(f"Warning: Unknown argument: {sys.argv[i]}")
            i += 1
    
    # Try to auto-detect the delimiter if not specified
    if delimiter == ',' and not '--tab' in sys.argv:
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if '\t' in first_line and ',' not in first_line:
                    delimiter = '\t'
                    print(f"Auto-detected tab-delimited format")
                elif ',' in first_line:
                    print(f"Using comma as delimiter")
        except Exception as e:
            print(f"Error reading file: {e}")
    
    tfidf_to_pca(input_file, output_file, tfidf_col, n_components, delimiter, matrix_column)