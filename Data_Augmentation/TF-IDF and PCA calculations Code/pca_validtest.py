#!/usr/bin/env python3
"""
Script to compute PCA vectors for validation and test sets by:
1. Finding the 5 closest TF-IDF neighbors from the training set for each row
2. Averaging the pre-computed PCA vectors of these neighbors
3. Adding this averaged PCA vector to each row in valid/test
"""

import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
import sys
import os
import json
import time
from joblib import load

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
                continue
    
    # Create a sparse vector
    row_indices = [0] * len(indices)  # All in the same row (just one vector)
    shape = (1, num_features)
    
    return sp.csr_matrix((data, (row_indices, indices)), shape=shape)

def parse_pca_string(pca_str):
    """
    Convert a string representation of a PCA vector back to a numpy array.
    
    Args:
        pca_str (str): JSON string representation of PCA vector
        
    Returns:
        numpy.ndarray: PCA vector
    """
    if not pca_str or pd.isna(pca_str):
        return None
    
    try:
        return np.array(json.loads(pca_str))
    except (json.JSONDecodeError, TypeError):
        print(f"Warning: Could not parse PCA string: {pca_str[:50]}...")
        return None

def compute_pca_via_neighbors(train_file, target_file, output_file, 
                             tfidf_col='tfidf_vector_str', 
                             pca_col='pca_vector',
                             n_neighbors=5,
                             delimiter=','):
    """
    Compute PCA vectors for a target file (validation or test) by averaging PCA vectors 
    of nearest neighbors from training set.
    
    Args:
        train_file (str): Path to training CSV file with TF-IDF and PCA vectors
        target_file (str): Path to target CSV file (validation/test) with TF-IDF vectors
        output_file (str): Path to output CSV file
        tfidf_col (str): Column name containing the TF-IDF string vectors
        pca_col (str): Column name for PCA vectors in training file and to use in output
        n_neighbors (int): Number of nearest neighbors to find
        delimiter (str): Delimiter used in the CSV files
    """
    start_time = time.time()
    
    # Check if files exist
    if not os.path.exists(train_file):
        print(f"Error: Training file not found: {train_file}")
        return False
    
    if not os.path.exists(target_file):
        print(f"Error: Target file not found: {target_file}")
        return False
    
    # Load the training data
    print(f"Loading training data from {train_file}...")
    train_df = pd.read_csv(train_file, sep=delimiter)
    
    if tfidf_col not in train_df.columns:
        print(f"Error: Column '{tfidf_col}' not found in the training file")
        return False
    
    if pca_col not in train_df.columns:
        print(f"Error: Column '{pca_col}' not found in the training file")
        return False
    
    print(f"Found {len(train_df)} rows in the training file")
    
    # Load the target data (validation or test)
    print(f"Loading target data from {target_file}...")
    target_df = pd.read_csv(target_file, sep=delimiter)
    
    if tfidf_col not in target_df.columns:
        print(f"Error: Column '{tfidf_col}' not found in the target file")
        return False
    
    print(f"Found {len(target_df)} rows in the target file")
    
    # Determine the number of features by examining the TF-IDF strings in training data
    max_feature_idx = 0
    for tfidf_str in train_df[tfidf_col].dropna():
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
    
    # Convert training TF-IDF strings to sparse vectors
    print("Converting training TF-IDF strings to sparse vectors...")
    train_sparse_matrices = []
    for idx, tfidf_str in enumerate(train_df[tfidf_col]):
        if idx % 2000 == 0 and idx > 0:
            print(f"  Processed {idx}/{len(train_df)} training rows...")
        train_sparse_matrices.append(parse_tfidf_string(tfidf_str, num_features))
    
    # Combine training sparse matrices
    print("Combining training sparse matrices...")
    X_train_sparse = sp.vstack(train_sparse_matrices)
    
    # Convert target TF-IDF strings to sparse vectors
    print("Converting target TF-IDF strings to sparse vectors...")
    target_sparse_matrices = []
    for idx, tfidf_str in enumerate(target_df[tfidf_col]):
        if idx % 2000 == 0 and idx > 0:
            print(f"  Processed {idx}/{len(target_df)} target rows...")
        target_sparse_matrices.append(parse_tfidf_string(tfidf_str, num_features))
    
    # Combine target sparse matrices
    print("Combining target sparse matrices...")
    X_target_sparse = sp.vstack(target_sparse_matrices)
    
    # Initialize NearestNeighbors model and fit on training data
    print("Fitting NearestNeighbors model on training data...")
    nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', n_jobs=-1)
    nn_model.fit(X_train_sparse)
    
    # Parse the PCA vectors from training data
    print("Parsing PCA vectors from training data...")
    train_pca_vectors = []
    for pca_str in train_df[pca_col]:
        train_pca_vectors.append(parse_pca_string(pca_str))
    
    # Find nearest neighbors for each target row and compute PCA by averaging
    print(f"Finding {n_neighbors} nearest neighbors for each target row...")
    target_pca_vectors = []
    
    batch_size = 1000  # Process in batches to avoid memory issues
    num_batches = (len(target_df) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(target_df))
        print(f"Processing batch {batch_idx+1}/{num_batches} (rows {start_idx}-{end_idx})...")
        
        # Get the batch of sparse matrices
        batch_sparse = X_target_sparse[start_idx:end_idx]
        
        # Find nearest neighbors for the batch
        distances, indices = nn_model.kneighbors(batch_sparse)
        
        # Compute PCA vectors by averaging neighbors
        for i, neighbor_indices in enumerate(indices):
            # Get the PCA vectors of the nearest neighbors
            neighbor_pca_vectors = [train_pca_vectors[idx] for idx in neighbor_indices 
                                  if train_pca_vectors[idx] is not None]
            
            if neighbor_pca_vectors:
                # Average the PCA vectors
                avg_pca_vector = np.mean(neighbor_pca_vectors, axis=0)
                target_pca_vectors.append(json.dumps(avg_pca_vector.tolist()))
            else:
                # No valid neighbors found
                target_pca_vectors.append(None)
    
    # Add the computed PCA vectors to the target DataFrame
    target_df[pca_col] = target_pca_vectors
    
    # Save the result
    print(f"Saving results to {output_file}...")
    target_df.to_csv(output_file, index=False, sep=delimiter)
    
    # Print summary
    elapsed_time = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"PCA Computation via Neighbors Summary")
    print(f"{'='*50}")
    print(f"Training file: {train_file}")
    print(f"Target file: {target_file}")
    print(f"Output file: {output_file}")
    print(f"Rows processed: {len(target_df)}")
    print(f"Neighbors used per row: {n_neighbors}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"\nComputation complete!")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python compute_pca_via_neighbors.py <train.csv> <target.csv> <output.csv> [--tfidf TFIDF_COL] [--pca PCA_COL] [--neighbors N] [--tab]")
        print("\nThis script computes PCA vectors for validation/test by averaging PCA vectors of nearest neighbors from training set.")
        print("Parameters:")
        print("  train.csv: Training CSV file with TF-IDF and PCA vectors")
        print("  target.csv: Target CSV file (validation/test) with TF-IDF vectors")
        print("  output.csv: Output CSV file to save target data with added PCA vectors")
        print("  --tfidf TFIDF_COL: Optional column name containing TF-IDF strings (default: tfidf_vector_str)")
        print("  --pca PCA_COL: Optional column name for PCA vectors (default: pca_vector)")
        print("  --neighbors N: Optional number of neighbors to use (default: 5)")
        print("  --tab: Optional flag for tab-delimited files\n")
        sys.exit(1)
    
    train_file = sys.argv[1]
    target_file = sys.argv[2]
    output_file = sys.argv[3]
    
    # Parse optional arguments
    tfidf_col = 'tfidf_vector_str'  # Default column name
    pca_col = 'pca_vector'  # Default column name
    n_neighbors = 5  # Default number of neighbors
    delimiter = ','  # Default delimiter
    
    i = 4
    while i < len(sys.argv):
        if sys.argv[i] == '--tfidf' and i + 1 < len(sys.argv):
            tfidf_col = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--pca' and i + 1 < len(sys.argv):
            pca_col = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--neighbors' and i + 1 < len(sys.argv):
            try:
                n_neighbors = int(sys.argv[i + 1])
                i += 2
            except ValueError:
                print(f"Error: Invalid number of neighbors: {sys.argv[i + 1]}")
                sys.exit(1)
        elif sys.argv[i] == '--tab':
            delimiter = '\t'
            i += 1
        else:
            print(f"Warning: Unknown argument: {sys.argv[i]}")
            i += 1
    
    # Try to auto-detect the delimiter if not specified
    if delimiter == ',' and not '--tab' in sys.argv:
        try:
            with open(train_file, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if '\t' in first_line and ',' not in first_line:
                    delimiter = '\t'
                    print(f"Auto-detected tab-delimited format")
                elif ',' in first_line:
                    print(f"Using comma as delimiter")
        except Exception as e:
            print(f"Error reading file: {e}")
    
    compute_pca_via_neighbors(
        train_file, target_file, output_file, 
        tfidf_col=tfidf_col, 
        pca_col=pca_col,
        n_neighbors=n_neighbors,
        delimiter=delimiter
    )