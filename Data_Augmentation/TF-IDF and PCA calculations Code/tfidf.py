import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

# Function to preprocess text
def preprocess_text(text):
    if pd.isna(text):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters but keep spaces between sections
    text = re.sub(r'inclusion criteria:|exclusion criteria:', ' ', text)
    # Remove bullet points and common formatting
    text = re.sub(r'[-•\*]\s+', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Load the dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {df.columns.tolist()}")
    return df

# Main function to generate TF-IDF vectors
def generate_tfidf_vectors(df):
    """
    Generate TF-IDF vectors for clinical trial criteria
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing clinical trial data with a 'criteria' column
        
    Returns:
    --------
    dict
        Dictionary containing the TF-IDF vectorizer, the sparse TF-IDF matrix,
        and the DataFrame with TF-IDF features added
    """
    # Preprocess the criteria text
    print("Preprocessing criteria text...")
    df['preprocessed_criteria'] = df['criteria'].apply(preprocess_text)
    
    # Generate TF-IDF for the dataset
    print("Generating TF-IDF matrix...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,      # Limit features to prevent memory issues
        stop_words='english',
        ngram_range=(1, 2),     # Include unigrams and bigrams
        min_df=2,               # Minimum document frequency
        max_df=0.9              # Maximum document frequency (to filter out very common terms)
    )
    
    # Fit and transform on the entire dataset
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['preprocessed_criteria'])
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Number of features: {len(tfidf_vectorizer.get_feature_names_out())}")
    
    # Store the feature names for reference
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # We'll create a version of the DataFrame that includes some sample TF-IDF features
    # (not all because there could be thousands)
    sample_features = min(10, len(feature_names))
    tfidf_df = df.copy()
    
    # Add a column that stores the sparse vector for each document
    tfidf_df['tfidf_vector'] = [vec for vec in tfidf_matrix]
    
    # Add a few sample TF-IDF features to the DataFrame for inspection
    for i in range(sample_features):
        feature_name = feature_names[i]
        column_name = f'tfidf_{feature_name}'
        tfidf_df[column_name] = tfidf_matrix.getcol(i).toarray().flatten()
    
    # Add top features for each document
    print("Extracting top TF-IDF terms for each document...")
    
    def get_top_features(vector, feature_names, top_n=5):
        # For a given sparse vector, return the top N features with their scores
        indices = vector.nonzero()[1]
        scores = zip(indices, [vector[0, x] for x in indices])
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        top_feats = [(feature_names[idx], score) for idx, score in scores[:top_n]]
        return top_feats
    
    tfidf_df['top_tfidf_terms'] = [
        get_top_features(tfidf_matrix[i], feature_names) 
        for i in range(tfidf_matrix.shape[0])
    ]
    
    return {
        'tfidf_vectorizer': tfidf_vectorizer,
        'tfidf_matrix': tfidf_matrix,
        'feature_names': feature_names,
        'tfidf_df': tfidf_df
    }

# Function to generate TF-IDF vector for a new clinical trial
def generate_tfidf_for_new_trial(new_criteria_text, tfidf_vectorizer):
    """
    Generate a TF-IDF vector for new clinical trial criteria text
    
    Parameters:
    -----------
    new_criteria_text : str
        The criteria text for a new clinical trial
    tfidf_vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer
        
    Returns:
    --------
    scipy.sparse.csr_matrix
        TF-IDF vector for the new clinical trial
    """
    # Preprocess the new text
    preprocessed_text = preprocess_text(new_criteria_text)
    
    # Transform to TF-IDF vector using fitted vectorizer
    new_tfidf = tfidf_vectorizer.transform([preprocessed_text])
    
    return new_tfidf

# Example usage
if __name__ == "__main__":
    # Set file path
    file_path = "total.csv"  # Update this to your file path
    
    # Load data
    df = load_data(file_path)
    
    # Generate TF-IDF vectors
    results = generate_tfidf_vectors(df)
    
    print(f"Generated TF-IDF vectors for {len(results['tfidf_df'])} clinical trials")
    print(f"TF-IDF matrix dimensions: {results['tfidf_matrix'].shape}")
    print(f"Number of features in TF-IDF vectors: {results['tfidf_matrix'].shape[1]}")
    
    # Convert each sparse vector to a string representation and save in the DataFrame
    print("Converting sparse TF-IDF vectors to string format for CSV storage...")
    
    def serialize_sparse_vector(sparse_vec):
        # Convert sparse vector to string representation: "idx1:val1,idx2:val2,..."
        indices = sparse_vec.indices
        data = sparse_vec.data
        return ','.join([f"{idx}:{val}" for idx, val in zip(indices, data)])
    
    # Replace the sparse matrix objects with their string representations
    tfidf_string_vectors = []
    for i in range(results['tfidf_matrix'].shape[0]):
        vec = results['tfidf_matrix'][i]
        tfidf_string_vectors.append(serialize_sparse_vector(vec))
    
    # Add the serialized vectors to the DataFrame
    results['tfidf_df']['tfidf_vector_str'] = tfidf_string_vectors
    
    # Save the results including the serialized TF-IDF vectors
    results['tfidf_df'].drop('tfidf_vector', axis=1).to_csv('clinical_trials_with_tfidf.csv', index=False)
    
    # Also save the full TF-IDF matrix as a sparse matrix file for efficient future use
    sp.save_npz('tfidf_matrix.npz', results['tfidf_matrix'])
    
    print("Process completed successfully!")
    
    # Example: Generate TF-IDF for a new clinical trial
    print("\nExample: Generating TF-IDF vector for a new clinical trial")
    new_criteria = """
    Inclusion Criteria:
    - Adults aged 18-65 years
    - Diagnosed with Type 2 Diabetes
    - HbA1c between 7.0% and 10.0%
    
    Exclusion Criteria:
    - Pregnant or breastfeeding women
    - History of severe hypoglycemia
    - Current use of insulin therapy
    """
    
    new_vector = generate_tfidf_for_new_trial(new_criteria, results['tfidf_vectorizer'])
    print(f"Generated TF-IDF vector with shape: {new_vector.shape}")
    
   