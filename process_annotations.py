import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
import shutil

def load_annotation_file(file_path):
    """Load and validate a single annotation file."""
    try:
        # Read the file with header
        df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
        
        # Validate required columns
        required_columns = ['ID', 'Label', 'Text']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"File {file_path} must contain columns: {required_columns}")
        
        # Clean labels and text
        df['Label'] = df['Label'].str.strip().str.replace('"', '')
        df['Text'] = df['Text'].str.strip().str.replace('"', '')
        
        # Convert ID to numeric, handling any non-numeric values
        df['ID'] = pd.to_numeric(df['ID'], errors='coerce')
        
        # Remove any rows with NaN IDs (invalid rows)
        df = df.dropna(subset=['ID'])
        
        # Convert IDs to integers
        df['ID'] = df['ID'].astype(int)
        
        return df
    
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None

def combine_annotations(annotation_dir='annotated'):
    """Combine all annotation files in the specified directory."""
    # Get all annotation files
    annotation_files = glob.glob(os.path.join(annotation_dir, '*.txt'))
    
    if not annotation_files:
        raise ValueError(f"No annotation files found in {annotation_dir}")
    
    # Load and combine all files
    dfs = []
    for file_path in annotation_files:
        df = load_annotation_file(file_path)
        if df is not None:
            dfs.append(df)
    
    if not dfs:
        raise ValueError("No valid annotation files found")
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Remove duplicates by ID, keeping the first occurrence
    combined_df = combined_df.drop_duplicates(subset=['ID'], keep='first')
    
    return combined_df

def create_train_dev_test_split(df, train_size=0.6, dev_size=0.2, test_size=0.2, random_state=42):
    """Create train, dev, and test splits."""
    # First split: separate out training data
    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        random_state=random_state,
        stratify=df['Label']
    )
    
    # Second split: divide remaining data into dev and test
    # Adjust dev_size to account for the reduced dataset
    adjusted_dev_size = dev_size / (1 - train_size)
    dev_df, test_df = train_test_split(
        temp_df,
        train_size=adjusted_dev_size,
        random_state=random_state,
        stratify=temp_df['Label']
    )
    
    return train_df, dev_df, test_df

def save_splits(train_df, dev_df, test_df, total_df, output_dir='splits'):
    """Save train, dev, and test splits to files."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save splits
    train_df.to_csv(os.path.join(output_dir, 'train.txt'), sep='\t', index=False)
    dev_df.to_csv(os.path.join(output_dir, 'dev.txt'), sep='\t', index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.txt'), sep='\t', index=False)
    
    print(f"Train split saved to {os.path.join(output_dir, 'train.txt')}")
    print(f"Dev split saved to {os.path.join(output_dir, 'dev.txt')}")
    print(f"Test split saved to {os.path.join(output_dir, 'test.txt')}")
    
    # Print split sizes
    print("\nSplit sizes:")
    print(f"Train: {len(train_df)} examples ({len(train_df)/len(total_df)*100:.1f}%)")
    print(f"Dev: {len(dev_df)} examples ({len(dev_df)/len(total_df)*100:.1f}%)")
    print(f"Test: {len(test_df)} examples ({len(test_df)/len(total_df)*100:.1f}%)")

def main():
    # Combine annotations
    print("Combining annotation files...")
    combined_df = combine_annotations()
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Total number of examples: {len(combined_df)}")
    print("\nLabel distribution:")
    print(combined_df['Label'].value_counts())
    
    # Create train/dev/test splits
    print("\nCreating train/dev/test splits...")
    train_df, dev_df, test_df = create_train_dev_test_split(combined_df)
    
    # Save splits
    save_splits(train_df, dev_df, test_df, combined_df)
    
    print("\nDone!")

if __name__ == "__main__":
    main() 