import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

def load_data(filename):
    """Load and validate data from a TSV file."""
    # Define valid labels
    valid_labels = {'Guilt-tripping', 'Exaggerated Claims', 'Scarcity', 'Time Pressure'}
    
    try:
        # Read the TSV file
        df = pd.read_csv(filename, sep='\t', encoding='utf-8')
        
        # Validate required columns
        required_columns = ['ID', 'Label', 'Text']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"File must contain columns: {required_columns}")
        
        # Clean labels
        df['Label'] = df['Label'].str.strip().str.replace('"', '')
        
        # Validate labels
        invalid_labels = set(df['Label'].unique()) - valid_labels
        if invalid_labels:
            raise ValueError(f"Invalid labels found: {invalid_labels}")
        
        # Convert ID to numeric
        df['ID'] = pd.to_numeric(df['ID'], errors='coerce')
        if df['ID'].isna().any():
            raise ValueError("Invalid ID values found")
        
        return df
    
    except Exception as e:
        raise ValueError(f"Error loading {filename}: {str(e)}")

def extract_features(text):
    """Extract features from text."""
    features = {}
    
    # Text length features
    features['char_count'] = len(text)
    features['word_count'] = len(text.split())
    
    # Punctuation features
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    
    # Word features
    words = text.lower().split()
    features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
    
    return features

class EnhancedLogisticRegression:
    def __init__(self):
        self.pipeline = None
        self.label_encoder = LabelEncoder()
        self.original_labels = None  # Store original string labels
        
    def fit(self, train_df):
        # Create feature matrices
        train_features = pd.DataFrame([extract_features(text) for text in train_df['Text']])
        
        # Create pipeline with both text and numeric features
        text_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=5000,
                stop_words='english'
            ))
        ])
        
        preprocessor = ColumnTransformer([
            ('text', text_pipeline, 'Text'),
            ('numeric', StandardScaler(), train_features.columns)
        ])
        
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            ))
        ])
        
        # Store original labels before encoding
        self.original_labels = sorted(train_df['Label'].astype(str).unique())
        
        # Fit the model
        self.pipeline.fit(
            pd.concat([train_df[['Text']], train_features], axis=1),
            self.label_encoder.fit_transform(train_df['Label'])
        )
        
    def predict(self, test_df):
        test_features = pd.DataFrame([extract_features(text) for text in test_df['Text']])
        return self.pipeline.predict(
            pd.concat([test_df[['Text']], test_features], axis=1)
        )
    
    def evaluate(self, test_df, results_dir='results'):
        y_pred = self.predict(test_df)
        y_true = self.label_encoder.transform(test_df['Label'])
        
        accuracy = accuracy_score(y_true, y_pred)
        n = len(test_df)
        z = stats.norm.ppf(0.975)
        ci = z * np.sqrt((accuracy * (1 - accuracy)) / n)
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Save results to file
        with open(os.path.join(results_dir, 'logistic_regression_results.txt'), 'w') as f:
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"95% Confidence Interval: [{accuracy - ci:.4f}, {accuracy + ci:.4f}]\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(y_true, y_pred, target_names=self.original_labels))
        
        # Print results to console
        print(f"Accuracy: {accuracy:.4f}")
        print(f"95% Confidence Interval: [{accuracy - ci:.4f}, {accuracy + ci:.4f}]")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.original_labels))
        
        # Plot and save confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', 
                    xticklabels=self.original_labels, yticklabels=self.original_labels)
        plt.title('Confusion Matrix - Enhanced Logistic Regression')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
        plt.close()

def run_model(train_file, test_file, results_dir='results'):
    """Run enhanced logistic regression on the given data files."""
    # Load data
    train_df = load_data(train_file)
    test_df = load_data(test_file)
    
    # Run Enhanced Logistic Regression
    print("\n=== Enhanced Logistic Regression ===")
    lr_model = EnhancedLogisticRegression()
    lr_model.fit(train_df)
    lr_model.evaluate(test_df, results_dir)

if __name__ == "__main__":
    run_model("splits/train.txt", "splits/test.txt") 