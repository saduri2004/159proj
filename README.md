# Persuasion Technique Classification

This project classifies text into different persuasion techniques using machine learning. The current implementation uses an enhanced logistic regression model with text and numeric features.

## Project Structure

```
.
├── annotated/              # Raw annotation files
│   ├── annotated-jenny.txt
│   ├── annotated-richard.txt
│   └── annotation-sasank.txt
├── splits/                # Processed train/dev/test splits
│   ├── train.txt
│   ├── dev.txt
│   └── test.txt
├── results/              # Model evaluation results
│   ├── logistic_regression_results.txt
│   └── confusion_matrix.png
├── enhanced_models.py    # Main model implementation
├── process_annotations.py # Script to process annotations
└── README.md            # This file
```

## Setup Instructions

1. Create and activate a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

If requirements.txt doesn't exist, install packages manually:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Usage

1. Process annotation files and create splits:

```bash
python process_annotations.py
```

This will:
- Combine all annotation files from the `annotated` directory
- Create train (60%), dev (20%), and test (20%) splits
- Save splits to the `splits` directory

2. Train and evaluate the model:

```bash
python enhanced_models.py
```

This will:
- Train the enhanced logistic regression model
- Evaluate on the test set
- Save results to the `results` directory

## Model Details

The enhanced logistic regression model uses:
- TF-IDF features for text
- Additional numeric features:
  - Character count
  - Word count
  - Punctuation counts
  - Average word length

## Label Categories

The model classifies text into four persuasion techniques:
1. Guilt-tripping
2. Exaggerated Claims
3. Scarcity
4. Time Pressure

## Results

Model evaluation results are saved in the `results` directory:
- `logistic_regression_results.txt`: Contains accuracy, confidence intervals, and classification report
- `confusion_matrix.png`: Visualization of model performance across classes 