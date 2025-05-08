"""
Exploratory Data Analysis - My Zoom Feedback Validation

This script performs exploratory data analysis on the train and evaluation datasets
for the Zoom feedback validation project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Set style for plots
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Define paths to the Excel files
project_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(project_dir, 'train.xlsx')
eval_path = os.path.join(project_dir, 'evaluation.xlsx')

print(f"Loading data from:\n- {train_path}\n- {eval_path}")

# Load the data
train_df = pd.read_excel(train_path)
eval_df = pd.read_excel(eval_path)

print(f"Train data shape: {train_df.shape}")
print(f"Evaluation data shape: {eval_df.shape}")

# Display the first few rows of training data
print("\nFirst few rows of training data:")
print(train_df.head())

# Display the first few rows of evaluation data
print("\nFirst few rows of evaluation data:")
print(eval_df.head())

# Check for missing values
print("\nMissing values in train data:")
print(train_df.isnull().sum())
print("\nMissing values in evaluation data:")
print(eval_df.isnull().sum())

# Analyze class distribution in training data
train_class_counts = train_df['label'].value_counts()
train_class_distribution = train_class_counts / len(train_df) * 100

print("\nClass counts in training data:")
print(train_class_counts)
print("\nClass distribution in training data (%):")
print(train_class_distribution)

# Analyze class distribution in evaluation data
eval_class_counts = eval_df['label'].value_counts()
eval_class_distribution = eval_class_counts / len(eval_df) * 100

print("\nClass counts in evaluation data:")
print(eval_class_counts)
print("\nClass distribution in evaluation data (%):")
print(eval_class_distribution)

# Visualize class distribution
plt.figure(figsize=(12, 5))

# Training data
plt.subplot(1, 2, 1)
train_class_counts.plot(kind='bar', color=['#ff9999', '#66b3ff'])
plt.title('Class Distribution in Training Data')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Invalid (0)', 'Valid (1)'], rotation=0)
for i, count in enumerate(train_class_counts):
    plt.text(i, count + 5, str(count), ha='center')

# Evaluation data
plt.subplot(1, 2, 2)
eval_class_counts.plot(kind='bar', color=['#ff9999', '#66b3ff'])
plt.title('Class Distribution in Evaluation Data')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Invalid (0)', 'Valid (1)'], rotation=0)
for i, count in enumerate(eval_class_counts):
    plt.text(i, count + 50, str(count), ha='center')

plt.tight_layout()
plt.savefig(os.path.join(project_dir, 'class_distribution.png'))
print("\nSaved class distribution plot to 'class_distribution.png'")

# Add text length to dataframes
train_df['text_length'] = train_df['text'].apply(lambda x: len(str(x)))
eval_df['text_length'] = eval_df['text'].apply(lambda x: len(str(x)))

# Add word count to dataframes
train_df['word_count'] = train_df['text'].apply(lambda x: len(str(x).split()))
eval_df['word_count'] = eval_df['text'].apply(lambda x: len(str(x).split()))

# Display descriptive statistics for text length and word count
print("\nTraining data text length statistics:")
print(train_df['text_length'].describe())
print("\nTraining data word count statistics:")
print(train_df['word_count'].describe())

print("\nEvaluation data text length statistics:")
print(eval_df['text_length'].describe())
print("\nEvaluation data word count statistics:")
print(eval_df['word_count'].describe())

# Function to clean text
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply text cleaning
train_df['cleaned_text'] = train_df['text'].apply(clean_text)
eval_df['cleaned_text'] = eval_df['text'].apply(clean_text)

# Function to get most common words
def get_common_words(texts, top_n=20, exclude_stopwords=True):
    # Combine all texts
    all_text = ' '.join(texts)
    
    # Tokenize
    tokens = word_tokenize(all_text)
    
    # Remove stopwords if requested
    if exclude_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
    
    # Count word frequencies
    word_counts = Counter(tokens)
    
    # Get top N words
    top_words = word_counts.most_common(top_n)
    
    return top_words

# Get common words for valid and invalid feedback in training data
valid_texts = train_df[train_df['label'] == 1]['cleaned_text'].tolist()
invalid_texts = train_df[train_df['label'] == 0]['cleaned_text'].tolist()

valid_common_words = get_common_words(valid_texts)
invalid_common_words = get_common_words(invalid_texts)

print("\nMost common words in valid feedback:")
for word, count in valid_common_words:
    print(f"  {word}: {count}")

print("\nMost common words in invalid feedback:")
for word, count in invalid_common_words:
    print(f"  {word}: {count}")

# Plot common words
plt.figure(figsize=(15, 10))

# Valid feedback
plt.subplot(2, 1, 1)
words, counts = zip(*valid_common_words)
plt.barh(range(len(words)), counts, color='#66b3ff')
plt.yticks(range(len(words)), words)
plt.gca().invert_yaxis()  # Invert y-axis to have most common word at top
plt.title('Most Common Words in Valid Feedback')
plt.xlabel('Count')

# Invalid feedback
plt.subplot(2, 1, 2)
words, counts = zip(*invalid_common_words)
plt.barh(range(len(words)), counts, color='#ff9999')
plt.yticks(range(len(words)), words)
plt.gca().invert_yaxis()  # Invert y-axis to have most common word at top
plt.title('Most Common Words in Invalid Feedback')
plt.xlabel('Count')

plt.tight_layout()
plt.savefig(os.path.join(project_dir, 'common_words.png'))
print("\nSaved common words plot to 'common_words.png'")

# Check if the 'reason' column contains meaningful data
print("\nNumber of non-null reasons in training data:", train_df['reason'].count())
print("Number of non-null reasons in evaluation data:", eval_df['reason'].count())

# Calculate correlation between text length/word count and label
train_corr_length = train_df['text_length'].corr(train_df['label'])
train_corr_words = train_df['word_count'].corr(train_df['label'])

eval_corr_length = eval_df['text_length'].corr(eval_df['label'])
eval_corr_words = eval_df['word_count'].corr(eval_df['label'])

print("\nCorrelation between text length and label (train):", train_corr_length)
print("Correlation between word count and label (train):", train_corr_words)
print("Correlation between text length and label (eval):", eval_corr_length)
print("Correlation between word count and label (eval):", eval_corr_words)

# Summary statistics for labels
print("\nSummary of findings:")
print(f"1. Training data has {len(train_df)} samples, Evaluation data has {len(eval_df)} samples")
print(f"2. Class balance in training: {train_class_distribution[1]:.1f}% valid, {train_class_distribution[0]:.1f}% invalid")
print(f"3. Class balance in evaluation: {eval_class_distribution[1]:.1f}% valid, {eval_class_distribution[0]:.1f}% invalid")
print(f"4. Average text length in training: {train_df['text_length'].mean():.1f} characters")
print(f"5. Average word count in training: {train_df['word_count'].mean():.1f} words")

print("\nAnalysis complete. Results saved to the project directory.")
