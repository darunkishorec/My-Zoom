"""
Data preprocessing module for the My Zoom project.
This module handles loading, cleaning, preprocessing, and augmenting the feedback data.
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class DataPreprocessor:
    def __init__(self, max_length=128, tokenizer_name='bert-base-uncased'):
        """
        Initialize the data preprocessor.
        
        Args:
            max_length (int): Maximum sequence length for tokenization
            tokenizer_name (str): Name of the pretrained tokenizer to use
        """
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.stop_words = set(stopwords.words('english'))
        
    def load_data(self, train_path, eval_path=None):
        """
        Load the training and evaluation data.
        
        Args:
            train_path (str): Path to the training data
            eval_path (str): Path to the evaluation data (optional)
            
        Returns:
            tuple: Training dataframe and evaluation dataframe (if provided)
        """
        train_df = pd.read_excel(train_path)
        
        if eval_path:
            eval_df = pd.read_excel(eval_path)
            return train_df, eval_df
        
        return train_df
    
    def clean_text(self, text):
        """
        Clean and preprocess text data.
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove special characters and numbers (keep only letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, text):
        """
        Remove stopwords from text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with stopwords removed
        """
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word not in self.stop_words]
        return ' '.join(filtered_text)
    
    def preprocess_data(self, df, clean=True, remove_stop=False):
        """
        Preprocess the dataframe.
        
        Args:
            df (DataFrame): Input dataframe
            clean (bool): Whether to clean the text
            remove_stop (bool): Whether to remove stopwords
            
        Returns:
            DataFrame: Preprocessed dataframe
        """
        df = df.copy()
        
        # Clean text
        if clean:
            df['cleaned_text'] = df['text'].apply(self.clean_text)
        else:
            df['cleaned_text'] = df['text']
        
        # Remove stopwords if requested
        if remove_stop:
            df['cleaned_text'] = df['cleaned_text'].apply(self.remove_stopwords)
        
        return df
    
    def tokenize_data(self, texts, labels=None):
        """
        Tokenize the text data for BERT.
        
        Args:
            texts (list): List of text strings
            labels (list): List of labels (optional)
            
        Returns:
            dict: Tokenized data
        """
        encoding = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        if labels is not None:
            encoding['labels'] = labels
            
        return encoding
    
    def split_train_val(self, df, test_size=0.2, random_state=42):
        """
        Split dataframe into training and validation sets.
        
        Args:
            df (DataFrame): Input dataframe
            test_size (float): Proportion of data to use for validation
            random_state (int): Random seed
            
        Returns:
            tuple: Training and validation dataframes
        """
        train_df, val_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=df['label']
        )
        
        return train_df, val_df
    
    def augment_data(self, df, minority_class=0, augmentation_factor=1):
        """
        Augment data to handle class imbalance.
        
        Args:
            df (DataFrame): Input dataframe
            minority_class (int): The minority class to augment
            augmentation_factor (int): Factor by which to augment
            
        Returns:
            DataFrame: Augmented dataframe
        """
        df_minority = df[df['label'] == minority_class].copy()
        df_augmented = pd.concat([df, pd.concat([df_minority] * augmentation_factor, ignore_index=True)])
        
        return df_augmented.sample(frac=1).reset_index(drop=True)  # Shuffle
    
    def prepare_data_for_training(self, train_path, eval_path=None, clean=True, 
                                 remove_stop=False, augment=False, test_size=0.2):
        """
        Complete pipeline to prepare data for training.
        
        Args:
            train_path (str): Path to training data
            eval_path (str): Path to evaluation data (optional)
            clean (bool): Whether to clean the text
            remove_stop (bool): Whether to remove stopwords
            augment (bool): Whether to augment minority class
            test_size (float): Proportion of data for validation (if eval_path not provided)
            
        Returns:
            tuple: Processed data ready for training
        """
        # Load data
        if eval_path:
            train_df, eval_df = self.load_data(train_path, eval_path)
        else:
            train_df = self.load_data(train_path)
            train_df, eval_df = self.split_train_val(train_df, test_size)
        
        # Preprocess data
        train_df = self.preprocess_data(train_df, clean, remove_stop)
        eval_df = self.preprocess_data(eval_df, clean, remove_stop)
        
        # Augment training data if requested
        if augment:
            class_counts = train_df['label'].value_counts()
            minority_class = class_counts.idxmin()
            augmentation_factor = (class_counts.max() // class_counts.min()) - 1
            if augmentation_factor > 0:
                train_df = self.augment_data(train_df, minority_class, augmentation_factor)
        
        # Prepare data for BERT
        train_encodings = self.tokenize_data(
            train_df['cleaned_text'].tolist(), 
            train_df['label'].tolist()
        )
        
        eval_encodings = self.tokenize_data(
            eval_df['cleaned_text'].tolist(), 
            eval_df['label'].tolist()
        )
        
        return train_encodings, eval_encodings, train_df, eval_df

# Example usage
if __name__ == "__main__":
    import os
    
    # Get the project directory
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define data paths
    train_path = os.path.join(project_dir, "train.xlsx")
    eval_path = os.path.join(project_dir, "evaluation.xlsx")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Process the data
    train_encodings, eval_encodings, train_df, eval_df = preprocessor.prepare_data_for_training(
        train_path, eval_path, clean=True, remove_stop=False, augment=True
    )
    
    # Print some stats
    print(f"Training data shape: {train_df.shape}")
    print(f"Evaluation data shape: {eval_df.shape}")
    print(f"Class distribution in training data: {train_df['label'].value_counts().to_dict()}")
    print(f"Class distribution in evaluation data: {eval_df['label'].value_counts().to_dict()}")
