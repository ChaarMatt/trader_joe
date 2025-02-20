"""
Text processing utilities for Trader Joe.
"""

import re
from typing import Optional, List, Dict, Any
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    try:
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z.!? ]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    except Exception as e:
        print(f"Error cleaning text: {e}")
        return text

def tokenize_text(text: str) -> List[str]:
    """Tokenize text into words."""
    try:
        tokens = nltk.word_tokenize(text)
        tokens = [token for token in tokens if token.isalpha()]
        return tokens
    except Exception as e:
        print(f"Error tokenizing text: {e}")
        return []

def remove_stopwords(tokens: List[str]) -> List[str]:
    """Remove stopwords from tokenized text."""
    try:
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [token for token in tokens if token not in stop_words]
        return filtered_tokens
    except Exception as e:
        print(f"Error removing stopwords: {e}")
        return tokens

def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """Lemmatize tokens to base form."""
    try:
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return lemmatized_tokens
    except Exception as e:
        print(f"Error lemmatizing tokens: {e}")
        return tokens

if __name__ == '__main__':
    # Example usage
    text = "This is an example sentence with special characters like @, #, and numbers 123."
    print("Original text:", text)
    print("Cleaned text:", clean_text(text))
    print("Tokenized text:", tokenize_text(clean_text(text)))
    print("Filtered tokens:", remove_stopwords(tokenize_text(clean_text(text))))
    print("Lemmatized tokens:", lemmatize_tokens(remove_stopwords(tokenize_text(clean_text(text)))))
