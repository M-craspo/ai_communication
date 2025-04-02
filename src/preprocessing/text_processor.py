# This is a Python module
"""
Text preprocessing utilities for NLP tasks
"""
import re
import nltk
import logging
from typing import List, Dict, Any, Optional
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download necessary NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logging.error(f"Error downloading NLTK resources: {str(e)}")

# Configure logger
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """
    Class for text preprocessing tasks
    """
    def __init__(self):
        """Initialize the text preprocessor"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # If model not found, download it
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        self.stop_words = set(stopwords.words('english'))
        logger.info("TextPreprocessor initialized successfully")
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing special characters and extra whitespace
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        if not text:
            return []
            
        # Clean text before tokenization
        cleaned_text = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(cleaned_text)
        
        return tokens
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from list of tokens
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of tokens with stopwords removed
        """
        if not tokens:
            return []
            
        filtered_tokens = [token for token in tokens if token not in self.stop_words]
        
        return filtered_tokens
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text
        
        Args:
            text: Input text
            
        Returns:
            List of dictionaries containing entity text, label, and position
        """
        if not text:
            return []
            
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        return entities
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Simple rule-based sentiment analysis
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment score and label
        """
        if not text:
            return {'score': 0, 'label': 'neutral'}
            
        # This is a simplified version - in a real implementation,
        # we would use a more sophisticated sentiment analysis model
        positive_words = {'good', 'great', 'excellent', 'positive', 'happy', 'satisfied'}
        negative_words = {'bad', 'poor', 'negative', 'unhappy', 'dissatisfied'}
        
        tokens = self.tokenize(text)
        
        positive_count = sum(1 for token in tokens if token in positive_words)
        negative_count = sum(1 for token in tokens if token in negative_words)
        
        score = (positive_count - negative_count) / max(len(tokens), 1)
        
        if score > 0.05:
            label = 'positive'
        elif score < -0.05:
            label = 'negative'
        else:
            label = 'neutral'
        
        return {
            'score': score,
            'label': label
        }
    
    def preprocess_pipeline(self, text: str, remove_stops: bool = True) -> Dict[str, Any]:
        """
        Run complete preprocessing pipeline on text
        
        Args:
            text: Input text
            remove_stops: Whether to remove stopwords
            
        Returns:
            Dictionary with preprocessing results
        """
        if not text:
            return {
                'original': '',
                'cleaned': '',
                'tokens': [],
                'entities': [],
                'sentiment': {'score': 0, 'label': 'neutral'}
            }
            
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(cleaned_text)
        
        # Remove stopwords if requested
        if remove_stops:
            tokens = self.remove_stopwords(tokens)
        
        # Extract entities
        entities = self.extract_entities(text)
        
        # Analyze sentiment
        sentiment = self.analyze_sentiment(text)
        
        return {
            'original': text,
            'cleaned': cleaned_text,
            'tokens': tokens,
            'entities': entities,
            'sentiment': sentiment
        }
