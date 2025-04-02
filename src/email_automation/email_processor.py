# This is a Python module
"""
Email automation module for AI-powered communication tools
"""
import os
import logging
from typing import Dict, List, Any, Optional
import pandas as pd

from src.utils.gemini_api import GeminiAPI
from src.preprocessing.text_processor import TextPreprocessor
from src.config.config import EMAIL_CATEGORIES

# Configure logger
logger = logging.getLogger(__name__)

class EmailAutomation:
    """
    Class for email automation using Gemini model
    """
    def __init__(self, gemini_api: Optional[GeminiAPI] = None):
        """
        Initialize the email automation module
        
        Args:
            gemini_api: Optional GeminiAPI instance
        """
        self.gemini_api = gemini_api or GeminiAPI()
        self.text_processor = TextPreprocessor()
        logger.info("EmailAutomation initialized")
    
    def categorize_email(self, email_subject: str, email_body: str) -> str:
        """
        Categorize email based on subject and body
        
        Args:
            email_subject: Email subject
            email_body: Email body
            
        Returns:
            Category of the email
        """
        try:
            prompt = f"""
            Categorize the following email into one of these categories: {', '.join(EMAIL_CATEGORIES)}
            
            Subject: {email_subject}
            Body: {email_body}
            
            Return only the category name.
            """
            
            category = self.gemini_api.generate_text(prompt)
            # Clean up response to ensure it matches one of our categories
            category = category.strip()
            
            # Default to first category if not recognized
            if category not in EMAIL_CATEGORIES:
                logger.warning(f"Category '{category}' not recognized, defaulting to {EMAIL_CATEGORIES[0]}")
                category = EMAIL_CATEGORIES[0]
                
            return category
        except Exception as e:
            logger.error(f"Error categorizing email: {str(e)}")
            return EMAIL_CATEGORIES[0]  # Default to first category on error
    
    def generate_reply(self, email_subject: str, email_body: str, sender: str, 
                      category: Optional[str] = None) -> str:
        """
        Generate email reply based on subject, body, and sender
        
        Args:
            email_subject: Email subject
            email_body: Email body
            sender: Email sender
            category: Optional email category
            
        Returns:
            Generated email reply
        """
        try:
            # Categorize email if category not provided
            if not category:
                category = self.categorize_email(email_subject, email_body)
            
            # Analyze sentiment
            sentiment_result = self.text_processor.analyze_sentiment(email_body)
            sentiment = sentiment_result.get('label', 'neutral')
            
            # Extract entities
            entities = self.text_processor.extract_entities(email_body)
            entity_text = ", ".join([f"{e['text']} ({e['label']})" for e in entities[:5]])
            
            prompt = f"""
            Generate a professional email reply to the following email:
            
            From: {sender}
            Subject: {email_subject}
            Body: {email_body}
            
            Additional information:
            - Email category: {category}
            - Sentiment: {sentiment}
            - Key entities: {entity_text}
            
            The reply should be professional, concise, and address the specific points in the email.
            If the email is a question, provide a helpful answer.
            If the email is a request, acknowledge it and provide next steps.
            """
            
            reply = self.gemini_api.generate_text(prompt)
            return reply
        except Exception as e:
            logger.error(f"Error generating email reply: {str(e)}")
            return f"Error generating reply: {str(e)}"
    
    def suggest_follow_up(self, email_thread: List[Dict[str, str]]) -> str:
        """
        Suggest follow-up email based on email thread
        
        Args:
            email_thread: List of emails in thread with sender, subject, and body
            
        Returns:
            Suggested follow-up email
        """
        try:
            thread_text = ""
            for i, email in enumerate(email_thread):
                thread_text += f"Email {i+1}:\n"
                thread_text += f"From: {email.get('sender', 'Unknown')}\n"
                thread_text += f"Subject: {email.get('subject', 'No Subject')}\n"
                thread_text += f"Body: {email.get('body', '')}\n\n"
            
            prompt = f"""
            Based on the following email thread, suggest a follow-up email:
            
            {thread_text}
            
            The follow-up should be professional, concise, and move the conversation forward.
            """
            
            follow_up = self.gemini_api.generate_text(prompt)
            return follow_up
        except Exception as e:
            logger.error(f"Error suggesting follow-up: {str(e)}")
            return f"Error suggesting follow-up: {str(e)}"
    
    def analyze_email_sentiment(self, email_body: str) -> Dict[str, Any]:
        """
        Analyze sentiment of email body
        
        Args:
            email_body: Email body text
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            return self.text_processor.analyze_sentiment(email_body)
        except Exception as e:
            logger.error(f"Error analyzing email sentiment: {str(e)}")
            return {"error": str(e)}
    
    def batch_process_emails(self, emails_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a batch of emails from DataFrame
        
        Args:
            emails_df: DataFrame with email data
            
        Returns:
            DataFrame with added columns for category and reply
        """
        try:
            # Create copy to avoid modifying original
            result_df = emails_df.copy()
            
            # Add columns for category and reply
            result_df['Predicted_Category'] = None
            result_df['Generated_Reply'] = None
            
            for idx, row in result_df.iterrows():
                subject = row.get('Subject', '')
                body = row.get('Email_Body', '')
                sender = row.get('Sender', '')
                
                # Categorize email
                category = self.categorize_email(subject, body)
                result_df.at[idx, 'Predicted_Category'] = category
                
                # Generate reply
                reply = self.generate_reply(subject, body, sender, category)
                result_df.at[idx, 'Generated_Reply'] = reply
                
                logger.info(f"Processed email {idx+1}/{len(result_df)}")
            
            return result_df
        except Exception as e:
            logger.error(f"Error batch processing emails: {str(e)}")
            raise
