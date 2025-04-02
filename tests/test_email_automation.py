# This is a Python module
"""
Test script for email automation module
"""
import os
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
sys.path.append(str(project_root))

import pandas as pd
from src.email_automation.email_processor import EmailAutomation
from src.utils.gemini_api import GeminiAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, 'email_test.log'))
    ]
)

logger = logging.getLogger(__name__)

def main():
    """
    Test email automation functionality
    """
    try:
        logger.info("Starting email automation test")
        
        # Initialize GeminiAPI
        gemini_api = GeminiAPI()
        
        # Initialize EmailAutomation
        email_automation = EmailAutomation(gemini_api)
        
        # Load email data
        # Load email data
        data_dir = os.path.join(project_root, 'data')
        # Check if file exists in project data directory, otherwise try other locations
        email_data_path = "/Users/mahmoudahmed/Desktop/Revolve AI/ai_communication_tools/data/email_communication_data.csv"
        if not os.path.exists(email_data_path):
            # Try the current directory
            email_data_path = 'email_communication_data.csv'

                
        if not os.path.exists(email_data_path):
            logger.error(f"Email data file not found at {email_data_path}")
            return
        
        emails_df = pd.read_csv(email_data_path)
        logger.info(f"Loaded {len(emails_df)} emails from {email_data_path}")
        
        # Test email categorization
        logger.info("Testing email categorization")
        sample_email = emails_df.iloc[0]
        subject = sample_email.get('Subject', '')
        body = sample_email.get('Email_Body', '')
        
        category = email_automation.categorize_email(subject, body)
        logger.info(f"Categorized email as: {category}")
        
        # Test email reply generation
        logger.info("Testing email reply generation")
        sender = sample_email.get('Sender', '')
        
        reply = email_automation.generate_reply(subject, body, sender, category)
        logger.info(f"Generated reply: {reply[:100]}...")
        
        # Test sentiment analysis
        logger.info("Testing email sentiment analysis")
        sentiment = email_automation.analyze_email_sentiment(body)
        logger.info(f"Email sentiment: {sentiment}")
        
        # Test batch processing (limited to 3 emails for testing)
        logger.info("Testing batch email processing")
        test_batch = emails_df.head(3)
        processed_batch = email_automation.batch_process_emails(test_batch)
        
        # Save test results
        results_dir = os.path.join(project_root, 'tests', 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        results_path = os.path.join(results_dir, 'email_test_results.csv')
        processed_batch.to_csv(results_path, index=False)
        
        logger.info(f"Email test results saved to {results_path}")
        logger.info("Email automation test completed successfully")
        
    except Exception as e:
        logger.error(f"Error in email automation test: {str(e)}")
        raise

if __name__ == "__main__":
    main()
