# This is a Python module
"""
Main script to run the data processing pipeline
"""
import os
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
sys.path.append(str(project_root))

from src.preprocessing.data_processor import DataProcessor
from src.preprocessing.text_processor import TextPreprocessor
from src.utils.gemini_api import GeminiAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, 'data_processing.log'))
    ]
)

logger = logging.getLogger(__name__)

def main():
    """
    Run the data processing pipeline
    """
    try:
        # Initialize data processor
        data_dir = os.path.join(project_root, 'data')
        os.makedirs(os.path.join(data_dir, 'processed'), exist_ok=True)
        
        # Copy CSV files to data directory if they don't exist there
        source_dir = '/home/ubuntu/upload'
        for filename in ['email_communication_data.csv', 'chatbot_conversation_data.csv', 'business_report_data.csv']:
            source_path = os.path.join(source_dir, filename)
            dest_path = os.path.join(data_dir, filename)
            if not os.path.exists(dest_path) and os.path.exists(source_path):
                logger.info(f"Copying {filename} to data directory")
                with open(source_path, 'r') as src, open(dest_path, 'w') as dst:
                    dst.write(src.read())
        
        logger.info("Initializing data processor")
        data_processor = DataProcessor(data_dir)
        
        # Process all data
        logger.info("Processing all datasets")
        processed_data = data_processor.process_all_data()
        
        # Save processed data
        logger.info("Saving processed data")
        data_processor.save_processed_data()
        
        # Prepare training data
        logger.info("Preparing training data")
        training_data = data_processor.prepare_training_data()
        
        # Initialize text processor
        logger.info("Initializing text processor")
        text_processor = TextPreprocessor()
        
        # Process sample text from each dataset for demonstration
        logger.info("Processing sample texts")
        
        # Sample email processing
        if 'email' in processed_data and not processed_data['email'].empty:
            sample_email = processed_data['email'].iloc[0]['Email_Body']
            email_preprocessing = text_processor.preprocess_pipeline(sample_email)
            logger.info(f"Processed sample email: {email_preprocessing['cleaned'][:50]}...")
            
            # Save sample processed email
            with open(os.path.join(data_dir, 'processed', 'sample_email_processed.txt'), 'w') as f:
                f.write(f"Original: {email_preprocessing['original']}\n")
                f.write(f"Cleaned: {email_preprocessing['cleaned']}\n")
                f.write(f"Tokens: {', '.join(email_preprocessing['tokens'])}\n")
                f.write(f"Entities: {email_preprocessing['entities']}\n")
                f.write(f"Sentiment: {email_preprocessing['sentiment']}\n")
        
        # Sample chatbot processing
        if 'chatbot' in processed_data and not processed_data['chatbot'].empty:
            sample_chat = processed_data['chatbot'].iloc[0]['User_Input']
            chat_preprocessing = text_processor.preprocess_pipeline(sample_chat)
            logger.info(f"Processed sample chat: {chat_preprocessing['cleaned'][:50]}...")
            
            # Save sample processed chat
            with open(os.path.join(data_dir, 'processed', 'sample_chat_processed.txt'), 'w') as f:
                f.write(f"Original: {chat_preprocessing['original']}\n")
                f.write(f"Cleaned: {chat_preprocessing['cleaned']}\n")
                f.write(f"Tokens: {', '.join(chat_preprocessing['tokens'])}\n")
                f.write(f"Entities: {chat_preprocessing['entities']}\n")
                f.write(f"Sentiment: {chat_preprocessing['sentiment']}\n")
        
        # Sample report processing
        if 'report' in processed_data and not processed_data['report'].empty:
            sample_report = processed_data['report'].iloc[0]['Summary']
            report_preprocessing = text_processor.preprocess_pipeline(sample_report)
            logger.info(f"Processed sample report: {report_preprocessing['cleaned'][:50]}...")
            
            # Save sample processed report
            with open(os.path.join(data_dir, 'processed', 'sample_report_processed.txt'), 'w') as f:
                f.write(f"Original: {report_preprocessing['original']}\n")
                f.write(f"Cleaned: {report_preprocessing['cleaned']}\n")
                f.write(f"Tokens: {', '.join(report_preprocessing['tokens'])}\n")
                f.write(f"Entities: {report_preprocessing['entities']}\n")
                f.write(f"Sentiment: {report_preprocessing['sentiment']}\n")
        
        # Initialize Gemini API (commented out since we don't have a real API key)
        # logger.info("Initializing Gemini API")
        # gemini_api = GeminiAPI()
        
        logger.info("Data processing pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in data processing pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
