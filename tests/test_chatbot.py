# This is a Python module
"""
Test script for chatbot module
"""
import os
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
sys.path.append(str(project_root))

import pandas as pd
from src.chatbot.chatbot_processor import ChatbotProcessor
from src.utils.gemini_api import GeminiAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, 'chatbot_test.log'))
    ]
)

logger = logging.getLogger(__name__)

def main():
    """
    Test chatbot functionality
    """
    try:
        logger.info("Starting chatbot test")
        
        # Initialize GeminiAPI
        gemini_api = GeminiAPI()
        
        # Initialize ChatbotProcessor
        chatbot = ChatbotProcessor(gemini_api)
        
        # Load chatbot data
        data_dir = os.path.join(project_root, 'data')
        # Check if file exists in project data directory, otherwise use upload directory
        chatbot_data_path = os.path.join(data_dir, 'chatbot_conversation_data.csv')
        if not os.path.exists(chatbot_data_path):
            chatbot_data_path = '/home/ubuntu/upload/chatbot_conversation_data.csv'
        
        if not os.path.exists(chatbot_data_path):
            logger.error(f"Chatbot data file not found at {chatbot_data_path}")
            return
        
        conversations_df = pd.read_csv(chatbot_data_path)
        logger.info(f"Loaded {len(conversations_df)} conversations from {chatbot_data_path}")
        
        # Test intent detection
        logger.info("Testing intent detection")
        sample_input = conversations_df.iloc[0]['User_Input']
        
        intent_info = chatbot.detect_intent(sample_input)
        logger.info(f"Detected intent: {intent_info['intent']} (confidence: {intent_info['confidence']})")
        
        # Test response generation
        logger.info("Testing response generation")
        response = chatbot.generate_response(sample_input)
        logger.info(f"Generated response: {response[:100]}...")
        
        # Test conversation history
        logger.info("Testing conversation history")
        history = chatbot.get_history()
        logger.info(f"Conversation history has {len(history)} messages")
        
        # Test business inquiry handling
        logger.info("Testing business inquiry handling")
        business_inquiry = "What AI tools do you offer for email automation?"
        business_context = {
            "company": "AI Communication Tools Inc.",
            "product_line": "Email automation, chatbots, report generation",
            "pricing_tier": "Enterprise"
        }
        
        inquiry_response = chatbot.handle_business_inquiry(business_inquiry, business_context)
        logger.info(f"Business inquiry response: {inquiry_response[:100]}...")
        
        # Test batch processing (limited to 3 conversations for testing)
        logger.info("Testing batch conversation processing")
        test_batch = conversations_df.head(3)
        processed_batch = chatbot.batch_process_conversations(test_batch)
        
        # Save test results
        results_dir = os.path.join(project_root, 'tests', 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        results_path = os.path.join(results_dir, 'chatbot_test_results.csv')
        processed_batch.to_csv(results_path, index=False)
        
        logger.info(f"Chatbot test results saved to {results_path}")
        logger.info("Chatbot test completed successfully")
        
    except Exception as e:
        logger.error(f"Error in chatbot test: {str(e)}")
        raise

if __name__ == "__main__":
    main()
