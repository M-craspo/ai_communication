# This is a Python module
"""
Chatbot module for AI-powered communication tools
"""
import os
import logging
from typing import Dict, List, Any, Optional
import pandas as pd

from src.utils.gemini_api import GeminiAPI
from src.preprocessing.text_processor import TextPreprocessor
from src.config.config import CHATBOT_MAX_HISTORY, CHATBOT_RESPONSE_TYPES

# Configure logger
logger = logging.getLogger(__name__)

class ChatbotProcessor:
    """
    Class for chatbot functionality using Gemini model
    """
    def __init__(self, gemini_api: Optional[GeminiAPI] = None):
        """
        Initialize the chatbot processor
        
        Args:
            gemini_api: Optional GeminiAPI instance
        """
        self.gemini_api = gemini_api or GeminiAPI()
        self.text_processor = TextPreprocessor()
        self.conversation_history = []
        logger.info("ChatbotProcessor initialized")
    
    def detect_intent(self, user_input: str) -> Dict[str, Any]:
        """
        Detect user intent from input
        
        Args:
            user_input: User's input message
            
        Returns:
            Dictionary with intent information
        """
        try:
            prompt = f"""
            Analyze the following user message and identify the primary intent.
            Return a JSON with intent name and confidence score.
            
            User message: {user_input}
            
            Possible intents: greeting, question, request, complaint, feedback, other
            """
            
            # In a real implementation, we would parse the JSON response
            # For now, using a simplified approach
            
            # Process text to help with intent detection
            preprocessed = self.text_processor.preprocess_pipeline(user_input)
            
            # Simple rule-based intent detection as fallback
            intent = "other"
            confidence = 0.6
            
            if any(word in user_input.lower() for word in ['hi', 'hello', 'hey']):
                intent = "greeting"
                confidence = 0.9
            elif '?' in user_input:
                intent = "question"
                confidence = 0.8
            elif any(word in user_input.lower() for word in ['can', 'could', 'please']):
                intent = "request"
                confidence = 0.7
            elif any(word in user_input.lower() for word in ['bad', 'issue', 'problem', 'wrong']):
                intent = "complaint"
                confidence = 0.7
            
            return {
                "intent": intent,
                "confidence": confidence,
                "entities": preprocessed.get('entities', [])
            }
        except Exception as e:
            logger.error(f"Error detecting intent: {str(e)}")
            return {"intent": "other", "confidence": 0.5, "entities": []}
    
    def generate_response(self, user_input: str) -> str:
        """
        Generate chatbot response to user input
        
        Args:
            user_input: User's input message
            
        Returns:
            Generated chatbot response
        """
        try:
            # Add user message to history
            self.conversation_history.append({"role": "user", "content": user_input})
            
            # Trim history if needed
            if len(self.conversation_history) > CHATBOT_MAX_HISTORY:
                self.conversation_history = self.conversation_history[-CHATBOT_MAX_HISTORY:]
            
            # Detect intent
            intent_info = self.detect_intent(user_input)
            intent = intent_info.get("intent", "other")
            
            # Create conversation history text
            history_text = ""
            for msg in self.conversation_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                history_text += f"{role}: {content}\n"
            
            prompt = f"""
            You are an AI assistant for business communication. Respond to the following user message:
            
            Conversation history:
            {history_text}
            
            User intent: {intent}
            
            Your response should be helpful, concise, and professional. If you don't know the answer, 
            acknowledge that and offer to help with something else.
            """
            
            response = self.gemini_api.generate_text(prompt)
            
            # Add response to history
            self.conversation_history.append({"role": "assistant", "content": response})
            
            return response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            error_response = "I apologize, but I'm having trouble processing your request right now. How else can I assist you?"
            self.conversation_history.append({"role": "assistant", "content": error_response})
            return error_response
    
    def clear_history(self) -> None:
        """
        Clear conversation history
        """
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_history(self) -> List[Dict[str, str]]:
        """
        Get conversation history
        
        Returns:
            List of conversation messages
        """
        return self.conversation_history
    
    def handle_business_inquiry(self, inquiry: str, business_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Handle specific business inquiry with additional context
        
        Args:
            inquiry: Business inquiry text
            business_context: Optional business context information
            
        Returns:
            Response to business inquiry
        """
        try:
            context_text = ""
            if business_context:
                context_text = "Business context:\n"
                for key, value in business_context.items():
                    context_text += f"- {key}: {value}\n"
            
            prompt = f"""
            You are a business assistant. Respond to the following business inquiry with accurate information:
            
            Inquiry: {inquiry}
            
            {context_text}
            
            Your response should be professional, informative, and actionable.
            """
            
            response = self.gemini_api.generate_text(prompt)
            return response
        except Exception as e:
            logger.error(f"Error handling business inquiry: {str(e)}")
            return "I apologize, but I'm having trouble processing your business inquiry right now. Please try again later or contact our support team for assistance."
    
    def analyze_conversation(self, conversation: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Analyze conversation for insights
        
        Args:
            conversation: List of conversation messages
            
        Returns:
            Dictionary with conversation analysis
        """
        try:
            conversation_text = ""
            for msg in conversation:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                conversation_text += f"{role}: {content}\n"
            
            prompt = f"""
            Analyze the following conversation between a user and an AI assistant.
            Identify key topics, sentiment, and potential action items.
            
            Conversation:
            {conversation_text}
            
            Return a JSON with the following fields:
            - topics: list of main topics discussed
            - sentiment: overall sentiment (positive, negative, neutral)
            - action_items: list of potential action items
            - satisfaction: estimated user satisfaction (high, medium, low)
            """
            
            # In a real implementation, we would parse the JSON response
            # For now, returning a simplified version
            return {
                "topics": ["business automation", "AI tools"],
                "sentiment": "positive",
                "action_items": ["Follow up with more information"],
                "satisfaction": "high"
            }
        except Exception as e:
            logger.error(f"Error analyzing conversation: {str(e)}")
            return {"error": str(e)}
    
    def batch_process_conversations(self, conversations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a batch of conversations from DataFrame
        
        Args:
            conversations_df: DataFrame with conversation data
            
        Returns:
            DataFrame with added column for AI response
        """
        try:
            # Create copy to avoid modifying original
            result_df = conversations_df.copy()
            
            # Add column for generated response
            result_df['Generated_Response'] = None
            
            for idx, row in result_df.iterrows():
                user_input = row.get('User_Input', '')
                
                # Clear history for each new conversation
                self.clear_history()
                
                # Generate response
                response = self.generate_response(user_input)
                result_df.at[idx, 'Generated_Response'] = response
                
                logger.info(f"Processed conversation {idx+1}/{len(result_df)}")
            
            return result_df
        except Exception as e:
            logger.error(f"Error batch processing conversations: {str(e)}")
            raise
