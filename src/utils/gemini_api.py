import os
import google.generativeai as genai
from typing import List, Dict, Any, Optional
import logging

from src.config.config import GEMINI_API_KEY, GEMINI_MODEL, TEMPERATURE, MAX_TOKENS, TOP_P, TOP_K

# Configure logger
logger = logging.getLogger(__name__)

class GeminiAPI:
    """
    Wrapper class for Gemini API interactions
    """
    def __init__(self, api_key: Optional[str] = None):

        self.api_key = api_key or GEMINI_API_KEY
        self.model = GEMINI_MODEL
        self._initialize_api()
        
    def _initialize_api(self):
        """Initialize the Gemini API with the provided key"""
        try:
            genai.configure(api_key=self.api_key)
            self.model_instance = genai.GenerativeModel(self.model)
            logger.info(f"Gemini API initialized with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API: {str(e)}")
            raise
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the Gemini model
        
        Args:
            prompt: The input prompt for text generation
            **kwargs: Additional parameters to override defaults
            
        Returns:
            Generated text response
        """
        try:
            generation_config = {
                "temperature": kwargs.get("temperature", TEMPERATURE),
                "top_p": kwargs.get("top_p", TOP_P),
                "top_k": kwargs.get("top_k", TOP_K),
                "max_output_tokens": kwargs.get("max_tokens", MAX_TOKENS),
            }
            
            response = self.model_instance.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            return response.text
        except Exception as e:
            logger.error(f"Text generation failed: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of the provided text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        prompt = f"""
        Analyze the sentiment of the following text and return a JSON with:
        - sentiment: (positive, negative, or neutral)
        - confidence: (a number between 0 and 1)
        - tone: (formal, informal, friendly, urgent, etc.)
        
        Text: {text}
        """
        
        try:
            response = self.generate_text(prompt)
            # In a real implementation, we would parse the JSON response
            # For now, returning a simplified version
            return {
                "sentiment": "positive",  # Placeholder
                "confidence": 0.8,        # Placeholder
                "tone": "formal"          # Placeholder
            }
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            return {"error": str(e)}
    
    def generate_email_reply(self, email_content: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate an email reply based on the provided email content
        
        Args:
            email_content: The email content to reply to
            context: Optional context information
            
        Returns:
            Generated email reply
        """
        prompt = f"""
        Generate a professional email reply to the following email:
        
        {email_content}
        """
        
        if context:
            prompt += f"\n\nAdditional context: {context}"
            
        return self.generate_text(prompt)
    
    def generate_chatbot_response(self, user_input: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Generate a chatbot response based on user input and conversation history
        
        Args:
            user_input: The user's input message
            conversation_history: Optional list of previous messages
            
        Returns:
            Generated chatbot response
        """
        history_text = ""
        if conversation_history:
            for msg in conversation_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                history_text += f"{role}: {content}\n"
        
        prompt = f"""
        You are an AI assistant for business communication. Respond to the following user message:
        
        {history_text}
        User: {user_input}
        Assistant:
        """
        
        return self.generate_text(prompt)
    
    def summarize_business_report(self, report_text: str) -> str:
        """
        Summarize a business report
        
        Args:
            report_text: The report text to summarize
            
        Returns:
            Summarized report
        """
        prompt = f"""
        Summarize the following business report, highlighting key insights, trends, and action items:
        
        {report_text}
        """
        
        return self.generate_text(prompt)
