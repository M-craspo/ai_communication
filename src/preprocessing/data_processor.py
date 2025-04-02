# This is a Python module
"""
Data processing utilities for AI-powered communication tools
"""
import pandas as pd
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Configure logger
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Class for processing and preparing data for AI models
    """
    def __init__(self, data_dir: str):
        """
        Initialize the data processor
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        self.processed_data = {}
        
    def load_csv_data(self, filename: str) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Args:
            filename: Name of the CSV file
            
        Returns:
            Pandas DataFrame containing the data
        """
        try:
            file_path = self.data_dir / filename
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded data from {filename}")
            return df
        except Exception as e:
            logger.error(f"Error loading data from {filename}: {str(e)}")
            raise
    
    def preprocess_email_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess email communication data
        
        Args:
            df: DataFrame containing email data
            
        Returns:
            Preprocessed DataFrame
        """
        try:
            # Remove duplicates
            df = df.drop_duplicates()
            
            # Handle missing values
            df = df.fillna({
                'Subject': 'No Subject',
                'Email_Body': '',
                'Category': 'Uncategorized'
            })
            
            # Create additional features
            df['Email_Length'] = df['Email_Body'].apply(len)
            
            # Normalize categories
            df['Category'] = df['Category'].str.lower().str.capitalize()
            
            logger.info("Email data preprocessing completed")
            return df
        except Exception as e:
            logger.error(f"Error preprocessing email data: {str(e)}")
            raise
    
    def preprocess_chatbot_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess chatbot conversation data
        
        Args:
            df: DataFrame containing chatbot data
            
        Returns:
            Preprocessed DataFrame
        """
        try:
            # Remove duplicates
            df = df.drop_duplicates()
            
            # Handle missing values
            df = df.fillna({
                'User_Input': '',
                'AI_Response': ''
            })
            
            # Create additional features
            df['Input_Length'] = df['User_Input'].apply(len)
            df['Response_Length'] = df['AI_Response'].apply(len)
            
            logger.info("Chatbot data preprocessing completed")
            return df
        except Exception as e:
            logger.error(f"Error preprocessing chatbot data: {str(e)}")
            raise
    
    def preprocess_business_report_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess business report data
        
        Args:
            df: DataFrame containing business report data
            
        Returns:
            Preprocessed DataFrame
        """
        try:
            # Remove duplicates
            df = df.drop_duplicates()
            
            # Handle missing values
            df = df.fillna({
                'Report_Type': 'General',
                'Summary': '',
                'Key_Findings': ''
            })
            
            # Create additional features
            df['Summary_Length'] = df['Summary'].apply(len)
            df['Findings_Length'] = df['Key_Findings'].apply(len)
            
            # Normalize report types
            df['Report_Type'] = df['Report_Type'].str.lower().str.capitalize()
            
            logger.info("Business report data preprocessing completed")
            return df
        except Exception as e:
            logger.error(f"Error preprocessing business report data: {str(e)}")
            raise
    
    def process_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Process all available data files
        
        Returns:
            Dictionary of processed DataFrames
        """
        try:
            # Process email data
            email_df = self.load_csv_data('email_communication_data.csv')
            processed_email_df = self.preprocess_email_data(email_df)
            self.processed_data['email'] = processed_email_df
            
            # Process chatbot data
            chatbot_df = self.load_csv_data('chatbot_conversation_data.csv')
            processed_chatbot_df = self.preprocess_chatbot_data(chatbot_df)
            self.processed_data['chatbot'] = processed_chatbot_df
            
            # Process business report data
            report_df = self.load_csv_data('business_report_data.csv')
            processed_report_df = self.preprocess_business_report_data(report_df)
            self.processed_data['report'] = processed_report_df
            
            logger.info("All data processed successfully")
            return self.processed_data
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise
    
    def save_processed_data(self, output_dir: Optional[str] = None) -> None:
        """
        Save processed data to CSV files
        
        Args:
            output_dir: Directory to save processed data files
        """
        if output_dir is None:
            output_dir = self.data_dir / 'processed'
        else:
            output_dir = Path(output_dir)
            
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            for data_type, df in self.processed_data.items():
                output_file = output_dir / f"processed_{data_type}_data.csv"
                df.to_csv(output_file, index=False)
                logger.info(f"Saved processed {data_type} data to {output_file}")
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise
    
    def prepare_training_data(self) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Prepare training and testing datasets
        
        Returns:
            Dictionary of (training_df, testing_df) tuples for each data type
        """
        training_data = {}
        
        try:
            for data_type, df in self.processed_data.items():
                # Split data into training (80%) and testing (20%)
                train_size = int(0.8 * len(df))
                train_df = df.iloc[:train_size]
                test_df = df.iloc[train_size:]
                
                training_data[data_type] = (train_df, test_df)
                logger.info(f"Prepared training and testing data for {data_type}")
                
            return training_data
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            raise
