# This is a Python module
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "src" / "config"

# Gemini API configuration
GEMINI_API_KEY = "write your gemini api key here"  
GEMINI_MODEL = "gemini-2.0-pro-exp-02-05"  

# Data processing configuration
MAX_TOKENS = 1024
TEMPERATURE = 0.2
TOP_P = 0.95
TOP_K = 40

# Email automation configuration
EMAIL_CATEGORIES = ["Business", "Support", "Meeting", "Finance"]
EMAIL_TEMPLATES_DIR = CONFIG_DIR / "email_templates"

# Chatbot configuration
CHATBOT_MAX_HISTORY = 10
CHATBOT_RESPONSE_TYPES = ["Informational", "Action", "Clarification"]

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FILE = PROJECT_ROOT / "logs" / "app.log"
