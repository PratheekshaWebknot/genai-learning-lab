"""
Configuration file for text generation tool.
Contains temperature values, model names, and output paths.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Temperature values to test
TEMPERATURES = [0.2, 0.7, 1.2]

# HuggingFace model configuration
HF_MODEL_NAME = "gpt2"  # Using GPT-2 as default (small, fast)
# Alternative options: "distilgpt2", "gpt2-medium", "EleutherAI/gpt-neo-125M"

# API model configuration
API_PROVIDER = "openai"  # Primary provider: "openai" or "groq"
API_MODEL_NAME = "gpt-3.5-turbo"  # For OpenAI
# For Groq, valid models include: "llama-3-8b-8192", "llama-3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"

# Fallback configuration (if primary provider fails)
ENABLE_FALLBACK = True  # Set to False to disable fallback
FALLBACK_PROVIDER = "groq"  # Fallback provider if primary fails
FALLBACK_MODEL_NAME = "llama-3.1-8b-instant"  # Model for fallback provider (valid Groq model)

# API Keys (loaded from environment)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Output file path
OUTPUT_FILE = OUTPUT_DIR / "sample_outputs.json"

# Generation parameters
MAX_NEW_TOKENS = 150  # Maximum tokens to generate
TOP_P = 0.9  # Nucleus sampling parameter
TOP_K = 50  # Top-k sampling parameter

