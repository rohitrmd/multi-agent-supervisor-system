from dotenv import load_dotenv
import os

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Agent Configuration
SUPERVISOR_MODEL = "gpt-4"
SUPERVISOR_TEMPERATURE = 0

# Other settings can be added here as needed
