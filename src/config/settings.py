from dotenv import load_dotenv
import os

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")  # For Stable Diffusion
REMOVE_BG_API_KEY = os.getenv("REMOVE_BG_API_KEY")  # For background removal

# Agent Configuration
SUPERVISOR_MODEL = "gpt-4"
SUPERVISOR_TEMPERATURE = 0 