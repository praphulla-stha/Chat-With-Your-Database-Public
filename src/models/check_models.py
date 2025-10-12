import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure with your API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("GOOGLE_API_KEY not found in .env file.")
    exit()

try:
    genai.configure(api_key=api_key)
    print("API Key configured. Fetching available models...\n")

    # List all models and filter for the ones that support generateContent
    print("--- Models available for your API key ---")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
    print("-----------------------------------------")

except Exception as e:
    print(f"An error occurred: {e}")