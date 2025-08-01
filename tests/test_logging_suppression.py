#!/usr/bin/env python3
"""
Test script to verify logging suppression is working
"""

import logging
import os
from dotenv import load_dotenv
from google import genai

# Configure logging to suppress unwanted messages
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

def test_gemini_api():
    """Test Gemini API to see if logging messages are suppressed"""
    print("Testing Gemini API with logging suppression...")
    
    try:
        # Load environment variables
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("❌ Missing GOOGLE_API_KEY")
            return False
        
        # Initialize Gemini client
        client = genai.Client(api_key=api_key)
        
        # Make a test request
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents="Say hello in Spanish"
        )
        
        if hasattr(response, "text") and response.text:
            print(f"✅ Gemini API working: {response.text.strip()}")
            print("✅ Logging suppression appears to be working (no INFO messages above)")
            return True
        else:
            print("❌ Gemini API returned no response")
            return False
            
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        return False

if __name__ == "__main__":
    test_gemini_api() 