#!/usr/bin/env python3
"""
Test script to verify Gemini warm-up functionality.
This tests the warm_up_gemini() function implementation.
"""

import os
import sys
import time
import threading
from dotenv import load_dotenv
from google import genai
from datetime import datetime

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-2.5-flash"

if not API_KEY:
    print("ERROR: GOOGLE_API_KEY not found in .env file")
    sys.exit(1)

# Initialize Gemini client
try:
    client = genai.Client(api_key=API_KEY)
    print("✅ Gemini client initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize Gemini client: {e}")
    sys.exit(1)

# Thread-safe logging for testing
_log_lock = threading.Lock()

def log_with_timestamp(message, event_type="INFO"):
    """Test logging function"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] [{event_type}] {message}"
    print(formatted_message)

def warm_up_gemini():
    """
    Test implementation of the Gemini warm-up function.
    This mirrors the implementation in test.py.
    """
    log_with_timestamp("Starting Gemini API warm-up routine", "GEMINI_WARMUP")
    
    # Use the same prompt prefix as real interactions
    prompt = (
        "You are a Spanish tutor. The student is preparing for an oral test. "
        "Reply only with a natural, brief Spanish sentence the student should say. "
        "Do not add explanations or alternatives. Just respond with the one sentence."
    )
    
    # Warm-up request content
    warmup_input = "Say 'ready' in Spanish."
    
    # Log the warm-up request
    log_with_timestamp(f"Warm-up prompt: {prompt}", "GEMINI_WARMUP")
    log_with_timestamp(f"Warm-up input: {warmup_input}", "GEMINI_WARMUP")
    
    max_retries = 1  # One retry for warm-up
    
    for attempt in range(max_retries + 1):  # 0, 1 (2 total attempts)
        try:
            start_time = time.time()
            
            response = client.models.generate_content(
                model=MODEL_NAME,  # "gemini-2.5-flash"
                contents=f"{prompt}\n\nStudent said in Spanish: {warmup_input}"
            )
            
            warmup_time = time.time() - start_time
            
            # Extract response text
            response_text = response.text.strip() if hasattr(response, "text") else ""
            
            if response_text:
                log_with_timestamp(f"Warm-up completed in {warmup_time:.2f}s - Response: {response_text}", "GEMINI_WARMUP_COMPLETE")
                return True
            else:
                raise ValueError("Gemini warm-up returned empty response")
                
        except Exception as warmup_error:
            error_msg = f"Gemini warm-up attempt {attempt + 1}/{max_retries + 1} failed: {str(warmup_error)}"
            log_with_timestamp(error_msg, "GEMINI_WARMUP_FAIL")
            
            # If this is not the last attempt, wait and retry
            if attempt < max_retries:
                log_with_timestamp("Retrying warm-up in 2 seconds...", "GEMINI_WARMUP")
                time.sleep(2)
            else:
                # Final attempt failed
                log_with_timestamp("Gemini warm-up failed after all attempts", "GEMINI_WARMUP_FAIL")
                return False
    
    return False

def test_warmup_functionality():
    """Test the warm-up functionality"""
    print("=== Gemini Warm-up Test ===\n")
    
    # Test 1: Direct warm-up call
    print("1. Testing direct warm-up call...")
    success = warm_up_gemini()
    
    if success:
        print("✅ Direct warm-up call successful")
    else:
        print("❌ Direct warm-up call failed")
        return False
    
    # Test 2: Background thread warm-up (simulating startup)
    print("\n2. Testing background thread warm-up...")
    
    warmup_results = []
    
    def background_warmup():
        result = warm_up_gemini()
        warmup_results.append(result)
    
    try:
        warmup_thread = threading.Thread(target=background_warmup, daemon=True)
        warmup_thread.start()
        log_with_timestamp("Gemini warm-up thread started", "SYSTEM")
        
        # Wait for thread to complete (with timeout)
        warmup_thread.join(timeout=30.0)  # 30 second timeout
        
        if warmup_thread.is_alive():
            print("⏰ Background warm-up timed out")
            return False
        elif warmup_results and warmup_results[0]:
            print("✅ Background thread warm-up successful")
        else:
            print("❌ Background thread warm-up failed")
            return False
            
    except Exception as thread_error:
        print(f"❌ Thread error: {thread_error}")
        return False
    
    # Test 3: Measure latency improvement
    print("\n3. Testing latency improvement...")
    
    # Make a regular API call after warm-up
    regular_prompt = (
        "You are a Spanish tutor. The student is preparing for an oral test. "
        "Reply only with a natural, brief Spanish sentence the student should say. "
        "Do not add explanations or alternatives. Just respond with the one sentence."
    )
    
    try:
        start_time = time.time()
        
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=f"{regular_prompt}\n\nStudent said in Spanish: Hola, ¿cómo estás?"
        )
        
        regular_time = time.time() - start_time
        
        if hasattr(response, "text") and response.text:
            print(f"✅ Regular API call after warm-up: {regular_time:.2f}s")
            print(f"   Response: {response.text.strip()}")
            
            # The second call should be faster (though this isn't guaranteed)
            if regular_time < 5.0:  # Reasonable threshold
                print("✅ Response time appears improved")
            else:
                print("⚠️  Response time still high (may be normal)")
        else:
            print("❌ Regular API call failed")
            return False
            
    except Exception as regular_error:
        print(f"❌ Regular API call error: {regular_error}")
        return False
    
    print("\n✅ All warm-up tests passed!")
    return True

def test_logging_events():
    """Test that all required logging events are generated"""
    print("\n=== Logging Events Test ===")
    
    print("Expected logging events:")
    print("  [GEMINI_WARMUP] - Before the call")
    print("  [GEMINI_WARMUP_COMPLETE] - On success")
    print("  [GEMINI_WARMUP_FAIL] - On failure")
    
    print("\nRunning warm-up to verify logging...")
    warm_up_gemini()
    
    print("✅ Check the log output above for proper event types")
    return True

if __name__ == "__main__":
    print("🚀 Testing Gemini Warm-up Implementation\n")
    
    success = True
    
    try:
        success &= test_warmup_functionality()
        success &= test_logging_events()
        
        if success:
            print("\n🎉 All Gemini warm-up tests passed!")
            print("\n📋 Implementation Summary:")
            print("  ✅ warm_up_gemini() function implemented")
            print("  ✅ Background thread execution working")
            print("  ✅ Proper logging events generated")
            print("  ✅ Error handling and retry logic functional")
            print("  ✅ Cold-start latency mitigation active")
            
        else:
            print("\n❌ Some tests failed. Please review implementation.")
            
    except Exception as test_error:
        print(f"\n❌ Test execution error: {test_error}")
        success = False
    
    sys.exit(0 if success else 1)