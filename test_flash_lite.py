#!/usr/bin/env python3
"""
Test script to verify gemini-2.5-flash-lite model performance and latency.
This tests the new optimized model configuration for live Spanish exam use.
"""

import os
import sys
import time
from dotenv import load_dotenv
from google import genai
from datetime import datetime

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-2.5-flash-lite"

if not API_KEY:
    print("ERROR: GOOGLE_API_KEY not found in .env file")
    sys.exit(1)

# Initialize Gemini client
try:
    client = genai.Client(api_key=API_KEY)
    print(f"✅ Gemini client initialized successfully")
    print(f"🚀 Using model: {MODEL_NAME} (optimized for live exam latency)")
except Exception as e:
    print(f"❌ Failed to initialize Gemini client: {e}")
    sys.exit(1)

def test_model_latency():
    """Test the latency of the new flash-lite model"""
    print("\n=== Gemini Flash Lite Latency Test ===")
    
    # Spanish tutor prompt (same as main app)
    prompt = (
        "You are a Spanish tutor. The student is preparing for an oral test. "
        "Reply only with a natural, brief Spanish sentence the student should say. "
        "Do not add explanations or alternatives. Just respond with the one sentence."
    )
    
    # Test inputs (typical Spanish exam responses)
    test_inputs = [
        "Hola, ¿cómo estás?",
        "Me llamo Juan y tengo veinte años.",
        "Estudio español en la universidad.",
        "Me gusta leer libros y escuchar música.",
        "¿Puedes repetir la pregunta, por favor?"
    ]
    
    response_times = []
    
    for i, spanish_input in enumerate(test_inputs, 1):
        print(f"\n{i}. Testing with: '{spanish_input}'")
        
        try:
            start_time = time.time()
            
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=f"{prompt}\n\nStudent said in Spanish: {spanish_input}"
            )
            
            response_time = time.time() - start_time
            response_times.append(response_time)
            
            if hasattr(response, "text") and response.text:
                response_text = response.text.strip()
                print(f"   ⏱️  Response time: {response_time:.2f}s")
                print(f"   🤖 Response: {response_text}")
                
                # Check if within live exam target
                if response_time <= 2.0:
                    print("   ✅ Within 2s live exam target")
                else:
                    print("   ⚠️  Exceeds 2s live exam target")
            else:
                print("   ❌ Empty response received")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Calculate statistics
    if response_times:
        avg_time = sum(response_times) / len(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        
        print(f"\n📊 Performance Summary:")
        print(f"   Average response time: {avg_time:.2f}s")
        print(f"   Fastest response: {min_time:.2f}s")
        print(f"   Slowest response: {max_time:.2f}s")
        print(f"   Total tests: {len(response_times)}")
        
        # Live exam readiness assessment
        fast_responses = sum(1 for t in response_times if t <= 2.0)
        success_rate = (fast_responses / len(response_times)) * 100
        
        print(f"\n🎯 Live Exam Readiness:")
        print(f"   Responses ≤ 2s: {fast_responses}/{len(response_times)} ({success_rate:.1f}%)")
        
        if success_rate >= 80:
            print("   ✅ READY for live Spanish exam")
        elif success_rate >= 60:
            print("   ⚠️  MARGINAL - may need further optimization")
        else:
            print("   ❌ NOT READY - requires optimization")
            
        return success_rate >= 80
    
    return False

def test_model_functionality():
    """Test that the model produces appropriate Spanish tutor responses"""
    print("\n=== Model Functionality Test ===")
    
    prompt = (
        "You are a Spanish tutor. The student is preparing for an oral test. "
        "Reply only with a natural, brief Spanish sentence the student should say. "
        "Do not add explanations or alternatives. Just respond with the one sentence."
    )
    
    test_case = "Buenos días, profesor."
    
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=f"{prompt}\n\nStudent said in Spanish: {test_case}"
        )
        
        if hasattr(response, "text") and response.text:
            response_text = response.text.strip()
            print(f"Input: {test_case}")
            print(f"Response: {response_text}")
            
            # Basic validation checks
            checks = {
                "Non-empty": len(response_text) > 0,
                "Reasonable length": 5 <= len(response_text) <= 200,
                "Single sentence": response_text.count('.') <= 2,
                "Spanish characters": any(c in response_text for c in "ñáéíóúü¿¡"),
            }
            
            print("\n✅ Response Quality Checks:")
            for check, passed in checks.items():
                status = "✅" if passed else "❌"
                print(f"   {status} {check}")
            
            return all(checks.values())
        else:
            print("❌ No response received")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🧠 Testing Gemini 2.5 Flash Lite for Live Spanish Exam")
    print("=" * 50)
    
    success = True
    
    try:
        # Test functionality first
        functionality_ok = test_model_functionality()
        
        # Test latency performance
        latency_ok = test_model_latency()
        
        success = functionality_ok and latency_ok
        
        print("\n" + "=" * 50)
        if success:
            print("🎉 Flash Lite model is READY for live Spanish exam!")
            print("\n📋 Optimization Summary:")
            print("  ✅ Model switched to gemini-2.5-flash-lite")
            print("  ✅ Response times optimized for live use")
            print("  ✅ Spanish tutor functionality verified")
            print("  ✅ Performance tracking implemented")
        else:
            print("❌ Flash Lite model needs further optimization")
            
    except Exception as test_error:
        print(f"\n❌ Test execution error: {test_error}")
        success = False
    
    sys.exit(0 if success else 1)