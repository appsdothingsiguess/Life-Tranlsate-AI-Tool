#!/usr/bin/env python3
"""
Test script to validate Whisper native audio format support.
Tests if faster-whisper accepts 48kHz float32 audio directly without external resampling.
"""

import os
import sys
import numpy as np
import torch
from faster_whisper import WhisperModel
from scipy.signal import resample
from datetime import datetime
import traceback

# Test configuration
SAMPLE_RATE_48K = 48000
SAMPLE_RATE_16K = 16000
TEST_DURATION = 3.0  # 3 seconds of test audio
LOG_FILE = "whisper_format_validation.log"

def log_validation_result(message, event_type="VALIDATION"):
    """Log validation results with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] [{event_type}] {message}"
    
    print(formatted_message)
    
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as log_file:
            log_file.write(formatted_message + "\n")
            log_file.flush()
    except Exception as e:
        print(f"[ERROR] Failed to write to log file: {e}")

def generate_test_audio(sample_rate, duration):
    """Generate synthetic test audio for validation"""
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples, False)
    
    # Generate a mix of frequencies to simulate speech
    frequency1 = 440  # A4 note
    frequency2 = 880  # A5 note
    frequency3 = 220  # A3 note
    
    audio = (
        0.3 * np.sin(2 * np.pi * frequency1 * t) +
        0.2 * np.sin(2 * np.pi * frequency2 * t) +
        0.1 * np.sin(2 * np.pi * frequency3 * t)
    )
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, 0.05, samples)
    audio = audio + noise
    
    # Ensure float32 format
    return audio.astype(np.float32)

def test_whisper_format_support():
    """Test if Whisper accepts different audio formats natively"""
    
    log_validation_result("=== Starting Whisper Audio Format Validation ===", "SYSTEM")
    
    # Initialize Whisper model
    try:
        compute_type = "float16" if torch.cuda.is_available() else "int8"
        log_validation_result(f"Initializing Whisper model with compute_type: {compute_type}", "SYSTEM")
        model = WhisperModel("small", compute_type=compute_type)
        log_validation_result("Whisper model loaded successfully", "SYSTEM")
    except Exception as e:
        log_validation_result(f"Failed to load Whisper model: {e}", "ERROR")
        return False
    
    # Test results storage
    test_results = {
        "48khz_float32_direct": None,
        "16khz_float32_resampled": None,
        "resampling_required": None
    }
    
    # Generate test audio at 48kHz
    log_validation_result("Generating 48kHz float32 test audio", "VALIDATION")
    audio_48k = generate_test_audio(SAMPLE_RATE_48K, TEST_DURATION)
    log_validation_result(f"Generated audio: {len(audio_48k)} samples, {audio_48k.dtype}, {len(audio_48k)/SAMPLE_RATE_48K:.2f}s duration", "VALIDATION")
    
    # Test 1: Try 48kHz float32 audio directly
    log_validation_result("TEST 1: Testing 48kHz float32 audio directly with Whisper", "VALIDATION")
    try:
        segments, info = model.transcribe(
            audio_48k,
            language="es",
            task="transcribe",
            beam_size=1,  # Faster for testing
            temperature=0.0,
            no_speech_threshold=0.6
        )
        
        # Try to extract segments
        segment_texts = []
        for segment in segments:
            segment_texts.append(segment.text.strip())
        
        result_text = " ".join(segment_texts)
        test_results["48khz_float32_direct"] = {
            "success": True,
            "text": result_text,
            "language": getattr(info, 'language', 'unknown'),
            "error": None
        }
        
        log_validation_result(f"SUCCESS: 48kHz direct transcription worked", "VALIDATION_SUCCESS")
        log_validation_result(f"Detected language: {getattr(info, 'language', 'unknown')}", "VALIDATION_SUCCESS")
        log_validation_result(f"Transcription result: '{result_text}'", "VALIDATION_SUCCESS")
        
    except Exception as e:
        test_results["48khz_float32_direct"] = {
            "success": False,
            "text": None,
            "language": None,
            "error": str(e)
        }
        log_validation_result(f"FAILED: 48kHz direct transcription failed: {e}", "VALIDATION_FAIL")
        log_validation_result(f"Error traceback: {traceback.format_exc()}", "VALIDATION_FAIL")
    
    # Test 2: Try 16kHz resampled audio (traditional approach)
    log_validation_result("TEST 2: Testing 16kHz resampled audio with Whisper", "VALIDATION")
    try:
        # Resample to 16kHz using scipy
        original_length = len(audio_48k)
        new_length = int(original_length * SAMPLE_RATE_16K / SAMPLE_RATE_48K)
        audio_16k = resample(audio_48k, new_length).astype(np.float32)
        
        log_validation_result(f"Resampled audio: {len(audio_16k)} samples, {audio_16k.dtype}, {len(audio_16k)/SAMPLE_RATE_16K:.2f}s duration", "VALIDATION")
        
        segments, info = model.transcribe(
            audio_16k,
            language="es",
            task="transcribe",
            beam_size=1,  # Faster for testing
            temperature=0.0,
            no_speech_threshold=0.6
        )
        
        # Try to extract segments
        segment_texts = []
        for segment in segments:
            segment_texts.append(segment.text.strip())
        
        result_text = " ".join(segment_texts)
        test_results["16khz_float32_resampled"] = {
            "success": True,
            "text": result_text,
            "language": getattr(info, 'language', 'unknown'),
            "error": None
        }
        
        log_validation_result(f"SUCCESS: 16kHz resampled transcription worked", "VALIDATION_SUCCESS")
        log_validation_result(f"Detected language: {getattr(info, 'language', 'unknown')}", "VALIDATION_SUCCESS")
        log_validation_result(f"Transcription result: '{result_text}'", "VALIDATION_SUCCESS")
        
    except Exception as e:
        test_results["16khz_float32_resampled"] = {
            "success": False,
            "text": None,
            "language": None,
            "error": str(e)
        }
        log_validation_result(f"FAILED: 16kHz resampled transcription failed: {e}", "VALIDATION_FAIL")
        log_validation_result(f"Error traceback: {traceback.format_exc()}", "VALIDATION_FAIL")
    
    # Analyze results and make recommendation
    log_validation_result("=== VALIDATION RESULTS ANALYSIS ===", "ANALYSIS")
    
    direct_48k_works = test_results["48khz_float32_direct"]["success"]
    resampled_16k_works = test_results["16khz_float32_resampled"]["success"]
    
    if direct_48k_works:
        log_validation_result("✅ RECOMMENDATION: Whisper accepts 48kHz float32 audio directly", "RECOMMENDATION")
        log_validation_result("✅ OPTIMIZATION: Skip scipy resampling entirely for better performance", "RECOMMENDATION")
        test_results["resampling_required"] = False
        
        if resampled_16k_works:
            log_validation_result("ℹ️  NOTE: Both 48kHz direct and 16kHz resampled work, but 48kHz direct is preferred", "RECOMMENDATION")
        else:
            log_validation_result("ℹ️  NOTE: Only 48kHz direct works, 16kHz resampling had issues", "RECOMMENDATION")
            
    elif resampled_16k_works:
        log_validation_result("⚠️  FALLBACK: Whisper requires 16kHz input, must use scipy resampling", "RECOMMENDATION")
        log_validation_result("⚠️  PERFORMANCE: Manual resampling required, slight latency increase expected", "RECOMMENDATION")
        test_results["resampling_required"] = True
        
    else:
        log_validation_result("❌ ERROR: Neither 48kHz direct nor 16kHz resampled worked", "RECOMMENDATION")
        log_validation_result("❌ CRITICAL: Whisper transcription validation failed completely", "RECOMMENDATION")
        test_results["resampling_required"] = None
    
    # Log final validation summary
    log_validation_result("=== VALIDATION SUMMARY ===", "SUMMARY")
    log_validation_result(f"48kHz direct support: {'YES' if direct_48k_works else 'NO'}", "SUMMARY")
    log_validation_result(f"16kHz resampled support: {'YES' if resampled_16k_works else 'NO'}", "SUMMARY")
    log_validation_result(f"Resampling required: {'NO' if not test_results['resampling_required'] else 'YES' if test_results['resampling_required'] else 'UNKNOWN'}", "SUMMARY")
    
    return test_results

def main():
    """Main validation function"""
    print("🔍 Starting Whisper Audio Format Validation...")
    print(f"📝 Logging to: {LOG_FILE}")
    
    # Initialize log file
    try:
        with open(LOG_FILE, "w", encoding="utf-8") as log_file:
            log_file.write(f"=== Whisper Format Validation Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    except Exception as e:
        print(f"Warning: Could not initialize log file: {e}")
    
    # Run validation tests
    results = test_whisper_format_support()
    
    if results:
        print("\n✅ Validation completed successfully!")
        print(f"📝 Check {LOG_FILE} for detailed results")
        
        # Print quick summary
        if results["resampling_required"] is False:
            print("🚀 RESULT: Whisper supports 48kHz directly - no resampling needed!")
        elif results["resampling_required"] is True:
            print("⚠️  RESULT: Whisper requires 16kHz - scipy resampling needed")
        else:
            print("❌ RESULT: Validation failed - check logs for details")
            return 1
    else:
        print("❌ Validation failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())