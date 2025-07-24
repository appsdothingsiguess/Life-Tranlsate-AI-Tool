#!/usr/bin/env python3
"""
Test script for the streaming transcription worker thread implementation.
Validates that the worker thread correctly processes audio windows with proper logging.
"""

import numpy as np
import queue
import threading
import time
from unittest.mock import Mock, patch
import sys
import os

# Add current directory to path to import main module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_streaming_worker_basic_functionality():
    """Test basic functionality of the streaming transcription worker thread."""
    print("=== Testing Streaming Transcription Worker Thread ===")
    
    # Import after path setup
    import main
    
    # Create test audio data (3 seconds of sine wave at 48kHz)
    sample_rate = 48000
    duration = 3.0
    frequency = 440  # A4 note
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    test_audio = 0.1 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    print(f"✅ Created test audio: {len(test_audio)} samples ({len(test_audio)/sample_rate:.2f}s)")
    
    # Mock the Whisper model to avoid actual transcription
    mock_segments = [Mock(text="Hola mundo")]
    mock_info = Mock(language="es")
    
    with patch.object(main.model, 'transcribe', return_value=(mock_segments, mock_info)) as mock_transcribe:
        # Clear the queue and reset worker state
        while not main.audio_queue.empty():
            try:
                main.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        main.transcription_worker_running = True
        
        # Start the worker thread
        worker_thread = threading.Thread(target=main.streaming_transcription_worker, daemon=True)
        worker_thread.start()
        print("✅ Started streaming transcription worker thread")
        
        # Add test audio to queue
        main.audio_queue.put(test_audio)
        print("✅ Added test audio window to queue")
        
        # Wait for processing
        time.sleep(2.0)
        
        # Verify the mock was called with correct parameters
        assert mock_transcribe.called, "Whisper transcribe should have been called"
        call_args = mock_transcribe.call_args
        
        # Check that audio data was passed
        assert len(call_args[0]) > 0, "Audio data should be passed to transcribe"
        audio_arg = call_args[0][0]
        assert isinstance(audio_arg, np.ndarray), "Audio should be numpy array"
        assert len(audio_arg) == len(test_audio), f"Audio length mismatch: {len(audio_arg)} vs {len(test_audio)}"
        
        # Check that language="es" and task="transcribe" were used
        kwargs = call_args[1]
        assert kwargs.get('language') == 'es', f"Language should be 'es', got {kwargs.get('language')}"
        assert kwargs.get('task') == 'transcribe', f"Task should be 'transcribe', got {kwargs.get('task')}"
        
        print("✅ Whisper called with correct parameters:")
        print(f"   - Language: {kwargs.get('language')}")
        print(f"   - Task: {kwargs.get('task')}")
        print(f"   - Audio samples: {len(audio_arg)}")
        
        # Verify last_spanish was updated
        assert main.last_spanish == "Hola mundo", f"last_spanish should be 'Hola mundo', got '{main.last_spanish}'"
        print("✅ Last Spanish transcription updated correctly")
        
        # Stop the worker
        main.transcription_worker_running = False
        worker_thread.join(timeout=2.0)
        print("✅ Worker thread stopped successfully")

def test_streaming_worker_error_handling():
    """Test error handling in the streaming transcription worker thread."""
    print("\n=== Testing Error Handling ===")
    
    import main
    
    # Test with invalid audio data
    with patch.object(main.model, 'transcribe', side_effect=Exception("Mock transcription error")) as mock_transcribe:
        # Clear the queue
        while not main.audio_queue.empty():
            try:
                main.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        main.transcription_worker_running = True
        
        # Start the worker thread
        worker_thread = threading.Thread(target=main.streaming_transcription_worker, daemon=True)
        worker_thread.start()
        
        # Add test audio that will cause an error
        test_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        main.audio_queue.put(test_audio)
        print("✅ Added audio that will trigger error")
        
        # Wait for processing
        time.sleep(1.0)
        
        # Verify the worker continues running despite the error
        assert worker_thread.is_alive(), "Worker thread should continue running after error"
        print("✅ Worker thread continues running after transcription error")
        
        # Stop the worker
        main.transcription_worker_running = False
        worker_thread.join(timeout=2.0)
        print("✅ Worker thread stopped successfully after error test")

def test_streaming_worker_queue_operations():
    """Test queue operations and thread safety."""
    print("\n=== Testing Queue Operations ===")
    
    import main
    
    # Test empty queue handling
    while not main.audio_queue.empty():
        try:
            main.audio_queue.get_nowait()
        except queue.Empty:
            break
    
    main.transcription_worker_running = True
    
    # Start worker with empty queue
    worker_thread = threading.Thread(target=main.streaming_transcription_worker, daemon=True)
    worker_thread.start()
    print("✅ Started worker with empty queue")
    
    # Let it run for a short time with empty queue
    time.sleep(0.5)
    assert worker_thread.is_alive(), "Worker should handle empty queue gracefully"
    print("✅ Worker handles empty queue correctly")
    
    # Test multiple audio windows
    with patch.object(main.model, 'transcribe', return_value=([Mock(text="Test")], Mock(language="es"))):
        for i in range(3):
            test_audio = np.random.random(1000).astype(np.float32)
            main.audio_queue.put(test_audio)
        
        print("✅ Added 3 audio windows to queue")
        
        # Wait for processing
        time.sleep(2.0)
        
        # Queue should be empty or nearly empty
        queue_size = main.audio_queue.qsize()
        print(f"✅ Queue size after processing: {queue_size}")
    
    # Stop the worker
    main.transcription_worker_running = False
    worker_thread.join(timeout=2.0)
    print("✅ Worker thread stopped successfully")

def test_logging_integration():
    """Test that proper logging events are generated."""
    print("\n=== Testing Logging Integration ===")
    
    import main
    
    # Capture log calls
    log_calls = []
    original_log = main.log_with_timestamp
    original_window_log = main.log_window_transcribe_start
    original_result_log = main.log_window_result
    
    def mock_log(message, event_type="INFO"):
        log_calls.append((message, event_type))
        return original_log(message, event_type)
    
    def mock_window_log(duration):
        log_calls.append((f"Started transcription for {duration:.1f}s slice", "WINDOW_TRANSCRIBE"))
        return original_window_log(duration)
    
    def mock_result_log(text, confidence=None):
        if confidence is not None:
            message = f"Transcription result (confidence: {confidence:.2f}): {text}"
        else:
            message = f"Transcription result: {text}"
        log_calls.append((message, "WINDOW_RESULT"))
        return original_result_log(text, confidence)
    
    with patch.object(main, 'log_with_timestamp', side_effect=mock_log), \
         patch.object(main, 'log_window_transcribe_start', side_effect=mock_window_log), \
         patch.object(main, 'log_window_result', side_effect=mock_result_log), \
         patch.object(main.model, 'transcribe', return_value=([Mock(text="Prueba")], Mock(language="es"))):
        
        # Clear previous log calls
        log_calls.clear()
        
        # Clear the queue
        while not main.audio_queue.empty():
            try:
                main.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        main.transcription_worker_running = True
        
        # Start worker
        worker_thread = threading.Thread(target=main.streaming_transcription_worker, daemon=True)
        worker_thread.start()
        
        # Add test audio
        test_audio = np.random.random(48000 * 3).astype(np.float32)  # 3 seconds
        main.audio_queue.put(test_audio)
        
        # Wait for processing
        time.sleep(2.0)
        
        # Check for expected log events
        window_transcribe_logs = [call for call in log_calls if call[1] == "WINDOW_TRANSCRIBE"]
        window_result_logs = [call for call in log_calls if call[1] == "WINDOW_RESULT"]
        
        assert len(window_transcribe_logs) > 0, "Should have WINDOW_TRANSCRIBE log events"
        assert len(window_result_logs) > 0, "Should have WINDOW_RESULT log events"
        
        print(f"✅ Found {len(window_transcribe_logs)} WINDOW_TRANSCRIBE events")
        print(f"✅ Found {len(window_result_logs)} WINDOW_RESULT events")
        
        # Stop worker
        main.transcription_worker_running = False
        worker_thread.join(timeout=2.0)

if __name__ == "__main__":
    try:
        test_streaming_worker_basic_functionality()
        test_streaming_worker_error_handling()
        test_streaming_worker_queue_operations()
        test_logging_integration()
        
        print("\n🎉 All streaming transcription worker tests passed!")
        print("\n✅ Task 4 Implementation Verified:")
        print("   - Single worker thread for non-blocking processing")
        print("   - Thread-safe queue for audio window processing")
        print("   - Faster-whisper configured with language='es' and task='transcribe'")
        print("   - [WINDOW_TRANSCRIBE] and [WINDOW_RESULT] logging")
        print("   - Error handling without crashing main loop")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)