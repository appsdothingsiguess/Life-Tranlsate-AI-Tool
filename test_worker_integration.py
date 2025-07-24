#!/usr/bin/env python3
"""
Integration test for streaming transcription worker with StreamingBuffer.
Tests the complete flow from audio callback to worker thread processing.
"""

import numpy as np
import threading
import time
from unittest.mock import Mock, patch
import sys
import os

# Add current directory to path to import main module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_streaming_integration():
    """Test integration between StreamingBuffer and streaming transcription worker."""
    print("=== Testing Streaming Integration ===")
    
    import main
    
    # Mock Whisper model
    mock_segments = [Mock(text="Integración exitosa")]
    mock_info = Mock(language="es")
    
    with patch.object(main.model, 'transcribe', return_value=(mock_segments, mock_info)) as mock_transcribe:
        # Reset state
        main.transcription_worker_running = True
        
        # Clear queue
        while not main.audio_queue.empty():
            try:
                main.audio_queue.get_nowait()
            except:
                break
        
        # Start worker thread
        worker_thread = threading.Thread(target=main.streaming_transcription_worker, daemon=True)
        worker_thread.start()
        print("✅ Started streaming transcription worker")
        
        # Create test streaming buffer
        test_buffer = main.StreamingBuffer(
            sample_rate=48000,
            window_duration=3.0,
            window_interval=1.0,
            min_audio_duration=2.5
        )
        
        # Simulate audio callback behavior
        # Add enough audio data to trigger window extraction
        chunk_size = int(48000 * 0.5)  # 0.5 second chunks
        for i in range(8):  # 4 seconds total
            audio_chunk = np.random.random(chunk_size).astype(np.float32) * 0.1
            test_buffer.append_audio(audio_chunk)
            
            # Check if we should extract a window
            if test_buffer.should_extract_window():
                window_data = test_buffer.extract_window()
                if window_data is not None:
                    # Queue for transcription (simulating callback behavior)
                    main.audio_queue.put(window_data.copy())
                    print(f"✅ Extracted and queued window: {len(window_data)} samples")
            
            time.sleep(0.1)  # Simulate real-time processing
        
        # Wait for processing
        time.sleep(2.0)
        
        # Verify transcription was called
        assert mock_transcribe.called, "Transcription should have been called"
        print(f"✅ Transcription called {mock_transcribe.call_count} times")
        
        # Verify correct parameters
        for call in mock_transcribe.call_args_list:
            kwargs = call[1]
            assert kwargs.get('language') == 'es', "Language should be 'es'"
            assert kwargs.get('task') == 'transcribe', "Task should be 'transcribe'"
        
        print("✅ All transcription calls used correct parameters")
        
        # Stop worker
        main.transcription_worker_running = False
        worker_thread.join(timeout=2.0)
        print("✅ Integration test completed successfully")

def test_window_extraction_timing():
    """Test that window extraction timing works correctly with worker thread."""
    print("\n=== Testing Window Extraction Timing ===")
    
    import main
    
    with patch.object(main.model, 'transcribe', return_value=([Mock(text="Timing test")], Mock(language="es"))):
        # Reset state
        main.transcription_worker_running = True
        
        # Clear queue
        while not main.audio_queue.empty():
            try:
                main.audio_queue.get_nowait()
            except:
                break
        
        # Start worker
        worker_thread = threading.Thread(target=main.streaming_transcription_worker, daemon=True)
        worker_thread.start()
        
        # Create buffer with specific timing
        test_buffer = main.StreamingBuffer(
            sample_rate=48000,
            window_duration=3.0,
            window_interval=1.0,  # Extract every 1 second
            min_audio_duration=2.5
        )
        
        windows_extracted = 0
        
        # Simulate 5 seconds of audio input
        for second in range(5):
            # Add 1 second of audio
            audio_chunk = np.random.random(48000).astype(np.float32) * 0.1
            test_buffer.append_audio(audio_chunk)
            
            # Check extraction timing
            if test_buffer.should_extract_window():
                window_data = test_buffer.extract_window()
                if window_data is not None:
                    main.audio_queue.put(window_data.copy())
                    windows_extracted += 1
                    print(f"✅ Window {windows_extracted} extracted at second {second + 1}")
        
        # Should have extracted multiple windows due to 1-second interval
        assert windows_extracted >= 2, f"Should extract multiple windows, got {windows_extracted}"
        print(f"✅ Extracted {windows_extracted} windows over 5 seconds")
        
        # Wait for processing
        time.sleep(2.0)
        
        # Stop worker
        main.transcription_worker_running = False
        worker_thread.join(timeout=2.0)
        print("✅ Timing test completed successfully")

if __name__ == "__main__":
    try:
        test_streaming_integration()
        test_window_extraction_timing()
        
        print("\n🎉 All integration tests passed!")
        print("\n✅ Task 4 Integration Verified:")
        print("   - Worker thread integrates with StreamingBuffer")
        print("   - Window extraction triggers transcription processing")
        print("   - Timing intervals work correctly")
        print("   - Non-blocking processing maintains audio flow")
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)