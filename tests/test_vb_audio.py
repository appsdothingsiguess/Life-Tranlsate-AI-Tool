#!/usr/bin/env python3
"""
Test script to verify VB-Audio detection is working properly.
"""

import sounddevice as sd
import numpy as np

def test_vb_audio_detection():
    """Test the VB-Audio device detection logic."""
    print("=== VB-Audio Detection Test ===")
    print()
    
    try:
        devices = sd.query_devices()
        print(f"Found {len(devices)} total audio devices")
        print()
        
        # Test the detection logic from main.py
        def find_vb_audio_device():
            """Copy of the detection logic from main.py"""
            try:
                # Priority 1: Look for "CABLE Output (VB-Audio Virtual Cable)"
                for i, device in enumerate(devices):
                    if device['max_input_channels'] > 0:
                        name = device['name'].lower()
                        if 'cable output' in name and 'vb-audio virtual cable' in name:
                            return i, device['name']
                
                # Priority 2: Look for any VB-Audio device with "CABLE Output"
                for i, device in enumerate(devices):
                    if device['max_input_channels'] > 0:
                        name = device['name'].lower()
                        if 'cable output' in name and 'vb-audio' in name:
                            return i, device['name']
                
                # Priority 3: Look for any VB-Audio device
                for i, device in enumerate(devices):
                    if device['max_input_channels'] > 0:
                        name = device['name'].lower()
                        if 'vb-audio' in name:
                            return i, device['name']
                
                return None, None
            except Exception as e:
                print(f"Error: {e}")
                return None, None
        
        # Run the detection
        device_index, device_name = find_vb_audio_device()
        
        if device_index is not None:
            print(f"✅ VB-Audio device found!")
            print(f"   Index: {device_index}")
            print(f"   Name: {device_name}")
            print()
            
            # Test audio recording with this device
            print("Testing audio recording with VB-Audio device...")
            try:
                sample_rate = 48000
                duration = 10
                
                print(f"Recording {duration} seconds...")
                recording = sd.rec(
                    int(duration * sample_rate), 
                    samplerate=sample_rate, 
                    channels=1, 
                    dtype='float32',
                    device=device_index
                )
                sd.wait()
                
                # Calculate RMS
                rms = np.sqrt(np.mean(recording**2))
                print(f"Audio RMS level: {rms:.6f}")
                
                if rms > 0.001:
                    print("✅ Audio input is working - detected sound")
                else:
                    print("⚠️  Audio input is very quiet - check audio routing")
                    
            except Exception as e:
                print(f"❌ Error testing audio recording: {e}")
                
        else:
            print("❌ No VB-Audio device found")
            print()
            print("Available input devices:")
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    print(f"  {i}: {device['name']}")
        
        print()
        print("=== Windows Configuration Check ===")
        print("Make sure:")
        print("1. VB-Audio Virtual Cable is installed")
        print("2. 'CABLE Output (VB-Audio Virtual Cable)' is set as Default Device")
        print("3. Your physical microphone is set as Default Communications Device")
        print("4. Audio is being routed through VB-Audio (play some audio)")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_vb_audio_detection() 