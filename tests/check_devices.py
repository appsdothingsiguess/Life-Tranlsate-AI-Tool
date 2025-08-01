#!/usr/bin/env python3
"""
Simple script to list all available audio input devices.
"""

import sounddevice as sd

def list_audio_devices():
    """List all available audio input devices."""
    print("=== AVAILABLE AUDIO INPUT DEVICES ===")
    print()
    
    try:
        devices = sd.query_devices()
        
        input_devices = []
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append((i, device))
        
        if not input_devices:
            print("❌ No audio input devices found!")
            return
        
        print(f"Found {len(input_devices)} input device(s):")
        print()
        
        for i, device in input_devices:
            print(f"  {i:2d}: {device['name']}")
            print(f"      Channels: {device['max_input_channels']}")
            print(f"      Sample Rates: {device['default_samplerate']} Hz")
            print()
        
        # Show default device
        try:
            default_device = sd.default.device[0]
            print(f"Default input device: {default_device}")
        except:
            print("No default input device set")
        
        print()
        print("💡 Look for 'VB-Audio Virtual Cable' or similar virtual audio device")
        print("💡 If not found, you may need to install VB-Audio Virtual Cable")
        
    except Exception as e:
        print(f"❌ Error listing devices: {e}")

if __name__ == "__main__":
    list_audio_devices() 