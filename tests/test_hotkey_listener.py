#!/usr/bin/env python3
"""
Simple test for hotkey listener functionality.
Tests msvcrt key detection and debouncing logic.
"""

import msvcrt
import time
import threading

def test_hotkey_listener():
    """Test the hotkey listener with debouncing."""
    print("Testing hotkey listener (press g, r, h, q keys, or ESC to exit)...")
    
    # Debouncing - track last press time for each key
    debounce_timers = {}
    debounce_delay = 0.3  # 300ms debounce
    
    while True:
        try:
            # Check if a key is available (non-blocking)
            if msvcrt.kbhit():
                try:
                    # Get the key press
                    key = msvcrt.getch().decode('utf-8').lower()
                    current_time = time.time()
                    
                    # Exit on ESC
                    if ord(key) == 27:  # ESC key
                        print("ESC pressed, exiting...")
                        break
                    
                    # Check debouncing for this specific key
                    if key in debounce_timers:
                        time_since_last = current_time - debounce_timers[key]
                        if time_since_last < debounce_delay:
                            # Key is debounced, ignore this press
                            print(f"Debounced key '{key}' (last press {time_since_last:.3f}s ago)")
                            continue
                    
                    # Update debounce timer for this key
                    debounce_timers[key] = current_time
                    
                    # Process the key press
                    if key in ['g', 'r', 'h', 'q']:
                        print(f"✅ Hotkey '{key}' pressed and processed")
                    else:
                        print(f"Other key '{key}' pressed (ignored)")
                        
                except UnicodeDecodeError:
                    # Handle special keys that can't be decoded
                    print("Special key pressed (ignored)")
                except Exception as key_error:
                    print(f"Error processing key press: {key_error}")
            
            # Small sleep to prevent excessive CPU usage
            time.sleep(0.01)  # 10ms sleep
            
        except KeyboardInterrupt:
            print("\nCtrl+C pressed, exiting...")
            break
        except Exception as listener_error:
            print(f"Critical hotkey listener error: {listener_error}")
            break

if __name__ == "__main__":
    test_hotkey_listener()