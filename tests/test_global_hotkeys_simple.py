#!/usr/bin/env python3
"""
Simple test script to verify global hotkeys work when console is not focused.
"""

import time
import sys
from windows_hotkeys import WindowsGlobalHotkeyManager, HOTKEY_IDS

def test_dispatch(hotkey_id: int):
    """Test dispatch function that prints which hotkey was pressed."""
    action_map = {v: k for k, v in HOTKEY_IDS.items()}
    if hotkey_id in action_map:
        action = action_map[hotkey_id]
        print(f"🎯 Hotkey pressed: {action} (ID: {hotkey_id})")
    else:
        print(f"❓ Unknown hotkey ID: {hotkey_id}")

def main():
    print("=== Global Hotkey Test ===")
    print("Press Ctrl+Alt+Shift+G to send")
    print("Press Ctrl+Alt+Shift+R to repeat") 
    print("Press Ctrl+Alt+Shift+H for help")
    print("Press Ctrl+Alt+Shift+F7 to skip")
    print("Press Ctrl+C to exit")
    print()
    print("💡 Try switching to another application (Chrome, Notepad, etc.)")
    print("   and press the hotkeys - they should still work!")
    print()
    
    # Create hotkey manager
    manager = WindowsGlobalHotkeyManager(test_dispatch)
    
    try:
        # Start listening
        if manager.start_listening():
            print("✅ Global hotkeys active - listening for 30 seconds...")
            print("   (Switch to another app and try the hotkeys)")
            
            # Listen for 30 seconds
            start_time = time.time()
            while time.time() - start_time < 30:
                time.sleep(0.1)
                
            print("\n⏰ Test completed after 30 seconds")
        else:
            print("❌ Failed to start global hotkeys")
            return 1
            
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"❌ Test error: {e}")
        return 1
    finally:
        manager.stop_listening()
        print("✅ Hotkey manager stopped")
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 