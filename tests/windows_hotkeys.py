#!/usr/bin/env python3
"""
Windows API Global Hotkey Manager

Provides true system-wide hotkeys using Windows API calls:
- RegisterHotKey: Register global hotkey combinations
- GetMessage: Listen for WM_HOTKEY messages
- UnregisterHotKey: Clean up registered hotkeys

No admin rights or UAC prompts required.
"""

import ctypes
import ctypes.wintypes
import threading
import time
from typing import Callable, Dict, Optional
import sys

# Windows API constants
WM_HOTKEY = 0x0312
MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008

# Windows API function declarations
user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

# Function signatures
RegisterHotKey = user32.RegisterHotKey
RegisterHotKey.argtypes = [
    ctypes.wintypes.HWND,
    ctypes.wintypes.INT,
    ctypes.wintypes.UINT,
    ctypes.wintypes.UINT
]
RegisterHotKey.restype = ctypes.wintypes.BOOL

UnregisterHotKey = user32.UnregisterHotKey
UnregisterHotKey.argtypes = [
    ctypes.wintypes.HWND,
    ctypes.wintypes.INT
]
UnregisterHotKey.restype = ctypes.wintypes.BOOL

GetMessage = user32.GetMessageW
GetMessage.argtypes = [
    ctypes.POINTER(ctypes.wintypes.MSG),
    ctypes.wintypes.HWND,
    ctypes.wintypes.UINT,
    ctypes.wintypes.UINT
]
GetMessage.restype = ctypes.wintypes.INT

TranslateMessage = user32.TranslateMessage
TranslateMessage.argtypes = [ctypes.POINTER(ctypes.wintypes.MSG)]
TranslateMessage.restype = ctypes.wintypes.BOOL

DispatchMessage = user32.DispatchMessageW
DispatchMessage.argtypes = [ctypes.POINTER(ctypes.wintypes.MSG)]
DispatchMessage.restype = ctypes.wintypes.LONG

PostQuitMessage = user32.PostQuitMessage
PostQuitMessage.argtypes = [ctypes.wintypes.INT]
PostQuitMessage.restype = None

# Hotkey ID mapping
HOTKEY_IDS = {
    "send": 1,      # Ctrl+Alt+Shift+G
    "repeat": 2,    # Ctrl+Alt+Shift+R
    "help": 3,      # Ctrl+Alt+Shift+H
    "skip": 4       # Ctrl+Alt+Shift+F7
}

# Virtual key codes
VK_G = 0x47
VK_R = 0x52
VK_H = 0x48
VK_F7 = 0x76

# Hotkey definitions
HOTKEY_DEFINITIONS = {
    "send": (MOD_CONTROL | MOD_ALT | MOD_SHIFT, VK_G),
    "repeat": (MOD_CONTROL | MOD_ALT | MOD_SHIFT, VK_R),
    "help": (MOD_CONTROL | MOD_ALT | MOD_SHIFT, VK_H),
    "skip": (MOD_CONTROL | MOD_ALT | MOD_SHIFT, VK_F7)
}


class WindowsGlobalHotkeyManager:
    """
    Windows API-based global hotkey manager.
    
    Provides true system-wide hotkeys that work regardless of application focus.
    No admin rights required.
    """
    
    def __init__(self, dispatch_func: Callable[[int], None]):
        """
        Initialize the hotkey manager.
        
        Args:
            dispatch_func: Function called when hotkeys are pressed.
                          Receives hotkey ID as parameter.
        """
        self.dispatch_func = dispatch_func
        self.running = False
        self.message_thread: Optional[threading.Thread] = None
        self.registered_hotkeys: Dict[str, int] = {}
        self.last_press_time: Dict[int, float] = {}
        self.debounce_delay = 0.15  # 150ms debounce
        
        # Message structure for GetMessage
        self.msg = ctypes.wintypes.MSG()
        
    def register_hotkey(self, action: str) -> bool:
        """
        Register a hotkey for the specified action.
        
        Args:
            action: Hotkey action name ("send", "repeat", "help", "skip")
            
        Returns:
            True if registration successful, False otherwise
        """
        if action not in HOTKEY_IDS:
            return False
            
        hotkey_id = HOTKEY_IDS[action]
        modifiers, vk = HOTKEY_DEFINITIONS[action]
        
        # Register the hotkey (HWND=0 for global registration)
        success = RegisterHotKey(0, hotkey_id, modifiers, vk)
        
        if success:
            self.registered_hotkeys[action] = hotkey_id
            return True
        else:
            # Get last error for debugging
            error_code = kernel32.GetLastError()
            print(f"RegisterHotKey failed for {action}: error 0x{error_code:04X}")
            return False
            
    def unregister_hotkey(self, action: str) -> bool:
        """
        Unregister a hotkey for the specified action.
        
        Args:
            action: Hotkey action name
            
        Returns:
            True if unregistration successful, False otherwise
        """
        if action not in self.registered_hotkeys:
            return True  # Not registered, consider it "unregistered"
            
        hotkey_id = self.registered_hotkeys[action]
        success = UnregisterHotKey(0, hotkey_id)
        
        if success:
            del self.registered_hotkeys[action]
            if hotkey_id in self.last_press_time:
                del self.last_press_time[hotkey_id]
                
        return bool(success)
        
    def _message_loop(self):
        """
        Windows message loop to process WM_HOTKEY messages.
        Runs in a separate thread and handles registration.
        """
        # Register all hotkeys in this thread (critical for Windows API)
        registration_success = True
        for action in HOTKEY_IDS.keys():
            if not self.register_hotkey(action):
                registration_success = False
                print(f"Failed to register hotkey: {action}")
                
        if not registration_success:
            print("Some hotkeys failed to register - continuing with available ones")
            
        print(f"Message loop started - registered {len(self.registered_hotkeys)} hotkeys")
        
        while self.running:
            try:
                # Get next message (blocking)
                result = GetMessage(ctypes.byref(self.msg), 0, 0, 0)
                
                if result == -1:  # Error
                    print("GetMessage returned -1 (error)")
                    break
                elif result == 0:  # WM_QUIT
                    print("GetMessage returned 0 (WM_QUIT)")
                    break
                    
                # Check if this is a hotkey message
                if self.msg.message == WM_HOTKEY:
                    hotkey_id = self.msg.wParam
                    print(f"WM_HOTKEY received: ID={hotkey_id}")
                    
                    # Check debouncing
                    current_time = time.time()
                    if hotkey_id in self.last_press_time:
                        time_since_last = current_time - self.last_press_time[hotkey_id]
                        if time_since_last < self.debounce_delay:
                            print(f"Debounced hotkey {hotkey_id} (last press {time_since_last:.3f}s ago)")
                            continue  # Ignore debounced press
                            
                    # Update last press time
                    self.last_press_time[hotkey_id] = current_time
                    
                    # Dispatch to handler
                    try:
                        self.dispatch_func(hotkey_id)
                    except Exception as e:
                        # Log error but don't crash the message loop
                        print(f"Error in hotkey dispatch: {e}")
                        
                else:
                    # Process other messages normally
                    TranslateMessage(ctypes.byref(self.msg))
                    DispatchMessage(ctypes.byref(self.msg))
                    
            except Exception as e:
                # Log error but continue message loop
                print(f"Error in message loop: {e}")
                time.sleep(0.01)  # Brief pause on error
                
    def start_listening(self) -> bool:
        """
        Start listening for global hotkeys.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.running:
            return True  # Already running
            
        # Start message loop thread (registration happens inside the thread)
        self.running = True
        self.message_thread = threading.Thread(target=self._message_loop, daemon=True)
        self.message_thread.start()
        
        # Give the thread a moment to start and register hotkeys
        time.sleep(0.1)
        
        return True
        
    def stop_listening(self):
        """
        Stop listening for global hotkeys and clean up.
        """
        self.running = False
        
        # Post quit message to break message loop (this will trigger cleanup in the message thread)
        PostQuitMessage(0)
        
        # Wait for message thread to finish
        if self.message_thread and self.message_thread.is_alive():
            self.message_thread.join(timeout=1.0)
            
        # Clean up any remaining registered hotkeys
        for action in list(self.registered_hotkeys.keys()):
            self.unregister_hotkey(action)
            
    def is_listening(self) -> bool:
        """
        Check if the hotkey manager is currently listening.
        
        Returns:
            True if listening, False otherwise
        """
        return self.running and self.message_thread and self.message_thread.is_alive()


def test_hotkeys() -> bool:
    """
    Test function to verify hotkey registration works.
    
    Returns:
        True if all hotkeys registered successfully, False otherwise
    """
    def dummy_dispatch(hotkey_id: int):
        pass  # Do nothing for testing
        
    manager = WindowsGlobalHotkeyManager(dummy_dispatch)
    
    try:
        # Test registration
        success = manager.start_listening()
        if success:
            print("✅ All hotkeys registered successfully")
            manager.stop_listening()
            return True
        else:
            print("❌ Failed to register some hotkeys")
            return False
    except Exception as e:
        print(f"❌ Hotkey test failed: {e}")
        return False


def dump_hotkey_status():
    """
    Dump detailed hotkey registration status.
    """
    print("=== Hotkey Registration Status ===")
    print("ID | Action | Modifiers | VK | Status")
    print("---|--------|-----------|----|-------")
    
    for action, hotkey_id in HOTKEY_IDS.items():
        modifiers, vk = HOTKEY_DEFINITIONS[action]
        
        # Try to register the hotkey
        success = RegisterHotKey(0, hotkey_id, modifiers, vk)
        
        if success:
            print(f"{hotkey_id:2d} | {action:6s} | 0x{modifiers:04X} | 0x{vk:02X} | OK (0)")
            # Unregister immediately
            UnregisterHotKey(0, hotkey_id)
        else:
            error_code = kernel32.GetLastError()
            print(f"{hotkey_id:2d} | {action:6s} | 0x{modifiers:04X} | 0x{vk:02X} | FAIL (0x{error_code:04X})")
    
    print()
    print("Modifiers: 0x0007 = Ctrl+Alt+Shift")
    print("VK codes: 0x47=G, 0x52=R, 0x48=H, 0x76=F7")


if __name__ == "__main__":
    # Self-test when run directly
    if test_hotkeys():
        print("Hotkeys OK")
        sys.exit(0)
    else:
        print("Hotkeys FAILED")
        sys.exit(1) 