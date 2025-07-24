# Implementation Plan

- [x] 1. Add minimal shared state and hotkey infrastructure





  - Create global `hotkey_state` dict with last_transcription, last_gemini, gemini_busy fields
  - Add single `hotkey_lock = threading.Lock()` for thread safety
  - Implement simple hotkey listener thread using msvcrt for Windows non-blocking input
  - Add 300ms debouncing with simple timestamp tracking per key
  - _Requirements: 1.1, 2.1, 2.2, 3.1, 3.4_

- [x] 2. Implement hotkey handlers (`g`, `r`, `h`)
  - Insert the three functions EXACTLY between the markers below.
  - Do NOT modify anything else in the file.
  - Log all actions with existing `log_event`.
  - Use the global `state` dict and `lock` that already exist.
  - _Requirements: 1.2, 1.3, 1.4, 4.1, 4.2, 4.4, 5.1, 6.1, 6.2_

- [ ] 3. Integrate hotkeys with existing transcription system
  - Remove blocking display_transcription_and_prompt() calls from streaming_transcription_worker()
  - Update transcription display to show hotkey instructions instead of input prompts
  - Store successful transcriptions in hotkey_state["last_transcription"] with timestamp
  - Ensure transcription loop continues uninterrupted while hotkeys are available
  - _Requirements: 1.6, 3.1, 7.4_

- [ ] 4. Add manual Gemini processing with busy state protection
  - Modify existing call_gemini_api() to work with hotkey triggers
  - Implement gemini_busy flag protection to prevent multiple simultaneous calls
  - Add visual response formatting with clear separators for Gemini output
  - Store Gemini responses in hotkey_state["last_gemini"] for repeat functionality
  - _Requirements: 4.2, 4.3, 4.5, 5.2, 7.1_

- [ ] 5. Add error handling and graceful shutdown
  - Implement Ctrl+C handling to stop audio, hotkey thread, and exit cleanly
  - Add error recovery for hotkey thread crashes without stopping main transcription
  - Handle empty transcription and busy state scenarios with appropriate logging
  - Ensure proper thread cleanup and resource deallocation on shutdown
  - _Requirements: 6.4, 7.2, 7.3, 7.5_

- [ ] 6. Start hotkey thread and integrate with main application
  - Initialize hotkey_state and hotkey_lock globals at startup
  - Start hotkey listener thread alongside existing transcription worker
  - Remove or disable existing user_input_worker thread (replaced by hotkeys)
  - Test complete system with live transcription and hotkey interactions
  - _Requirements: 1.1, 1.6, 7.4_