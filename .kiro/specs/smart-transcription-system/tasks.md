# Implementation Plan

- [x] 1. Set up streaming constants and window-specific logging





  - Define minimal streaming constants (WINDOW_DURATION=3.0, WINDOW_INTERVAL=1.0, MIN_AUDIO_DURATION=2.5)
  - Add logging methods for [WINDOW_TRANSCRIBE], [WINDOW_RESULT], [WINDOW_SKIP] events with timestamps
  - Keep existing thread-safe logging infrastructure
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

- [x] 2. Create unified StreamingBuffer class for audio management





  - Implement single StreamingBuffer class that combines rolling buffer and window extraction
  - Use deque for efficient circular buffer with automatic oldest-data eviction
  - Extract 3.0-second windows directly in audio callback (no separate timer threads)
  - Skip windows with less than 2.5 seconds of data and log [WINDOW_SKIP]
  - _Requirements: 1.1, 1.2, 1.4, 4.3_

- [x] 3. Validate Whisper native audio format support





  - Test if faster-whisper accepts 48kHz float32 audio directly without external resampling
  - If Whisper handles resampling internally, skip scipy resampling entirely
  - Only implement manual resampling if Whisper requires 16kHz input
  - Log audio format validation results during startup
  - Add fallback logging if Whisper rejects 48kHz input - log exactly when/why it fell back to scipy
  - _Requirements: 1.3, 5.4_

- [x] 4. Build minimal streaming transcription worker thread





  - Create single worker thread for non-blocking transcription processing
  - Use thread-safe queue for audio window processing
  - Configure faster-whisper with language="es" and task="transcribe"
  - Log [WINDOW_TRANSCRIBE] start and [WINDOW_RESULT] completion
  - Handle transcription errors without crashing main loop
  - _Requirements: 2.1, 2.2, 2.4, 4.1, 4.2, 5.1, 5.2_

- [x] 5. Implement simple duplicate filtering





  - Add basic check: skip if current transcription is identical to last result within 1.5s window
  - Store only last transcription text and timestamp for comparison
  - No fuzzy matching or complex similarity algorithms
  - Log when duplicates are skipped
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [x] 6. Replace silence detection with continuous streaming callback





  - Remove all RMS threshold and silence detection logic completely
  - Implement audio callback that feeds StreamingBuffer and triggers window extraction directly
  - Extract windows every 1.0 second within the callback or main polling loop
  - Maintain 48kHz mono audio capture from VB-Audio Virtual Cable
  - _Requirements: 1.1, 1.2, 2.3_

- [x] 7. Add performance monitoring for live exam latency





  - Target ~1.5 second total latency from speech to transcription result
  - Log processing time for each transcription window with log_transcription_latency(window_id, duration)
  - Buffer timestamps and calculate rolling average over last 5-10 windows for smoothed performance tracking
  - Add latency warnings when processing exceeds 1.5s target
  - Test with continuous Spanish speech to validate real-time performance
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [x] 8. Test complete streaming pipeline with existing integrations





  - Maintain existing VB-Audio Virtual Cable input (device index 37)
  - Keep faster-whisper float16 integration and Gemini 2.5 Flash Lite
  - Preserve dotenv configuration and error handling
  - Run end-to-end test with live Spanish audio input
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_