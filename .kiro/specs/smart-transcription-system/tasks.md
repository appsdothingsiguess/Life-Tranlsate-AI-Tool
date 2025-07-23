# Implementation Plan

- [x] 1. Set up core constants and logging infrastructure





  - Define system constants (RMS_THRESHOLD, SILENCE_SECS, LOG_FILE, GEMINI_PROMPT, MODEL_NAME)
  - Implement thread-safe logging function with timestamp formatting
  - Create log file with proper error handling
  - _Requirements: 4.1, 4.5, 4.6_

- [x] 2. Implement RMS-based pause detection and audio buffering






  - Compute RMS for audio level detection
  - Track is_speaking state and silence_start timing
  - Flip state after SILENCE_SECS threshold is reached
  - Log SPEECH_START and SPEECH_END events with timestamps
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 3. Create transcription processing worker thread





  - Run `process_audio()` in a background thread to handle queued audio chunks
  - Pull from `audio_queue` (thread-safe Queue) and call faster-whisper transcription
  - Configure faster-whisper with `language="es"` and `task="transcribe"` (no English translation)
  - Log `[TRANSCRIBE_ES]` event with the Spanish result
  - Store last transcription in `last_spanish` for repeat prompt functionality
  - Handle empty results and exceptions gracefully with `[ERROR]` logs
  - _Requirements: 1.3, 3.2, 4.1, 4.2, 6.2_


- [x] 4. Build user confirmation and Gemini AI interaction system





  - Show transcription to user and prompt for action
  - Only call Gemini if confirmed (Enter)
  - Use controlled, single-sentence Gemini prompts
  - Support repeat functionality ('r') for last transcription
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 5.1, 5.2, 5.3, 5.4_

- [x] 4.1 Implement user confirmation interface


  - Display Spanish transcription after silence is detected and processed
  - Prompt user with: `[Press Enter = Gemini, q = skip, r = repeat]`
  - Block for input, but do not call Gemini until Enter is pressed
  - Log `[USER_CONFIRM]` or `[USER_SKIP]` or `[USER_REPEAT]`
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [x] 4.2 Integrate Gemini API with strict prompt and logging


  - Use Gemini 2.5 Flash model with this prompt:  
    `"You are a Spanish tutor. The student is preparing for an oral test. Reply only with a natural, brief Spanish sentence the student should say. Do not add explanations or alternatives. Just respond with the one sentence."`
  - Retry up to 2 times on failure (1s and 2s backoff)
  - Log full request prompt and full response in `[GEMINI_PROMPT]` and `[GEMINI_REPLY]`
  - Handle API errors gracefully without crash
  - _Requirements: 3.1, 3.2, 3.3, 4.3, 4.4_

- [x] 4.3 Add repeat-last-response functionality


  - Store last Spanish transcription and Gemini reply in global variables
  - On 'r' key, resend Gemini request with previous transcription
  - If no previous prompt exists, log `[USER_REPEAT]` with warning
  - Display repeated response and log `[GEMINI_REPLY_REPEAT]`
  - _Requirements: 5.1, 5.2, 5.3, 5.4_


- [x] 5. Implement comprehensive error handling and cleanup





  - Add try/catch blocks around all major operations
  - Handle KeyboardInterrupt for graceful shutdown
  - Implement proper thread cleanup and resource management
  - Add error logging for all failure scenarios
  - _Requirements: 4.1, 4.5_

- [x] 6. Integrate with existing tech stack and test end-to-end





  - Configure VB-Audio Virtual Cable as input source
  - Verify faster-whisper float16 integration
  - Test google-genai SDK integration
  - Validate dotenv configuration loading
  - Run complete workflow testing (speak → silence → transcribe → confirm → AI response)
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_
