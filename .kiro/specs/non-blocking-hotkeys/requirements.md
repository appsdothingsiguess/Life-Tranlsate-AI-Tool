# Requirements Document

## Introduction

This feature transforms the manual Gemini interaction system from blocking input prompts to a non-blocking keyboard listener with hotkeys. The system will allow users to trigger Gemini responses, reprint last replies, and access help without interrupting the continuous audio transcription loop, providing a seamless live exam experience.

## Requirements

### Requirement 1

**User Story:** As a Spanish student using the live transcription system, I want to press hotkeys to interact with Gemini without stopping the audio processing, so that I can get AI feedback while maintaining continuous transcription.

#### Acceptance Criteria

1. WHEN the system starts THEN it SHALL initialize a non-blocking keyboard listener thread
2. WHEN a user presses 'g' THEN the system SHALL send the most recent transcription to Gemini
3. WHEN a user presses 'r' THEN the system SHALL reprint the last Gemini reply
4. WHEN a user presses 'h' THEN the system SHALL show help/hotkeys information
5. WHEN a user presses 'q' THEN the system SHALL skip/ignore the current transcription
6. WHEN hotkeys are pressed THEN the main audio/transcribe loop SHALL continue uninterrupted

### Requirement 2

**User Story:** As a user of the hotkey system, I want key press debouncing to prevent accidental double triggers, so that I don't send duplicate requests to Gemini.

#### Acceptance Criteria

1. WHEN a hotkey is pressed THEN the system SHALL ignore repeat presses within 300ms
2. WHEN debouncing is active THEN the system SHALL log ignored repeat key presses
3. WHEN the debounce period expires THEN the system SHALL accept new key presses for that hotkey
4. WHEN implementing debouncing THEN the system SHALL track timing per individual key

### Requirement 3

**User Story:** As a user interacting with the system, I want shared state management for transcriptions and Gemini responses, so that hotkeys can access the latest data safely across threads.

#### Acceptance Criteria

1. WHEN transcription completes THEN the system SHALL store the result as `last_transcription` with timestamp
2. WHEN Gemini responds THEN the system SHALL store the response as `last_gemini_reply`
3. WHEN storing Gemini responses THEN the system SHALL maintain a deque of the last 5 replies
4. WHEN accessing shared state THEN the system SHALL use thread-safe mechanisms (locks or queues)
5. WHEN multiple threads access shared data THEN the system SHALL prevent race conditions

### Requirement 4

**User Story:** As a user pressing the 'g' hotkey, I want manual Gemini triggering with proper error handling, so that I can get AI responses reliably without system crashes.

#### Acceptance Criteria

1. WHEN 'g' is pressed AND no transcription is available THEN the system SHALL log "[HOTKEY_G] No transcription available"
2. WHEN 'g' is pressed AND a Gemini call is in progress THEN the system SHALL log "[HOTKEY_G] Busy" and ignore the request
3. WHEN 'g' is pressed AND transcription is available THEN the system SHALL enqueue the Gemini request
4. WHEN Gemini processing starts THEN the system SHALL log "[MANUAL_GEMINI]" with the request
5. WHEN Gemini responds THEN the system SHALL print the response in a visually distinct block
6. WHEN Gemini processing completes THEN the system SHALL log "[GEMINI_REPLY]" and "[GEMINI_RESPONSE_TIME]"

### Requirement 5

**User Story:** As a user of the system, I want clear visual display of Gemini responses, so that I can easily distinguish AI replies from transcription output.

#### Acceptance Criteria

1. WHEN displaying Gemini responses THEN the system SHALL use clear separators like "========= GEMINI RESPONSE ========="
2. WHEN 'r' is pressed THEN the system SHALL reprint the last reply using the same visual format
3. WHEN logging responses THEN the system SHALL include full response text and elapsed time
4. WHEN no previous reply exists THEN the system SHALL display an appropriate message
5. WHEN displaying responses THEN the system SHALL ensure readability in the console output

### Requirement 6

**User Story:** As a developer maintaining the system, I want comprehensive logging of hotkey events, so that I can monitor system behavior and debug issues.

#### Acceptance Criteria

1. WHEN hotkeys are pressed THEN the system SHALL log events with types "[HOTKEY_G]", "[HOTKEY_R]", "[HOTKEY_Q]", "[HOTKEY_H]"
2. WHEN Gemini requests are made THEN the system SHALL log request and response times
3. WHEN hotkey errors occur THEN the system SHALL log "[ERROR_HOTKEY]" without crashing the main pipeline
4. WHEN the keyboard listener thread crashes THEN the system SHALL log the error and keep the main system running
5. WHEN logging hotkey events THEN the system SHALL continue existing window/transcription logs

### Requirement 7

**User Story:** As a user of the system, I want robust error handling and race condition prevention, so that the hotkey system works reliably without interfering with transcription.

#### Acceptance Criteria

1. WHEN multiple 'g' presses occur THEN the system SHALL prevent multiple simultaneous Gemini calls using a busy flag or lock
2. WHEN EOF or keyboard interrupts occur THEN the system SHALL handle them cleanly and stop the listener thread
3. WHEN the hotkey thread crashes THEN the system SHALL log the error and maintain main pipeline operation
4. WHEN implementing thread safety THEN the system SHALL use appropriate synchronization mechanisms
5. WHEN handling errors THEN the system SHALL prioritize system stability over individual hotkey operations