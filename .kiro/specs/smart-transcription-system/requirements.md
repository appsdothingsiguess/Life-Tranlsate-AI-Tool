# Requirements Document

## Introduction

This feature transforms the Spanish oral exam transcription tool from silence-based detection to a continuous overlapping streaming transcription system. The system will provide real-time transcription with minimal latency suitable for live oral exams, eliminating the need to wait for silence periods and ensuring continuous feedback.

## Requirements

### Requirement 1

**User Story:** As a Spanish student using the oral exam tool during a live exam, I want continuous real-time transcription without waiting for silence, so that I can receive immediate feedback on my speech.

#### Acceptance Criteria

1. WHEN the system starts THEN it SHALL record audio at 48kHz mono from the capture device
2. WHEN 1.0 second has elapsed THEN the system SHALL extract the latest 3.0 seconds of audio from the rolling buffer
3. WHEN extracting audio chunks THEN the system SHALL resample each 3.0s slice from 48kHz to 16kHz before transcription
4. WHEN there is sufficient audio data THEN the system SHALL send the 3.0s chunk to Whisper for transcription without waiting for silence

### Requirement 2

**User Story:** As a Spanish student using the live transcription system, I want efficient background processing that doesn't block the main audio loop, so that transcription happens smoothly without audio dropouts.

#### Acceptance Criteria

1. WHEN transcription is needed THEN the system SHALL use a worker thread to perform transcription processing
2. WHEN the main audio loop is running THEN it SHALL NOT be blocked by transcription operations
3. WHEN there is less than 2.5 seconds of audio in the buffer THEN the system SHALL skip transcription
4. WHEN transcription is in progress THEN the system SHALL continue capturing new audio without interruption

### Requirement 3

**User Story:** As a Spanish student using the live transcription system, I want to avoid duplicate transcriptions from overlapping audio chunks, so that I receive clean, non-repetitive feedback.

#### Acceptance Criteria

1. WHEN processing overlapping audio chunks THEN the system SHALL implement content deduplication to avoid duplicate transcriptions
2. WHEN transcription results are similar to recent outputs THEN the system SHALL skip displaying duplicate content
3. WHEN implementing deduplication THEN the system SHALL use either content comparison or minimum time gap between outputs
4. WHEN transcription overlap occurs THEN the system SHALL handle it gracefully without confusing the user

### Requirement 4

**User Story:** As a user debugging or monitoring the streaming transcription system, I want detailed logging of all window-based transcription events, so that I can track system performance and transcription quality.

#### Acceptance Criteria

1. WHEN starting transcription of a 3s slice THEN the system SHALL log "[WINDOW_TRANSCRIBE] Started transcription for 3s slice" with timestamp
2. WHEN transcription completes THEN the system SHALL log "[WINDOW_RESULT] Transcription result: <text>" with timestamp
3. WHEN skipping transcription due to insufficient audio THEN the system SHALL log "[WINDOW_SKIP] Not enough audio data" with timestamp
4. WHEN any system event occurs THEN the system SHALL include timestamps in all log entries
5. WHEN the log file doesn't exist THEN the system SHALL create it automatically
6. WHEN logging transcription events THEN the system SHALL log to both console and log.txt file

### Requirement 5

**User Story:** As a Spanish student using the live transcription system, I want transcription to happen within approximately 1.5 seconds of speaking, so that I can receive near real-time feedback during my oral exam practice.

#### Acceptance Criteria

1. WHEN audio is captured THEN the system SHALL process and transcribe it within approximately 1.5 seconds
2. WHEN using Whisper for transcription THEN the system SHALL configure it with task="transcribe" and language="es"
3. WHEN transcription latency exceeds acceptable limits THEN the system SHALL log performance warnings
4. WHEN optimizing for live use THEN the system SHALL prioritize speed over perfect accuracy

### Requirement 6

**User Story:** As a user of the transcription system, I want reliable audio processing using the existing tech stack, so that the enhanced features work seamlessly with my current setup.

#### Acceptance Criteria

1. WHEN the system starts THEN it SHALL use VB-Audio Virtual Cable as the audio input source
2. WHEN processing audio THEN the system SHALL use faster-whisper with float16 precision
3. WHEN making API calls THEN the system SHALL use the latest google-genai SDK
4. WHEN handling environment variables THEN the system SHALL use dotenv for configuration
5. WHEN processing audio signals THEN the system SHALL use sounddevice and scipy for audio handling
6. WHEN running on Windows 11 with Python 3.11 THEN all components SHALL function correctly