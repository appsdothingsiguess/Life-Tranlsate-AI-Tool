# Requirements Document

## Introduction

This feature enhances the existing Spanish oral exam transcription tool by implementing intelligent audio processing, proper sentence detection, controlled AI interaction, and comprehensive logging. The system will prevent over-transcription of partial speech, require explicit user confirmation before AI calls, and provide detailed logging of all interactions.

## Requirements

### Requirement 1

**User Story:** As a Spanish student using the oral exam tool, I want the system to detect complete sentences instead of fragmenting my speech, so that I get accurate transcriptions of full thoughts.

#### Acceptance Criteria

1. WHEN the user speaks THEN the system SHALL wait for 2-3 seconds of silence before processing the audio
2. WHEN audio contains mid-sentence pauses THEN the system SHALL NOT trigger transcription until the complete sentence is finished
3. WHEN a complete sentence is detected THEN the system SHALL transcribe the entire audio segment as one unit
4. WHEN the user is still speaking THEN the system SHALL continue buffering audio without premature transcription

### Requirement 2

**User Story:** As a Spanish student, I want to control when the AI generates responses, so that I can review my transcribed speech before getting AI feedback.

#### Acceptance Criteria

1. WHEN transcription is complete THEN the system SHALL display the transcribed text and wait for user confirmation
2. WHEN the user presses Enter THEN the system SHALL send the transcription to Gemini for response
3. WHEN no user confirmation is given THEN the system SHALL NOT automatically call the Gemini API
4. WHEN the user chooses not to proceed THEN the system SHALL allow them to continue with new speech input

### Requirement 3

**User Story:** As a Spanish student, I want the AI to provide concise, natural Spanish responses, so that I can practice realistic conversational exchanges.

#### Acceptance Criteria

1. WHEN sending prompts to Gemini THEN the system SHALL use the exact prompt: "You are a Spanish tutor. The student is preparing for an oral test. Reply only with a natural, brief Spanish sentence the student should say. Do not add explanations or alternatives. Just respond with the one sentence."
2. WHEN Gemini responds THEN the system SHALL use the "gemini-2.5-flash" model
3. WHEN receiving AI responses THEN the system SHALL display only the single Spanish sentence response
4. WHEN the AI provides verbose feedback THEN the system SHALL handle it appropriately by using the constrained prompt

### Requirement 4

**User Story:** As a user debugging or reviewing my practice sessions, I want comprehensive logging of all system activities, so that I can track what was transcribed and what responses were generated.

#### Acceptance Criteria

1. WHEN any transcription occurs THEN the system SHALL log the transcribed Spanish input to both console and log.txt file
2. WHEN Whisper provides translations THEN the system SHALL log the translation output
3. WHEN sending requests to Gemini THEN the system SHALL log the exact prompt and parameters sent
4. WHEN receiving Gemini responses THEN the system SHALL log the complete response received
5. WHEN any system event occurs THEN the system SHALL include timestamps in all log entries
6. WHEN the log file doesn't exist THEN the system SHALL create it automatically

### Requirement 5

**User Story:** As a user who wants to practice with previous prompts, I want the ability to repeat the last Gemini call, so that I can get alternative responses without speaking again.

#### Acceptance Criteria

1. WHEN the user presses a designated hotkey THEN the system SHALL repeat the last successful Gemini API call
2. WHEN no previous prompt exists THEN the system SHALL inform the user that no previous prompt is available
3. WHEN repeating a prompt THEN the system SHALL log this action clearly
4. WHEN using the repeat function THEN the system SHALL use the same prompt text as the previous call

### Requirement 6

**User Story:** As a user of the transcription system, I want reliable audio processing using the existing tech stack, so that the enhanced features work seamlessly with my current setup.

#### Acceptance Criteria

1. WHEN the system starts THEN it SHALL use VB-Audio Virtual Cable as the audio input source
2. WHEN processing audio THEN the system SHALL use faster-whisper with float16 precision
3. WHEN making API calls THEN the system SHALL use the latest google-genai SDK
4. WHEN handling environment variables THEN the system SHALL use dotenv for configuration
5. WHEN processing audio signals THEN the system SHALL use sounddevice and scipy for audio handling
6. WHEN running on Windows 11 with Python 3.11 THEN all components SHALL function correctly