# Design Document

## Overview

The smart transcription system enhances the existing Spanish oral exam tool by implementing intelligent audio processing with pause detection, controlled AI interaction, and comprehensive logging. The system transforms the current continuous processing approach into a sentence-aware, user-controlled workflow that provides better transcription accuracy and learning experience.

## Architecture

The system follows a modular architecture with clear separation of concerns:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Audio Input   │───▶│  Audio Processor │───▶│   Transcriber   │
│   (VB-Cable)    │    │  (Pause Detect)  │    │  (Whisper)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│     Logger      │◀───│  Main Controller │◀───│ User Interface  │
│   (File + CLI)  │    │   (Orchestrator) │    │ (Confirmation)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   AI Interface  │
                       │   (Gemini API)  │
                       └─────────────────┘
```

## Components and Interfaces

### 1. Audio Processor Component

**Purpose:** Handles real-time audio capture with intelligent pause detection

**Key Classes:**
- `AudioProcessor`: Main audio handling class
- `PauseDetector`: Implements silence detection algorithm
- `AudioBuffer`: Manages audio data buffering

**Interfaces:**
```python
class AudioProcessor:
    def __init__(self, sample_rate: int, device_index: int, pause_threshold: float)
    def start_listening(self) -> None
    def stop_listening(self) -> None
    def set_audio_callback(self, callback: Callable) -> None

class PauseDetector:
    def __init__(self, silence_duration: float, threshold: float)
    def detect_pause(self, audio_chunk: np.ndarray) -> bool
    def reset(self) -> None
```

### 2. Transcription Component

**Purpose:** Manages Whisper model integration and transcription processing

**Key Classes:**
- `TranscriptionService`: Handles Whisper model operations
- `TranscriptionResult`: Data structure for transcription results

**Interfaces:**
```python
class TranscriptionService:
    def __init__(self, model_size: str, compute_type: str)
    def transcribe(self, audio: np.ndarray) -> TranscriptionResult
    def translate(self, audio: np.ndarray) -> TranscriptionResult

@dataclass
class TranscriptionResult:
    spanish_text: str
    english_translation: str
    confidence: float
    timestamp: datetime
```

### 3. AI Interface Component

**Purpose:** Manages Gemini API interactions with proper prompt formatting

**Key Classes:**
- `GeminiService`: Handles API calls and response processing
- `PromptManager`: Manages prompt templates and history

**Interfaces:**
```python
class GeminiService:
    def __init__(self, api_key: str, model: str)
    def generate_response(self, prompt: str) -> str
    def repeat_last_prompt(self) -> str

class PromptManager:
    def format_spanish_tutor_prompt(self, english_input: str) -> str
    def get_last_prompt(self) -> Optional[str]
    def store_prompt(self, prompt: str) -> None
```

### 4. Logging Component

**Purpose:** Provides comprehensive logging to file and console

**Key Classes:**
- `SystemLogger`: Centralized logging service
- `LogEntry`: Structured log entry format

**Interfaces:**
```python
class SystemLogger:
    def __init__(self, log_file: str)
    def log_transcription(self, spanish: str, english: str) -> None
    def log_ai_request(self, prompt: str) -> None
    def log_ai_response(self, response: str) -> None
    def log_system_event(self, event: str) -> None

@dataclass
class LogEntry:
    timestamp: datetime
    event_type: str
    content: str
    metadata: Dict[str, Any]
```

### 5. User Interface Component

**Purpose:** Handles user interactions and confirmations

**Key Classes:**
- `UserInterface`: Manages console interactions
- `HotkeyManager`: Handles keyboard shortcuts

**Interfaces:**
```python
class UserInterface:
    def display_transcription(self, result: TranscriptionResult) -> None
    def confirm_ai_request(self) -> bool
    def display_ai_response(self, response: str) -> None

class HotkeyManager:
    def register_hotkey(self, key: str, callback: Callable) -> None
    def start_listening(self) -> None
```

## Data Models

### Audio Processing Models
```python
@dataclass
class AudioConfig:
    sample_rate: int = 48000
    target_rate: int = 16000
    channels: int = 1
    device_index: int
    block_duration: float = 2.0
    window_duration: float = 4.0
    pause_threshold: float = 2.5  # seconds of silence

@dataclass
class AudioSegment:
    data: np.ndarray
    timestamp: datetime
    duration: float
    is_complete_sentence: bool
```

### Transcription Models
```python
@dataclass
class WhisperConfig:
    model_size: str = "small"
    compute_type: str = "float16"
    beam_size: int = 5
    temperature: float = 0.0
    vad_filter: bool = True
    no_speech_threshold: float = 0.5
```

### AI Service Models
```python
@dataclass
class GeminiConfig:
    model: str = "gemini-2.5-flash"
    api_key: str
    max_retries: int = 3
    timeout: float = 30.0

@dataclass
class AIResponse:
    content: str
    timestamp: datetime
    prompt_used: str
    model: str
    success: bool
```

## Error Handling

### Audio Processing Errors
- **Device Not Found**: Graceful fallback to default device with user notification
- **Audio Stream Interruption**: Automatic reconnection with buffered audio preservation
- **Buffer Overflow**: Intelligent buffer management with oldest data eviction

### Transcription Errors
- **Model Loading Failure**: Clear error message with fallback options
- **Transcription Timeout**: Configurable timeout with partial result handling
- **Empty Audio**: Skip processing with appropriate logging

### AI Service Errors
- **API Key Issues**: Clear authentication error messages
- **Rate Limiting**: Exponential backoff with user notification
- **Network Failures**: Retry logic with offline mode indication
- **Empty Responses**: Fallback messaging with error logging

### Logging Errors
- **File Permission Issues**: Fallback to console-only logging
- **Disk Space**: Log rotation and cleanup mechanisms
- **Concurrent Access**: Thread-safe logging implementation

## Testing Strategy

### Unit Testing
- **Audio Processing**: Mock audio input for pause detection testing
- **Transcription Service**: Test with known audio samples
- **AI Interface**: Mock API responses for various scenarios
- **Logging**: Verify log format and file operations

### Integration Testing
- **End-to-End Flow**: Complete sentence processing pipeline
- **Error Scenarios**: Network failures, device disconnections
- **Performance**: Audio processing latency and memory usage

### Manual Testing
- **Real Audio Input**: Test with actual Spanish speech
- **User Interaction**: Confirm Enter key behavior
- **Hotkey Functionality**: Test repeat prompt feature
- **Log Verification**: Ensure all events are properly logged

### Test Data
- **Audio Samples**: Various Spanish sentences with different pause patterns
- **API Responses**: Mock Gemini responses for consistent testing
- **Error Conditions**: Simulated failures for robustness testing

## Performance Considerations

### Audio Processing
- **Buffer Management**: Efficient circular buffer implementation
- **Real-time Processing**: Non-blocking audio callback design
- **Memory Usage**: Automatic cleanup of processed audio segments

### AI API Optimization
- **Request Batching**: Single API call per confirmed transcription
- **Response Caching**: Store last prompt for repeat functionality
- **Connection Pooling**: Reuse HTTP connections for API calls

### Logging Optimization
- **Asynchronous Logging**: Non-blocking log writes
- **Log Rotation**: Automatic file size management
- **Structured Format**: JSON logging for easy parsing