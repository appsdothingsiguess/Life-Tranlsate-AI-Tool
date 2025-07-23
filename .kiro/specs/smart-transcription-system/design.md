# Design Document

## Overview

The overlapping streaming transcription system transforms the Spanish oral exam tool from silence-based detection to a continuous sliding window approach. The system captures audio at 48kHz, processes 3-second overlapping chunks every 1 second, resamples to 16kHz, and provides real-time transcription with minimal latency suitable for live oral exams.

## Architecture

The system follows a streaming architecture with continuous audio processing:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Audio Input   │───▶│ Rolling Buffer   │───▶│ Window Extractor│
│  (48kHz Mono)   │    │ (Continuous)     │    │ (Every 1.0s)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Resampler     │◀───│ Audio Validator  │◀───│ 3.0s Audio Slice│
│ (48kHz→16kHz)   │    │ (≥2.5s check)   │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                                              
         ▼                                              
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Worker Thread   │───▶│   Transcriber    │───▶│  Deduplicator   │
│ (Non-blocking)  │    │ (Whisper ES)     │    │ (Content Filter)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │     Logger      │    │ Result Display  │
                       │ (Window Events) │    │ (Live Output)   │
                       └─────────────────┘    └─────────────────┘
```

## Components and Interfaces

### 1. Streaming Audio Processor Component

**Purpose:** Handles continuous audio capture with rolling buffer and sliding window extraction

**Key Classes:**
- `StreamingAudioProcessor`: Main streaming audio handling class
- `RollingBuffer`: Manages continuous audio data buffering
- `WindowExtractor`: Extracts 3-second slices every 1 second

**Interfaces:**
```python
class StreamingAudioProcessor:
    def __init__(self, sample_rate: int, device_index: int, window_interval: float)
    def start_streaming(self) -> None
    def stop_streaming(self) -> None
    def set_window_callback(self, callback: Callable) -> None

class RollingBuffer:
    def __init__(self, max_duration: float, sample_rate: int)
    def append(self, audio_chunk: np.ndarray) -> None
    def get_latest_window(self, duration: float) -> np.ndarray
    def has_sufficient_data(self, min_duration: float) -> bool

class WindowExtractor:
    def __init__(self, window_duration: float, interval: float)
    def extract_window(self, buffer: RollingBuffer) -> Optional[np.ndarray]
    def should_extract(self) -> bool
```

### 2. Streaming Transcription Component

**Purpose:** Manages Whisper model integration for continuous window-based transcription

**Key Classes:**
- `StreamingTranscriptionService`: Handles Whisper model operations for streaming audio
- `WindowTranscriptionResult`: Data structure for window-based transcription results
- `ContentDeduplicator`: Filters duplicate transcriptions from overlapping windows

**Interfaces:**
```python
class StreamingTranscriptionService:
    def __init__(self, model_size: str, compute_type: str)
    def transcribe_window(self, audio: np.ndarray) -> WindowTranscriptionResult
    def configure_for_streaming(self, language: str, task: str) -> None

@dataclass
class WindowTranscriptionResult:
    spanish_text: str
    confidence: float
    timestamp: datetime
    window_duration: float
    is_duplicate: bool

class ContentDeduplicator:
    def __init__(self, similarity_threshold: float, min_time_gap: float)
    def is_duplicate(self, new_text: str, timestamp: datetime) -> bool
    def add_result(self, text: str, timestamp: datetime) -> None
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

### 4. Streaming Logging Component

**Purpose:** Provides comprehensive logging for window-based transcription events

**Key Classes:**
- `StreamingLogger`: Centralized logging service for streaming events
- `WindowLogEntry`: Structured log entry format for window events

**Interfaces:**
```python
class StreamingLogger:
    def __init__(self, log_file: str)
    def log_window_transcribe_start(self, window_duration: float) -> None
    def log_window_result(self, text: str, confidence: float) -> None
    def log_window_skip(self, reason: str) -> None
    def log_system_event(self, event: str) -> None

@dataclass
class WindowLogEntry:
    timestamp: datetime
    event_type: str  # WINDOW_TRANSCRIBE, WINDOW_RESULT, WINDOW_SKIP
    content: str
    window_metadata: Dict[str, Any]
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

### Streaming Audio Processing Models
```python
@dataclass
class StreamingAudioConfig:
    sample_rate: int = 48000
    target_rate: int = 16000
    channels: int = 1
    device_index: int
    window_duration: float = 3.0  # 3-second windows
    window_interval: float = 1.0  # Extract every 1 second
    min_audio_duration: float = 2.5  # Skip if less than 2.5s
    buffer_max_duration: float = 10.0  # Rolling buffer size

@dataclass
class AudioWindow:
    data: np.ndarray
    timestamp: datetime
    duration: float
    sample_rate: int
    is_resampled: bool
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

### Streaming Audio Processing Errors
- **Device Not Found**: Graceful fallback to default device with user notification
- **Audio Stream Interruption**: Automatic reconnection with rolling buffer preservation
- **Buffer Overflow**: Circular buffer management with oldest data eviction
- **Window Extraction Failure**: Skip current window and continue with next interval
- **Resampling Errors**: Log error and skip problematic audio window

### Streaming Transcription Errors
- **Model Loading Failure**: Clear error message with fallback options
- **Window Transcription Timeout**: Skip current window and continue processing
- **Insufficient Audio Data**: Log WINDOW_SKIP event and wait for next window
- **Worker Thread Failure**: Restart transcription worker with error logging

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