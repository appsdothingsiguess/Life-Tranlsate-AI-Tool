# Design Document

## Overview

The non-blocking hotkeys system transforms the current manual Gemini interaction from blocking `input()` prompts to a seamless keyboard listener that operates alongside the continuous audio transcription loop. The system uses a dedicated keyboard listener thread with thread-safe state management to provide instant hotkey responses without interrupting audio processing.

## Architecture

### Thread Architecture
The system extends the existing 3-thread architecture:
1. **Main Thread**: Audio capture and streaming buffer management
2. **Transcription Worker Thread**: Audio processing and Whisper transcription (existing)
3. **Keyboard Listener Thread**: Non-blocking hotkey capture and processing (new)
4. **Gemini Worker Thread**: Asynchronous Gemini API calls (new)

### Key Components
- **HotkeyListener**: Dedicated thread for non-blocking keyboard input
- **SharedState**: Thread-safe storage for transcriptions and Gemini responses
- **GeminiWorker**: Asynchronous Gemini processing with busy state management
- **ResponseDisplay**: Formatted console output for Gemini responses

## Components and Interfaces

### 1. HotkeyListener Class

```python
class HotkeyListener:
    def __init__(self, shared_state, gemini_worker):
        self.shared_state = shared_state
        self.gemini_worker = gemini_worker
        self.running = True
        self.debounce_timers = {}  # Per-key debounce tracking
        self.debounce_delay = 0.3  # 300ms debounce
        
    def start_listener(self):
        # Start keyboard listener thread using msvcrt for Windows
        
    def handle_keypress(self, key):
        # Process individual key presses with debouncing
        
    def is_debounced(self, key):
        # Check if key is within debounce window
```

**Implementation Strategy**: Use Windows `msvcrt.kbhit()` and `msvcrt.getch()` for non-blocking keyboard input, as it's native to Windows and doesn't require additional dependencies.

### 2. SharedState Class

```python
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.last_transcription = ""
        self.last_transcription_timestamp = None
        self.last_gemini_reply = ""
        self.gemini_history = deque(maxlen=5)  # Last 5 replies
        
    def update_transcription(self, text, timestamp):
        # Thread-safe transcription update
        
    def update_gemini_reply(self, reply):
        # Thread-safe Gemini response update
        
    def get_last_transcription(self):
        # Thread-safe transcription retrieval
```

### 3. GeminiWorker Class

```python
class GeminiWorker:
    def __init__(self, shared_state):
        self.shared_state = shared_state
        self.busy = False
        self.busy_lock = threading.Lock()
        self.request_queue = queue.Queue()
        
    def is_busy(self):
        # Check if Gemini call is in progress
        
    def enqueue_request(self, transcription_text):
        # Add Gemini request to processing queue
        
    def process_requests(self):
        # Worker thread method for processing Gemini requests
```

### 4. ResponseDisplay Class

```python
class ResponseDisplay:
    @staticmethod
    def print_gemini_response(response_text, is_repeat=False):
        # Print with visual separators
        separator = "========= GEMINI RESPONSE ========="
        print(f"\n{separator}")
        print(response_text)
        print("=" * len(separator))
        
    @staticmethod
    def print_help():
        # Display hotkey help information
        
    @staticmethod
    def print_history(history):
        # Display last 5 Gemini responses
```

## Data Models

### Hotkey Actions
```python
class HotkeyAction(Enum):
    GEMINI_REQUEST = "g"
    REPEAT_REPLY = "r"
    SHOW_HELP = "h"
    SKIP_IGNORE = "q"
```

### Shared State Structure
```python
@dataclass
class TranscriptionData:
    text: str
    timestamp: datetime
    
@dataclass
class GeminiResponse:
    text: str
    timestamp: datetime
    response_time: float
    is_repeat: bool
```

## Error Handling

### Keyboard Listener Resilience
- **EOF/Interrupt Handling**: Graceful shutdown on Ctrl+C or EOF
- **Thread Crash Recovery**: Log errors and continue main pipeline operation
- **Key Processing Errors**: Log individual key processing failures without stopping listener

### Gemini Integration Safety
- **Busy State Protection**: Prevent multiple simultaneous Gemini calls
- **API Error Handling**: Retry logic and graceful degradation
- **Queue Management**: Handle full queues and processing failures

### Thread Safety Measures
- **Lock Hierarchy**: Consistent lock ordering to prevent deadlocks
- **Timeout Handling**: Prevent indefinite blocking on shared resources
- **Resource Cleanup**: Proper thread shutdown and resource deallocation

## Testing Strategy

### Unit Tests
1. **HotkeyListener Tests**
   - Key debouncing functionality
   - Thread lifecycle management
   - Error handling scenarios

2. **SharedState Tests**
   - Thread-safe read/write operations
   - Concurrent access scenarios
   - Data consistency validation

3. **GeminiWorker Tests**
   - Busy state management
   - Queue processing logic
   - API error handling

### Integration Tests
1. **End-to-End Hotkey Flow**
   - Complete 'g' key press to Gemini response cycle
   - Repeat functionality with 'r' key
   - Help display with 'h' key

2. **Concurrent Operation Tests**
   - Hotkeys during active transcription
   - Multiple rapid key presses
   - System stability under load

### Performance Tests
1. **Latency Measurements**
   - Key press to action response time
   - Gemini request processing time
   - System resource usage

2. **Stress Testing**
   - Rapid key press sequences
   - Long-running operation stability
   - Memory usage over time

## Implementation Phases

### Phase 1: Core Infrastructure
- Implement SharedState class with thread-safe operations
- Create HotkeyListener with basic key capture
- Add logging infrastructure for hotkey events

### Phase 2: Gemini Integration
- Implement GeminiWorker with busy state management
- Add request queuing and processing
- Integrate with existing Gemini API calls

### Phase 3: User Interface
- Implement ResponseDisplay with formatted output
- Add help system and history display
- Integrate with existing transcription display

### Phase 4: Error Handling & Polish
- Add comprehensive error handling
- Implement graceful shutdown procedures
- Performance optimization and testing

## Integration with Existing System

### Transcription Integration
- Replace current `display_transcription_and_prompt()` blocking calls
- Update `streaming_transcription_worker()` to use SharedState
- Maintain existing logging and duplicate filtering

### State Management Migration
- Migrate `last_spanish` and `last_gemini_response` to SharedState
- Update existing user input worker to use new architecture
- Preserve existing Gemini API integration

### Logging Continuity
- Extend existing logging system with hotkey events
- Maintain current log format and file handling
- Add performance metrics for hotkey response times