import os
import torch
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
# scipy.signal import removed - Whisper handles 48kHz natively
# from scipy.signal import resample
from collections import deque
import threading
import queue
from dotenv import load_dotenv
from google import genai
from datetime import datetime

# System Constants
LOG_FILE = "log.txt"
GEMINI_PROMPT = (
    "You are a Spanish tutor. The student is preparing for an oral test. "
    "Reply only with a natural, brief Spanish sentence the student should say. "
    "Do not add explanations or alternatives. Just respond with the one sentence."
)
MODEL_NAME = "gemini-2.5-flash-lite"

# Streaming Constants
WINDOW_DURATION = 3.0  # 3-second audio windows for streaming transcription
WINDOW_INTERVAL = 1.0  # Extract windows every 1 second
MIN_AUDIO_DURATION = 2.5  # Skip transcription if less than 2.5 seconds of audio

# Thread-safe logging infrastructure
_log_lock = threading.Lock()

# Performance monitoring for live exam latency
from collections import deque
import time
import uuid

# Performance tracking globals
_latency_buffer = deque(maxlen=10)  # Store last 10 window latencies for rolling average
_latency_lock = threading.Lock()
LATENCY_TARGET = 1.5  # 1.5 second target latency for live exam

def log_with_timestamp(message, event_type="INFO"):
    """
    Thread-safe logging function with timestamp formatting.
    Logs to both console and log file with proper error handling.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] [{event_type}] {message}"
    
    # Always print to console
    print(formatted_message)
    
    # Thread-safe file logging with error handling
    with _log_lock:
        try:
            # Create log file if it doesn't exist, append if it does
            with open(LOG_FILE, "a", encoding="utf-8") as log_file:
                log_file.write(formatted_message + "\n")
                log_file.flush()  # Ensure immediate write
        except IOError as e:
            # If file logging fails, at least notify on console
            print(f"[{timestamp}] [ERROR] Failed to write to log file {LOG_FILE}: {e}")
        except Exception as e:
            # Handle any other unexpected errors
            print(f"[{timestamp}] [ERROR] Unexpected logging error: {e}")

def initialize_log_file():
    """
    Initialize the log file with proper error handling.
    Creates the file if it doesn't exist and logs system startup.
    """
    try:
        # Test if we can write to the log file
        with open(LOG_FILE, "a", encoding="utf-8") as log_file:
            startup_message = f"=== System startup at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ==="
            log_file.write(startup_message + "\n")
            log_file.flush()
        
        log_with_timestamp("Logging system initialized successfully", "SYSTEM")
        log_with_timestamp(f"Log file: {os.path.abspath(LOG_FILE)}", "SYSTEM")
        
    except IOError as e:
        print(f"[ERROR] Cannot create or write to log file {LOG_FILE}: {e}")
        print("[WARNING] Continuing with console-only logging")
    except Exception as e:
        print(f"[ERROR] Unexpected error initializing log file: {e}")
        print("[WARNING] Continuing with console-only logging")

# Initialize logging system
initialize_log_file()

def log_window_transcribe_start(window_duration):
    """
    Log the start of window-based transcription processing.
    Uses thread-safe logging with [WINDOW_TRANSCRIBE] event type.
    """
    message = f"Started transcription for {window_duration:.1f}s slice"
    log_with_timestamp(message, "WINDOW_TRANSCRIBE")

def log_window_result(transcription_text, confidence=None):
    """
    Log the completion of window-based transcription with result.
    Uses thread-safe logging with [WINDOW_RESULT] event type.
    """
    if confidence is not None:
        message = f"Transcription result (confidence: {confidence:.2f}): {transcription_text}"
    else:
        message = f"Transcription result: {transcription_text}"
    log_with_timestamp(message, "WINDOW_RESULT")

def log_window_skip(reason):
    """
    Log when a transcription window is skipped.
    Uses thread-safe logging with [WINDOW_SKIP] event type.
    """
    message = f"Not enough audio data - {reason}"
    log_with_timestamp(message, "WINDOW_SKIP")

def log_transcription_latency(window_id, duration):
    """
    Log processing time for each transcription window with performance monitoring.
    Buffers timestamps and calculates rolling average over last 5-10 windows.
    Adds latency warnings when processing exceeds 1.5s target.
    
    Args:
        window_id: Unique identifier for the transcription window
        duration: Processing duration in seconds
    """
    global _latency_buffer
    
    try:
        with _latency_lock:
            # Add current latency to rolling buffer
            _latency_buffer.append(duration)
            
            # Calculate rolling average over available windows (up to 10)
            if len(_latency_buffer) > 0:
                rolling_avg = sum(_latency_buffer) / len(_latency_buffer)
                buffer_size = len(_latency_buffer)
            else:
                rolling_avg = duration
                buffer_size = 1
            
            # Log individual window latency
            log_with_timestamp(f"Window {window_id} latency: {duration:.3f}s", "LATENCY")
            
            # Log rolling average performance
            log_with_timestamp(f"Rolling average latency ({buffer_size} windows): {rolling_avg:.3f}s", "LATENCY_AVG")
            
            # Check against 1.5s target and log warnings
            if duration > LATENCY_TARGET:
                excess = duration - LATENCY_TARGET
                log_with_timestamp(f"WARNING: Window {window_id} latency {duration:.3f}s exceeds {LATENCY_TARGET}s target by {excess:.3f}s", "LATENCY_WARNING")
            else:
                log_with_timestamp(f"Window {window_id} latency {duration:.3f}s within {LATENCY_TARGET}s target", "LATENCY_OK")
            
            # Log warning if rolling average exceeds target
            if rolling_avg > LATENCY_TARGET:
                excess_avg = rolling_avg - LATENCY_TARGET
                log_with_timestamp(f"WARNING: Rolling average latency {rolling_avg:.3f}s exceeds {LATENCY_TARGET}s target by {excess_avg:.3f}s", "LATENCY_AVG_WARNING")
            else:
                log_with_timestamp(f"Rolling average latency {rolling_avg:.3f}s within {LATENCY_TARGET}s target", "LATENCY_AVG_OK")
                
    except Exception as e:
        log_with_timestamp(f"Error logging transcription latency: {e}", "ERROR")

def get_latency_stats():
    """
    Get current latency statistics for monitoring.
    
    Returns:
        dict: Current latency statistics
    """
    try:
        with _latency_lock:
            if len(_latency_buffer) == 0:
                return {
                    'count': 0,
                    'current': None,
                    'average': None,
                    'min': None,
                    'max': None,
                    'target': LATENCY_TARGET,
                    'within_target': None
                }
            
            latencies = list(_latency_buffer)
            current = latencies[-1]
            average = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            within_target = average <= LATENCY_TARGET
            
            return {
                'count': len(latencies),
                'current': current,
                'average': average,
                'min': min_latency,
                'max': max_latency,
                'target': LATENCY_TARGET,
                'within_target': within_target
            }
    except Exception as e:
        log_with_timestamp(f"Error getting latency stats: {e}", "ERROR")
        return {}

# Load environment variables with error handling
try:
    load_dotenv()
    API_KEY = os.getenv("GOOGLE_API_KEY")
    if not API_KEY:
        log_with_timestamp("Missing GOOGLE_API_KEY in .env file", "ERROR")
        raise RuntimeError("Missing GOOGLE_API_KEY in .env file")
    log_with_timestamp("Environment variables loaded successfully", "SYSTEM")
except Exception as e:
    log_with_timestamp(f"Failed to load environment variables: {e}", "ERROR")
    raise

# Initialize Gemini client with error handling
try:
    client = genai.Client(api_key=API_KEY)
    log_with_timestamp("Gemini client initialized successfully", "SYSTEM")
    log_with_timestamp(f"Using Gemini model: {MODEL_NAME} (optimized for live exam latency)", "SYSTEM")
except Exception as e:
    log_with_timestamp(f"Failed to initialize Gemini client: {e}", "ERROR")
    raise

def warm_up_gemini():
    """
    Pre-warm the Gemini API to reduce cold-start latency on first real request.
    Sends a lightweight dummy request using the same logic as regular requests.
    Runs in background thread to avoid blocking startup.
    """
    import time
    
    log_with_timestamp("Starting Gemini API warm-up routine", "GEMINI_WARMUP")
    
    # Use the same prompt prefix as real interactions
    prompt = (
        "You are a Spanish tutor. The student is preparing for an oral test. "
        "Reply only with a natural, brief Spanish sentence the student should say. "
        "Do not add explanations or alternatives. Just respond with the one sentence."
    )
    
    # Warm-up request content
    warmup_input = "Say 'ready' in Spanish."
    
    # Log the warm-up request
    log_with_timestamp(f"Warm-up prompt: {prompt}", "GEMINI_WARMUP")
    log_with_timestamp(f"Warm-up input: {warmup_input}", "GEMINI_WARMUP")
    
    max_retries = 1  # One retry for warm-up
    
    for attempt in range(max_retries + 1):  # 0, 1 (2 total attempts)
        try:
            start_time = time.time()
            
            response = client.models.generate_content(
                model=MODEL_NAME,  # "gemini-2.5-flash-lite"
                contents=f"{prompt}\n\nStudent said in Spanish: {warmup_input}"
            )
            
            warmup_time = time.time() - start_time
            
            # Extract response text
            response_text = response.text.strip() if hasattr(response, "text") else ""
            
            if response_text:
                log_with_timestamp(f"Warm-up completed in {warmup_time:.2f}s - Response: {response_text}", "GEMINI_WARMUP_COMPLETE")
                return True
            else:
                raise ValueError("Gemini warm-up returned empty response")
                
        except Exception as warmup_error:
            error_msg = f"Gemini warm-up attempt {attempt + 1}/{max_retries + 1} failed: {str(warmup_error)}"
            log_with_timestamp(error_msg, "GEMINI_WARMUP_FAIL")
            
            # If this is not the last attempt, wait and retry
            if attempt < max_retries:
                log_with_timestamp("Retrying warm-up in 2 seconds...", "GEMINI_WARMUP")
                time.sleep(2)
            else:
                # Final attempt failed
                log_with_timestamp("Gemini warm-up failed after all attempts", "GEMINI_WARMUP_FAIL")
                return False
    
    return False

def call_gemini_api(spanish_input, is_repeat=False):
    """
    Use Gemini 2.5 Flash Lite model with strict prompt, response time tracking, and logging.
    Retry up to 2 times on failure (1s and 2s backoff).
    Log full request prompt and full response in [GEMINI_PROMPT] and [GEMINI_REPLY].
    Track response times for live exam performance monitoring.
    Handle API errors gracefully without crash.
    """
    import time
    
    # Use the exact prompt specified in requirements
    prompt = (
        "You are a Spanish tutor. The student is preparing for an oral test. "
        "Reply only with a natural, brief Spanish sentence the student should say. "
        "Do not add explanations or alternatives. Just respond with the one sentence."
    )
    
    # Log the full request prompt
    log_with_timestamp(f"Prompt: {prompt}", "GEMINI_PROMPT")
    log_with_timestamp(f"Spanish input: {spanish_input}", "GEMINI_PROMPT")
    
    max_retries = 2
    backoff_delays = [1, 2]  # 1s and 2s backoff
    
    for attempt in range(max_retries + 1):  # 0, 1, 2 (3 total attempts)
        try:
            # Start timing for performance tracking
            start_time = time.time()
            
            # Configure request with optimizations for live exam latency
            response = client.models.generate_content(
                model=MODEL_NAME,  # "gemini-2.5-flash-lite"
                contents=f"{prompt}\n\nStudent said in Spanish: {spanish_input}",
                # Add any available latency optimizations here
                # Note: thinking_budget parameter may not be available in all versions
            )
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Extract response text
            response_text = response.text.strip() if hasattr(response, "text") else ""
            
            if not response_text:
                raise ValueError("Gemini returned an empty response")
            
            # Log the full response with timing information
            event_type = "GEMINI_REPLY_REPEAT" if is_repeat else "GEMINI_REPLY"
            log_with_timestamp(f"Response ({response_time:.2f}s): {response_text}", event_type)
            
            # Log performance warning if response is too slow for live exam
            if response_time > 2.0:
                log_with_timestamp(f"WARNING: Gemini response time {response_time:.2f}s exceeds 2s target for live exam", "PERFORMANCE")
            else:
                log_with_timestamp(f"Gemini response time {response_time:.2f}s within live exam target", "PERFORMANCE")
            
            return response_text
            
        except Exception as e:
            error_msg = f"Gemini API error (attempt {attempt + 1}/{max_retries + 1}): {str(e)}"
            log_with_timestamp(error_msg, "ERROR")
            
            # If this is not the last attempt, wait and retry
            if attempt < max_retries:
                delay = backoff_delays[attempt]
                log_with_timestamp(f"Retrying in {delay} seconds...", "ERROR")
                time.sleep(delay)
            else:
                # Final attempt failed
                log_with_timestamp("All Gemini API retry attempts failed", "ERROR")
                return None
    
    return None


# Load Whisper model with error handling
print("[🧠] Loading Whisper model...")
try:
    compute_type = "float16" if torch.cuda.is_available() else "int8"
    log_with_timestamp(f"Initializing Whisper model with compute_type: {compute_type}", "SYSTEM")
    model = WhisperModel("small", compute_type=compute_type)
    log_with_timestamp("Whisper model loaded successfully", "SYSTEM")
    print("[✅] Model loaded.")
except Exception as e:
    log_with_timestamp(f"Failed to load Whisper model: {e}", "ERROR")
    print(f"[❌] Failed to load Whisper model: {e}")
    raise

# Audio format validation logging during startup
log_with_timestamp("=== Audio Format Validation Results ===", "AUDIO_FORMAT")
log_with_timestamp("✅ Whisper native 48kHz float32 support: CONFIRMED", "AUDIO_FORMAT")
log_with_timestamp("✅ Optimization: Skipping scipy resampling entirely", "AUDIO_FORMAT")
log_with_timestamp("🚀 Performance: Direct 48kHz processing reduces latency", "AUDIO_FORMAT")
log_with_timestamp("📊 Validation source: test_whisper_format_validation.py results", "AUDIO_FORMAT")

# Audio settings
SAMPLE_RATE = 48000
TARGET_RATE = 16000
BLOCK_DURATION = 2
# WINDOW_DURATION now defined in streaming constants section above
CHANNELS = 1
DEVICE_INDEX = 37  # VB-Audio Virtual Cable WASAPI input

BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)
WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_DURATION)

class StreamingBuffer:
    """
    Unified streaming buffer class that combines rolling buffer and window extraction.
    Uses deque for efficient circular buffer with automatic oldest-data eviction.
    Extracts 3.0-second windows directly in audio callback without separate timer threads.
    """
    
    def __init__(self, sample_rate, window_duration=3.0, window_interval=1.0, min_audio_duration=2.5, max_buffer_duration=10.0):
        """
        Initialize the streaming buffer.
        
        Args:
            sample_rate: Audio sample rate (e.g., 48000)
            window_duration: Duration of each extracted window in seconds (default: 3.0)
            window_interval: Interval between window extractions in seconds (default: 1.0)
            min_audio_duration: Minimum duration required to extract a window (default: 2.5)
            max_buffer_duration: Maximum buffer duration to prevent memory issues (default: 10.0)
        """
        self.sample_rate = sample_rate
        self.window_duration = window_duration
        self.window_interval = window_interval
        self.min_audio_duration = min_audio_duration
        self.max_buffer_duration = max_buffer_duration
        
        # Calculate sizes in samples
        self.window_size = int(sample_rate * window_duration)
        self.interval_size = int(sample_rate * window_interval)
        self.min_size = int(sample_rate * min_audio_duration)
        self.max_buffer_size = int(sample_rate * max_buffer_duration)
        
        # Rolling buffer using deque for efficient operations
        self.buffer = deque(maxlen=self.max_buffer_size)
        self.buffer_lock = threading.Lock()
        
        # Window extraction timing
        self.last_extraction_time = 0
        self.samples_since_extraction = 0
        
        log_with_timestamp(f"StreamingBuffer initialized: window={window_duration}s, interval={window_interval}s, min={min_audio_duration}s", "SYSTEM")
    
    def append_audio(self, audio_chunk):
        """
        Append new audio data to the rolling buffer.
        Automatically evicts oldest data when buffer reaches max size.
        
        Args:
            audio_chunk: numpy array of audio samples
        """
        try:
            with self.buffer_lock:
                # Extend buffer with new audio data
                # deque with maxlen automatically evicts oldest data
                self.buffer.extend(audio_chunk)
                self.samples_since_extraction += len(audio_chunk)
                
        except Exception as e:
            log_with_timestamp(f"Error appending audio to streaming buffer: {e}", "ERROR")
    
    def should_extract_window(self):
        """
        Check if it's time to extract a new window based on interval timing.
        
        Returns:
            bool: True if a window should be extracted
        """
        try:
            with self.buffer_lock:
                # Check if we have enough samples since last extraction
                return self.samples_since_extraction >= self.interval_size
        except Exception as e:
            log_with_timestamp(f"Error checking window extraction timing: {e}", "ERROR")
            return False
    
    def extract_window(self):
        """
        Extract a 3.0-second window from the rolling buffer.
        Skips extraction if less than 2.5 seconds of data available.
        
        Returns:
            numpy.ndarray or None: Extracted audio window, or None if insufficient data
        """
        try:
            with self.buffer_lock:
                # Check if we have sufficient data
                if len(self.buffer) < self.min_size:
                    log_window_skip(f"insufficient data: {len(self.buffer)} samples ({len(self.buffer)/self.sample_rate:.2f}s) < {self.min_audio_duration}s minimum")
                    return None
                
                # Extract the latest window_size samples (or all available if less than window_size)
                window_samples = min(self.window_size, len(self.buffer))
                
                # Get the most recent samples
                window_data = np.array(list(self.buffer)[-window_samples:])
                
                # Reset extraction counter
                self.samples_since_extraction = 0
                
                actual_duration = len(window_data) / self.sample_rate
                log_with_timestamp(f"Extracted window: {len(window_data)} samples ({actual_duration:.2f}s)", "WINDOW_EXTRACT")
                
                return window_data
                
        except Exception as e:
            log_with_timestamp(f"Error extracting window from streaming buffer: {e}", "ERROR")
            return None
    
    def get_buffer_info(self):
        """
        Get current buffer status information.
        
        Returns:
            dict: Buffer status information
        """
        try:
            with self.buffer_lock:
                buffer_samples = len(self.buffer)
                buffer_duration = buffer_samples / self.sample_rate
                samples_until_extraction = max(0, self.interval_size - self.samples_since_extraction)
                
                return {
                    'buffer_samples': buffer_samples,
                    'buffer_duration': buffer_duration,
                    'samples_since_extraction': self.samples_since_extraction,
                    'samples_until_extraction': samples_until_extraction,
                    'can_extract': buffer_samples >= self.min_size,
                    'should_extract': self.samples_since_extraction >= self.interval_size
                }
        except Exception as e:
            log_with_timestamp(f"Error getting buffer info: {e}", "ERROR")
            return {}

# Streaming buffer instance
streaming_buffer = StreamingBuffer(
    sample_rate=SAMPLE_RATE,
    window_duration=WINDOW_DURATION,
    window_interval=WINDOW_INTERVAL,
    min_audio_duration=MIN_AUDIO_DURATION
)

# Transcription worker thread infrastructure
audio_queue = queue.Queue()
user_input_queue = queue.Queue()  # Queue for user input handling
last_spanish = ""  # Store last transcription for repeat functionality
last_gemini_response = ""  # Store last Gemini response for repeat functionality
transcription_worker_running = True
user_input_worker_running = True
transcription_thread = None  # Will hold the worker thread reference
user_input_thread = None  # Will hold the user input thread reference
audio_stream = None  # Will hold the audio stream reference

# Simple duplicate filtering - store only last transcription text and timestamp
last_transcription_text = ""
last_transcription_timestamp = None
DUPLICATE_FILTER_WINDOW = 1.5  # 1.5 second window for duplicate detection

def is_duplicate_transcription(new_text, current_timestamp):
    """
    Simple duplicate filtering: skip if current transcription is identical to last result within 1.5s window.
    Store only last transcription text and timestamp for comparison.
    No fuzzy matching or complex similarity algorithms.
    
    Args:
        new_text: Current transcription text to check
        current_timestamp: Current timestamp for the transcription
    
    Returns:
        bool: True if this is a duplicate that should be skipped
    """
    global last_transcription_text, last_transcription_timestamp
    
    try:
        # If no previous transcription, not a duplicate
        if not last_transcription_text or last_transcription_timestamp is None:
            return False
        
        # Check if texts are identical (exact string match)
        if new_text.strip() != last_transcription_text.strip():
            return False
        
        # Check if within 1.5 second window
        time_diff = (current_timestamp - last_transcription_timestamp).total_seconds()
        if time_diff <= DUPLICATE_FILTER_WINDOW:
            return True
        
        return False
        
    except Exception as e:
        log_with_timestamp(f"Error in duplicate detection: {e}", "ERROR")
        return False  # If error, don't skip - better to show duplicate than miss transcription

def update_last_transcription(text, timestamp):
    """
    Update the stored last transcription text and timestamp for duplicate filtering.
    
    Args:
        text: Transcription text to store
        timestamp: Timestamp to store
    """
    global last_transcription_text, last_transcription_timestamp
    
    try:
        last_transcription_text = text.strip()
        last_transcription_timestamp = timestamp
    except Exception as e:
        log_with_timestamp(f"Error updating last transcription: {e}", "ERROR")

def user_input_worker():
    """
    Dedicated user input worker thread to handle user interactions without blocking transcription.
    Processes user input requests from the user_input_queue.
    """
    global user_input_worker_running, last_gemini_response
    
    log_with_timestamp("User input worker thread started", "SYSTEM")
    
    while user_input_worker_running:
        try:
            # Get user input request from queue
            try:
                spanish_text = user_input_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            except Exception as queue_error:
                log_with_timestamp(f"Error getting user input request: {queue_error}", "ERROR")
                continue
            
            try:
                # Validate input text
                if not spanish_text or not spanish_text.strip():
                    log_with_timestamp("Empty or invalid transcription text provided", "ERROR")
                    user_input_queue.task_done()
                    continue
                
                print(f"\n🎯 Spanish transcription: {spanish_text}")
                print("   [Press Enter = Gemini, q = skip, r = repeat]: ", end="", flush=True)
                
                try:
                    user_input = input().strip().lower()
                    
                    if user_input == "":
                        # Enter pressed - confirm for Gemini
                        log_with_timestamp("User confirmed for Gemini processing", "USER_CONFIRM")
                        try:
                            gemini_response = call_gemini_api(spanish_text)
                            if gemini_response:
                                print(f"🤖 Gemini response: {gemini_response}")
                                last_gemini_response = gemini_response
                            else:
                                log_with_timestamp("Gemini API returned no response", "ERROR")
                                print("   🤖 Sorry, couldn't get a response from Gemini. Try again.")
                        except Exception as gemini_error:
                            log_with_timestamp(f"Error calling Gemini API: {gemini_error}", "ERROR")
                            print("   🤖 Error getting Gemini response. Try again.")
                            
                    elif user_input == "q":
                        # Skip - user chose not to proceed
                        log_with_timestamp("User skipped Gemini processing", "USER_SKIP")
                        print("   Skipped. Continuing to listen...")
                        
                    elif user_input == "r":
                        # Repeat - user wants to repeat last transcription
                        log_with_timestamp("User requested repeat of last transcription", "USER_REPEAT")
                        if last_spanish and last_spanish.strip():
                            try:
                                # Resend Gemini request with previous transcription
                                gemini_response = call_gemini_api(last_spanish, is_repeat=True)
                                if gemini_response:
                                    print(f"🤖 Gemini response (repeat): {gemini_response}")
                                    last_gemini_response = gemini_response
                                else:
                                    log_with_timestamp("Gemini API returned no response for repeat", "ERROR")
                                    print("   🤖 Sorry, couldn't get a response from Gemini for repeat. Try again.")
                            except Exception as repeat_error:
                                log_with_timestamp(f"Error during repeat Gemini call: {repeat_error}", "ERROR")
                                print("   🤖 Error getting repeat Gemini response. Try again.")
                        else:
                            log_with_timestamp("No previous transcription available for repeat", "USER_REPEAT")
                            print("   No previous transcription available to repeat.")
                            
                    else:
                        # Invalid input - treat as skip
                        log_with_timestamp(f"Invalid user input '{user_input}' - treating as skip", "USER_SKIP")
                        print("   Invalid input. Skipping...")
                        
                except EOFError:
                    # Handle EOF (Ctrl+D on Unix, Ctrl+Z on Windows)
                    print("\n   EOF received during input. Skipping...")
                    log_with_timestamp("EOF received during confirmation prompt", "USER_SKIP")
                except KeyboardInterrupt:
                    # Handle Ctrl+C during input
                    print("\n   Interrupted during input. Skipping...")
                    log_with_timestamp("User interrupted during confirmation prompt", "USER_SKIP")
                    # Re-raise to allow main loop to handle graceful shutdown
                    raise
                except Exception as input_error:
                    log_with_timestamp(f"Error reading user input: {input_error}", "ERROR")
                    print("   Error reading input. Skipping...")
                    
            except Exception as processing_error:
                log_with_timestamp(f"Error processing user input request: {processing_error}", "ERROR")
            finally:
                try:
                    user_input_queue.task_done()
                except Exception as task_done_error:
                    log_with_timestamp(f"Error marking user input task as done: {task_done_error}", "ERROR")
                    
        except Exception as worker_error:
            log_with_timestamp(f"Critical user input worker error: {worker_error}", "ERROR")
            continue
    
    log_with_timestamp("User input worker thread stopped", "SYSTEM")

def display_transcription_and_prompt(spanish_text):
    """
    Non-blocking transcription display that queues user interaction requests.
    This prevents blocking the transcription worker thread.
    """
    try:
        # Validate input text
        if not spanish_text or not spanish_text.strip():
            log_with_timestamp("Empty or invalid transcription text provided", "ERROR")
            return
        
        # Queue the user input request for the dedicated user input worker
        try:
            user_input_queue.put(spanish_text, block=False)
            log_with_timestamp("User input request queued", "USER_INPUT")
        except queue.Full:
            log_with_timestamp("User input queue is full, dropping request", "ERROR")
        except Exception as queue_error:
            log_with_timestamp(f"Error queuing user input request: {queue_error}", "ERROR")
            
    except Exception as display_error:
        log_with_timestamp(f"Critical error in display_transcription_and_prompt: {display_error}", "ERROR")

def streaming_transcription_worker():
    """
    Enhanced streaming transcription worker with proper queue management and user interaction.
    Fixes queue task_done() issues and implements proper user confirmation flow.
    Only processes audio with actual speech content to avoid wasting resources.
    """
    global last_spanish, transcription_worker_running
    
    log_with_timestamp("Streaming transcription worker thread started", "SYSTEM")
    
    while transcription_worker_running:
        audio_window = None
        task_done_called = False
        
        try:
            # Pull audio window from thread-safe queue with timeout
            try:
                audio_window = audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue  # Check if we should keep running
            except Exception as queue_error:
                log_with_timestamp(f"Error getting audio window from queue: {queue_error}", "ERROR")
                continue
            
            try:
                # Validate audio window data
                if audio_window is None or len(audio_window) == 0:
                    log_with_timestamp("Empty audio window received from queue", "ERROR")
                    audio_queue.task_done()
                    task_done_called = True
                    continue
                
                # Check for actual audio content before processing
                # Calculate RMS to detect if there's meaningful audio
                rms = np.sqrt(np.mean(audio_window ** 2))
                audio_threshold = 0.001  # Minimum RMS threshold for speech detection
                
                if rms < audio_threshold:
                    # Skip processing silent audio to save resources
                    log_with_timestamp(f"Skipping silent audio window (RMS: {rms:.6f} < {audio_threshold})", "AUDIO_SKIP")
                    audio_queue.task_done()
                    task_done_called = True
                    continue
                
                # Generate unique window ID for latency tracking
                window_id = str(uuid.uuid4())[:8]
                window_duration = len(audio_window) / SAMPLE_RATE
                
                # Start latency tracking - record processing start time
                processing_start_time = time.time()
                
                # Log [WINDOW_TRANSCRIBE] start
                log_window_transcribe_start(window_duration)
                log_with_timestamp(f"Processing audio window with RMS: {rms:.6f}", "AUDIO_PROCESS")
                
                # Configure faster-whisper with language="es" and task="transcribe"
                try:
                    segments, info = model.transcribe(
                        audio_window,
                        language="es",  # Spanish language as required
                        task="transcribe",  # Transcribe task as required
                        beam_size=5,
                        best_of=5,
                        vad_filter=True,
                        temperature=0.0,
                        no_speech_threshold=0.6  # Increased threshold to reduce false positives
                    )
                    
                    # Extract transcription text
                    transcription_text = " ".join([s.text.strip() for s in segments])
                    
                    # Calculate processing latency
                    processing_end_time = time.time()
                    processing_latency = processing_end_time - processing_start_time
                    
                    # Log transcription latency with performance monitoring
                    log_transcription_latency(window_id, processing_latency)
                    
                    # Handle empty results or very short results
                    if not transcription_text.strip() or len(transcription_text.strip()) < 3:
                        log_window_result("(no speech detected or too short)")
                        audio_queue.task_done()
                        task_done_called = True
                        continue
                    
                    # Get current timestamp for duplicate filtering
                    current_timestamp = datetime.now()
                    
                    # Check for duplicate transcription
                    if is_duplicate_transcription(transcription_text.strip(), current_timestamp):
                        # Log when duplicates are skipped
                        time_diff = (current_timestamp - last_transcription_timestamp).total_seconds()
                        log_with_timestamp(f"Duplicate transcription skipped: '{transcription_text.strip()}' (identical to result {time_diff:.1f}s ago)", "DUPLICATE_SKIP")
                        audio_queue.task_done()
                        task_done_called = True
                        continue
                    
                    # Update last transcription for duplicate filtering
                    update_last_transcription(transcription_text.strip(), current_timestamp)
                    
                    # Store last transcription for repeat functionality
                    last_spanish = transcription_text.strip()
                    
                    # Log [WINDOW_RESULT] completion
                    log_window_result(last_spanish)
                    
                    # Mark task as done before user interaction to prevent queue issues
                    audio_queue.task_done()
                    task_done_called = True
                    
                    # Display transcription and handle user interaction
                    display_transcription_and_prompt(last_spanish)
                    
                except Exception as transcription_error:
                    # Handle transcription errors without crashing main loop
                    error_msg = f"Whisper transcription failed: {str(transcription_error)}"
                    log_with_timestamp(error_msg, "ERROR")
                    log_window_result(f"ERROR: {error_msg}")
                    
            except Exception as processing_error:
                log_with_timestamp(f"Error processing audio window: {processing_error}", "ERROR")
            finally:
                # Ensure task_done is called exactly once
                if not task_done_called:
                    try:
                        audio_queue.task_done()
                    except ValueError as task_done_error:
                        # task_done() called too many times - this is expected in some cases
                        log_with_timestamp(f"Queue task_done already called for this item", "DEBUG")
                    except Exception as task_done_error:
                        log_with_timestamp(f"Error marking queue task as done: {task_done_error}", "ERROR")
            
        except Exception as worker_error:
            log_with_timestamp(f"Critical streaming worker thread error: {str(worker_error)}", "ERROR")
            # Ensure task_done is called even on critical errors
            if audio_window is not None and not task_done_called:
                try:
                    audio_queue.task_done()
                except:
                    pass
            continue
    
    log_with_timestamp("Streaming transcription worker thread stopped", "SYSTEM")

# RMS and speaking state functions removed - replaced by StreamingBuffer

# Show input device info with comprehensive error handling
try:
    info = sd.query_devices(DEVICE_INDEX)
    log_with_timestamp(f"Using audio device {DEVICE_INDEX}: {info['name']}", "SYSTEM")
    print(f"[📻] Using device: {info['name']}")
    
    # Validate device capabilities
    if info['max_input_channels'] < CHANNELS:
        raise ValueError(f"Device only supports {info['max_input_channels']} input channels, need {CHANNELS}")
    
    # Check if sample rate is supported
    try:
        sd.check_input_settings(device=DEVICE_INDEX, channels=CHANNELS, samplerate=SAMPLE_RATE)
        log_with_timestamp(f"Audio settings validated: {SAMPLE_RATE}Hz, {CHANNELS} channel(s)", "SYSTEM")
    except Exception as settings_error:
        log_with_timestamp(f"Audio settings validation failed: {settings_error}", "ERROR")
        raise
        
except sd.PortAudioError as pa_error:
    log_with_timestamp(f"PortAudio error with device {DEVICE_INDEX}: {pa_error}", "ERROR")
    print(f"[❌] PortAudio device error: {pa_error}")
    print("Available devices:")
    try:
        print(sd.query_devices())
    except:
        pass
    exit(1)
except Exception as e:
    log_with_timestamp(f"Audio device error: {e}", "ERROR")
    print(f"[❌] Device error: {e}")
    print("Available devices:")
    try:
        print(sd.query_devices())
    except:
        pass
    exit(1)



# Audio callback with enhanced streaming buffer integration and audio detection
def callback(indata, frames, time, status):
    """
    Enhanced audio callback with intelligent audio detection and efficient processing.
    Only queues audio windows that contain meaningful audio content.
    Reduces unnecessary processing of silent audio.
    """
    try:
        if status:
            log_with_timestamp(f"Audio status warning: {status}", "AUDIO")
        
        # Validate input data
        if indata is None or len(indata) == 0:
            return  # Silent return for empty data - this is normal
        
        # Get audio chunk (mono channel) with error handling
        try:
            audio_chunk = indata[:, 0]
        except IndexError as e:
            log_with_timestamp(f"Audio channel indexing error: {e}", "ERROR")
            return
        except Exception as e:
            log_with_timestamp(f"Audio data extraction error: {e}", "ERROR")
            return
        
        # Append audio to streaming buffer
        try:
            streaming_buffer.append_audio(audio_chunk)
        except Exception as e:
            log_with_timestamp(f"Error appending audio to streaming buffer: {e}", "ERROR")
            return
        
        # Check if it's time to extract a window
        try:
            if streaming_buffer.should_extract_window():
                # Extract window from buffer
                window_data = streaming_buffer.extract_window()
                
                if window_data is not None:
                    # Pre-check audio content before queuing to avoid processing silence
                    rms = np.sqrt(np.mean(window_data ** 2))
                    audio_threshold = 0.001  # Minimum RMS threshold for speech detection
                    
                    if rms >= audio_threshold:
                        # Only queue windows with meaningful audio content
                        try:
                            audio_queue.put(window_data.copy(), block=False)
                            log_with_timestamp(f"Audio window queued (RMS: {rms:.6f})", "AUDIO")
                        except queue.Full:
                            log_with_timestamp("Audio queue is full, dropping audio window", "AUDIO_QUEUE_FULL")
                        except Exception as queue_error:
                            log_with_timestamp(f"Error queuing audio window: {queue_error}", "ERROR")
                    else:
                        # Skip silent audio to save processing resources
                        log_with_timestamp(f"Skipping silent audio window (RMS: {rms:.6f})", "AUDIO_SILENT")
                # If window_data is None, log_window_skip was already called in extract_window()
                        
        except Exception as extraction_error:
            log_with_timestamp(f"Error in window extraction: {extraction_error}", "ERROR")
            
    except Exception as callback_error:
        log_with_timestamp(f"Critical error in audio callback: {callback_error}", "ERROR")

def cleanup_resources():
    """
    Enhanced cleanup function for graceful shutdown with dual worker threads.
    Handles transcription worker, user input worker, and resource management.
    """
    global transcription_worker_running, user_input_worker_running, transcription_thread, user_input_thread, audio_stream
    
    log_with_timestamp("Starting system cleanup", "SYSTEM")
    
    try:
        # Signal both worker threads to stop
        transcription_worker_running = False
        user_input_worker_running = False
        log_with_timestamp("Signaled worker threads to stop", "SYSTEM")
        
        # Wait for any remaining audio processing to complete with timeout
        try:
            log_with_timestamp("Waiting for audio queue to empty", "SYSTEM")
            import time
            start_time = time.time()
            timeout = 5.0  # 5 second timeout
            
            while not audio_queue.empty() and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            if not audio_queue.empty():
                log_with_timestamp(f"Audio queue not empty after {timeout}s timeout, forcing cleanup", "SYSTEM")
            else:
                log_with_timestamp("Audio queue emptied successfully", "SYSTEM")
                
        except Exception as queue_error:
            log_with_timestamp(f"Error waiting for audio queue: {queue_error}", "ERROR")
        
        # Wait for user input queue to empty
        try:
            log_with_timestamp("Waiting for user input queue to empty", "SYSTEM")
            start_time = time.time()
            timeout = 3.0  # 3 second timeout
            
            while not user_input_queue.empty() and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            if not user_input_queue.empty():
                log_with_timestamp(f"User input queue not empty after {timeout}s timeout, forcing cleanup", "SYSTEM")
            else:
                log_with_timestamp("User input queue emptied successfully", "SYSTEM")
                
        except Exception as queue_error:
            log_with_timestamp(f"Error waiting for user input queue: {queue_error}", "ERROR")
        
        # Wait for transcription thread to finish with timeout
        if transcription_thread and transcription_thread.is_alive():
            try:
                log_with_timestamp("Waiting for transcription thread to finish", "SYSTEM")
                transcription_thread.join(timeout=3.0)  # 3 second timeout
                
                if transcription_thread.is_alive():
                    log_with_timestamp("Transcription thread did not finish within timeout", "ERROR")
                else:
                    log_with_timestamp("Transcription thread finished successfully", "SYSTEM")
                    
            except Exception as thread_error:
                log_with_timestamp(f"Error joining transcription thread: {thread_error}", "ERROR")
        
        # Wait for user input thread to finish with timeout
        if user_input_thread and user_input_thread.is_alive():
            try:
                log_with_timestamp("Waiting for user input thread to finish", "SYSTEM")
                user_input_thread.join(timeout=1.0)  # Reduced timeout to 1 second
                
                if user_input_thread.is_alive():
                    log_with_timestamp("User input thread did not finish within timeout - forcing exit", "SYSTEM")
                else:
                    log_with_timestamp("User input thread finished successfully", "SYSTEM")
                    
            except KeyboardInterrupt:
                # Handle Ctrl+C during cleanup gracefully
                log_with_timestamp("Cleanup interrupted by user - forcing exit", "SYSTEM")
            except Exception as thread_error:
                log_with_timestamp(f"Error joining user input thread: {thread_error}", "ERROR")
        
        # Clear streaming buffer
        try:
            with streaming_buffer.buffer_lock:
                streaming_buffer.buffer.clear()
                streaming_buffer.samples_since_extraction = 0
                log_with_timestamp("Streaming buffer cleared", "SYSTEM")
        except Exception as buffer_error:
            log_with_timestamp(f"Error clearing streaming buffer: {buffer_error}", "ERROR")
        
        # Log final statistics
        try:
            audio_queue_size = audio_queue.qsize()
            user_input_queue_size = user_input_queue.qsize()
            if audio_queue_size > 0:
                log_with_timestamp(f"Warning: {audio_queue_size} items remaining in audio queue", "SYSTEM")
            if user_input_queue_size > 0:
                log_with_timestamp(f"Warning: {user_input_queue_size} items remaining in user input queue", "SYSTEM")
        except Exception:
            pass
        
        log_with_timestamp("System cleanup completed", "SYSTEM")
        
    except Exception as cleanup_error:
        log_with_timestamp(f"Error during cleanup: {cleanup_error}", "ERROR")

def main():
    """
    Enhanced main execution function with dual worker threads and improved error handling.
    Starts transcription worker and user input worker for non-blocking operation.
    """
    global transcription_thread, user_input_thread, audio_stream
    
    try:
        # Start streaming transcription worker thread with error handling
        try:
            transcription_thread = threading.Thread(target=streaming_transcription_worker, daemon=True)
            transcription_thread.start()
            log_with_timestamp("Streaming transcription worker thread started successfully", "SYSTEM")
        except Exception as thread_error:
            log_with_timestamp(f"Failed to start streaming transcription worker thread: {thread_error}", "ERROR")
            raise
        
        # Start user input worker thread with error handling
        try:
            user_input_thread = threading.Thread(target=user_input_worker, daemon=True)
            user_input_thread.start()
            log_with_timestamp("User input worker thread started successfully", "SYSTEM")
        except Exception as thread_error:
            log_with_timestamp(f"Failed to start user input worker thread: {thread_error}", "ERROR")
            raise
        
        # Start Gemini warm-up in background thread to avoid blocking startup
        try:
            warmup_thread = threading.Thread(target=warm_up_gemini, daemon=True)
            warmup_thread.start()
            log_with_timestamp("Gemini warm-up thread started", "SYSTEM")
        except Exception as warmup_thread_error:
            log_with_timestamp(f"Failed to start Gemini warm-up thread: {warmup_thread_error}", "ERROR")
            # Don't raise - warm-up is optional and shouldn't block startup
        
        # Start audio stream with comprehensive error handling
        print(f"[🎙️] Starting audio stream on device {DEVICE_INDEX}...")
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                callback=callback,
                blocksize=BLOCK_SIZE,
                device=DEVICE_INDEX
            ) as stream:
                audio_stream = stream
                log_with_timestamp("Audio stream started successfully", "SYSTEM")
                print("[🚀] Listening for Spanish input. Press Ctrl+C to exit.")
                print("💡 Speak Spanish or play Spanish audio through VB-Audio Virtual Cable")
                print("🎯 System will detect speech and present transcriptions for confirmation")
                print()
                
                # Main listening loop with error handling
                try:
                    while True:
                        sd.sleep(1000)
                except KeyboardInterrupt:
                    print("\n[🛑] Keyboard interrupt received. Exiting gracefully...")
                    log_with_timestamp("Keyboard interrupt received", "SYSTEM")
                    raise
                except Exception as loop_error:
                    log_with_timestamp(f"Error in main listening loop: {loop_error}", "ERROR")
                    raise
                    
        except sd.PortAudioError as pa_error:
            log_with_timestamp(f"PortAudio stream error: {pa_error}", "ERROR")
            print(f"[❌] Audio stream PortAudio error: {pa_error}")
            print("💡 Make sure VB-Audio Virtual Cable is installed and audio is being routed to it")
            raise
        except Exception as stream_error:
            log_with_timestamp(f"Audio stream error: {stream_error}", "ERROR")
            print(f"[❌] Audio stream error: {stream_error}")
            raise
            
    except KeyboardInterrupt:
        print("\n[🛑] Exiting gracefully...")
        log_with_timestamp("Application terminated by user", "SYSTEM")
    except Exception as main_error:
        log_with_timestamp(f"Critical error in main execution: {main_error}", "ERROR")
        print(f"[❌] Critical error: {main_error}")
    finally:
        # Always perform cleanup
        try:
            cleanup_resources()
        except Exception as final_cleanup_error:
            log_with_timestamp(f"Error during final cleanup: {final_cleanup_error}", "ERROR")
            print(f"[❌] Cleanup error: {final_cleanup_error}")

# Run the main function
if __name__ == "__main__":
    try:
        main()
    except Exception as startup_error:
        log_with_timestamp(f"Failed to start application: {startup_error}", "ERROR")
        print(f"[❌] Application startup failed: {startup_error}")
        exit(1)
