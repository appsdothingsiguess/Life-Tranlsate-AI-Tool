"""
CHANGELOG
=========

v2.2.0 - Collision-Free Global Hotkeys
- Replaced F-keys with triple-modifier combinations (Ctrl+Alt+Shift+key)
- Eliminated input hijacking in other applications
- Updated hotkey mapping: Ctrl+Alt+Shift+S (send), Ctrl+Alt+Shift+R (repeat), etc.
- Added KEY_CONFIG for easy hotkey customization
- Maintained all UX improvements: instant transcription cue, empty buffer handling, error handling
- Added --keys dump CLI flag for key mapping verification

v2.1.0 - Global Hotkeys & UX Improvements
- Replaced console polling with pynput.keyboard.Listener for system-wide hotkeys
- Added instant transcription cue (🎙️ Transcribing…) on SPEECH_START
- Improved empty buffer handling with user-friendly warning
- Enhanced Gemini error handling with concise error messages
- Fixed banner order: warm-up completes before "Ready" status
- Removed F6 alternate send key to simplify hotkey mapping
- Updated all console messages for better UX flow

v2.0.0 - Previous version with F-key hotkeys and basic global support
"""

import os
import torch
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
# from scipy.signal import resample  # Commented out due to Python 3.13 compatibility issue
from collections import deque
import threading
import queue
from dotenv import load_dotenv
from google import genai
from datetime import datetime
import time
import re  # Added for filtering nonsense chunks
import logging  # Added for logging configuration

# Global hotkey library imports with cross-platform fallback
try:
    from pynput import keyboard as pynput_keyboard
    _HOTKEY_LIB = "pynput"
except ImportError:
    try:
        import keyboard  # Windows / X11 / Wayland
        _HOTKEY_LIB = "keyboard"
    except ImportError:
        import msvcrt  # Fallback to console-only hotkeys
        _HOTKEY_LIB = "msvcrt"

# Hotkey mapping configuration - collision-free triple-modifier combos
HOTKEY_MAP = {
    "send": "ctrl+alt+shift+s",
    "repeat": "ctrl+alt+shift+r", 
    "help": "ctrl+alt+shift+h",
    "skip": "ctrl+alt+shift+c"
}

# Key configuration for easy customization
KEY_CONFIG = {
    "modifiers": ["ctrl", "alt", "shift"],
    "keys": {
        "send": "s",
        "repeat": "r", 
        "help": "h",
        "skip": "c"
    }
}

# Global variables for cleanup
_pynput_listener = None

# Configure logging to suppress unwanted messages
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("faster_whisper").setLevel(logging.WARNING)

os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin")

# Custom resample function to replace scipy.signal.resample
def resample(audio_data, new_length):
    """
    Simple resample function to replace scipy.signal.resample
    Uses linear interpolation for audio resampling
    """
    try:
        original_length = len(audio_data)
        if original_length == new_length:
            return audio_data.astype(np.float32)  # Ensure float32 output
        
        # Create time arrays
        original_time = np.linspace(0, 1, original_length)
        new_time = np.linspace(0, 1, new_length)
        
        # Linear interpolation
        resampled = np.interp(new_time, original_time, audio_data)
        return resampled.astype(np.float32)  # Ensure float32 output
        
    except Exception as e:
        # Fallback: return original data if resampling fails
        print(f"Resampling failed: {e}, using original audio")
        return audio_data.astype(np.float32)  # Ensure float32 output

# System Constants
RMS_THRESHOLD = 0.012  # Optimized threshold for better responsiveness
SILENCE_SECS = 0.2     # Reduced from 0.3s to 0.2s for faster response
MAX_SPEECH_SECS = 3.2  # Soft-hold forced flush starts after 3.2s
AUTO_SEND_AFTER_SECS = None  # Auto-send disabled - manual send only

# Soft flush constants for intelligent chunking (user requirements)
SOFT_FLUSH_START_SECS = 3.2    # when grace window should begin
SOFT_HOLD_MS = 400             # grace-period length (0.4s)
SOFT_FLUSH_COUNTDOWN_SECS = 0.4  # 400ms countdown window
MIN_SILENCE_MS = 150           # silence needed to early-flush (0.15s)
SOFT_FLUSH_SILENCE_THRESHOLD = 0.15  # 150ms silence within countdown
HARD_FLUSH_TIMEOUT = 5.0       # absolute max duration of one speech chunk
SOFT_FLUSH_ABSOLUTE_CAP = 5.0  # Absolute hard-flush cap at 5s total
SEGMENT_MERGE_THRESHOLD = 0.5  # 500ms threshold for merging segments
LOG_FILE = "log.txt"
GEMINI_PROMPT = (
    "You are a university student currently in Spanish 2, responding naturally to another student in Spanish. Your goal is to reply with a simple, grammatically correct Spanish sentence, as someone who has completed Spanish 1 and is currently learning the concepts listed below would. "
    "Concepts you understand and should utilize: "
    "- Saber and conocer (and irregular yo forms: sé, conozco)\n"
    "- Preterite tense (regular -ar, -er, -ir verbs; irregulars like -car/-gar/-zar changes; creer, leer, oír, ver; common preterite adverbs)\n"
    "- Acabar de + [infinitive]\n"
    "- Basic Spanish pronunciation rules\n"
    "- Regular -AR, -ER, -IR verb conjugations (present tense)\n"
    "- Irregular present tense verbs: ser, estar, ir, tener, venir, decir, hacer, poner, salir, suponer, traer, ver, oír\n"
    "- Stem-changing verbs (e→ie, o→ue, e→i)\n"
    "- Direct object nouns and pronouns (placement)\n"
    "- Possession (using \"de,\" possessive adjectives, possessive pronouns, de + el = del)\n"
    "- Ser vs. Estar (DOCTOR, PLACE/LoCo mnemonics, adjective meaning changes)\n"
    "- Gustar (and similar verbs, where liked item is subject)\n"
    "- Forming questions (pitch, inversion, tags, interrogative words, por qué vs. porque)\n"
    "- Present Progressive (estar + present participle, irregulars, stem changes)\n"
    "- Descriptive adjectives and nationality (agreement, position, bueno/malo/grande/santo)\n"
    "- Numbers 31 and higher (patterns, cien/ciento, mil/millón)\n"
    "- Common verbs and expressions (especially with tener)\n"
    "When the other student provides Spanish text, reply *only* with a single, natural, brief Spanish sentence. Do not add explanations, alternatives, or any text beyond that one sentence."
)
MODEL_NAME = "gemini-2.5-flash-lite"

# Thread-safe logging infrastructure
_log_lock = threading.Lock()

# Console events for live exam interface
CONSOLE_EVENTS = {
    "INIT", "READY", "SPEECH_START", "BUFFER_UPDATE", 
    "GEMINI_REPLY", "READY_NEXT", "HELP", "ERROR", "TRANSCRIBE_ES"
}

# Live transcription performance monitoring removed - now using immediate processing

def print_console(message, event_type="INFO"):
    """
    Console output for live exam interface - only shows exam-relevant messages.
    All other messages go to log file only.
    """
    if event_type in CONSOLE_EVENTS:
        print(message)
    
    # Always log to file for diagnostics
    log_with_timestamp(message, event_type)

def log_with_timestamp(message, event_type="INFO"):
    """
    Thread-safe logging function with timestamp formatting.
    Logs to file only - console output handled by print_console().
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] [{event_type}] {message}"
    
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

def log_flush_event(reason, duration_seconds):
    """
    Log flush events with consistent format including chunk length.
    Args:
        reason: "soft" or "hard" 
        duration_seconds: length of the audio chunk in seconds
    """
    log_with_timestamp(f"FLUSH ({reason}) | len={duration_seconds:.2f}s", "AUDIO")

def print_startup_banner():
    """
    Display the live exam startup banner with instructions.
    """
    banner = f"""
═════════════════════════════════════════════════════════
LIVE SPANISH ORAL EXAM ASSISTANT
─────────────────────────────────────────────────────────
Global Hotkeys (work from any application):
{HOTKEY_MAP['send'].upper()}   → send current text to Gemini immediately
{HOTKEY_MAP['repeat'].upper()}   → repeat last Gemini reply
{HOTKEY_MAP['help'].upper()}   → show this help
{HOTKEY_MAP['skip'].upper()}   → clear/skip current buffer
Ctrl‑C → exit

Send mode:   manual only (press {HOTKEY_MAP['send'].upper()} to send)
Flush limit: 5.0 s per speech chunk
═════════════════════════════════════════════════════════
"""
    print(banner)

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

# Window-based logging functions removed - now using live transcription system

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
 "You are a university student currently in Spanish 2, responding naturally to another student in Spanish. Your goal is to reply with a simple, grammatically correct Spanish sentence, as someone who has completed Spanish 1 and is currently learning the concepts listed below would. "
    "Concepts you understand and should utilize: "
    "- Saber and conocer (and irregular yo forms: sé, conozco)\n"
    "- Preterite tense (regular -ar, -er, -ir verbs; irregulars like -car/-gar/-zar changes; creer, leer, oír, ver; common preterite adverbs)\n"
    "- Acabar de + [infinitive]\n"
    "- Basic Spanish pronunciation rules\n"
    "- Regular -AR, -ER, -IR verb conjugations (present tense)\n"
    "- Irregular present tense verbs: ser, estar, ir, tener, venir, decir, hacer, poner, salir, suponer, traer, ver, oír\n"
    "- Stem-changing verbs (e→ie, o→ue, e→i)\n"
    "- Direct object nouns and pronouns (placement)\n"
    "- Possession (using \"de,\" possessive adjectives, possessive pronouns, de + el = del)\n"
    "- Ser vs. Estar (DOCTOR, PLACE/LoCo mnemonics, adjective meaning changes)\n"
    "- Gustar (and similar verbs, where liked item is subject)\n"
    "- Forming questions (pitch, inversion, tags, interrogative words, por qué vs. porque)\n"
    "- Present Progressive (estar + present participle, irregulars, stem changes)\n"
    "- Descriptive adjectives and nationality (agreement, position, bueno/malo/grande/santo)\n"
    "- Numbers 31 and higher (patterns, cien/ciento, mil/millón)\n"
    "- Common verbs and expressions (especially with tener)\n"
    "When the other student provides Spanish text, reply *only* with a single, natural, brief Spanish sentence. Do not add explanations, alternatives, or any text beyond that one sentence."
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
                print_console(f"Gemini  warm‑up ({warmup_time:.2f}s)  ✔", "INIT")
                
                # Print ready banner after warm-up completes
                print_console("Ready!  ─ Ready to transcribe ─", "READY")
                print_console("─────────────────────────────────────────────────────────", "READY")
                
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
                # Final attempt failed - still show ready banner
                log_with_timestamp("Gemini warm-up failed after all attempts", "GEMINI_WARMUP_FAIL")
                print_console("Ready!  ─ Ready to transcribe ─", "READY")
                print_console("─────────────────────────────────────────────────────────", "READY")
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
    "You are a university student currently in Spanish 2, responding naturally to another student in Spanish. Your goal is to reply with a simple, grammatically correct Spanish sentence, as someone who has completed Spanish 1 and is currently learning the concepts listed below would. "
    "Concepts you understand and should utilize: "
    "- Saber and conocer (and irregular yo forms: sé, conozco)\n"
    "- Preterite tense (regular -ar, -er, -ir verbs; irregulars like -car/-gar/-zar changes; creer, leer, oír, ver; common preterite adverbs)\n"
    "- Acabar de + [infinitive]\n"
    "- Basic Spanish pronunciation rules\n"
    "- Regular -AR, -ER, -IR verb conjugations (present tense)\n"
    "- Irregular present tense verbs: ser, estar, ir, tener, venir, decir, hacer, poner, salir, suponer, traer, ver, oír\n"
    "- Stem-changing verbs (e→ie, o→ue, e→i)\n"
    "- Direct object nouns and pronouns (placement)\n"
    "- Possession (using \"de,\" possessive adjectives, possessive pronouns, de + el = del)\n"
    "- Ser vs. Estar (DOCTOR, PLACE/LoCo mnemonics, adjective meaning changes)\n"
    "- Gustar (and similar verbs, where liked item is subject)\n"
    "- Forming questions (pitch, inversion, tags, interrogative words, por qué vs. porque)\n"
    "- Present Progressive (estar + present participle, irregulars, stem changes)\n"
    "- Descriptive adjectives and nationality (agreement, position, bueno/malo/grande/santo)\n"
    "- Numbers 31 and higher (patterns, cien/ciento, mil/millón)\n"
    "- Common verbs and expressions (especially with tener)\n"
    "When the other student provides Spanish text, reply *only* with a single, natural, brief Spanish sentence. Do not add explanations, alternatives, or any text beyond that one sentence."

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

        # Live transcription system initialization
log_with_timestamp("=== Live Transcription System Ready ===", "AUDIO_FORMAT")
log_with_timestamp("✅ RMS-based speech detection enabled", "AUDIO_FORMAT")
log_with_timestamp("✅ Live buffer accumulation system initialized", "AUDIO_FORMAT")
log_with_timestamp("🚀 Performance: True live streaming with soft-hold flush", "AUDIO_FORMAT")
log_with_timestamp("📊 Settings: RMS_THRESHOLD=0.012, SILENCE_SECS=0.2, MAX_SPEECH_SECS=3.2", "AUDIO_FORMAT")
log_with_timestamp("🎯 Soft-hold: 3.2s start + 400ms countdown + 150ms silence threshold", "AUDIO_FORMAT")
log_with_timestamp("⏰ Absolute cap: 5.0s maximum speech duration (HARD_FLUSH_TIMEOUT)", "AUDIO_FORMAT")
log_with_timestamp("🔧 Whisper: beam=1, best_of=1, vad=True, no_speech=0.6", "AUDIO_FORMAT")
log_with_timestamp("❓ Question splitting: automatic split on '?' for separate Gemini calls", "AUDIO_FORMAT")
log_with_timestamp("📝 Flush logging: FLUSH (soft|hard) | len=X.XXs format", "AUDIO_FORMAT")

# Audio settings
SAMPLE_RATE = 48000
TARGET_RATE = 16000
BLOCK_DURATION = 0.5
# WINDOW_DURATION now defined in streaming constants section above
CHANNELS = 1
DEVICE_INDEX = 37  # VB-Audio Virtual Cable WASAPI input

BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)

# RMS-based pause detection state (from stablever.py)
is_speaking = False
silence_start = None
speech_start_time = None  # Track when current speech block began
audio_buffer = deque()
speech_buffer_lock = threading.Lock()

# Soft-hold forced flush state
soft_flush_countdown_start = None  # When countdown started
soft_flush_silence_start = None    # When silence started during countdown

# Live transcription buffer for accumulating text
live_buffer: list[str] = []
live_buffer_lock = threading.Lock()

# Transcription worker thread infrastructure
audio_queue = queue.Queue()
# user_input_queue removed - using hotkey system instead
last_spanish = ""  # Store last transcription for repeat functionality
# last_gemini_response removed - using hotkey_state["last_gemini"] instead
transcription_worker_running = True
# user_input_worker_running removed - using hotkey system instead
transcription_thread = None  # Will hold the worker thread reference
# user_input_thread removed - using hotkey system instead
audio_stream = None  # Will hold the audio stream reference

# Hotkey infrastructure - minimal shared state and thread safety
hotkey_state = {
    "last_transcription": "",
    "last_gemini": "",
    "gemini_busy": False,
    "question_counter": 0  # Track question index for logging
}
hotkey_lock = threading.Lock()  # Single lock for thread safety
hotkey_listener_running = False
hotkey_thread = None  # Will hold the hotkey listener thread reference

# Auto-send monitoring infrastructure
last_buffer_update_time = None
auto_send_monitor_running = False
auto_send_thread = None  # Will hold the auto-send monitor thread reference

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

def compute_rms(audio_chunk):
    """
    Compute RMS (Root Mean Square) for audio level detection.
    Returns the RMS value of the audio chunk.
    Includes error handling for invalid audio data.
    """
    try:
        if audio_chunk is None or len(audio_chunk) == 0:
            return 0.0
        
        # Validate audio data type
        if not isinstance(audio_chunk, np.ndarray):
            log_with_timestamp(f"Invalid audio chunk type: {type(audio_chunk)}", "ERROR")
            return 0.0
        
        # Check for NaN or infinite values
        if np.any(np.isnan(audio_chunk)) or np.any(np.isinf(audio_chunk)):
            log_with_timestamp("Audio chunk contains NaN or infinite values", "ERROR")
            return 0.0
        
        # Compute RMS
        mean_square = np.mean(audio_chunk ** 2)
        if mean_square < 0:
            log_with_timestamp(f"Negative mean square value: {mean_square}", "ERROR")
            return 0.0
        
        rms_value = np.sqrt(mean_square)
        return rms_value
        
    except Exception as rms_error:
        log_with_timestamp(f"Error computing RMS: {rms_error}", "ERROR")
        return 0.0

def update_speaking_state(audio_chunk):
    """
    Track is_speaking state and silence_start timing.
    Flip state after SILENCE_SECS threshold is reached.
    Log SPEECH_START and SPEECH_END events with timestamps.
    """
    global is_speaking, silence_start, speech_start_time, soft_flush_countdown_start, soft_flush_silence_start
    
    try:
        rms = compute_rms(audio_chunk)
        
        if rms is None or np.isnan(rms) or np.isinf(rms):
            log_with_timestamp(f"Invalid RMS value: {rms}", "ERROR")
            return False
        
        current_time = datetime.now()
        
        if rms > RMS_THRESHOLD:
            # Audio detected above threshold
            if not is_speaking:
                # Transition from silence to speech
                is_speaking = True
                silence_start = None
                speech_start_time = current_time  # Track when speech began
                log_with_timestamp(f"SPEECH_START detected (RMS: {rms:.4f})", "AUDIO")
        else:
            # Audio below threshold (silence)
            if is_speaking:
                # We were speaking, now we might be in silence
                if silence_start is None:
                    # First moment of silence
                    silence_start = current_time
                else:
                    # Check if we've been silent long enough
                    silence_duration = (current_time - silence_start).total_seconds()
                    if silence_duration >= SILENCE_SECS:
                        # Transition from speech to silence
                        is_speaking = False
                        speech_start_time = None  # Reset speech start time
                        # Reset soft-hold state when speech ends naturally
                        soft_flush_countdown_start = None
                        soft_flush_silence_start = None
                        # Reset speaking feedback flag so it can show again for next speech
                        # Note: This will be reset in the callback function when speech ends
                        log_with_timestamp(f"SPEECH_END detected after {silence_duration:.2f}s of silence (RMS: {rms:.4f})", "AUDIO")
                        return True  # Signal that speech segment is complete
        
        return False  # Speech segment not complete yet
        
    except Exception as state_error:
        log_with_timestamp(f"Critical error in update_speaking_state: {state_error}", "ERROR")
        return False

def is_nonsense_chunk(text):
    """
    Filter out nonsense chunks like repeated letters or very short text.
    Returns True if the chunk should be skipped.
    """
    try:
        if not text or not isinstance(text, str):
            return True
        
        # Strip and check length
        clean_text = text.strip()
        if len(clean_text) < 3:
            return True
        
        # Count alphabetic characters
        alpha_chars = sum(1 for c in clean_text if c.isalpha())
        if alpha_chars < 3:
            return True
        
        # Check for repeated letter patterns (rrrr, uuuuh, etc.)
        # Remove punctuation and check for 3+ repeated characters
        letters_only = re.sub(r'[^a-zA-Z]', '', clean_text.lower())
        if re.search(r'([a-z])\1{2,}', letters_only):
            return True
        
        return False
        
    except Exception as e:
        log_with_timestamp(f"Error in nonsense filter: {e}", "ERROR")
        return True  # Skip on error to be safe

# >>> HOTKEY_HANDLERS_START
def send_to_gemini(buffer_content):
    """
    Shared function to send content to Gemini, used by manual g-key only.
    Splits text on question marks and sends each question separately.
    Returns True if successful, False otherwise.
    """
    import time
    
    try:
        with hotkey_lock:
            busy = hotkey_state.get("gemini_busy", False)
        
        if busy:
            return False
        
        if not buffer_content:
            return False
        
        # Join all buffered text with spaces if it's a list
        if isinstance(buffer_content, list):
            full_text = " ".join(buffer_content)
        else:
            full_text = str(buffer_content)
        
        # Split text on question marks to handle multiple questions
        questions = [q.strip() for q in full_text.split('?') if q.strip()]
        
        if not questions:
            log_with_timestamp("No valid questions found in text", "ERROR")
            return False
        
        # Send each question separately
        success_count = 0
        for i, question in enumerate(questions):
            # Add question mark back if it was split
            if not question.endswith('?'):
                question += '?'
            
            with hotkey_lock:
                hotkey_state["gemini_busy"] = True
                hotkey_state["question_counter"] += 1
                question_index = hotkey_state["question_counter"]
            
            try:
                start = time.time()
                
                # Log the send request (file only, no console output)
                log_with_timestamp(f"Sending Q{question_index} to Gemini: {question[:50]}...", "USER_FEEDBACK")
                
                # Call Gemini with the question
                reply = call_gemini_api(question)
                elapsed = time.time() - start
                
                if reply is None:
                    raise ValueError(f"Gemini API returned no response for Q{question_index}")
                
                with hotkey_lock:
                    hotkey_state["last_gemini"] = reply
                    hotkey_state["gemini_busy"] = False
                
                log_with_timestamp(f"Q{question_index} response time: {elapsed:.2f}s", "GEMINI_RESPONSE_TIME")
                print_console(f"🤖  Q{question_index}  «{reply}»", "GEMINI_REPLY")
                success_count += 1
                
            except Exception as question_error:
                with hotkey_lock:
                    hotkey_state["gemini_busy"] = False
                log_with_timestamp(f"Error sending Q{question_index}: {str(question_error)}", "ERROR_GEMINI")
                print(f"🤖 ❌ Q{question_index} failed. Check logs for details.")
                continue
        
        # Clear the live buffer after all questions are processed
        with live_buffer_lock:
            live_buffer.clear()
        
        # Show ready status after successful send
        if success_count > 0:
            print_console("─────────────────────────────────────────────────────────", "READY_NEXT")
            print_console("🔄  Ready — start next answer", "READY_NEXT")
            print_console("─────────────────────────────────────────────────────────", "READY_NEXT")
        
        return success_count > 0
        
    except KeyboardInterrupt:
        with hotkey_lock:
            hotkey_state["gemini_busy"] = False
        log_with_timestamp("Gemini call interrupted by user", "ERROR_GEMINI")
        raise
    except Exception as handler_error:
        # User-friendly error handling - don't clear buffer so user can retry
        error_msg = str(handler_error)
        if "Empty response" in error_msg or not error_msg.strip():
            error_msg = "Empty response"
        
        print_console(f"❌  Gemini error: {error_msg}", "USER_FEEDBACK")
        log_with_timestamp(f"Critical error in send_to_gemini: {str(handler_error)}", "ERROR_SEND_GEMINI")
        
        try:
            with hotkey_lock:
                hotkey_state["gemini_busy"] = False
        except:
            pass
        return False

def handle_send_key(state, lock, log_event, call_gemini):
    """Send all live buffered text to Gemini if available and not busy."""
    log_event(f"[HOTKEY] {HOTKEY_MAP['send']} fired - sending buffer", "HOTKEY")
    
    try:
        # Get all live buffer content and strip whitespace
        with live_buffer_lock:
            if not live_buffer:
                print_console("⚠️  Nothing to send — say something first!", "USER_FEEDBACK")
                log_event("Nothing to send", "USER_FEEDBACK")
                return
            
            # Create a copy for processing
            buffer_copy = live_buffer.copy()
            full_text = " ".join(buffer_copy).strip()
            
            # Check if buffer is empty or only whitespace
            if not full_text:
                print_console("⚠️  Nothing to send — say something first!", "USER_FEEDBACK")
                log_event("Nothing to send", "USER_FEEDBACK")
                return
            
            word_count = len(full_text.split())
            chunk_count = len(buffer_copy)
        
        # Log what we're sending (file only)
        log_event(f"Manual send: {word_count}w / {chunk_count}c", "HOTKEY_SEND")
        
        # Use shared send function
        success = send_to_gemini(buffer_copy)
        if success:
            log_event("Manual send completed", "HOTKEY_SEND")
        else:
            log_event("Manual send failed", "HOTKEY_SEND")
            
    except Exception as handler_error:
        log_event(f"Critical error in handle_send_key: {str(handler_error)}", "ERROR_HOTKEY_HANDLER")

def handle_repeat_key(state, lock, log_event):
    """Reprint the last Gemini reply if available."""
    log_event(f"[HOTKEY] {HOTKEY_MAP['repeat']} fired - repeating last reply", "HOTKEY")
    
    with lock:
        reply = state.get("last_gemini", "")
    
    if not reply:
        log_event("No previous Gemini reply", "HOTKEY_REPEAT")
        return
    
    log_event("Reprinting last Gemini reply", "HOTKEY_REPEAT")
    print_console(f"🤖  Last reply: «{reply}»", "GEMINI_REPLY")

def handle_help_key(log_event):
    """Display help text for available hotkeys."""
    log_event(f"[HOTKEY] {HOTKEY_MAP['help']} fired - showing help", "HOTKEY")
    log_event("Help shown", "HOTKEY_HELP")
    print_startup_banner()

def handle_skip_key(log_event):
    """Skip/clear current transcription buffer."""
    log_event(f"[HOTKEY] {HOTKEY_MAP['skip']} fired - clearing buffer", "HOTKEY")
    
    try:
        with live_buffer_lock:
            buffer_size = len(live_buffer)
            live_buffer.clear()
        
        if buffer_size > 0:
            log_event(f"Buffer cleared ({buffer_size} chunks)", "HOTKEY_SKIP")
            print_console("🗑️  Buffer cleared", "HOTKEY_SKIP")
        else:
            log_event("Buffer was already empty", "HOTKEY_SKIP")
            
    except Exception as skip_error:
        log_event(f"Error clearing buffer: {skip_error}", "ERROR_HOTKEY_HANDLER")

# HANDLERS_OK
# <<< HOTKEY_HANDLERS_END

# Global hotkey debouncing state
_hotkey_debounce_timers = {}
_hotkey_debounce_delay = 0.3  # 300ms debounce

def _is_hotkey_debounced(key):
    """
    Check if a hotkey is within the debounce window.
    Returns True if the key should be ignored due to debouncing.
    """
    global _hotkey_debounce_timers, _hotkey_debounce_delay
    
    current_time = time.time()
    if key in _hotkey_debounce_timers:
        time_since_last = current_time - _hotkey_debounce_timers[key]
        if time_since_last < _hotkey_debounce_delay:
            log_with_timestamp(f"Debounced key '{key}' (last press {time_since_last:.3f}s ago)", "HOTKEY_DEBOUNCE")
            return True
    
    # Update debounce timer for this key
    _hotkey_debounce_timers[key] = current_time
    return False

# Global modifier state tracking
_modifier_state = {
    "ctrl": False,
    "alt": False,
    "shift": False
}

def _update_modifier_state(key, pressed):
    """Update modifier state based on key press/release."""
    global _modifier_state
    
    if hasattr(key, 'name'):
        key_name = key.name.lower()
        if key_name in _modifier_state:
            _modifier_state[key_name] = pressed

def _check_hotkey_combination(key):
    """Check if current key press matches any hotkey combination."""
    global _modifier_state
    
    # Get the pressed key character
    if hasattr(key, 'char') and key.char:
        pressed_key = key.char.lower()
    else:
        return None
    
    # Check if all required modifiers are pressed
    if (_modifier_state["ctrl"] and 
        _modifier_state["alt"] and 
        _modifier_state["shift"]):
        
        # Check if this key matches any of our hotkeys
        for action, key_char in KEY_CONFIG["keys"].items():
            if pressed_key == key_char:
                return action
    
    return None

def _on_hotkey_press(key):
    """
    Global hotkey press handler with modifier combination detection.
    """
    try:
        # Update modifier state
        _update_modifier_state(key, True)
        
        # Check for hotkey combination
        action = _check_hotkey_combination(key)
        
        if action:
            # Check debouncing
            if _is_hotkey_debounced(action):
                return
            
            # Process the hotkey action
            try:
                if action == "send":
                    handle_send_key(hotkey_state, hotkey_lock, log_with_timestamp, call_gemini_api)
                    
                elif action == "repeat":
                    handle_repeat_key(hotkey_state, hotkey_lock, log_with_timestamp)
                    
                elif action == "help":
                    handle_help_key(log_with_timestamp)
                    
                elif action == "skip":
                    handle_skip_key(log_with_timestamp)
                    
            except Exception as handler_error:
                log_with_timestamp(f"Error in hotkey handler for '{action}': {handler_error}", "ERROR_HOTKEY_HANDLER")
                # Continue running - don't let handler errors crash the listener
                
    except Exception as hotkey_error:
        log_with_timestamp(f"Error processing global hotkey: {hotkey_error}", "ERROR_HOTKEY_INPUT")

def _on_hotkey_release(key):
    """
    Global hotkey release handler for modifier state tracking.
    """
    try:
        # Update modifier state
        _update_modifier_state(key, False)
    except Exception as hotkey_error:
        log_with_timestamp(f"Error processing hotkey release: {hotkey_error}", "ERROR_HOTKEY_INPUT")

def _register_hotkey(label, key, func):
    """
    Register a hotkey with robust logging and error handling.
    """
    try:
        keyboard.add_hotkey(key, func)
        log_with_timestamp(f"[HOTKEY] Registered {label} on {key}", "SYSTEM")
        return True
    except Exception as e:
        log_with_timestamp(f"[HOTKEY] Failed to register {label} on {key}: {e}", "ERROR")
        return False

def _check_keyboard_privileges():
    """
    Check if keyboard library has sufficient privileges for global hotkeys.
    Returns True if privileges are sufficient, False otherwise.
    """
    try:
        # Test if we can detect a simple key press
        keyboard.is_pressed('shift')
        return True
    except (RuntimeError, PermissionError) as e:
        log_with_timestamp(f"[HOTKEY] Admin rights required for global hooks: {e}", "ERROR")
        print_console("[HOTKEY] Admin rights required for global hooks. Run PowerShell as Administrator.", "ERROR")
        return False
    except Exception as e:
        log_with_timestamp(f"[HOTKEY] Keyboard privilege check failed: {e}", "ERROR")
        return False

def start_global_hotkeys():
    """
    Start global hotkey listener using the appropriate library.
    Returns True if successful, False otherwise.
    """
    global hotkey_listener_running, _pynput_listener
    
    try:
        log_with_timestamp(f"Starting global hotkey listener using {_HOTKEY_LIB}", "SYSTEM")
        
        if _HOTKEY_LIB == "keyboard":
            # Keyboard library doesn't support complex modifier combinations well
            # Fall back to pynput or console hotkeys
            log_with_timestamp("Keyboard library doesn't support modifier combinations, falling back to console", "SYSTEM")
            return start_console_hotkeys()
            
        elif _HOTKEY_LIB == "pynput":
            # Register hotkeys with pynput library using modifier combinations
            def on_press(key):
                try:
                    _on_hotkey_press(key)
                except Exception as e:
                    log_with_timestamp(f"Pynput key press error: {e}", "ERROR_HOTKEY_INPUT")
            
            def on_release(key):
                try:
                    _on_hotkey_release(key)
                except Exception as e:
                    log_with_timestamp(f"Pynput key release error: {e}", "ERROR_HOTKEY_INPUT")
            
            # Start pynput listener in daemon thread
            def pynput_listener():
                try:
                    global _pynput_listener
                    _pynput_listener = pynput_keyboard.Listener(
                        on_press=on_press,
                        on_release=on_release
                    )
                    _pynput_listener.start()
                    _pynput_listener.join()
                except Exception as e:
                    log_with_timestamp(f"Pynput listener error: {e}", "ERROR_HOTKEY_THREAD")
            
            hotkey_thread = threading.Thread(target=pynput_listener, daemon=True)
            hotkey_thread.start()
            hotkey_listener_running = True
            
        else:
            # Fallback to console-only hotkeys (original msvcrt implementation)
            log_with_timestamp("Falling back to console-only hotkeys (msvcrt)", "SYSTEM")
            return start_console_hotkeys()
        
        log_with_timestamp("Global hotkey listener started successfully", "SYSTEM")
        print_console(f"[HOTKEY] Global hooks active ({HOTKEY_MAP['send'].upper()}=send • {HOTKEY_MAP['repeat'].upper()}=repeat • {HOTKEY_MAP['help'].upper()}=help • {HOTKEY_MAP['skip'].upper()}=skip)", "SYSTEM")
        return True
        
    except Exception as e:
        log_with_timestamp(f"Failed to start global hotkey listener: {e}", "ERROR")
        log_with_timestamp("Falling back to console-only hotkeys", "SYSTEM")
        return start_console_hotkeys()

def start_console_hotkeys():
    """
    Fallback console-only hotkey listener using msvcrt.
    Returns True if successful, False otherwise.
    """
    global hotkey_listener_running
    
    try:
        log_with_timestamp("Starting console-only hotkey listener (msvcrt)", "SYSTEM")
        
        def console_hotkey_worker():
            """
            Console-only hotkey listener thread using msvcrt.
            """
            global hotkey_listener_running
            
            # Debouncing - track last press time for each key
            debounce_timers = {}
            debounce_delay = 0.3  # 300ms debounce
            
            log_with_timestamp("Console hotkey listener thread started", "SYSTEM")
            
            while hotkey_listener_running:
                try:
                    # Check if a key is available (non-blocking)
                    if msvcrt.kbhit():
                        try:
                            # Get the key press
                            key = msvcrt.getch().decode('utf-8').lower()
                            current_time = time.time()
                            
                            # Check debouncing for this specific key
                            if key in debounce_timers:
                                time_since_last = current_time - debounce_timers[key]
                                if time_since_last < debounce_delay:
                                    # Key is debounced, ignore this press
                                    log_with_timestamp(f"Debounced key '{key}' (last press {time_since_last:.3f}s ago)", "HOTKEY_DEBOUNCE")
                                    continue
                            
                            # Update debounce timer for this key
                            debounce_timers[key] = current_time
                            
                            # Process the key press with individual error handling
                            try:
                                # Convert to uppercase for comparison
                                key_upper = key.upper()
                                
                                # Note: Console hotkeys are limited to single keys
                                # For full modifier combination support, use global hotkeys
                                if key_upper == 'S':
                                    handle_send_key(hotkey_state, hotkey_lock, log_with_timestamp, call_gemini_api)
                                    
                                elif key_upper == 'R':
                                    handle_repeat_key(hotkey_state, hotkey_lock, log_with_timestamp)
                                    
                                elif key_upper == 'H':
                                    handle_help_key(log_with_timestamp)
                                    
                                elif key_upper == 'C':
                                    handle_skip_key(log_with_timestamp)
                                    
                                else:
                                    # Ignore other keys silently
                                    pass
                                    
                            except Exception as handler_error:
                                log_with_timestamp(f"Error in hotkey handler for '{key}': {handler_error}", "ERROR_HOTKEY_HANDLER")
                                # Continue running - don't let handler errors crash the listener
                                
                        except UnicodeDecodeError:
                            # Handle special keys that can't be decoded
                            log_with_timestamp("Special key pressed (ignored)", "HOTKEY_SPECIAL")
                        except EOFError:
                            # Handle EOF gracefully
                            log_with_timestamp("EOF received in hotkey listener, shutting down", "HOTKEY_EOF")
                            break
                        except KeyboardInterrupt:
                            # Handle Ctrl+C in hotkey thread
                            log_with_timestamp("Keyboard interrupt in hotkey listener, shutting down", "HOTKEY_INTERRUPT")
                            break
                        except Exception as key_error:
                            log_with_timestamp(f"Error processing key press: {key_error}", "ERROR_HOTKEY_INPUT")
                    
                    # Small sleep to prevent excessive CPU usage
                    time.sleep(0.01)  # 10ms sleep
                    
                except KeyboardInterrupt:
                    # Handle Ctrl+C at thread level
                    log_with_timestamp("Console hotkey listener thread interrupted by user", "SYSTEM")
                    break
                except Exception as listener_error:
                    log_with_timestamp(f"Critical console hotkey listener error: {listener_error}", "ERROR_HOTKEY_THREAD")
                    # Continue running even on critical errors to maintain system stability
                    time.sleep(0.1)  # Longer sleep on error
                    continue
            
            log_with_timestamp("Console hotkey listener thread stopped gracefully", "SYSTEM")
        
        # Start console hotkey worker thread
        hotkey_thread = threading.Thread(target=console_hotkey_worker, daemon=True)
        hotkey_thread.start()
        hotkey_listener_running = True
        
        log_with_timestamp("Console hotkey listener started successfully", "SYSTEM")
        print_console(f"[HOTKEY] Console hot-keys active ({HOTKEY_MAP['send'].upper()} / {HOTKEY_MAP['repeat'].upper()} / {HOTKEY_MAP['help'].upper()} / {HOTKEY_MAP['skip'].upper()}) – requires console focus", "SYSTEM")
        return True
        
    except Exception as e:
        log_with_timestamp(f"Failed to start console hotkey listener: {e}", "ERROR")
        return False

# user_input_worker() and display_transcription_and_prompt() functions removed
# These have been replaced by the non-blocking hotkey system

def auto_send_monitor():
    """
    Auto-send monitor thread - DISABLED.
    Auto-send feature has been disabled. Only manual send via 'g' key is available.
    """
    global auto_send_monitor_running
    
    log_with_timestamp("Auto-send monitor thread disabled - manual send only", "SYSTEM")
    
    # Keep thread alive but do nothing
    while auto_send_monitor_running:
        try:
            time.sleep(1.0)  # Sleep for 1 second to reduce CPU usage
        except Exception:
            break
    
    log_with_timestamp("Auto-send monitor thread stopped", "SYSTEM")

def streaming_transcription_worker():
    """
    Live transcription worker using RMS-based speech detection.
    Processes complete speech segments immediately after silence detection.
    Filters out nonsense chunks and accumulates valid text in live_buffer.
    """
    global last_spanish, transcription_worker_running, last_buffer_update_time
    
    log_with_timestamp("Live transcription worker thread started", "SYSTEM")
    
    while transcription_worker_running:
        audio_data = None
        task_done_called = False
        
        try:
            # Pull complete speech segment from queue
            try:
                audio_data = audio_queue.get(timeout=1.0)
                log_with_timestamp("Processing audio chunk", "USER_FEEDBACK")
            except queue.Empty:
                continue
            except Exception as queue_error:
                log_with_timestamp(f"Error getting audio from queue: {queue_error}", "ERROR")
                continue
            
            try:
                # Validate audio data
                if audio_data is None or len(audio_data) == 0:
                    log_with_timestamp("Empty audio data received", "ERROR")
                    audio_queue.task_done()
                    task_done_called = True
                    continue
                
                # Start processing timing
                processing_start_time = time.time()
                
                # Resample audio if needed
                try:
                    if SAMPLE_RATE != TARGET_RATE:
                        original_length = len(audio_data)
                        new_length = int(original_length * TARGET_RATE / SAMPLE_RATE)
                        audio_data = resample(audio_data, new_length)
                        log_with_timestamp(f"Audio resampled from {original_length} to {new_length} samples", "AUDIO")
                except Exception as resample_error:
                    log_with_timestamp(f"Audio resampling failed: {resample_error}", "ERROR")
                    audio_queue.task_done()
                    task_done_called = True
                    continue
                
                # Transcribe with Whisper
                try:
                    print_console("🔄 Processing audio...", "TRANSCRIBE_ES")
                    log_with_timestamp("Starting live transcription", "TRANSCRIBE_ES")
                    
                    segments, info = model.transcribe(
                        audio_data,
                        language="es",
                        task="transcribe",
                        beam_size=1,  # Reduced from 2 for speed (-70ms)
                        best_of=1,    # Reduced from 2 for speed (-70ms)
                        vad_filter=True,
                        temperature=0.0,
                        no_speech_threshold=0.6  # Slightly more permissive (-20ms)
                    )
                    
                    # Extract text
                    transcription_text = " ".join([s.text.strip() for s in segments])
                    
                    # Calculate processing time
                    processing_time = time.time() - processing_start_time
                    
                    # Handle empty results
                    if not transcription_text.strip():
                        log_with_timestamp("Empty transcription result - no speech detected", "TRANSCRIBE_ES")
                        audio_queue.task_done()
                        task_done_called = True
                        continue
                    
                    # Check no_speech_probability threshold
                    if hasattr(info, 'no_speech_prob') and info.no_speech_prob > 0.66:
                        log_with_timestamp(f"Skipping low-confidence transcription (no_speech_prob: {info.no_speech_prob:.3f})", "TRANSCRIBE_ES")
                        audio_queue.task_done()
                        task_done_called = True
                        continue
                    
                    # Filter out nonsense chunks
                    clean_text = transcription_text.strip()
                    if is_nonsense_chunk(clean_text):
                        log_with_timestamp(f"Filtered nonsense chunk: '{clean_text}'", "TRANSCRIBE_ES")
                        audio_queue.task_done()
                        task_done_called = True
                        continue
                    
                    # Check for duplicates
                    current_timestamp = datetime.now()
                    if is_duplicate_transcription(clean_text, current_timestamp):
                        time_diff = (current_timestamp - last_transcription_timestamp).total_seconds()
                        log_with_timestamp(f"Duplicate transcription skipped: '{clean_text}' (identical to result {time_diff:.1f}s ago)", "DUPLICATE_SKIP")
                        audio_queue.task_done()
                        task_done_called = True
                        continue
                    
                    # Update tracking
                    update_last_transcription(clean_text, current_timestamp)
                    last_spanish = clean_text
                    
                    # Add to live buffer and update hotkey state
                    with live_buffer_lock:
                        live_buffer.append(clean_text)
                        # Update buffer timing (auto-send disabled)
                        last_buffer_update_time = datetime.now()
                        
                        # Show transcription completion and buffer status
                        print_console("✅ Transcription complete", "TRANSCRIBE_ES")
                        
                        # Show friendly buffer status (only if >= 20 chars)
                        total_chars = len(" ".join(live_buffer))
                        if total_chars >= 20:
                            print_console(f"📄  {total_chars} chars captured   |   {HOTKEY_MAP['send'].upper()} to send", "BUFFER_UPDATE")
                        else:
                            # Clear the transcribing cue for short transcriptions
                            print_console("📄  Buffer updated", "BUFFER_UPDATE")
                    
                    with hotkey_lock:
                        hotkey_state["last_transcription"] = clean_text
                    
                    # Log the live text (keep detailed logging)
                    log_with_timestamp(f"{clean_text}", "LIVE_ES")
                    
                    # Mark task as done
                    audio_queue.task_done()
                    task_done_called = True
                    
                except Exception as transcription_error:
                    error_msg = f"Whisper transcription failed: {str(transcription_error)}"
                    log_with_timestamp(error_msg, "ERROR")
                    
            except Exception as processing_error:
                log_with_timestamp(f"Error processing audio data: {processing_error}", "ERROR")
            finally:
                if not task_done_called:
                    try:
                        audio_queue.task_done()
                    except Exception:
                        pass
            
        except Exception as worker_error:
            log_with_timestamp(f"Critical transcription worker error: {str(worker_error)}", "ERROR")
            if audio_data is not None and not task_done_called:
                try:
                    audio_queue.task_done()
                except:
                    pass
            continue
    
    log_with_timestamp("Live transcription worker thread stopped", "SYSTEM")

# Live transcription system using RMS-based speech detection

# Show input device info with comprehensive error handling
try:
    info = sd.query_devices(DEVICE_INDEX)
    log_with_timestamp(f"Using audio device {DEVICE_INDEX}: {info['name']}", "SYSTEM")
    
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
    print_console(f"❌ PortAudio device error: {pa_error}", "ERROR")
    print_console("Available devices:", "ERROR")
    try:
        print_console(str(sd.query_devices()), "ERROR")
    except:
        pass
    exit(1)
except Exception as e:
    log_with_timestamp(f"Audio device error: {e}", "ERROR")
    print_console(f"❌ Device error: {e}", "ERROR")
    print_console("Available devices:", "ERROR")
    try:
        print_console(str(sd.query_devices()), "ERROR")
    except:
        pass
    exit(1)



# Audio callback with RMS-based pause detection (from stablever.py)
def callback(indata, frames, time, status):
    """
    Audio callback that implements RMS-based pause detection and audio buffering.
    Continuously buffers audio and processes complete speech segments.
    """
    global audio_buffer, is_speaking, silence_start, speech_start_time, soft_flush_countdown_start, soft_flush_silence_start
    
    try:
        if status:
            log_with_timestamp(f"Audio status warning: {status}", "AUDIO")
        
        # Validate input data
        if indata is None or len(indata) == 0:
            log_with_timestamp("Empty audio input received", "ERROR")
            return
        
        # Get audio chunk (mono channel)
        try:
            audio_chunk = indata[:, 0]
        except IndexError as e:
            log_with_timestamp(f"Audio channel indexing error: {e}", "ERROR")
            return
        except Exception as e:
            log_with_timestamp(f"Audio data extraction error: {e}", "ERROR")
            return
        
        # Update speaking state and check if speech segment is complete
        try:
            speech_complete = update_speaking_state(audio_chunk)
            
            # 🎙️ Live feedback when speech starts - use direct print for immediate display
            if is_speaking and not hasattr(callback, '_speaking_feedback_shown'):
                print("🎙️  Transcribing …")
                log_with_timestamp("User started speaking", "USER_FEEDBACK")
                callback._speaking_feedback_shown = True
                
        except Exception as e:
            log_with_timestamp(f"Error in speaking state detection: {e}", "ERROR")
            return
        
        # Always buffer audio when we're speaking or just finished speaking
        try:
            with speech_buffer_lock:
                if is_speaking or speech_complete:
                    audio_buffer.extend(audio_chunk)
                
                # If speech segment is complete, process the buffered audio
                if speech_complete and len(audio_buffer) > 0:
                    # Convert buffer to numpy array for processing
                    try:
                        complete_audio = np.array(audio_buffer)
                        log_with_timestamp(f"Processing complete speech segment ({len(complete_audio)} samples, {len(complete_audio)/SAMPLE_RATE:.2f}s)", "AUDIO")
                        
                        # Queue audio for transcription worker thread
                        try:
                            audio_queue.put(complete_audio.copy(), block=False)
                            log_with_timestamp("Audio queued for live transcription", "AUDIO")
                        except queue.Full:
                            log_with_timestamp("Audio queue is full, dropping audio segment", "ERROR")
                        except Exception as queue_error:
                            log_with_timestamp(f"Error queuing audio: {queue_error}", "ERROR")
                        
                        # Clear the buffer for next speech segment
                        audio_buffer.clear()
                        
                        # Reset speaking feedback flag so it can show again for next speech
                        if hasattr(callback, '_speaking_feedback_shown'):
                            callback._speaking_feedback_shown = False
                        
                    except Exception as processing_error:
                        log_with_timestamp(f"Error processing complete audio segment: {processing_error}", "ERROR")
                        # Clear buffer to prevent corruption
                        audio_buffer.clear()
                        
        except Exception as buffer_error:
            log_with_timestamp(f"Error in audio buffering: {buffer_error}", "ERROR")
        
        # Soft-hold forced flush logic with proper timing constants
        if is_speaking and speech_start_time:
            duration = (datetime.now() - speech_start_time).total_seconds()
            
            # Check absolute 5s cap first (HARD_FLUSH_TIMEOUT)
            if duration >= HARD_FLUSH_TIMEOUT:
                # Force-complete the segment at absolute cap
                is_speaking = False
                silence_start = None
                speech_start_time = None
                soft_flush_countdown_start = None
                soft_flush_silence_start = None
                
                # Log flush event with proper format
                log_flush_event("hard", duration)
                
                # Process buffered audio
                with speech_buffer_lock:
                    if len(audio_buffer) > 0:
                        try:
                            complete_audio = np.array(audio_buffer)
                            audio_duration = len(complete_audio) / SAMPLE_RATE
                            
                            # Queue audio for transcription worker thread
                            try:
                                audio_queue.put(complete_audio.copy(), block=False)
                                log_with_timestamp("Audio queued for live transcription (absolute cap)", "AUDIO")
                            except queue.Full:
                                log_with_timestamp("Audio queue is full, dropping absolute cap segment", "ERROR")
                            except Exception as queue_error:
                                log_with_timestamp(f"Error queuing absolute cap audio: {queue_error}", "ERROR")
                            
                            # Clear the buffer for next speech segment
                            audio_buffer.clear()
                            
                            # Reset speaking feedback flag so it can show again for next speech
                            if hasattr(callback, '_speaking_feedback_shown'):
                                callback._speaking_feedback_shown = False
                            
                        except Exception as processing_error:
                            log_with_timestamp(f"Error processing absolute cap segment: {processing_error}", "ERROR")
                            # Clear buffer to prevent corruption
                            audio_buffer.clear()
                            
                            # Reset speaking feedback flag so it can show again for next speech
                            if hasattr(callback, '_speaking_feedback_shown'):
                                callback._speaking_feedback_shown = False
                return
            
            # Start soft-hold countdown after MAX_SPEECH_SECS (3.2s)
            if duration >= MAX_SPEECH_SECS and soft_flush_countdown_start is None:
                soft_flush_countdown_start = datetime.now()
                log_with_timestamp(f"SOFT_HOLD_START after {duration:.1f}s - {SOFT_HOLD_MS}ms countdown begins", "AUDIO")
            
            # Handle countdown period
            if soft_flush_countdown_start is not None:
                countdown_elapsed = (datetime.now() - soft_flush_countdown_start).total_seconds()
                
                # Check for silence during countdown
                rms = compute_rms(audio_chunk)
                if rms <= RMS_THRESHOLD:
                    if soft_flush_silence_start is None:
                        soft_flush_silence_start = datetime.now()
                        log_with_timestamp("SOFT_HOLD_SILENCE_START - silence detected during countdown", "AUDIO")
                else:
                    # Reset silence tracking if speech resumes
                    soft_flush_silence_start = None
                
                # Check if we have enough silence to trigger flush (MIN_SILENCE_MS)
                if soft_flush_silence_start is not None:
                    silence_duration = (datetime.now() - soft_flush_silence_start).total_seconds()
                    if silence_duration >= (MIN_SILENCE_MS / 1000.0):  # Convert ms to seconds
                        # Soft-hold flush triggered by sufficient silence
                        is_speaking = False
                        silence_start = None
                        speech_start_time = None
                        soft_flush_countdown_start = None
                        soft_flush_silence_start = None
                        
                        # Log flush event with proper format
                        log_flush_event("soft", duration)
                        
                        # Process buffered audio (reuse same code path)
                        with speech_buffer_lock:
                            if len(audio_buffer) > 0:
                                try:
                                    complete_audio = np.array(audio_buffer)
                                    audio_duration = len(complete_audio) / SAMPLE_RATE
                                    
                                    # Queue audio for transcription worker thread
                                    try:
                                        audio_queue.put(complete_audio.copy(), block=False)
                                        log_with_timestamp("Audio queued for live transcription (soft-hold)", "AUDIO")
                                    except queue.Full:
                                        log_with_timestamp("Audio queue is full, dropping soft-hold segment", "ERROR")
                                    except Exception as queue_error:
                                        log_with_timestamp(f"Error queuing soft-hold audio: {queue_error}", "ERROR")
                                    
                                    # Clear the buffer for next speech segment
                                    audio_buffer.clear()
                                    
                                    # Reset speaking feedback flag so it can show again for next speech
                                    if hasattr(callback, '_speaking_feedback_shown'):
                                        callback._speaking_feedback_shown = False
                                    
                                except Exception as processing_error:
                                    log_with_timestamp(f"Error processing soft-hold segment: {processing_error}", "ERROR")
                                    # Clear buffer to prevent corruption
                                    audio_buffer.clear()
                                    
                                    # Reset speaking feedback flag so it can show again for next speech
                                    if hasattr(callback, '_speaking_feedback_shown'):
                                        callback._speaking_feedback_shown = False
                        return
                
                # Hard flush if countdown expires without sufficient silence
                if countdown_elapsed >= (SOFT_HOLD_MS / 1000.0):  # Convert ms to seconds
                    # Force-complete the segment
                    is_speaking = False
                    silence_start = None
                    speech_start_time = None
                    soft_flush_countdown_start = None
                    soft_flush_silence_start = None
                    
                    # Log flush event with proper format
                    log_flush_event("hard", duration)
                    
                    # Process buffered audio (reuse same code path)
                    with speech_buffer_lock:
                        if len(audio_buffer) > 0:
                            try:
                                complete_audio = np.array(audio_buffer)
                                audio_duration = len(complete_audio) / SAMPLE_RATE
                                
                                # Queue audio for transcription worker thread
                                try:
                                    audio_queue.put(complete_audio.copy(), block=False)
                                    log_with_timestamp("Audio queued for live transcription (hard flush)", "AUDIO")
                                except queue.Full:
                                    log_with_timestamp("Audio queue is full, dropping hard flush segment", "ERROR")
                                except Exception as queue_error:
                                    log_with_timestamp(f"Error queuing hard flush audio: {queue_error}", "ERROR")
                                
                                # Clear the buffer for next speech segment
                                audio_buffer.clear()
                                
                                # Reset speaking feedback flag so it can show again for next speech
                                if hasattr(callback, '_speaking_feedback_shown'):
                                    callback._speaking_feedback_shown = False
                                
                            except Exception as processing_error:
                                log_with_timestamp(f"Error processing hard flush segment: {processing_error}", "ERROR")
                                # Clear buffer to prevent corruption
                                audio_buffer.clear()
                                
                                # Reset speaking feedback flag so it can show again for next speech
                                if hasattr(callback, '_speaking_feedback_shown'):
                                    callback._speaking_feedback_shown = False
            
    except Exception as callback_error:
        log_with_timestamp(f"Critical error in audio callback: {callback_error}", "ERROR")

def cleanup_resources():
    """
    Enhanced cleanup function for graceful shutdown with all worker threads.
    Handles transcription worker, hotkey listener, auto-send monitor (disabled), and resource management.
    """
    global transcription_worker_running, hotkey_listener_running, auto_send_monitor_running
    global transcription_thread, hotkey_thread, auto_send_thread, audio_stream
    
    log_with_timestamp("Starting system cleanup", "SYSTEM")
    
    try:
        # Signal all worker threads to stop
        transcription_worker_running = False
        hotkey_listener_running = False
        auto_send_monitor_running = False
        log_with_timestamp("Signaled worker threads to stop", "SYSTEM")
        
        # Unhook global hotkeys if using keyboard library
        try:
            if _HOTKEY_LIB == "keyboard":
                keyboard.unhook_all_hotkeys()
                log_with_timestamp("Global hotkeys unhooked successfully", "SYSTEM")
        except ImportError:
            # keyboard library not available, skip unhooking
            pass
        except Exception as unhook_error:
            log_with_timestamp(f"Error unhooking global hotkeys: {unhook_error}", "ERROR")
        
        # Stop pynput listener if using pynput library
        try:
            if _HOTKEY_LIB == "pynput" and _pynput_listener is not None:
                _pynput_listener.stop()
                log_with_timestamp("Pynput listener stopped successfully", "SYSTEM")
        except Exception as pynput_error:
            log_with_timestamp(f"Error stopping pynput listener: {pynput_error}", "ERROR")
        
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
        
        # Wait for hotkey listener thread to finish with timeout
        if hotkey_thread and hotkey_thread.is_alive():
            try:
                log_with_timestamp("Waiting for hotkey listener thread to finish", "SYSTEM")
                hotkey_thread.join(timeout=1.0)  # 1 second timeout
                
                if hotkey_thread.is_alive():
                    log_with_timestamp("Hotkey listener thread did not finish within timeout", "SYSTEM")
                else:
                    log_with_timestamp("Hotkey listener thread finished successfully", "SYSTEM")
                    
            except Exception as thread_error:
                log_with_timestamp(f"Error joining hotkey listener thread: {thread_error}", "ERROR")
        else:
            log_with_timestamp("No hotkey thread to join (global hotkeys may be using library threads)", "SYSTEM")
        
        # Wait for auto-send monitor thread to finish with timeout
        if auto_send_thread and auto_send_thread.is_alive():
            try:
                log_with_timestamp("Waiting for auto-send monitor thread to finish", "SYSTEM")
                auto_send_thread.join(timeout=1.0)  # 1 second timeout
                
                if auto_send_thread.is_alive():
                    log_with_timestamp("Auto-send monitor thread did not finish within timeout", "SYSTEM")
                else:
                    log_with_timestamp("Auto-send monitor thread finished successfully", "SYSTEM")
                    
            except Exception as thread_error:
                log_with_timestamp(f"Error joining auto-send monitor thread: {thread_error}", "ERROR")
        
        # Clear audio buffer and live buffer
        try:
            with speech_buffer_lock:
                audio_buffer.clear()
                log_with_timestamp("Audio buffer cleared", "SYSTEM")
        except Exception as buffer_error:
            log_with_timestamp(f"Error clearing audio buffer: {buffer_error}", "ERROR")
        
        try:
            with live_buffer_lock:
                live_buffer.clear()
                log_with_timestamp("Live buffer cleared", "SYSTEM")
        except Exception as live_error:
            log_with_timestamp(f"Error clearing live buffer: {live_error}", "ERROR")
        
        # Log final statistics
        try:
            audio_queue_size = audio_queue.qsize()
            if audio_queue_size > 0:
                log_with_timestamp(f"Warning: {audio_queue_size} items remaining in audio queue", "SYSTEM")
        except Exception:
            pass
        
        log_with_timestamp("System cleanup completed", "SYSTEM")
        
    except Exception as cleanup_error:
        log_with_timestamp(f"Error during cleanup: {cleanup_error}", "ERROR")

def main():
    """
    Enhanced main execution function with robust error handling and graceful shutdown.
    Starts transcription worker, hotkey listener, and auto-send monitor (disabled) for non-blocking operation.
    """
    global transcription_thread, hotkey_thread, auto_send_thread, audio_stream
    global hotkey_listener_running, auto_send_monitor_running
    
    # Display system startup information
    log_with_timestamp("=== SPANISH TRANSCRIPTION SYSTEM STARTING ===", "SYSTEM")
    log_with_timestamp("Shared state initialized: hotkey_state and hotkey_lock ready", "SYSTEM")
    log_with_timestamp("Thread architecture: transcription worker + hotkey listener + auto-send monitor (disabled)", "SYSTEM")
    
    try:
        # Initialize and start core system threads
        log_with_timestamp("Starting system threads...", "SYSTEM")
        
        # Start streaming transcription worker thread with error handling
        try:
            transcription_thread = threading.Thread(target=streaming_transcription_worker, daemon=True)
            transcription_thread.start()
            log_with_timestamp("✅ Streaming transcription worker thread started successfully", "SYSTEM")
        except Exception as thread_error:
            log_with_timestamp(f"❌ Failed to start streaming transcription worker thread: {thread_error}", "ERROR")
            raise
        
        # Start global hotkey listener with error handling
        try:
            hotkey_success = start_global_hotkeys()
            if hotkey_success:
                log_with_timestamp("✅ Global hotkey listener started successfully", "SYSTEM")
            else:
                log_with_timestamp("❌ Failed to start global hotkey listener", "ERROR")
                raise RuntimeError("Failed to start global hotkey listener")
        except Exception as hotkey_error:
            log_with_timestamp(f"❌ Failed to start global hotkey listener: {hotkey_error}", "ERROR")
            raise
        
        # Start auto-send monitor thread (disabled) with error handling
        try:
            auto_send_monitor_running = True
            auto_send_thread = threading.Thread(target=auto_send_monitor, daemon=True)
            auto_send_thread.start()
            log_with_timestamp("✅ Auto-send monitor thread started (disabled)", "SYSTEM")
        except Exception as thread_error:
            log_with_timestamp(f"❌ Failed to start auto-send monitor thread: {thread_error}", "ERROR")
            raise
        
        # Verify all threads are alive
        if transcription_thread.is_alive() and auto_send_thread.is_alive():
            log_with_timestamp("✅ All system threads running successfully", "SYSTEM")
        else:
            raise RuntimeError("One or more system threads failed to start properly")
        
        # Show initialization status
        print_console("[Init] Loading models …", "INIT")
        print_console("Whisper small‑fp16         ✔", "INIT")
        
        # Start Gemini warm-up in background thread to avoid blocking startup
        try:
            warmup_thread = threading.Thread(target=warm_up_gemini, daemon=True)
            warmup_thread.start()
            log_with_timestamp("✅ Gemini warm-up thread started", "SYSTEM")
            print_console("Warming up Gemini …", "INIT")
        except Exception as warmup_thread_error:
            log_with_timestamp(f"⚠️ Failed to start Gemini warm-up thread: {warmup_thread_error}", "ERROR")
            # Don't raise - warm-up is optional and shouldn't block startup
        
        # Start audio stream with comprehensive error handling
        log_with_timestamp("Initializing audio stream...", "SYSTEM")
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
                log_with_timestamp("✅ Audio stream started successfully", "SYSTEM")
                log_with_timestamp("=== SYSTEM FULLY OPERATIONAL ===", "SYSTEM")
                
                # Display startup banner
                print_startup_banner()
                
                # Main listening loop with enhanced shutdown handling
                try:
                    while True:
                        sd.sleep(100)  # Shorter sleep for more responsive shutdown
                except KeyboardInterrupt:
                    print_console("\n🛑 Keyboard interrupt received. Shutting down gracefully...", "ERROR")
                    log_with_timestamp("Keyboard interrupt received - initiating graceful shutdown", "SYSTEM")
                    raise
                except EOFError:
                    print_console("\n🛑 EOF received. Shutting down gracefully...", "ERROR")
                    log_with_timestamp("EOF received - initiating graceful shutdown", "SYSTEM")
                    raise
                except Exception as loop_error:
                    log_with_timestamp(f"Error in main listening loop: {loop_error}", "ERROR")
                    print_console(f"❌ Main loop error: {loop_error}", "ERROR")
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
        print_console("\n🛑 Shutdown initiated by user...", "ERROR")
        log_with_timestamp("Application terminated by user (Ctrl+C)", "SYSTEM")
    except EOFError:
        print_console("\n🛑 EOF received, shutting down...", "ERROR")
        log_with_timestamp("Application terminated by EOF", "SYSTEM")
    except Exception as main_error:
        log_with_timestamp(f"Critical error in main execution: {main_error}", "ERROR")
        print_console(f"❌ Critical error: {main_error}", "ERROR")
        import traceback
        log_with_timestamp(f"Full traceback: {traceback.format_exc()}", "ERROR")
    finally:
        # Always perform cleanup with enhanced error handling
        print_console("🔧 Cleaning up resources...", "ERROR")
        try:
            cleanup_resources()
            print_console("✅ Cleanup completed successfully.", "ERROR")
        except KeyboardInterrupt:
            print_console("\n⚠️ Cleanup interrupted - forcing immediate exit", "ERROR")
            log_with_timestamp("Cleanup interrupted by user - forcing exit", "SYSTEM")
        except Exception as final_cleanup_error:
            log_with_timestamp(f"Error during final cleanup: {final_cleanup_error}", "ERROR")
            print_console(f"❌ Cleanup error: {final_cleanup_error}", "ERROR")
            print_console("⚠️ Some resources may not have been cleaned up properly", "ERROR")

def dump_key_mapping():
    """Print the current key mapping configuration."""
    print("=== Current Key Mapping ===")
    print(f"Modifiers: {' + '.join(KEY_CONFIG['modifiers'])}")
    print()
    print("Hotkeys:")
    for action, key in KEY_CONFIG["keys"].items():
        combo = f"{' + '.join(KEY_CONFIG['modifiers'])} + {key.upper()}"
        print(f"  {combo:<20} → {action}")
    print()
    print("To customize hotkeys, edit the KEY_CONFIG dictionary in main.py")
    print("Format: KEY_CONFIG['keys']['action'] = 'new_key'")

# Run the main function
if __name__ == "__main__":
    import sys
    
    # Check for --keys dump flag
    if "--keys" in sys.argv and "dump" in sys.argv:
        dump_key_mapping()
        sys.exit(0)
    
    try:
        main()
    except Exception as startup_error:
        log_with_timestamp(f"Failed to start application: {startup_error}", "ERROR")
        print_console(f"❌ Application startup failed: {startup_error}", "ERROR")
        exit(1)
