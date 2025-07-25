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
import msvcrt
import time
import re  # Added for filtering nonsense chunks
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
RMS_THRESHOLD = 0.008  # Lower threshold for breath-level latency
SILENCE_SECS = 0.3     # Breath-level pause detection for immediate response
MAX_SPEECH_SECS = 4.0  # Flush every 4 seconds if still speaking
AUTO_SEND_AFTER_SECS = 8  # seconds of silence before auto-fire
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

# Live transcription performance monitoring removed - now using immediate processing

def log_with_timestamp(message, event_type="INFO"):
    """
    Thread-safe logging function with timestamp formatting.
    Logs to both console and log file with proper error handling.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] [{event_type}] {message}"
    
    # Only print to console for certain event types (suppress internal diagnostics)
    console_events = {"ERROR", "SYSTEM", "GEMINI_REPLY", "GEMINI_REPLY_REPEAT", "AUTO_SEND", "HOTKEY_G", "HOTKEY_R", "HOTKEY_H"}
    if event_type in console_events:
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
log_with_timestamp("🚀 Performance: True live streaming with forced flush", "AUDIO_FORMAT")
log_with_timestamp("📊 Settings: RMS_THRESHOLD=0.008, SILENCE_SECS=0.3, MAX_SPEECH_SECS=4.0", "AUDIO_FORMAT")

# Audio settings
SAMPLE_RATE = 48000
TARGET_RATE = 16000
BLOCK_DURATION = 2
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
    "gemini_busy": False
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
    global is_speaking, silence_start, speech_start_time
    
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
    Shared function to send content to Gemini, used by both manual g-key and auto-send.
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
        
        word_count = len(full_text.split())
        
        with hotkey_lock:
            hotkey_state["gemini_busy"] = True
        
        try:
            start = time.time()
            
            # Call Gemini with the content
            reply = call_gemini_api(full_text)
            elapsed = time.time() - start
            
            if reply is None:
                raise ValueError("Gemini API returned no response")
            
            with hotkey_lock:
                hotkey_state["last_gemini"] = reply
                hotkey_state["gemini_busy"] = False
            
            # Clear the live buffer after successful send
            with live_buffer_lock:
                live_buffer.clear()
            
            log_with_timestamp(f"{elapsed:.2f}s", "GEMINI_RESPONSE_TIME")
            print(f"🤖 Gemini: {reply}")
            
            return True
            
        except KeyboardInterrupt:
            with hotkey_lock:
                hotkey_state["gemini_busy"] = False
            log_with_timestamp("Gemini call interrupted by user", "ERROR_GEMINI")
            raise
        except Exception as gemini_error:
            with hotkey_lock:
                hotkey_state["gemini_busy"] = False
            log_with_timestamp(f"Gemini API error: {str(gemini_error)}", "ERROR_GEMINI")
            print("🤖 ❌ Gemini request failed. Check logs for details.")
            return False
            
    except Exception as handler_error:
        log_with_timestamp(f"Critical error in send_to_gemini: {str(handler_error)}", "ERROR_SEND_GEMINI")
        try:
            with hotkey_lock:
                hotkey_state["gemini_busy"] = False
        except:
            pass
        return False

def handle_g_key(state, lock, log_event, call_gemini):
    """Send all live buffered text to Gemini if available and not busy."""
    try:
        # Get all live buffer content
        with live_buffer_lock:
            if not live_buffer:
                log_event("No live text available", "HOTKEY_G")
                return
            
            # Create a copy for processing
            buffer_copy = live_buffer.copy()
            word_count = len(" ".join(buffer_copy).split())
            chunk_count = len(buffer_copy)
        
        # Log what we're sending
        log_event(f"{word_count}w / {chunk_count}c", "SEND_GEMINI")
        
        # Use shared send function
        success = send_to_gemini(buffer_copy)
        if success:
            log_event("Manual send completed", "HOTKEY_G")
        else:
            log_event("Manual send failed", "HOTKEY_G")
            
    except Exception as handler_error:
        log_event(f"Critical error in handle_g_key: {str(handler_error)}", "ERROR_HOTKEY_HANDLER")

def handle_r_key(state, lock, log_event):
    """Reprint the last Gemini reply if available."""
    with lock:
        reply = state.get("last_gemini", "")
    
    if not reply:
        log_event("No previous Gemini reply", "HOTKEY_R")
        return
    
    log_event("Reprinting last Gemini reply", "HOTKEY_R")
    print(f"🤖 Gemini: {reply}")

def handle_h_key(log_event):
    """Display help text for available hotkeys."""
    help_text = (
        "\n[Hotkeys]\n"
        "  g  → Send all live text to Gemini\n"
        "  r  → Repeat last Gemini reply\n"
        "  h  → Show this help\n"
        "  q  → Skip / ignore\n"
    )
    log_event("Help shown", "HOTKEY_H")
    print(help_text)

# HANDLERS_OK
# <<< HOTKEY_HANDLERS_END


def hotkey_listener_worker():
    """
    Enhanced hotkey listener thread with comprehensive error handling and graceful shutdown.
    Implements 300ms debouncing with timestamp tracking per key.
    Handles g, r, h, q keys with thread-safe shared state access.
    All exceptions are caught and logged without crashing the main system.
    """
    global hotkey_listener_running
    
    # Debouncing - track last press time for each key
    debounce_timers = {}
    debounce_delay = 0.3  # 300ms debounce
    
    log_with_timestamp("Hotkey listener thread started with enhanced error handling", "SYSTEM")
    
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
                        if key == 'g':
                            handle_g_key(hotkey_state, hotkey_lock, log_with_timestamp, call_gemini_api)
                            
                        elif key == 'r':
                            handle_r_key(hotkey_state, hotkey_lock, log_with_timestamp)
                            
                        elif key == 'h':
                            handle_h_key(log_with_timestamp)
                            
                        elif key == 'q':
                            log_with_timestamp("Hotkey 'q' pressed", "HOTKEY_Q")
                            # Handle q key - will be implemented in next task
                            
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
            log_with_timestamp("Hotkey listener thread interrupted by user", "SYSTEM")
            break
        except Exception as listener_error:
            log_with_timestamp(f"Critical hotkey listener error: {listener_error}", "ERROR_HOTKEY_THREAD")
            # Continue running even on critical errors to maintain system stability
            time.sleep(0.1)  # Longer sleep on error
            continue
    
    log_with_timestamp("Hotkey listener thread stopped gracefully", "SYSTEM")

# user_input_worker() and display_transcription_and_prompt() functions removed
# These have been replaced by the non-blocking hotkey system

def auto_send_monitor():
    """
    Auto-send monitor thread that checks for idle silence and auto-sends buffer content.
    Runs every 0.3s and triggers auto-send after AUTO_SEND_AFTER_SECS seconds of inactivity.
    """
    global auto_send_monitor_running, last_buffer_update_time
    
    log_with_timestamp("Auto-send monitor thread started", "SYSTEM")
    
    while auto_send_monitor_running:
        try:
            time.sleep(0.3)  # Check every 0.3 seconds
            
            # Check conditions for auto-send
            with live_buffer_lock:
                buffer_empty = len(live_buffer) == 0
                if buffer_empty:
                    continue
                
                buffer_copy = live_buffer.copy()
                char_count = len(" ".join(buffer_copy))
            
            with hotkey_lock:
                gemini_busy = hotkey_state.get("gemini_busy", False)
            
            if gemini_busy:
                continue
            
            # Check if enough time has passed since last buffer update
            if last_buffer_update_time is None:
                continue
            
            current_time = datetime.now()
            time_since_update = (current_time - last_buffer_update_time).total_seconds()
            
            if time_since_update >= AUTO_SEND_AFTER_SECS:
                # Trigger auto-send
                word_count = len(" ".join(buffer_copy).split())
                log_with_timestamp(f"Auto-sending {word_count} words after {time_since_update:.1f}s silence", "AUTO_SEND")
                print(f"📨 Auto-sent {word_count} words to Gemini – waiting for reply…")
                
                # Use shared send function
                success = send_to_gemini(buffer_copy)
                if success:
                    # Reset the update time to prevent immediate retrigger
                    last_buffer_update_time = None
                else:
                    log_with_timestamp("Auto-send failed", "AUTO_SEND")
            
        except Exception as monitor_error:
            log_with_timestamp(f"Error in auto-send monitor: {monitor_error}", "ERROR_AUTO_SEND")
            continue
    
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
                    log_with_timestamp("Starting live transcription", "TRANSCRIBE_ES")
                    
                    segments, info = model.transcribe(
                        audio_data,
                        language="es",
                        task="transcribe",
                        beam_size=5,
                        best_of=5,
                        vad_filter=True,
                        temperature=0.0,
                        no_speech_threshold=0.5
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
                        # Update buffer timing for auto-send
                        last_buffer_update_time = datetime.now()
                        
                        # Show friendly buffer status
                        total_chars = len(" ".join(live_buffer))
                        print(f"📄 Buffer: {total_chars} chars  (press g or wait 8 s)")
                    
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



# Audio callback with RMS-based pause detection (from stablever.py)
def callback(indata, frames, time, status):
    """
    Audio callback that implements RMS-based pause detection and audio buffering.
    Continuously buffers audio and processes complete speech segments.
    """
    global audio_buffer, is_speaking, silence_start, speech_start_time
    
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
                        
                    except Exception as processing_error:
                        log_with_timestamp(f"Error processing complete audio segment: {processing_error}", "ERROR")
                        # Clear buffer to prevent corruption
                        audio_buffer.clear()
                        
        except Exception as buffer_error:
            log_with_timestamp(f"Error in audio buffering: {buffer_error}", "ERROR")
        
        # Live-flush if speaking too long
        if is_speaking and speech_start_time:
            duration = (datetime.now() - speech_start_time).total_seconds()
            if duration >= MAX_SPEECH_SECS:
                # Force-complete the segment
                is_speaking = False
                silence_start = None
                speech_start_time = None
                log_with_timestamp(f"FORCED_FLUSH after {duration:.1f}s continuous speech", "AUDIO")
                
                # Process buffered audio (reuse same code path)
                with speech_buffer_lock:
                    if len(audio_buffer) > 0:
                        try:
                            complete_audio = np.array(audio_buffer)
                            log_with_timestamp(f"Processing forced flush segment ({len(complete_audio)} samples, {len(complete_audio)/SAMPLE_RATE:.2f}s)", "AUDIO")
                            
                            # Queue audio for transcription worker thread
                            try:
                                audio_queue.put(complete_audio.copy(), block=False)
                                log_with_timestamp("Audio queued for live transcription (forced flush)", "AUDIO")
                            except queue.Full:
                                log_with_timestamp("Audio queue is full, dropping forced flush segment", "ERROR")
                            except Exception as queue_error:
                                log_with_timestamp(f"Error queuing forced flush audio: {queue_error}", "ERROR")
                            
                            # Clear the buffer for next speech segment
                            audio_buffer.clear()
                            
                        except Exception as processing_error:
                            log_with_timestamp(f"Error processing forced flush segment: {processing_error}", "ERROR")
                            # Clear buffer to prevent corruption
                            audio_buffer.clear()
            
    except Exception as callback_error:
        log_with_timestamp(f"Critical error in audio callback: {callback_error}", "ERROR")

def cleanup_resources():
    """
    Enhanced cleanup function for graceful shutdown with all worker threads.
    Handles transcription worker, hotkey listener, auto-send monitor, and resource management.
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
    Starts transcription worker, hotkey listener, and auto-send monitor for non-blocking operation.
    """
    global transcription_thread, hotkey_thread, auto_send_thread, audio_stream
    global hotkey_listener_running, auto_send_monitor_running
    
    # Display system startup information
    log_with_timestamp("=== SPANISH TRANSCRIPTION SYSTEM STARTING ===", "SYSTEM")
    log_with_timestamp("Shared state initialized: hotkey_state and hotkey_lock ready", "SYSTEM")
    log_with_timestamp("Thread architecture: transcription worker + hotkey listener + auto-send monitor", "SYSTEM")
    
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
        
        # Start hotkey listener thread with error handling
        try:
            hotkey_listener_running = True
            hotkey_thread = threading.Thread(target=hotkey_listener_worker, daemon=True)
            hotkey_thread.start()
            log_with_timestamp("✅ Hotkey listener thread started successfully", "SYSTEM")
        except Exception as thread_error:
            log_with_timestamp(f"❌ Failed to start hotkey listener thread: {thread_error}", "ERROR")
            raise
        
        # Start auto-send monitor thread with error handling
        try:
            auto_send_monitor_running = True
            auto_send_thread = threading.Thread(target=auto_send_monitor, daemon=True)
            auto_send_thread.start()
            log_with_timestamp("✅ Auto-send monitor thread started successfully", "SYSTEM")
        except Exception as thread_error:
            log_with_timestamp(f"❌ Failed to start auto-send monitor thread: {thread_error}", "ERROR")
            raise
        
        # Verify all threads are alive
        if transcription_thread.is_alive() and hotkey_thread.is_alive() and auto_send_thread.is_alive():
            log_with_timestamp("✅ All system threads running successfully", "SYSTEM")
        else:
            raise RuntimeError("One or more system threads failed to start properly")
        
        # Start Gemini warm-up in background thread to avoid blocking startup
        try:
            warmup_thread = threading.Thread(target=warm_up_gemini, daemon=True)
            warmup_thread.start()
            log_with_timestamp("✅ Gemini warm-up thread started", "SYSTEM")
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
                
                # Display friendly startup banner
                print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                print("🎤  Live Spanish Transcriber READY")
                print("• Speak → text appears immediately")
                print("• [g] send  [r] repeat  [h] help")
                print("• Auto-send after 8 s silence")
                print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                
                # Main listening loop with enhanced shutdown handling
                try:
                    while True:
                        sd.sleep(100)  # Shorter sleep for more responsive shutdown
                except KeyboardInterrupt:
                    print("\n[🛑] Keyboard interrupt received. Shutting down gracefully...")
                    log_with_timestamp("Keyboard interrupt received - initiating graceful shutdown", "SYSTEM")
                    raise
                except EOFError:
                    print("\n[🛑] EOF received. Shutting down gracefully...")
                    log_with_timestamp("EOF received - initiating graceful shutdown", "SYSTEM")
                    raise
                except Exception as loop_error:
                    log_with_timestamp(f"Error in main listening loop: {loop_error}", "ERROR")
                    print(f"[❌] Main loop error: {loop_error}")
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
        print("\n[🛑] Shutdown initiated by user...")
        log_with_timestamp("Application terminated by user (Ctrl+C)", "SYSTEM")
    except EOFError:
        print("\n[🛑] EOF received, shutting down...")
        log_with_timestamp("Application terminated by EOF", "SYSTEM")
    except Exception as main_error:
        log_with_timestamp(f"Critical error in main execution: {main_error}", "ERROR")
        print(f"[❌] Critical error: {main_error}")
        import traceback
        log_with_timestamp(f"Full traceback: {traceback.format_exc()}", "ERROR")
    finally:
        # Always perform cleanup with enhanced error handling
        print("[🔧] Cleaning up resources...")
        try:
            cleanup_resources()
            print("[✅] Cleanup completed successfully.")
        except KeyboardInterrupt:
            print("\n[⚠️] Cleanup interrupted - forcing immediate exit")
            log_with_timestamp("Cleanup interrupted by user - forcing exit", "SYSTEM")
        except Exception as final_cleanup_error:
            log_with_timestamp(f"Error during final cleanup: {final_cleanup_error}", "ERROR")
            print(f"[❌] Cleanup error: {final_cleanup_error}")
            print("[⚠️] Some resources may not have been cleaned up properly")

# Run the main function
if __name__ == "__main__":
    try:
        main()
    except Exception as startup_error:
        log_with_timestamp(f"Failed to start application: {startup_error}", "ERROR")
        print(f"[❌] Application startup failed: {startup_error}")
        exit(1)
