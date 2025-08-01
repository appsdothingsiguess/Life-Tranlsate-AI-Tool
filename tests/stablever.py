import os
import torch
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from scipy.signal import resample
from collections import deque
import threading
import queue
from dotenv import load_dotenv
from google import genai
from datetime import datetime
import logging  # Added for logging configuration

# Configure logging to suppress unwanted messages
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("faster_whisper").setLevel(logging.WARNING)

# System Constants
RMS_THRESHOLD = 0.01
SILENCE_SECS = 2.5
LOG_FILE = "log.txt"
GEMINI_PROMPT = (
    "You are a Spanish tutor. The student is preparing for an oral test. "
    "Reply only with a natural, brief Spanish sentence the student should say. "
    "Do not add explanations or alternatives. Just respond with the one sentence."
)
MODEL_NAME = "gemini-2.5-flash-lite"

# Thread-safe logging infrastructure
_log_lock = threading.Lock()

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

# Audio settings
SAMPLE_RATE = 48000
TARGET_RATE = 16000
BLOCK_DURATION = 2
WINDOW_DURATION = 4
CHANNELS = 1
DEVICE_INDEX = 37  # VB-Audio Virtual Cable WASAPI input

BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)
WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_DURATION)

# RMS-based pause detection state
is_speaking = False
silence_start = None
audio_buffer = deque()
speech_buffer_lock = threading.Lock()

# Transcription worker thread infrastructure
audio_queue = queue.Queue()
last_spanish = ""  # Store last transcription for repeat functionality
last_gemini_response = ""  # Store last Gemini response for repeat functionality
transcription_worker_running = True
transcription_thread = None  # Will hold the worker thread reference
audio_stream = None  # Will hold the audio stream reference

def display_transcription_and_prompt(spanish_text):
    """
    Display Spanish transcription after silence is detected and processed.
    Prompt user with: [Press Enter = Gemini, q = skip, r = repeat]
    Block for input, but do not call Gemini until Enter is pressed.
    Log USER_CONFIRM or USER_SKIP or USER_REPEAT.
    Includes comprehensive error handling for all user interaction scenarios.
    """
    global last_gemini_response
    
    try:
        # Validate input text
        if not spanish_text or not spanish_text.strip():
            log_with_timestamp("Empty or invalid transcription text provided", "ERROR")
            return
        
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
            
    except Exception as display_error:
        log_with_timestamp(f"Critical error in display_transcription_and_prompt: {display_error}", "ERROR")
        print("   Critical error in user interface. Continuing...")

def process_audio():
    """
    Background worker thread function that processes queued audio chunks.
    Pulls from audio_queue and calls faster-whisper transcription.
    Configured for Spanish transcription (no English translation).
    Logs TRANSCRIBE_ES events and handles errors gracefully.
    Includes comprehensive error handling for all transcription operations.
    """
    global last_spanish, transcription_worker_running
    
    log_with_timestamp("Transcription worker thread started", "SYSTEM")
    
    while transcription_worker_running:
        try:
            # Pull from audio_queue with timeout to allow periodic checks
            try:
                audio_data = audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue  # Check if we should keep running
            except Exception as queue_error:
                log_with_timestamp(f"Error getting audio from queue: {queue_error}", "ERROR")
                continue
            
            try:
                # Validate audio data
                if audio_data is None or len(audio_data) == 0:
                    log_with_timestamp("Empty audio data received from queue", "ERROR")
                    audio_queue.task_done()
                    continue
                
                # Resample audio if needed with error handling
                try:
                    if SAMPLE_RATE != TARGET_RATE:
                        original_length = len(audio_data)
                        new_length = int(original_length * TARGET_RATE / SAMPLE_RATE)
                        audio_data = resample(audio_data, new_length)
                        log_with_timestamp(f"Audio resampled from {original_length} to {new_length} samples", "AUDIO")
                except Exception as resample_error:
                    log_with_timestamp(f"Audio resampling failed: {resample_error}", "ERROR")
                    audio_queue.task_done()
                    continue
                
                # Configure faster-whisper for Spanish transcription (no translation)
                try:
                    log_with_timestamp("Starting Whisper transcription", "TRANSCRIBE_ES")
                    segments, info = model.transcribe(
                        audio_data,
                        language="es",  # Spanish language
                        task="transcribe",  # Transcribe only, no translation
                        beam_size=5,
                        best_of=5,
                        vad_filter=True,
                        temperature=0.0,
                        no_speech_threshold=0.5
                    )
                    
                    # Extract Spanish transcription with error handling
                    try:
                        spanish_text = " ".join([s.text.strip() for s in segments])
                        log_with_timestamp(f"Transcription completed, detected language: {info.language if hasattr(info, 'language') else 'unknown'}", "TRANSCRIBE_ES")
                    except Exception as segment_error:
                        log_with_timestamp(f"Error extracting text from segments: {segment_error}", "ERROR")
                        spanish_text = ""
                    
                    # Handle empty results
                    if not spanish_text.strip():
                        log_with_timestamp("Empty transcription result - no speech detected", "TRANSCRIBE_ES")
                        audio_queue.task_done()
                        continue
                    
                    # Store last transcription for repeat functionality
                    last_spanish = spanish_text.strip()
                    
                    # Log TRANSCRIBE_ES event with the Spanish result
                    log_with_timestamp(f"Spanish transcription: {last_spanish}", "TRANSCRIBE_ES")
                    
                    # Display transcription and prompt for user action
                    try:
                        display_transcription_and_prompt(last_spanish)
                    except Exception as display_error:
                        log_with_timestamp(f"Error displaying transcription: {display_error}", "ERROR")
                    
                except Exception as transcription_error:
                    log_with_timestamp(f"Whisper transcription failed: {str(transcription_error)}", "ERROR")
                
            except Exception as processing_error:
                log_with_timestamp(f"Error processing audio data: {processing_error}", "ERROR")
            finally:
                # Always mark task as done to prevent queue blocking
                try:
                    audio_queue.task_done()
                except Exception as task_done_error:
                    log_with_timestamp(f"Error marking queue task as done: {task_done_error}", "ERROR")
            
        except Exception as worker_error:
            log_with_timestamp(f"Critical worker thread error: {str(worker_error)}", "ERROR")
            # Continue running unless explicitly stopped
            continue
    
    log_with_timestamp("Transcription worker thread stopped", "SYSTEM")

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
        
        # Compute RMS with error handling
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
    Includes comprehensive error handling for all state tracking operations.
    """
    global is_speaking, silence_start
    
    try:
        # Compute RMS with error handling
        rms = compute_rms(audio_chunk)
        
        # Validate RMS value
        if rms is None or np.isnan(rms) or np.isinf(rms):
            log_with_timestamp(f"Invalid RMS value: {rms}", "ERROR")
            return False
        
        try:
            current_time = datetime.now()
        except Exception as time_error:
            log_with_timestamp(f"Error getting current time: {time_error}", "ERROR")
            return False
        
        if rms > RMS_THRESHOLD:
            # Audio detected above threshold
            if not is_speaking:
                # Transition from silence to speech
                try:
                    is_speaking = True
                    silence_start = None
                    log_with_timestamp(f"SPEECH_START detected (RMS: {rms:.4f})", "AUDIO")
                except Exception as speech_start_error:
                    log_with_timestamp(f"Error handling speech start: {speech_start_error}", "ERROR")
        else:
            # Audio below threshold (silence)
            if is_speaking:
                # We were speaking, now we might be in silence
                try:
                    if silence_start is None:
                        # First moment of silence
                        silence_start = current_time
                    else:
                        # Check if we've been silent long enough
                        try:
                            silence_duration = (current_time - silence_start).total_seconds()
                            if silence_duration >= SILENCE_SECS:
                                # Transition from speech to silence
                                is_speaking = False
                                log_with_timestamp(f"SPEECH_END detected after {silence_duration:.2f}s of silence (RMS: {rms:.4f})", "AUDIO")
                                return True  # Signal that speech segment is complete
                        except Exception as duration_error:
                            log_with_timestamp(f"Error calculating silence duration: {duration_error}", "ERROR")
                            # Reset silence tracking on error
                            silence_start = current_time
                except Exception as silence_error:
                    log_with_timestamp(f"Error handling silence detection: {silence_error}", "ERROR")
        
        return False  # Speech segment not complete yet
        
    except Exception as state_error:
        log_with_timestamp(f"Critical error in update_speaking_state: {state_error}", "ERROR")
        return False

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



# Audio callback with comprehensive error handling
def callback(indata, frames, time, status):
    """
    Audio callback that implements RMS-based pause detection and audio buffering.
    Continuously buffers audio and processes complete speech segments.
    Includes comprehensive error handling for all audio processing operations.
    """
    global audio_buffer
    
    try:
        if status:
            log_with_timestamp(f"Audio status warning: {status}", "AUDIO")
        
        # Validate input data
        if indata is None or len(indata) == 0:
            log_with_timestamp("Empty audio input received", "ERROR")
            return
        
        # Get audio chunk (mono channel) with error handling
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
                            log_with_timestamp("Audio queued for transcription processing", "AUDIO")
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
            
    except Exception as callback_error:
        log_with_timestamp(f"Critical error in audio callback: {callback_error}", "ERROR")

def cleanup_resources():
    """
    Comprehensive cleanup function for graceful shutdown.
    Handles thread cleanup, resource management, and proper logging.
    """
    global transcription_worker_running, transcription_thread, audio_stream
    
    log_with_timestamp("Starting system cleanup", "SYSTEM")
    
    try:
        # Signal worker thread to stop
        transcription_worker_running = False
        log_with_timestamp("Signaled transcription worker to stop", "SYSTEM")
        
        # Wait for any remaining audio processing to complete with timeout
        try:
            log_with_timestamp("Waiting for audio queue to empty", "SYSTEM")
            # Add a small timeout to prevent hanging
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
        
        # Clear audio buffer
        try:
            with speech_buffer_lock:
                audio_buffer.clear()
                log_with_timestamp("Audio buffer cleared", "SYSTEM")
        except Exception as buffer_error:
            log_with_timestamp(f"Error clearing audio buffer: {buffer_error}", "ERROR")
        
        # Log final statistics
        try:
            queue_size = audio_queue.qsize()
            if queue_size > 0:
                log_with_timestamp(f"Warning: {queue_size} items remaining in audio queue", "SYSTEM")
        except Exception:
            pass
        
        log_with_timestamp("System cleanup completed", "SYSTEM")
        
    except Exception as cleanup_error:
        log_with_timestamp(f"Error during cleanup: {cleanup_error}", "ERROR")

def main():
    """
    Main execution function with comprehensive error handling and cleanup.
    Handles all major operations with proper resource management.
    """
    global transcription_thread, audio_stream
    
    try:
        # Start transcription worker thread with error handling
        try:
            transcription_thread = threading.Thread(target=process_audio, daemon=True)
            transcription_thread.start()
            log_with_timestamp("Transcription worker thread started successfully", "SYSTEM")
        except Exception as thread_error:
            log_with_timestamp(f"Failed to start transcription worker thread: {thread_error}", "ERROR")
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
