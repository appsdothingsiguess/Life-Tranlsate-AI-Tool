# Live Language Transcription Tool

A real-time language speech-to-text transcription system with AI-powered response generation, designed for language learning and conversation practice. This tool provides immediate transcription of any audio on your PC and generates contextual responses using Google's Gemini AI.

## 🎯 What It Does

This application is a sophisticated real-time transcription and conversation assistant that:

- **Real-time Speech Recognition**: Continuously listens to language speech and transcribes it using OpenAI's Whisper model
- **AI-Powered Responses**: Generates natural Spanish responses using Google's Gemini 2.5 Flash Lite model
- **Live Buffer System**: Accumulates transcribed text in real-time and sends it to AI for responses
- **Smart Audio Detection**: Uses RMS-based speech detection to identify when you're speaking vs. silent
- **Auto-send Functionality**: Automatically sends accumulated text after 8 seconds of silence
- **Hotkey Controls**: Provides keyboard shortcuts for manual control and interaction

## 🏗️ System Architecture

The application uses a multi-threaded architecture with the following components:

- **Audio Stream Thread**: Captures audio from VB-Audio Virtual Cable
- **Transcription Worker Thread**: Processes audio chunks and converts speech to text
- **Hotkey Listener Thread**: Monitors keyboard input for user commands
- **Auto-send Monitor Thread**: Manages automatic sending of accumulated text
- **Gemini API Thread**: Handles AI response generation

## 📋 Prerequisites

### Software Requirements

1. **Python 3.10** - The application is written in Python and need 3.10 for certain libraries.
2. **VB-Audio Virtual Cable** - Required for audio routing
   - Download from: https://vb-audio.com/Cable/
   - This creates a virtual audio device that routes system audio to the application
3. **CUDA-capable GPU (Optional)** - For faster Whisper processing
   - NVIDIA GPU with CUDA support recommended
   - CPU-only mode available but slower
4. **CTranslate2** - Required for faster-whisper (automatically installed with faster-whisper)

### Hardware Requirements

- **Microphone** - For speech input
- **Audio Output** - Speakers or headphones to hear responses
- **Sufficient RAM** - At least 4GB recommended for smooth operation
- **Sufficent GPU** - At least a Nvidia GTX 3000s series or above.

### API Requirements

- **Google AI API Key** - Required for Gemini AI responses
  - Get your API key from: https://makersuite.google.com/app/apikey

## 🚀 Installation & Setup

### 1. Clone or Download the Project

```bash
git clone <repository-url>
cd translate-tool
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

If no requirements.txt exists, install these packages manually:

```bash
pip install torch sounddevice numpy faster-whisper scipy python-dotenv google-genai
```

### Faster-Whisper Setup

This application uses **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** - a reimplementation of OpenAI's Whisper model using CTranslate2, which is up to 4 times faster than the original Whisper for the same accuracy while using less memory.

#### Model Configuration
- **Model Size**: "small" (244M parameters) - optimized for speed and accuracy
- **Model Download**: Automatically downloads on first run (~244MB)
- **Compute Type**: 
  - `float16` if CUDA GPU is available (faster)
  - `int8` if CPU-only (slower but works on all systems)
- **Language**: Configured for Spanish (`language="es"`)

#### Transcription Settings
- `beam_size=5` and `best_of=5` for accuracy
- `vad_filter=True` for voice activity detection (Silero VAD model)
- `temperature=0.0` for deterministic results
- `no_speech_threshold=0.5` for noise filtering

#### Performance Benefits
- **4x faster** than original OpenAI Whisper
- **Lower memory usage**
- **GPU acceleration** with CUDA support
- **Voice Activity Detection** to filter out silence

**First Run**: The Whisper model will be downloaded automatically (~244MB). Ensure you have a stable internet connection.

### 3. Set Up Audio Routing

1. **Install VB-Audio Virtual Cable**
   - Download and install VB-Audio Virtual Cable
   - Restart your computer after installation

2. **Configure Audio Settings**
   - Open Windows Sound Settings
   - Set "CABLE Input (VB-Audio Virtual Cable)" as your default microphone
   - Ensure your actual microphone is routed to the virtual cable

3. **Verify Device Index**
   - The application uses device index 37 by default
   - You may need to adjust this in the code if your system has different audio devices
   - Run `python check_devices.py` to see available audio devices

### 4. Install Faster-Whisper Dependencies

The faster-whisper library requires specific dependencies for optimal performance:

```bash
# For CPU-only usage
pip install faster-whisper

# For GPU acceleration (CUDA)
pip install faster-whisper[cuda]
```

**Note**: If you encounter issues with CUDA installation, the CPU version will work but will be slower.

### 5. Configure API Key

Create a `.env` file in the project directory:

```env
GOOGLE_API_KEY=your_google_ai_api_key_here
```

### 6. Test the Setup

**Optional: Test Whisper Setup First**
```bash
python test_whisper_format_validation.py
```
This will validate that Whisper is working correctly with your audio setup.

**Run the Application:**
```bash
python main.py
```

## 🎮 How to Use

### Starting the Application

1. Ensure VB-Audio Virtual Cable is properly configured
2. Make sure your microphone is working and routed to the virtual cable
3. Run the application: `python main.py`
4. Wait for the startup banner to appear

### Basic Operation

Once the application is running, you'll see:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎤  Live Spanish Transcriber READY
• Speak → text appears immediately
• [g] send  [r] repeat  [h] help
• Auto-send after 8 s silence
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Speaking and Transcription

1. **Start Video/Audio**: Begin playing a video or someform of audio
2. **Real-time Display**: The transcribed speech will appear as text in real-time
3. **Buffer Accumulation**: Text accumulates in a buffer as you speak
4. **Auto-send**: After 8 seconds of silence, the text is automatically sent to Gemini AI
5. **AI Response**: You'll receive a natural Spanish response

### Keyboard Controls

| Key Combination | Function |
|----------------|----------|
| `Ctrl+Alt+Shift+S` | **Send manually** - Send accumulated text to Gemini AI immediately |
| `Ctrl+Alt+Shift+R` | **Repeat** - Show the last Gemini response again |
| `Ctrl+Alt+Shift+H` | **Help** - Display this help information |
| `Ctrl+Alt+Shift+C` | **Skip/Ignore** - Clear current transcription buffer |
| `Ctrl+C` | **Exit** - Gracefully shut down the application |

### How to Test Hotkeys Globally

1. **Start the application**: `python main.py`
2. **Verify global hotkeys are active**: Look for the message `[HOTKEY] Global hooks active (CTRL+ALT+SHIFT+S=send • CTRL+ALT+SHIFT+R=repeat • CTRL+ALT+SHIFT+H=help • CTRL+ALT+SHIFT+C=skip)`
3. **Switch to another application**: Open a browser, text editor, or any other application
4. **Test hotkeys**: Press the key combinations while the other application is focused
5. **Verify functionality**: The hotkeys should work even without console focus

**Note**: If you see `[HOTKEY] Console hot-keys active` instead, the global hotkey libraries failed to load. Install them with:
```bash
pip install pynput keyboard
```

### Customizing Hotkeys

To customize the hotkey combinations, edit the `KEY_CONFIG` dictionary in `main.py`:

```python
KEY_CONFIG = {
    "modifiers": ["ctrl", "alt", "shift"],  # Change modifier keys
    "keys": {
        "send": "s",      # Change to any key
        "repeat": "r",    # Change to any key
        "help": "h",      # Change to any key
        "skip": "c"       # Change to any key
    }
}
```

To view the current key mapping, run:
```bash
python main.py --keys dump
```

### Understanding the Interface

- **Buffer Status**: Shows character count and available actions
  ```
  📄 Buffer: 45 chars  (press g or wait 8 s)
  ```

- **AI Responses**: Displayed with the 🤖 emoji
  ```
  🤖 Gemini: ¡Hola! ¿Cómo estás hoy?
  ```

- **System Messages**: Important status updates and errors
  ```
  [2024-01-15 14:30:25] [SYSTEM] Audio stream started successfully
  ```

## ⚙️ Configuration Options

### Audio Settings

You can modify these constants in `main.py`:

```python
RMS_THRESHOLD = 0.008      # Audio sensitivity (lower = more sensitive)
SILENCE_SECS = 0.3         # Silence duration to end speech segment
MAX_SPEECH_SECS = 4.0      # Maximum speech duration before forced flush
AUTO_SEND_AFTER_SECS = 8   # Seconds of silence before auto-send
SAMPLE_RATE = 48000        # Audio sample rate
DEVICE_INDEX = 37          # Audio device index
```

### AI Response Settings

The Gemini AI is configured to respond as a Spanish 2 student with knowledge of:
- Basic verb conjugations
- Preterite tense
- Stem-changing verbs
- Direct object pronouns
- Ser vs. Estar
- And other Spanish 2 concepts

## 🔧 Troubleshooting

### Common Issues

1. **"PortAudio device error"**
   - Ensure VB-Audio Virtual Cable is installed
   - Check that audio is being routed to the virtual cable
   - Verify the device index in the code matches your system

2. **"Missing GOOGLE_API_KEY"**
   - Create a `.env` file with your API key
   - Ensure the API key is valid and has access to Gemini models

3. **"Failed to load Whisper model"**
   - Check your internet connection (first run downloads the model)
   - Ensure you have sufficient disk space (~244MB for the model)
   - Verify PyTorch installation
   - Check CUDA installation if using GPU
   - Try running `python test_whisper_format_validation.py` to test Whisper setup
   - Ensure faster-whisper is properly installed: `pip install faster-whisper[cuda]` for GPU support

4. **No audio transcription**
   - Check microphone permissions
   - Verify audio routing to VB-Audio Virtual Cable
   - Adjust RMS_THRESHOLD if audio is too quiet/loud

5. **Slow or no AI responses**
   - Check your internet connection
   - Verify API key is valid and has sufficient quota
   - Check the log file for detailed error messages

### Log Files

The application creates a `log.txt` file with detailed information about:
- System startup and initialization
- Audio processing events
- Transcription results
- AI API calls and responses
- Error messages and debugging information

Check this file if you encounter issues.

## 📁 File Structure

```
translate-tool/
├── main.py                           # Main application file
├── check_devices.py                  # Audio device detection utility
├── test_whisper_format_validation.py # Whisper setup validation script
├── simple_test.py                    # Simple Whisper test script
├── stablever.py                      # Stable version backup
├── log.txt                           # Application log file
├── whisper_format_validation.log     # Whisper validation log
├── .env                              # Environment variables (create this)
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## 🔒 Privacy & Security

- **Local Processing**: Audio transcription happens locally using faster-whisper
- **API Calls**: Only transcribed text is sent to Google's Gemini API
- **No Storage**: Audio is not recorded or stored permanently
- **Logging**: Only system events and text are logged (no audio)
- **Open Source**: Uses the open-source faster-whisper implementation from [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)

## 🚀 Performance Tips

1. **Have clear audio** for best results; do not have music playing or other system sounds.
2. **Verify sound settings.** you may need to adjust your sound settings.
3. **Minimize background noise** for optimal results
4. **Close unnecessary applications** to free up system resources
5. **Use wired connections** for more stable audio routing
6. **GPU Acceleration**: Install CUDA for faster Whisper processing
   - Use `pip install faster-whisper[cuda]` for GPU support
   - 4x faster than CPU-only mode
7. **Model Optimization**: The "small" model balances speed and accuracy
8. **Voice Activity Detection**: The built-in VAD filter automatically removes silence

## 🤝 Contributing

This tool is designed for educational purposes. Feel free to:
- Report bugs and issues
- Suggest improvements
- Fork and modify for your own needs

## 📄 License

This project is provided as-is for educational and personal use.

## 🆘 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the `log.txt` file for detailed error information
3. Ensure all prerequisites are properly installed
4. Verify your audio routing configuration

---

**Note**: This tool is designed for language learning and conversation practice. Always use AI responses responsibly and verify information when needed for academic or professional purposes. 
