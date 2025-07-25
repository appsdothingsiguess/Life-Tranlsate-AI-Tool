import os
import torch
import sounddevice as sd
from faster_whisper import WhisperModel
from dotenv import load_dotenv
from google import genai
from pathlib import Path
import ctypes

print("=== System Diagnostic ===\n")

# Check PyTorch GPU + cuDNN
print("→ Checking CUDA + cuDNN...")
assert torch.cuda.is_available(), "CUDA not available"
lib = Path(torch.__file__).parent / "lib" / "cudnn_ops64_9.dll"
assert lib.exists(), f"Missing cuDNN DLL: {lib}"
ctypes.CDLL(str(lib))
print("✅ CUDA and cuDNN are working")

# Check audio input
print("→ Checking VB-CABLE input...")
device = 37  # change if needed
info = sd.query_devices(device)
assert info['max_input_channels'] > 0, "Device has no input channels"
sd.check_input_settings(device=device, samplerate=48000)
print(f"✅ Audio input: {info['name']}")

# Check Whisper model
print("→ Loading Whisper...")
model = WhisperModel("small", compute_type="float16")
print("✅ Whisper model loaded")

# Check Gemini API
print("→ Checking Gemini...")
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
assert api_key, "Missing GOOGLE_API_KEY"
client = genai.Client(api_key=api_key)
resp = client.models.generate_content(model="gemini-2.5-flash-lite", contents="Say hola.")
assert hasattr(resp, "text") and resp.text.strip(), "Gemini test failed"
print("✅ Gemini API working")

print("\n✅ All checks passed! Ready to launch.")
