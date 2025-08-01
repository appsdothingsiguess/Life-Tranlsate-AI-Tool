# Windows Audio Configuration Guide

## 🔧 VB-Audio Virtual Cable Setup

### Step 1: Install VB-Audio Virtual Cable
1. Download from: https://vb-audio.com/Cable/
2. Install with default settings
3. Restart your computer

### Step 2: Configure Windows Sound Settings

#### Open Sound Settings:
1. Press `Win + R`
2. Type `sound settings` and press Enter
3. Click "More sound settings" on the right

#### Configure Recording Devices:
1. Go to the **Recording** tab
2. You should see:
   - `CABLE Output (VB-Audio Virtual Cable)` - This is what we want to use
   - Your physical microphone (e.g., "Realtek Microphone Array")

#### Set Default Devices:
1. **Right-click** on `CABLE Output (VB-Audio Virtual Cable)`
2. Select **"Set as Default Device"** (NOT "Set as Default Communications Device")
3. Keep your physical microphone as **"Default Communications Device"** for Zoom/Discord

### Step 3: Test Audio Routing

#### Test 1: Check if audio is flowing to VB-Audio
1. Open any audio source (YouTube, Spotify, etc.)
2. Play some audio
3. The audio should now be routed through VB-Audio Cable
4. Run `python check_devices.py` to verify device detection

#### Test 2: Verify transcription is working
1. Run `python main.py`
2. Speak some Spanish words
3. You should see "🎙️ Transcribing …" appear
4. Check the log file for audio levels and device detection

## 🚨 Common Issues & Solutions

### Issue: "No VB-Audio devices found"
**Solution:**
- Reinstall VB-Audio Virtual Cable
- Make sure to restart after installation
- Check Device Manager for any yellow warning icons

### Issue: Audio levels are very low
**Solution:**
- Check Windows volume mixer
- Make sure audio source is playing
- Verify VB-Audio is set as default recording device

### Issue: Transcription not working
**Solution:**
- Check log.txt for detailed error messages
- Verify audio is being routed to VB-Audio
- Test with `python check_devices.py`

### Issue: "Device not found" errors
**Solution:**
- Run as Administrator
- Check Windows privacy settings for microphone access
- Disable and re-enable VB-Audio in Device Manager

## 📊 Expected Device Names

Your system should show these devices:
- `CABLE Output (VB-Audio Virtual Cable)` - **This is what we want**
- `CABLE Input (VB-Audio Virtual Cable)` - Output device
- Your physical microphone (keep as communications device)

## 🔍 Debugging Commands

```bash
# Check all audio devices
python check_devices.py

# Test transcription directly
python -c "import sounddevice as sd; print('Default input:', sd.default.device[0])"

# Check if VB-Audio is working
python -c "import sounddevice as sd; devices = sd.query_devices(); [print(f'{i}: {d[\"name\"]}') for i, d in enumerate(devices) if 'vb-audio' in d['name'].lower()]"
```

## ✅ Verification Checklist

- [ ] VB-Audio Virtual Cable installed
- [ ] `CABLE Output (VB-Audio Virtual Cable)` appears in recording devices
- [ ] VB-Audio is set as **Default Device** (not communications)
- [ ] Physical microphone is **Default Communications Device**
- [ ] Audio source is playing and routed through VB-Audio
- [ ] `python check_devices.py` shows VB-Audio devices
- [ ] `python main.py` detects audio and shows "🎙️ Transcribing …"
- [ ] Log file shows proper device detection and audio levels 