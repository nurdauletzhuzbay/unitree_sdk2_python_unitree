#!/usr/bin/env python3
"""
Temirbek Voice Assistant - Python Version
External microphone (USB) + External speaker (USB) + Piper TTS + AlemAI STT/LLM
"""

import os
import sys
import signal
import subprocess
import threading
import time
import select
import struct
import wave
import tempfile
from typing import List, Dict, Optional
from dataclasses import dataclass
import requests
import pyaudio
import json

# ============================================================================
# CONFIGURATION
# ============================================================================
ALSA_OUTPUT_DEVICE = "plughw:CARD=Audio,DEV=0"  # External speaker (Moshi)
ALSA_INPUT_DEVICE = "plughw:CARD=Device,DEV=0"  # External microphone (JMTek)
PIPER_MODEL = "/opt/piper/piper/kk_KZ-iseke-x_low.onnx"
SAMPLE_RATE = 16000
MAX_HISTORY_TURNS = 5
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1

# ============================================================================
# GLOBAL STATE
# ============================================================================
g_running = True
g_is_recording = False
g_stop_recording = False

@dataclass
class ConversationTurn:
    user_message: str
    assistant_message: str

conversation_history: List[ConversationTurn] = []

# ============================================================================
# SIGNAL HANDLER
# ============================================================================
def signal_handler(sig, frame):
    global g_running, g_is_recording
    g_running = False
    g_is_recording = False
    print("\nShutting down...")

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ============================================================================
# UNITREE G1 AUDIO CLIENT (LED CONTROL)
# ============================================================================
class G1AudioClient:
    """Simple wrapper for G1 LED control via DDS"""
    
    def __init__(self, network_interface: str):
        self.network_interface = network_interface
        # Import unitree SDK if available
        try:
            from unitree.robot.channel.channel_factory import ChannelFactory
            from unitree.robot.g1.audio.g1_audio_client import AudioClient
            
            ChannelFactory.Instance().Init(0, network_interface)
            self.client = AudioClient()
            self.client.Init()
            self.client.SetTimeout(10.0)
            self.available = True
        except ImportError:
            print("Warning: Unitree SDK not available. LED control disabled.")
            self.available = False
    
    def led_control(self, r: int, g: int, b: int):
        """Control G1 head LED"""
        if self.available:
            try:
                self.client.LedControl(r, g, b)
            except Exception as e:
                print(f"LED control error: {e}")

# ============================================================================
# SPEECH-TO-TEXT (AlemAI Whisper)
# ============================================================================
def create_wav_file(audio_pcm: List[int], filename: str):
    """Create WAV file from PCM data"""
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(SAMPLE_RATE)
        
        # Convert list of ints to bytes
        audio_bytes = struct.pack(f'{len(audio_pcm)}h', *audio_pcm)
        wav_file.writeframes(audio_bytes)

def transcribe_audio(audio_pcm: List[int]) -> str:
    """Transcribe audio using AlemAI Whisper"""
    if not g_running:
        return ""
    
    api_key = os.getenv("ALEMAI_STT_API_KEY")
    if not api_key:
        print("ERROR: ALEMAI_STT_API_KEY not set")
        return ""
    
    # Create temporary WAV file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        wav_file = tmp_file.name
    
    try:
        create_wav_file(audio_pcm, wav_file)
        
        # Send to AlemAI
        url = "https://llm.alem.ai/v1/audio/transcriptions"
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        
        with open(wav_file, 'rb') as f:
            files = {
                'file': ('audio.wav', f, 'audio/wav')
            }
            data = {
                'model': 'speech-to-text-kk'
            }
            
            response = requests.post(
                url,
                headers=headers,
                files=files,
                data=data,
                timeout=15
            )
        
        if response.status_code == 200 and g_running:
            result = response.json()
            return result.get('text', '')
        else:
            print(f"STT error: {response.status_code}")
            return ""
            
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""
    finally:
        if os.path.exists(wav_file):
            os.remove(wav_file)

# ============================================================================
# LLM (AlemAI)
# ============================================================================
def get_llm_response(user_text: str) -> str:
    """Get response from AlemAI LLM"""
    if not g_running:
        return ""
    
    api_key = os.getenv("ALEMAI_LLM_API_KEY")
    if not api_key:
        print("ERROR: ALEMAI_LLM_API_KEY not set")
        return ""
    
    try:
        # Build messages with conversation history
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Temirbek, a helpful robot assistant. User speaks Kazakh. "
                    "Reply in Kazakh briefly (1-2 sentences). "
                    "Remember the conversation context."
                )
            }
        ]
        
        # Add conversation history
        start_idx = max(0, len(conversation_history) - MAX_HISTORY_TURNS)
        for turn in conversation_history[start_idx:]:
            messages.append({"role": "user", "content": turn.user_message})
            messages.append({"role": "assistant", "content": turn.assistant_message})
        
        # Add current message
        messages.append({"role": "user", "content": user_text})
        
        # Send to AlemAI
        url = "https://llm.alem.ai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": "alemllm",
            "messages": messages
        }
        
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200 and g_running:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            print(f"LLM error: {response.status_code}")
            return ""
            
    except Exception as e:
        print(f"LLM error: {e}")
        return ""

# ============================================================================
# TEXT-TO-SPEECH (Piper)
# ============================================================================
def generate_speech(text: str) -> Optional[List[int]]:
    """Generate speech using Piper TTS"""
    if not g_running:
        return None
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
            wav_file = tmp_wav.name
        
        with tempfile.NamedTemporaryFile(suffix='.pcm', delete=False) as tmp_pcm:
            pcm_file = tmp_pcm.name
        
        # Generate speech with Piper
        cmd = f"echo '{text}' | /opt/piper/piper/piper -m {PIPER_MODEL} -f {wav_file}"
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
        
        # Convert to PCM
        cmd = f"ffmpeg -y -i {wav_file} -ar {SAMPLE_RATE} -ac 1 -f s16le {pcm_file}"
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
        
        # Read PCM data
        if g_running and os.path.exists(pcm_file):
            with open(pcm_file, 'rb') as f:
                pcm_data = f.read()
            
            # Convert bytes to list of int16
            audio = list(struct.unpack(f'{len(pcm_data)//2}h', pcm_data))
            return audio
        
        return None
        
    except Exception as e:
        print(f"TTS error: {e}")
        return None
    finally:
        for f in [wav_file, pcm_file]:
            if os.path.exists(f):
                os.remove(f)

# ============================================================================
# AUDIO RECORDING (External USB Microphone via arecord)
# ============================================================================
def record_audio() -> List[int]:
    """Record audio from USB microphone using arecord"""
    global g_is_recording, g_stop_recording
    
    pcm = []
    
    cmd = [
        'arecord',
        '-D', ALSA_INPUT_DEVICE,
        '-f', 'S16_LE',
        '-r', str(SAMPLE_RATE),
        '-c', '1',
        '-q'
    ]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        
        print("Recording... (Press ENTER to stop)")
        
        while g_running and g_is_recording and not g_stop_recording:
            # Read in chunks
            chunk = process.stdout.read(2048)
            if not chunk:
                break
            
            # Convert bytes to int16
            samples = struct.unpack(f'{len(chunk)//2}h', chunk)
            pcm.extend(samples)
        
        process.terminate()
        process.wait(timeout=1)
        
        duration = len(pcm) / SAMPLE_RATE
        print(f"Recording stopped ({duration:.1f}s)")
        
    except Exception as e:
        print(f"Recording error: {e}")
    
    return pcm

# ============================================================================
# AUDIO PLAYBACK (External USB Speaker via PyAudio)
# ============================================================================
def play_audio(audio_data: List[int]) -> bool:
    """Play audio through USB speaker using PyAudio"""
    if not audio_data or not g_running:
        return False
    
    try:
        # Convert list to bytes
        audio_bytes = struct.pack(f'{len(audio_data)}h', *audio_data)
        
        # Use aplay for playback (more reliable with ALSA devices)
        cmd = [
            'aplay',
            '-D', ALSA_OUTPUT_DEVICE,
            '-f', 'S16_LE',
            '-r', str(SAMPLE_RATE),
            '-c', '1',
            '-q'
        ]
        
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        
        process.communicate(input=audio_bytes)
        return True
        
    except Exception as e:
        print(f"Playback error: {e}")
        return False

# ============================================================================
# INPUT MONITORING THREAD
# ============================================================================
def input_monitor_thread():
    """Monitor stdin for ENTER key to stop recording"""
    global g_stop_recording
    
    while g_running:
        if g_is_recording:
            # Check if input is available
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if ready:
                sys.stdin.readline()
                g_stop_recording = True
        time.sleep(0.05)

# ============================================================================
# MAIN
# ============================================================================
def main():
    global g_running, g_is_recording, g_stop_recording, conversation_history
    
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} [NetworkInterface]")
        return 1
    
    # Check API keys
    if not os.getenv("ALEMAI_STT_API_KEY"):
        print("ERROR: ALEMAI_STT_API_KEY not set")
        return 1
    
    if not os.getenv("ALEMAI_LLM_API_KEY"):
        print("ERROR: ALEMAI_LLM_API_KEY not set")
        return 1
    
    # Initialize robot LED control
    audio_client = G1AudioClient(sys.argv[1])
    
    print("\n" + "=" * 40)
    print("  Temirbek Voice Assistant")
    print("=" * 40)
    print("  STT: AlemAI (Kazakh)")
    print("  LLM: AlemLLM (Kazakh)")
    print("  TTS: Piper - Iseke (Male Kazakh)")
    print("  Input: USB Microphone")
    print("  Output: USB Speaker")
    print(f"  Memory: {MAX_HISTORY_TURNS} turns")
    print("=" * 40)
    print("Commands:")
    print("  ENTER: Start/stop recording")
    print("  Say 'clear': Reset memory")
    print("  Ctrl+C: Quit")
    print("=" * 40 + "\n")
    
    # Ready indicator
    audio_client.led_control(0, 255, 0)
    time.sleep(1)
    audio_client.led_control(0, 0, 0)
    
    print("Ready.\n")
    
    # Start input monitoring thread
    input_thread = threading.Thread(target=input_monitor_thread, daemon=True)
    input_thread.start()
    
    # Main loop
    while g_running:
        print("Press ENTER to start recording...")
        
        # Wait for ENTER key
        ready, _, _ = select.select([sys.stdin], [], [], 0.5)
        
        if not g_running:
            break
        
        if ready:
            sys.stdin.readline()
            
            if not g_running:
                break
            
            # Record
            g_is_recording = True
            g_stop_recording = False
            audio_client.led_control(255, 0, 0)  # Red = recording
            
            audio = record_audio()
            g_is_recording = False
            audio_client.led_control(0, 0, 0)
            
            if not g_running or not audio:
                break
            
            if len(audio) < SAMPLE_RATE * 0.5:
                print("Recording too short, skipping.")
                continue
            
            # Transcribe
            print("Transcribing...")
            transcription = transcribe_audio(audio)
            
            if not g_running or not transcription:
                print("Transcription failed.")
                continue
            
            print(f"You: {transcription}")
            
            # Check for clear command
            lower_trans = transcription.lower()
            if any(word in lower_trans for word in ['clear', 'тазала', 'жаңарт']):
                conversation_history.clear()
                print("Memory cleared.")
                continue
            
            # Get LLM response
            print("Thinking...")
            reply = get_llm_response(transcription)
            
            if not g_running:
                break
            
            if reply:
                print(f"Temirbek: {reply}")
                
                # Save to history
                turn = ConversationTurn(
                    user_message=transcription,
                    assistant_message=reply
                )
                conversation_history.append(turn)
                
                # Trim history
                if len(conversation_history) > MAX_HISTORY_TURNS:
                    conversation_history.pop(0)
                
                print(f"Memory: {len(conversation_history)} turns")
                
                # Generate and play speech
                print("Generating speech...")
                tts_audio = generate_speech(reply)
                
                if not g_running:
                    break
                
                if tts_audio:
                    audio_client.led_control(0, 100, 255)  # Blue = speaking
                    print("Speaking...")
                    play_audio(tts_audio)
                    audio_client.led_control(0, 0, 0)
                
                print("Done.\n")
    
    print("\nShutting down...")
    audio_client.led_control(0, 0, 0)
    print("Goodbye.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
