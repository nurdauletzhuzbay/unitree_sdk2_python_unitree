#!/usr/bin/env python3
"""
Temirbek Voice Assistant - OPTIMIZED Streaming Version
- Streams LLM responses word-by-word
- Generates and plays TTS in real-time
- Much faster response time
"""

import os
import sys
import signal
import subprocess
import threading
import time
import struct
import wave
import tempfile
import queue
from typing import List, Optional
from dataclasses import dataclass
import requests
import collections

# ============================================================================
# CONFIGURATION
# ============================================================================
ALSA_OUTPUT_DEVICE = "plughw:CARD=Audio,DEV=0"
ALSA_INPUT_DEVICE = "plughw:CARD=Device,DEV=0"
PIPER_MODEL = "/opt/piper/piper/kk_KZ-iseke-x_low.onnx"
SAMPLE_RATE = 16000
MAX_HISTORY_TURNS = 5

# VAD settings (high thresholds for motor noise)
SILENCE_THRESHOLD = 1500
SILENCE_DURATION = 2.0
MIN_SPEECH_DURATION = 1.5
SPEECH_START_THRESHOLD = 2500

# Streaming settings
SENTENCE_DELIMITERS = ['.', '!', '?', '।', ':', '\n']  # Wait for complete sentences
MIN_CHUNK_LENGTH = 15  # Minimum characters before generating TTS

# ============================================================================
# GLOBAL STATE
# ============================================================================
g_running = True
g_is_listening = True
g_is_speaking = False

@dataclass
class ConversationTurn:
    user_message: str
    assistant_message: str

conversation_history: List[ConversationTurn] = []
recent_transcriptions: collections.deque = collections.deque(maxlen=5)

HALLUCINATION_PHRASES = [
    "қасындастық елігі қазақстан",
    "қазақстан",
    "субтитры",
    "переклад",
]

# ============================================================================
# SIGNAL HANDLER
# ============================================================================
def signal_handler(sig, frame):
    global g_running
    g_running = False
    print("\nShutting down...")

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ============================================================================
# UNITREE G1 AUDIO CLIENT
# ============================================================================
class G1AudioClient:
    def __init__(self, network_interface: str):
        self.network_interface = network_interface
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
        if self.available:
            try:
                self.client.LedControl(r, g, b)
            except Exception as e:
                pass

# ============================================================================
# AUDIO UTILITIES
# ============================================================================
def calculate_rms(audio_chunk: bytes) -> float:
    try:
        count = len(audio_chunk) // 2
        shorts = struct.unpack(f'{count}h', audio_chunk)
        sum_squares = sum(s**2 for s in shorts)
        rms = (sum_squares / count) ** 0.5
        return rms
    except:
        return 0

def create_wav_file(audio_pcm: List[int], filename: str):
    """Create WAV file from PCM data"""
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        audio_bytes = struct.pack(f'{len(audio_pcm)}h', *audio_pcm)
        wav_file.writeframes(audio_bytes)

# ============================================================================
# SPEECH-TO-TEXT (AlemAI Whisper)
# ============================================================================
def is_hallucination(text: str) -> bool:
    if not text:
        return True
    
    text_lower = text.lower().strip()
    
    for phrase in HALLUCINATION_PHRASES:
        if phrase.lower() in text_lower:
            return True
    
    if text_lower in recent_transcriptions:
        count = sum(1 for t in recent_transcriptions if t == text_lower)
        if count >= 2:
            return True
    
    recent_transcriptions.append(text_lower)
    
    if len(text_lower) < 3:
        return True
    
    return False

def transcribe_audio(audio_pcm: List[int]) -> str:
    """Transcribe audio using AlemAI Whisper"""
    if not g_running:
        return ""
    
    api_key = os.getenv("ALEMAI_STT_API_KEY")
    if not api_key:
        print("ERROR: ALEMAI_STT_API_KEY not set")
        return ""
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        wav_file = tmp_file.name
    
    try:
        create_wav_file(audio_pcm, wav_file)
        
        url = "https://llm.alem.ai/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {api_key}"}
        
        with open(wav_file, 'rb') as f:
            files = {'file': ('audio.wav', f, 'audio/wav')}
            data = {'model': 'speech-to-text-kk'}
            
            response = requests.post(url, headers=headers, files=files, data=data, timeout=15)
        
        if response.status_code == 200 and g_running:
            result = response.json()
            transcription = result.get('text', '').strip()
            
            if is_hallucination(transcription):
                print(f"⚠️  Detected hallucination: '{transcription}' - ignoring")
                return ""
            
            return transcription
        else:
            return ""
            
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""
    finally:
        if os.path.exists(wav_file):
            os.remove(wav_file)

# ============================================================================
# LLM (AlemAI) - STREAMING VERSION
# ============================================================================
def get_llm_response_streaming(user_text: str, text_queue: queue.Queue):
    """Get streaming response from AlemAI LLM and put complete sentences in queue"""
    if not g_running:
        return
    
    api_key = os.getenv("ALEMAI_LLM_API_KEY")
    if not api_key:
        print("ERROR: ALEMAI_LLM_API_KEY not set")
        text_queue.put(None)
        return
    
    try:
        messages = [
            {
                "role": "system",
                "content": "You are Temirbek, a helpful robot assistant. User speaks Kazakh. Reply in Kazakh briefly (1-2 sentences)."
            }
        ]
        
        start_idx = max(0, len(conversation_history) - MAX_HISTORY_TURNS)
        for turn in conversation_history[start_idx:]:
            messages.append({"role": "user", "content": turn.user_message})
            messages.append({"role": "assistant", "content": turn.assistant_message})
        
        messages.append({"role": "user", "content": user_text})
        
        url = "https://llm.alem.ai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": "alemllm",
            "messages": messages,
            "stream": True
        }
        
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=30)
        
        if response.status_code == 200:
            full_text = ""
            buffer = ""
            
            for line in response.iter_lines():
                if not g_running:
                    break
                    
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]
                        
                        if data_str == '[DONE]':
                            # Send any remaining buffer
                            if buffer.strip():
                                text_queue.put(buffer.strip())
                                full_text += buffer
                            break
                        
                        try:
                            import json
                            data = json.loads(data_str)
                            
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                
                                if content:
                                    buffer += content
                                    
                                    # Check if buffer ends with sentence delimiter
                                    if any(buffer.rstrip().endswith(delim) for delim in SENTENCE_DELIMITERS):
                                        # Send complete sentence
                                        sentence = buffer.strip()
                                        if sentence:
                                            text_queue.put(sentence)
                                            full_text += sentence + " "
                                            buffer = ""
                                    
                                    # Also check if buffer is getting long without delimiter
                                    elif len(buffer) > 50:
                                        # Find last space to break at word boundary
                                        last_space = buffer.rfind(' ')
                                        if last_space > MIN_CHUNK_LENGTH:
                                            chunk = buffer[:last_space].strip()
                                            text_queue.put(chunk)
                                            full_text += chunk + " "
                                            buffer = buffer[last_space:].lstrip()
                        except:
                            pass
            
            # Signal completion
            text_queue.put(None)
            text_queue.put(('FULL_TEXT', full_text.strip()))
            
        else:
            print(f"LLM error: {response.status_code}")
            text_queue.put(None)
            
    except Exception as e:
        print(f"LLM error: {e}")
        text_queue.put(None)

# ============================================================================
# TEXT-TO-SPEECH (Piper) - FAST VERSION
# ============================================================================
def generate_speech_chunk(text: str) -> Optional[bytes]:
    """Generate speech for a text chunk and return raw PCM bytes"""
    if not g_running or not text:
        return None
    
    try:
        # Escape text for shell
        safe_text = text.replace("'", "'\\''")
        
        # Use temporary files for more reliable output
        with tempfile.NamedTemporaryFile(suffix='.pcm', delete=False) as tmp_pcm:
            pcm_file = tmp_pcm.name
        
        # Generate speech - using file output for reliability
        cmd = f"echo '{safe_text}' | /opt/piper/piper/piper -m {PIPER_MODEL} --output-raw -f {pcm_file} 2>/dev/null"
        
        result = subprocess.run(cmd, shell=True, timeout=5)
        
        if result.returncode == 0 and os.path.exists(pcm_file):
            with open(pcm_file, 'rb') as f:
                audio_data = f.read()
            
            os.remove(pcm_file)
            return audio_data if audio_data else None
        
        if os.path.exists(pcm_file):
            os.remove(pcm_file)
        
        return None
        
    except Exception as e:
        return None

# ============================================================================
# STREAMING TTS PLAYBACK
# ============================================================================
def play_audio_stream(audio_bytes: bytes):
    """Play audio bytes directly without file"""
    if not audio_bytes or not g_running:
        return
    
    try:
        cmd = [
            'aplay',
            '-D', ALSA_OUTPUT_DEVICE,
            '-f', 'S16_LE',
            '-r', str(SAMPLE_RATE),
            '-c', '1',
            '-q'
        ]
        
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        process.communicate(input=audio_bytes)
        
    except Exception as e:
        pass

def streaming_tts_worker(text_queue: queue.Queue, audio_client: G1AudioClient):
    """Worker thread that generates and plays TTS in real-time"""
    global g_is_speaking
    
    full_text_parts = []
    first_chunk = True
    
    while g_running:
        item = text_queue.get()
        
        if item is None:
            # End of stream
            g_is_speaking = False
            break
        
        if isinstance(item, tuple) and item[0] == 'FULL_TEXT':
            # This is the full text for history
            continue  # We already have full_text_parts
        
        # This is a text chunk to speak
        text_chunk = item
        full_text_parts.append(text_chunk)
        
        if not g_is_speaking:
            g_is_speaking = True
            audio_client.led_control(0, 100, 255)  # Blue = speaking
            if first_chunk:
                print("", end='', flush=True)  # Start speaking indicator
                first_chunk = False
        
        # Print the text as it comes
        print(text_chunk + " ", end='', flush=True)
        
        # Generate TTS (this takes time, which is good - prevents cutoff)
        audio_bytes = generate_speech_chunk(text_chunk)
        
        # Play immediately after generation
        if audio_bytes and g_running:
            play_audio_stream(audio_bytes)
    
    print()  # New line after speaking
    audio_client.led_control(0, 0, 0)
    
    # Return full text
    return ' '.join(full_text_parts)

# ============================================================================
# CONTINUOUS AUDIO RECORDING WITH VAD
# ============================================================================
def record_with_vad() -> Optional[List[int]]:
    """Record audio with voice activity detection"""
    global g_is_listening
    
    if g_is_speaking:
        return None
    
    cmd = [
        'arecord',
        '-D', ALSA_INPUT_DEVICE,
        '-f', 'S16_LE',
        '-r', str(SAMPLE_RATE),
        '-c', '1',
        '-q'
    ]
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=2048)
        
        speech_frames = []
        silence_frames = 0
        speech_started = False
        frames_per_check = int(SAMPLE_RATE * 0.1)
        
        print("🎤 Listening...")
        
        buffer = b''
        rms_history = []
        
        while g_running and g_is_listening and not g_is_speaking:
            chunk = process.stdout.read(2048)
            if not chunk:
                break
            
            buffer += chunk
            
            if len(buffer) >= frames_per_check * 2:
                check_chunk = buffer[:frames_per_check * 2]
                buffer = buffer[frames_per_check * 2:]
                
                rms = calculate_rms(check_chunk)
                
                if not speech_started and rms > SPEECH_START_THRESHOLD:
                    speech_started = True
                    silence_frames = 0
                    print("🗣️  Detected...")
                
                if speech_started:
                    samples = struct.unpack(f'{len(check_chunk)//2}h', check_chunk)
                    speech_frames.extend(samples)
                    rms_history.append(rms)
                    
                    if rms < SILENCE_THRESHOLD:
                        silence_frames += 1
                    else:
                        silence_frames = 0
                    
                    silence_duration = silence_frames * 0.1
                    if silence_duration >= SILENCE_DURATION:
                        break
        
        process.terminate()
        process.wait(timeout=1)
        
        if speech_frames and rms_history:
            duration = len(speech_frames) / SAMPLE_RATE
            avg_rms = sum(rms_history) / len(rms_history)
            max_rms = max(rms_history)
            
            if duration < MIN_SPEECH_DURATION:
                return None
            
            if avg_rms < SPEECH_START_THRESHOLD * 0.8:
                return None
            
            if max_rms < avg_rms * 1.3:
                print("⚠️  Constant noise, ignoring...")
                return None
            
            return speech_frames
        
        return None
        
    except Exception as e:
        return None

# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================
def process_conversation(audio_client: G1AudioClient):
    global conversation_history
    
    while g_running:
        audio_client.led_control(0, 255, 0)  # Green = listening
        audio = record_with_vad()
        audio_client.led_control(0, 0, 0)
        
        if not g_running or not audio:
            time.sleep(0.1)
            continue
        
        # Transcribe
        print("📝 Transcribing...")
        transcription = transcribe_audio(audio)
        
        if not g_running or not transcription:
            continue
        
        print(f"👤 You: {transcription}")
        
        # Check commands
        lower_trans = transcription.lower()
        if any(word in lower_trans for word in ['clear', 'тазала', 'жаңарт']):
            conversation_history.clear()
            print("🗑️  Memory cleared.")
            continue
        
        if any(word in lower_trans for word in ['stop', 'тоқта', 'exit']):
            break
        
        # Get streaming LLM response
        print("🤖 Temirbek: ", end='', flush=True)
        
        text_queue = queue.Queue()
        
        # Start LLM streaming in background
        llm_thread = threading.Thread(
            target=get_llm_response_streaming,
            args=(transcription, text_queue),
            daemon=True
        )
        llm_thread.start()
        
        # Process and speak chunks as they arrive
        full_response = streaming_tts_worker(text_queue, audio_client)
        
        llm_thread.join()
        
        if full_response and g_running:
            
            # Save to history
            turn = ConversationTurn(user_message=transcription, assistant_message=full_response)
            conversation_history.append(turn)
            
            if len(conversation_history) > MAX_HISTORY_TURNS:
                conversation_history.pop(0)
            
            print(f"💾 Memory: {len(conversation_history)} turns\n")
        
        time.sleep(0.3)

# ============================================================================
# MAIN
# ============================================================================
def main():
    global g_running
    
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} [NetworkInterface]")
        return 1
    
    if not os.getenv("ALEMAI_STT_API_KEY") or not os.getenv("ALEMAI_LLM_API_KEY"):
        print("ERROR: API keys not set")
        return 1
    
    audio_client = G1AudioClient(sys.argv[1])
    
    print("\n" + "=" * 50)
    print("  Temirbek - OPTIMIZED STREAMING MODE")
    print("=" * 50)
    print("  ⚡ Real-time word-by-word speech")
    print("  ⚡ Much faster responses!")
    print("=" * 50 + "\n")
    
    audio_client.led_control(0, 255, 0)
    time.sleep(1)
    audio_client.led_control(0, 0, 0)
    
    print("✅ Ready!\n")
    
    try:
        process_conversation(audio_client)
    except KeyboardInterrupt:
        pass
    finally:
        g_running = False
        audio_client.led_control(0, 0, 0)
        print("\n👋 Goodbye!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
