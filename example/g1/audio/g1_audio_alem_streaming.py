#!/usr/bin/env python3
"""
Temirbek Voice Assistant - Continuous Listening Version
Streaming Edge TTS (audio plays immediately while being generated)
External microphone (USB) + External speaker (USB) + Edge TTS + AlemAI STT/LLM
"""

import os
import sys
import signal
import subprocess
import threading
import time
import struct
import tempfile
import asyncio
from typing import List, Optional
from dataclasses import dataclass
import requests
import collections

# ============================================================================
# CONFIGURATION
# ============================================================================
ALSA_OUTPUT_DEVICE = "plughw:CARD=Audio,DEV=0"
ALSA_INPUT_DEVICE = "plughw:CARD=Device,DEV=0"

EDGE_TTS_VOICE = "kk-KZ-DauletNeural"   # Male Kazakh
SAMPLE_RATE = 16000

MAX_HISTORY_TURNS = 5
CHUNK_SIZE = 1024

# VAD
SILENCE_THRESHOLD = 1500
SILENCE_DURATION = 1.0
MIN_SPEECH_DURATION = 0.8
SPEECH_START_THRESHOLD = 2200

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
recent_transcriptions = collections.deque(maxlen=5)

HALLUCINATION_PHRASES = [
    "қазақстан",
    "елігі",
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
# UNITREE LED CLIENT (optional)
# ============================================================================
class G1AudioClient:
    def __init__(self, network_interface: str):
        try:
            from unitree.robot.channel.channel_factory import ChannelFactory
            from unitree.robot.g1.audio.g1_audio_client import AudioClient
            ChannelFactory.Instance().Init(0, network_interface)
            self.client = AudioClient()
            self.client.Init()
            self.available = True
        except Exception:
            self.available = False

    def led_control(self, r: int, g: int, b: int):
        if self.available:
            try:
                self.client.LedControl(r, g, b)
            except Exception:
                pass

# ============================================================================
# AUDIO UTILS
# ============================================================================
def calculate_rms(audio_chunk: bytes) -> float:
    if not audio_chunk:
        return 0.0
    count = len(audio_chunk) // 2
    samples = struct.unpack(f"{count}h", audio_chunk)
    return (sum(s*s for s in samples) / count) ** 0.5

# ============================================================================
# STT (AlemAI)
# ============================================================================
def transcribe_audio(audio_pcm: List[int]) -> str:
    api_key = os.getenv("ALEMAI_STT_API_KEY")
    if not api_key:
        return ""

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name

    try:
        with open(wav_path, "wb") as wf:
            wf.write(struct.pack(f"{len(audio_pcm)}h", *audio_pcm))

        r = requests.post(
            "https://llm.alem.ai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": open(wav_path, "rb")},
            data={"model": "speech-to-text-kk"},
            timeout=15,
        )

        if r.status_code == 200:
            txt = r.json().get("text", "").strip()
            return "" if txt.lower() in recent_transcriptions else txt
        return ""
    finally:
        os.remove(wav_path)

# ============================================================================
# LLM
# ============================================================================
def get_llm_response(user_text: str) -> str:
    api_key = os.getenv("ALEMAI_LLM_API_KEY")
    if not api_key:
        return ""

    messages = [{
        "role": "system",
        "content": "You are Temirbek, a Kazakh-speaking robot assistant. Reply briefly in Kazakh."
    }]

    for t in conversation_history[-MAX_HISTORY_TURNS:]:
        messages.append({"role": "user", "content": t.user_message})
        messages.append({"role": "assistant", "content": t.assistant_message})

    messages.append({"role": "user", "content": user_text})

    r = requests.post(
        "https://llm.alem.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": "alemllm", "messages": messages},
        timeout=10,
    )

    if r.status_code == 200:
        return r.json()["choices"][0]["message"]["content"]
    return ""

# ============================================================================
# 🔥 STREAMING EDGE TTS (OPTION 2)
# ============================================================================
def speak_streaming_edge_tts(text: str):
    global g_is_speaking
    if not text.strip():
        return

    g_is_speaking = True

    def _run():
        import edge_tts

        ffmpeg = subprocess.Popen(
            ["ffmpeg", "-i", "pipe:0", "-ar", str(SAMPLE_RATE), "-ac", "1",
             "-f", "s16le", "pipe:1"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

        aplay = subprocess.Popen(
            ["aplay", "-D", ALSA_OUTPUT_DEVICE, "-f", "S16_LE",
             "-r", str(SAMPLE_RATE), "-c", "1", "-q"],
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

        def pcm_forward():
            while True:
                data = ffmpeg.stdout.read(4096)
                if not data:
                    break
                aplay.stdin.write(data)

        threading.Thread(target=pcm_forward, daemon=True).start()

        async def producer():
            communicate = edge_tts.Communicate(text=text, voice=EDGE_TTS_VOICE)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    ffmpeg.stdin.write(chunk["data"])
            ffmpeg.stdin.close()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(producer())
        loop.close()

        ffmpeg.wait()
        aplay.stdin.close()
        aplay.wait()

    try:
        _run()
    finally:
        g_is_speaking = False

# ============================================================================
# VAD RECORDING
# ============================================================================
def record_with_vad() -> Optional[List[int]]:
    cmd = ["arecord", "-D", ALSA_INPUT_DEVICE, "-f", "S16_LE",
           "-r", str(SAMPLE_RATE), "-c", "1", "-q"]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    buf = b""
    speech = []
    silence = 0
    speaking = False

    while g_running and not g_is_speaking:
        chunk = proc.stdout.read(CHUNK_SIZE * 2)
        if not chunk:
            break
        buf += chunk

        if len(buf) >= SAMPLE_RATE // 10 * 2:
            frame = buf[:SAMPLE_RATE // 10 * 2]
            buf = buf[len(frame):]
            rms = calculate_rms(frame)

            if rms > SPEECH_START_THRESHOLD:
                speaking = True
                silence = 0

            if speaking:
                speech.extend(struct.unpack(f"{len(frame)//2}h", frame))
                silence = silence + 1 if rms < SILENCE_THRESHOLD else 0
                if silence * 0.1 >= SILENCE_DURATION:
                    break

    proc.terminate()
    return speech if len(speech) / SAMPLE_RATE >= MIN_SPEECH_DURATION else None

# ============================================================================
# MAIN LOOP
# ============================================================================
def process_conversation(client: G1AudioClient):
    while g_running:
        client.led_control(0, 255, 0)
        audio = record_with_vad()
        client.led_control(0, 0, 0)

        if not audio:
            continue

        print("📝 Transcribing...")
        text = transcribe_audio(audio)
        if not text:
            continue

        print("👤", text)
        reply = get_llm_response(text)
        print("🤖", reply)

        conversation_history.append(ConversationTurn(text, reply))
        conversation_history[:] = conversation_history[-MAX_HISTORY_TURNS:]

        client.led_control(0, 100, 255)
        speak_streaming_edge_tts(reply)
        client.led_control(0, 0, 0)

# ============================================================================
# MAIN
# ============================================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: script.py <network_interface>")
        return

    client = G1AudioClient(sys.argv[1])
    print("✅ Temirbek ready (Streaming TTS)")
    process_conversation(client)

if __name__ == "__main__":
    main()
