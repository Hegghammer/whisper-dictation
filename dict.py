"""
Simple transcription app for use with any backend that supports the OpenAI SDK. It notably works with [Speaches](https://github.com/speaches-ai/speaches/) for offline Whisper models.

Minimal requirements:
  pip install sounddevice numpy pynput scipy openai
Environment:
  Set the vars DICTATION_BASE_URL and DICTATION_API_KEY.
  If you are using Speaches, this will be your server URL and the api_key you set for your Speaches instance.
Usage:
  python dict.py $WHISPER_MODEL
  If you are using Speaches, this can be one of the aliases you set in `model_aliases.json` 
"""

import os
import re
import sys
import tempfile
import sounddevice as sd
import numpy as np
from pynput.keyboard import Controller as KeyboardController, Key, Listener
from scipy.io import wavfile
from openai import OpenAI, OpenAIError

CUSTOM_SPELLING="""
Names: Gloucestershire, Kyrkjsæterøra
Here in London we honour high-calibre travellers, never take offence, and never apologise. It's 4 June 2023.
""" # NB: 224 tokens max
HOTKEY = "ctrl_r" 
WHISPER_MODEL = sys.argv[1] if len(sys.argv) > 1 else "whisper-1"
API_KEY = os.environ.get("DICTATION_API_KEY")
BASE_URL = os.environ.get("DICTATION_BASE_URL", None)
SAMPLE_RATE = 16000

REPLACEMENTS_REAL = [
    (r'\bactual new ?line\b[,.?]? ?', 'NEW_LINE_PLACEHOLDER'),
    (r'\bactual inverted comma\b[,.?]? ?', 'INVERTED_COMMA_PLACEHOLDER'),
    (r'\bactual comma\b[,.?]? ?', 'COMMA_PLACEHOLDER'),
    (r'\bactual full stop\b[,.?]? ?', 'FULL_STOP_PLACEHOLDER'),
]

REPLACEMENTS = [
    (r'\bnew ?line\b[,.?]? ?', '\n'),
    (r'\binverted comma\b[,.?]?', '"'),
    (r'\bcomma\b[,.?]? ?', ', '),
    (r'\bfull stop\b[,.?]? ?', '. '),
]

PLACEHOLDERS = [
    ('NEW_LINE_PLACEHOLDER', 'new line '),
    ('INVERTED_COMMA_PLACEHOLDER', 'inverted comma '),
    ('COMMA_PLACEHOLDER', 'comma '),
    ('FULL_STOP_PLACEHOLDER', 'full stop '),
]

if not API_KEY:
    print("Error: Please set the OPENAI_API_KEY environment variable.")
    sys.exit(1)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
keyboard_controller = KeyboardController()

try:
    KEY = getattr(Key, HOTKEY)
except AttributeError:
    print(f"Error: HOTKEY '{HOTKEY}' is not a valid pynput.keyboard.Key")
    sys.exit(1)

recording = False
audio_data = []

def apply_whisper(filepath: str) -> str:
    with open(filepath, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=audio_file,
            prompt=CUSTOM_SPELLING
        )
    return response.text

def start_recording():
    global recording, audio_data
    recording = True
    audio_data = []
    print("Listening...")

def clean_transcript(transcript):
    temp = transcript
    for pattern, replacement in REPLACEMENTS_REAL:
        temp = re.sub(pattern, replacement, temp, flags=re.IGNORECASE)
    for pattern, replacement in REPLACEMENTS:
        temp = re.sub(pattern, replacement, temp, flags=re.IGNORECASE)
    for placeholder, text in PLACEHOLDERS:
        temp = temp.replace(placeholder, text)
    return temp

def stop_recording_and_process():
    global recording, audio_data
    recording = False
    print("Transcribing...")
    if not audio_data:
        print("No audio recorded")
        return
    audio_np = np.concatenate(audio_data, axis=0)
    audio_int16 = (audio_np * np.iinfo(np.int16).max).astype(np.int16)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as tmpfile:
        wavfile.write(tmpfile.name, SAMPLE_RATE, audio_int16)
        try:
            transcript = apply_whisper(tmpfile.name).strip()
            if transcript:
                cleaned = clean_transcript(transcript)
                # remove duplicate punctuation
                cleaned = re.sub(r'([,.!?])(?:\s*\1)+', r'\1', cleaned)
                cleaned = re.sub(r',\s*\.', '.', cleaned)
                cleaned = re.sub(r'\s+([,.!?])', r'\1', cleaned)
                # fix spaces around inverted commas
                cleaned = re.sub(r'"\s+', '"', cleaned)
                cleaned = re.sub(r'\s+"', '"', cleaned)
                cleaned = re.sub(r'",', '"', cleaned)
                cleaned = re.sub(r',"', ' "', cleaned)
                keyboard_controller.type(cleaned + " ")
        except OpenAIError as e:
            print(f"OpenAI error: {e}")
        except Exception as e:
            print(f"Error: {e}")

def on_press(key):
    global recording
    if key == KEY and not recording:
        start_recording()

def on_release(key):
    global recording
    if key == KEY and recording:
        stop_recording_and_process()

def callback(indata, frames, time, status):
    if status:
        print(f"Audio status: {status}")
    if recording:
        audio_data.append(indata.copy())

def main():
    print(f"Ready to transcribe with {WHISPER_MODEL}.")
    print(f"Hold '{HOTKEY}' to transcribe (Ctrl+C to quit).")
    try:
        with Listener(on_press=on_press, on_release=on_release) as listener:
            with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, callback=callback):
                listener.join()
    except KeyboardInterrupt:
        print("\nExiting on user interrupt.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
