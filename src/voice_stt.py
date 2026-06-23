"""
voice_stt.py
Speech-to-Text (STT) — Whisper is the production engine.

  transcribe_audio()        — OpenAI Whisper "base", runs fully locally.
                              No API key, no internet, no ffmpeg required.
                              Audio loaded via miniaudio + scipy; model
                              weights (~145 MB) cached after first download.
                              Benchmarked at 80% end-to-end pipeline accuracy
                              vs 66.7% for Google on the same 15-question set.

  transcribe_audio_google() — Google free web API (SpeechRecognition).
                              Kept for comparison. Requires internet; struggles
                              with uncommon/synthetic names.
"""

import os
import speech_recognition as sr


def transcribe_audio(file_path: str, model_name: str = "base") -> str:
    """
    Transcribe a .wav or .mp3 file using OpenAI Whisper (production engine).
    Runs fully locally — no API key, no internet, no ffmpeg required.
    Audio is loaded via miniaudio and resampled to 16 kHz via scipy.

    Model weights (~145 MB for "base") are downloaded on first call and
    cached in ~/.cache/whisper; all subsequent calls skip the download.

    Args:
        file_path:  Path to the audio file (wav or mp3).
        model_name: Whisper model size. "base" is the default (good
                    accuracy/speed on a laptop). Try "small" or "medium"
                    for higher accuracy at the cost of slower inference.

    Returns:
        Transcribed text string, or an "ERROR: ..." string on failure.
    """
    try:
        import warnings
        import numpy as np
        import miniaudio
        from scipy.signal import resample_poly
        from math import gcd
        import whisper

        if not os.path.exists(file_path):
            return f"ERROR: Could not find audio file at {file_path}"

        # Decode audio with miniaudio (handles both wav and mp3, no ffmpeg needed)
        decoded = miniaudio.decode_file(file_path)
        samples = np.frombuffer(decoded.samples, dtype=np.int16).astype(np.float32)

        # Convert stereo -> mono by averaging channels
        if decoded.nchannels == 2:
            samples = samples.reshape(-1, 2).mean(axis=1)

        # Normalize int16 range to [-1, 1]
        samples /= 32768.0

        # Resample to 16000 Hz (Whisper's required sample rate)
        src_rate = decoded.sample_rate
        target_rate = 16000
        if src_rate != target_rate:
            g = gcd(src_rate, target_rate)
            samples = resample_poly(samples, target_rate // g, src_rate // g).astype(np.float32)

        model = whisper.load_model(model_name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # suppress FP16 CPU warning
            result = model.transcribe(samples, fp16=False)

        return result["text"].strip()

    except Exception as e:
        return f"ERROR: Whisper transcription failed — {e}"


# ── Google backend (kept for comparison) ──────────────────────────────────────

def transcribe_audio_google(file_path: str) -> str:
    """
    Transcribe a .wav file using Google's free web speech API.
    Requires an internet connection. Kept as a reference/comparison backend;
    transcribe_audio() (Whisper) is preferred for production use.
    """
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)
    except FileNotFoundError:
        return f"ERROR: Could not find audio file at {file_path}"
    except sr.UnknownValueError:
        return "ERROR: Google could not understand the audio (too quiet/unclear)."
    except sr.RequestError as e:
        return f"ERROR: Could not reach Google's speech API — check your internet connection. ({e})"
    except Exception as e:
        return f"ERROR: Something unexpected went wrong — {e}"


# ──────────────────────────────────────────────────────────────────
# Everything below this line only runs when you execute this file
# directly, e.g.  python src/voice_stt.py
# It will NOT run when another file does `import voice_stt`.
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_path = os.path.join("voice_samples", "sample_question.wav")

    if not os.path.exists(sample_path):
        print(f"No sample file found at: {sample_path}")
        print()
        print("This script needs a real recorded .wav file to test with.")
        print("See the 'How to get a sample .wav file' section in the")
        print("instructions your teammate sent you for two easy options.")
    else:
        print(f"Transcribing {sample_path} ...")
        result = transcribe_audio(sample_path)
        print("Transcribed text:")
        print(result)

