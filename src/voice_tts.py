"""
voice_tts.py
Text-to-Speech (TTS) proof of concept.

This file has ONE job: take a string of text and turn it into a
spoken audio file (.mp3), using the FREE gTTS (Google Text-to-Speech)
library. No paid API key is needed, but it DOES need an internet
connection, since the text gets sent to Google's servers to generate
the audio.
"""

import os
from gtts import gTTS


def speak_answer(text, output_path):
    """
    Takes a string of text and saves it as a spoken .mp3 file.

    Steps:
    1. Make sure the folder for output_path exists.
    2. Use gTTS to turn the text into speech.
    3. Save the result to output_path.

    If something goes wrong (no internet, empty text, bad path), we
    return a clear error message instead of letting the program crash.
    """
    try:
        # Make sure the destination folder exists before we save into it
        folder = os.path.dirname(output_path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)

        # Convert the text to speech and save it to disk
        speech = gTTS(text=text, lang="en")
        speech.save(output_path)

        return f"Saved spoken audio to {output_path}"

    except Exception as e:
        return f"ERROR: Could not generate speech — {e}"


# ──────────────────────────────────────────────────────────────────
# Everything below this line only runs when you execute this file
# directly, e.g.  python src/voice_tts.py
# It will NOT run when another file does `import voice_tts`.
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_text = "The patient's average heart rate in wave 5 was 77 beats per minute"
    sample_output = os.path.join("voice_samples", "sample_answer.mp3")

    print(f'Converting text to speech: "{sample_text}"')
    result = speak_answer(sample_text, sample_output)
    print(result)
