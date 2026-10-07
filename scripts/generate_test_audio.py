"""
Generate spoken test audio for the benchmark conversations (Windows only).

Each conversation in data/benchmark_dataset.json is read aloud with the Windows built-in
text-to-speech voices - one voice for the doctor, another for the patient - and saved as a
16 kHz WAV file in data/test_audio/. Because the words are known, the audio can be used to
measure Whisper's word error rate (see scripts/evaluate_asr.py).

Usage:
    python scripts/generate_test_audio.py
"""
import re
import json
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

ROOT_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_PATH = ROOT_DIR / "data" / "benchmark_dataset.json"
OUT_DIR = ROOT_DIR / "data" / "test_audio"
VOICES = {"Doctor": "Microsoft David Desktop", "Patient": "Microsoft Zira Desktop"}

POWERSHELL = r"""
Add-Type -AssemblyName System.Speech
$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SetOutputToWaveFile($args[1], $format)
$synth.SpeakSsml([System.IO.File]::ReadAllText($args[0]))
$synth.Dispose()
"""


def to_ssml(transcript):
    parts = []
    for line in transcript.split("\n"):
        match = re.match(r"^\s*(Doctor|Patient)\s*:\s*(.*)$", line)
        if match:
            speaker, text = match.groups()
            parts.append(f'<voice name="{VOICES[speaker]}">{escape(text)}</voice><break time="400ms"/>')
    return ('<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
            + "".join(parts) + "</speak>")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    script_path = OUT_DIR / "_synthesize.ps1"
    script_path.write_text(POWERSHELL, encoding="utf-8")

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)

    for case in cases:
        ssml_path = OUT_DIR / f"{case['id']}.ssml"
        wav_path = OUT_DIR / f"{case['id']}.wav"
        ssml_path.write_text(to_ssml(case["transcript"]), encoding="utf-8")
        subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script_path),
                        str(ssml_path), str(wav_path)], check=True)
        ssml_path.unlink()
        print(f"  {wav_path.name}  ({wav_path.stat().st_size // 1024} KB)  {case['title']}")

    script_path.unlink()
    print(f"\nAudio written to {OUT_DIR}")


if __name__ == "__main__":
    main()
