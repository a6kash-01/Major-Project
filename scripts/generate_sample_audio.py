import wave
import math
import struct
from pathlib import Path

def generate_consultation_tone_audio(output_path: str, duration_sec: int = 10):
    """
    Generates a clean PCM WAV audio file simulating speech frequencies.
    This guarantees a valid, non-corrupt WAV file for Whisper testing.
    """
    sample_rate = 16000  # Standard 16kHz audio format for Whisper ASR
    num_samples = sample_rate * duration_sec
    
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(path), "w") as wav_file:
        wav_file.setnchannels(1)       # Mono
        wav_file.setsampwidth(2)      # 16-bit
        wav_file.setframerate(sample_rate)

        # Synthesize audio with speech-range formants (150Hz, 300Hz, 800Hz) and gentle amplitude modulations
        for i in range(num_samples):
            t = float(i) / sample_rate
            # Speech formant envelope modulation
            envelope = 0.6 * math.sin(2 * math.pi * 0.5 * t) + 0.4 * math.cos(2 * math.pi * 1.2 * t)
            envelope = max(0.1, abs(envelope))
            
            signal = (
                0.5 * math.sin(2 * math.pi * 180 * t) +   # Fundamental male/female pitch
                0.3 * math.sin(2 * math.pi * 440 * t) +   # Vowel formant 1
                0.2 * math.sin(2 * math.pi * 950 * t)     # Vowel formant 2
            )
            
            sample_val = int(signal * envelope * 12000)
            sample_val = max(-32768, min(32767, sample_val))
            
            data = struct.pack("<h", sample_val)
            wav_file.writeframesraw(data)

    print(f"[SampleGenerator] Sample audio file generated successfully at: {path}")

if __name__ == "__main__":
    target = Path(__file__).resolve().parent.parent / "app" / "static" / "samples" / "sample_doctor_patient.wav"
    generate_consultation_tone_audio(str(target), duration_sec=8)
