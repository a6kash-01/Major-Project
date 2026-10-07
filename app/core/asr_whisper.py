import os
import shutil

# Fix Windows Anaconda OpenMP & Intel Fortran runtime signal handler conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

import torch
import whisper
from typing import Dict, Any
from app.config import WHISPER_MODEL, WHISPER_PROMPT, DEVICE, logger

# Ensure FFmpeg executable is located and added to System PATH for Whisper audio decoding
try:
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_exe)
    target_exe = os.path.join(ffmpeg_dir, "ffmpeg.exe")
    if not os.path.exists(target_exe):
        shutil.copyfile(ffmpeg_exe, target_exe)
    if ffmpeg_dir not in os.environ["PATH"]:
        os.environ["PATH"] = ffmpeg_dir + os.path.pathsep + os.environ["PATH"]
    logger.info(f"[WhisperASR] FFmpeg binary initialized cleanly at: {target_exe}")
except Exception as e:
    logger.warning(f"[WhisperASR] FFmpeg initialization check: {e}")

class WhisperASR:
    """
    OpenAI Whisper Automatic Speech Recognition Engine.
    Converts doctor-patient conversation audio into text transcript.
    """
    def __init__(self, model_name: str = WHISPER_MODEL, device: str = DEVICE, prompt: str = WHISPER_PROMPT):
        self.model_name = model_name
        self.device = device
        # Optional context sentence that steers Whisper towards clinical vocabulary and number formats
        self.prompt = prompt
        self.model = None

    def _load_model(self):
        if self.model is None:
            logger.info(f"[WhisperASR] Loading Whisper model '{self.model_name}' on device '{self.device}'...")
            try:
                self.model = whisper.load_model(self.model_name, device=self.device)
            except Exception as e:
                logger.warning(f"[WhisperASR] GPU load failed ({e}), falling back to CPU...")
                self.device = "cpu"
                self.model = whisper.load_model(self.model_name, device="cpu")
            logger.info(f"[WhisperASR] Whisper model '{self.model_name}' loaded successfully.")

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe an audio file to text using OpenAI Whisper.
        Returns dict containing full transcript, segments, and language.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found at path: {audio_path}")
            
        self._load_model()
        
        logger.info(f"[WhisperASR] Transcribing audio file: {audio_path}")
        result = self.model.transcribe(
            audio_path, fp16=(self.device == "cuda"), initial_prompt=self.prompt or None
        )
        
        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "id": seg.get("id"),
                "start": round(seg.get("start", 0.0), 2),
                "end": round(seg.get("end", 0.0), 2),
                "text": seg.get("text", "").strip()
            })
            
        return {
            "transcript": result.get("text", "").strip(),
            "language": result.get("language", "en"),
            "segments": segments
        }

# Global Singleton Instance
whisper_asr = WhisperASR()
