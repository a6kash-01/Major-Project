import os
import logging
from pathlib import Path

# Setup standardized application logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("clinical_ai")

# Fix Windows Anaconda OpenMP & Intel Fortran runtime signal handler conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

import torch

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
STATIC_DIR = BASE_DIR / "app" / "static"
TEMPLATES_DIR = BASE_DIR / "app" / "templates"

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "clinical_reports.db"
BENCHMARK_PATH = DATA_DIR / "benchmark_dataset.json"

# Hardware Device Detection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Whisper ASR Config
# Models: 'tiny', 'base', 'small', 'medium', 'large'. Word error rate on the spoken benchmark
# (scripts/evaluate_asr.py): tiny 8.1%, base 5.4%, small 2.6%. Small is quick only on a GPU.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small" if DEVICE == "cuda" else "base")
# Context sentence given to Whisper before decoding (empty = none)
WHISPER_PROMPT = os.getenv("WHISPER_PROMPT", "")

# Summarization Models Config
# Pretrained checkpoints already fine-tuned (by their authors) on SAMSum dialogue summaries; scored best
# on the clinical benchmark (see scripts/compare_models.py). The original project used
# facebook/bart-large-cnn and t5-base; set BART_MODEL / T5_MODEL to switch back for comparison.
BART_MODEL = os.getenv("BART_MODEL", "philschmid/bart-large-cnn-samsum")
T5_MODEL = os.getenv("T5_MODEL", "philschmid/flan-t5-base-samsum")
T5_PROMPT = os.getenv("T5_PROMPT", "summarize: ")

# Chunking Config for Long Conversations
MAX_CHUNK_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 50

# Security & Upload Limit (50 MB)
MAX_UPLOAD_SIZE = 50 * 1024 * 1024

# Filler words list to clean in preprocessing
FILLER_WORDS = [
    r"\bum\b", r"\buh\b", r"\buhh\b", r"\bumm\b", r"\bhmm\b", r"\berr\b",
    r"\bah\b", r"\bhhh\b", r"\blike,", r"\byou know,", r",\s*you know(?=[.!?])",
    r"\bI mean,", r"\bso, yeah\b", r"\bkind of\b", r"\bsort of\b"
]
