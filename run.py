import os
import sys
import subprocess
from pathlib import Path

# Auto-detect and switch to project virtual environment (.venv) if current Python lacks dependencies
venv_python = Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe"
if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
    try:
        import uvicorn
    except ImportError:
        script_path = str(Path(__file__).resolve())
        args = [str(venv_python), script_path] + sys.argv[1:]
        result = subprocess.run(args)
        sys.exit(result.returncode)

# Prevent Windows Intel Fortran (libifcoremd.dll) and OpenMP (libiomp5md.dll) signal handler crashes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

import uvicorn
from app.main import app

if __name__ == "__main__":
    print("==========================================================================")
    print(" Starting AI Powered Clinical Report Generation Server...")
    print(" Address: http://127.0.0.1:8000")
    print(" Pipeline: Whisper ASR -> Text Preprocessing -> BART/T5 -> Clinical Report -> SQLite")
    print("==========================================================================")
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
