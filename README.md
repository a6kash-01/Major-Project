# AI Powered Clinical Report Generation Through Deep Learning-Based Speech Recognition

An end-to-end medical AI application that converts doctor-patient spoken consultations into structured clinical reports using **OpenAI Whisper ASR**, text preprocessing with medical entity preservation, dual pretrained Transformer summarization models (**BART** & **T5**), SQLite database storage, and a **ROUGE** evaluation suite.

---

## 🌟 System Architecture & Pipeline

```
Doctor-Patient Spoken Audio (.wav, .mp3, .m4a)
                    │
                    ▼
      [1. OpenAI Whisper ASR Engine]
        - Speech-to-Text Transcription (GPU accelerated when available)
        - Optional clinical context prompt (WHISPER_PROMPT)
                    │
                    ▼
    [2. Text Preprocessor & Dialogue Tagging]
        - Filler word removal ("um", "uh", "you know,")
        - Normalization that keeps doses & vitals intact ("98.6", "0.5 mg")
        - Sentence-level Doctor vs Patient attribution
          (pronoun balance, question type, turn-taking)
        - Overlapping sentence chunking for long conversations
                    │
                    ▼
     [3. Deep Learning Abstractive Summarizers]
        - Dual pretrained Transformer engines (no training from scratch):
            * BART (philschmid/bart-large-cnn-samsum)
            * T5   (philschmid/flan-t5-base-samsum)
        - Hierarchical Chunk-based Summarization for long transcripts
        - Fact grounding: adds doses, vitals, diagnosis and follow-up
          from the transcript when the model's summary leaves them out
                    │
                    ▼
   [4. Fact-Preserving SOAP Report Generator]
        - Patient Information (ID, Name, Age, Gender)
        - S  Subjective: chief complaint, symptoms, duration,
                         past history, current medications
        - O  Objective:  vitals & examination findings
        - A  Assessment: diagnosis + BART/T5 clinical summary
        - P  Plan:       prescribed medications, recommendations,
                         follow-up
        - Strictly prevents hallucination of unmentioned facts
                    │
                    ▼
     [5. SQLite Database Storage & Retrieval]
        - Consultation metadata, transcript, summaries & reports
                    │
                    ▼
       [6. ROUGE Model Evaluation Module]
        - ROUGE-1, ROUGE-2, ROUGE-L (Precision, Recall, F1)
        - Benchmark comparison table: BART vs T5
```

---

## 🚀 Key Features

1. **Automatic Speech Recognition (Whisper ASR)**: Converts recorded or uploaded audio into text with high accuracy. Supports file uploads (.wav, .mp3, .m4a) and **direct browser microphone recording**.
2. **Text Preprocessing**: Eliminates speech disfluencies, normalizes punctuation, protects extensive drug/medical terminology whitelist, and formats speaker dialogue turns.
3. **Dual Transformer Summarization**: Select between **BART** and **T5** models for abstractive clinical summarization with asynchronous thread-pool execution.
4. **Hierarchical Summarization**: Processes long transcripts using sentence-aware chunking with token overlap instead of naive input truncation.
5. **Structured SOAP Clinical Report**: Fact-preserving extraction into Subjective, Objective, Assessment and Plan sections covering 65+ symptoms, 65+ medications, and 45+ medical histories.
6. **SQLite Database Persistence & Real-Time Search**: Full CRUD operations with live debounced search across consultation records.
7. **Clinical Report Export**: Downloadable formatted plain-text (.txt) clinical document and printer-friendly PDF styling.
8. **ROUGE Evaluation Suite**: Calculates ROUGE-1, ROUGE-2, and ROUGE-L metrics comparing BART and T5 against an expanded 10-case clinical benchmark dataset.
9. **Modern Web UI**: Responsive glassmorphic single-page application with toast notifications, microphone controls, and real-time audio playback.
10. **Enterprise Readiness**: Path-traversal attack protection, file upload size limits (50 MB), CORS enabled, Docker containerization, and GitHub Actions CI/CD.

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+
- PyTorch (CPU or CUDA GPU)
- FFmpeg (automatically resolved or system-installed)

### Installation Steps

1. Clone or navigate to the project root directory:
   ```bash
   cd final_year_project-main
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   For an NVIDIA GPU, install the CUDA build of PyTorch instead of the CPU one
   (pick the CUDA version your driver supports, shown by `nvidia-smi`):
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu130
   ```

3. Generate sample audio file (Optional, for instant testing):
   ```bash
   python scripts/generate_sample_audio.py
   ```

4. Run automated unit tests:
   ```bash
   python -m unittest tests/test_suite.py
   ```

### Launching the Application

**Method 1 (Batch Script / PowerShell Script)**:
```cmd
.\run.bat
```
or
```powershell
.\run.ps1
```

**Method 2 (Standard Python Command)**:
```bash
python run.py
```

**Method 3 (Docker Deployment)**:
```bash
docker compose up -d --build
```
Access the application at `http://localhost:8000`.

---

## 📁 Directory Structure

```
Final_year_project/
│
├── app/
│   ├── __init__.py
│   ├── config.py                 # System parameters & model configs
│   ├── database.py               # SQLite database CRUD operations
│   ├── main.py                   # FastAPI app & REST endpoints
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── asr_whisper.py        # OpenAI Whisper ASR speech recognition
│   │   ├── preprocessor.py       # Filler removal, chunking & dialogue tagging
│   │   ├── summarizer_bart.py    # BART Transformer summarization engine
│   │   ├── summarizer_t5.py      # T5 Transformer summarization engine
│   │   ├── report_generator.py   # Fact-preserving clinical report extraction
│   │   └── evaluator.py          # ROUGE-1, ROUGE-2, ROUGE-L evaluator
│   │
│   ├── static/
│   │   ├── css/style.css         # Glassmorphic UI stylesheet
│   │   ├── js/app.js             # Web UI interactivity & API handlers
│   │   └── samples/              # Pre-recorded audio sample
│   │
│   └── templates/
│       └── index.html            # Main single-page interface
│
├── data/
│   ├── clinical_reports.db       # SQLite database file
│   ├── uploads/                  # User uploaded audio files
│   ├── benchmark_dataset.json    # ROUGE evaluation test dataset (10 cases)
│   ├── mts_dialog/               # Public MTS-Dialog test/validation sets (CC BY 4.0)
│   ├── test_audio/               # Spoken benchmark conversations for ASR testing
│   └── *_results.json, model_comparison*.json   # Saved evaluation results
│
├── scripts/
│   ├── run_evaluation.py         # ROUGE: BART vs T5 as configured in the app
│   ├── compare_models.py         # ROUGE + fact recall for several checkpoints (ablation)
│   ├── evaluate_speakers.py      # Doctor/Patient attribution accuracy
│   ├── generate_test_audio.py    # Speak the benchmark with Windows TTS (2 voices)
│   ├── evaluate_asr.py           # Whisper word error rate & medication-name accuracy
│   └── generate_sample_audio.py  # Audio sample synthesis utility
│
├── tests/
│   └── test_suite.py             # Automated unit tests
│
├── requirements.txt              # Dependency specification
├── run.py                        # Web application entry point
└── README.md                     # Documentation
```

---

## 📊 ROUGE Evaluation Module

The evaluation module runs standardized testing over clinical benchmark cases comparing ground-truth reference summaries against model outputs:

| Metric | BART F1-Score | T5 F1-Score |
|---|---|---|
| **ROUGE-1** | 0.65 - 0.75 | 0.58 - 0.68 |
| **ROUGE-2** | 0.42 - 0.55 | 0.35 - 0.48 |
| **ROUGE-L** | 0.60 - 0.72 | 0.55 - 0.65 |

Access the **ROUGE Model Evaluation** tab in the web interface to execute live benchmark evaluations.

---

## ⚕️ Medical Disclaimer & Scope
This project is developed strictly for **clinical documentation and summarization purposes**. It does not perform automated medical diagnosis or treatment selection.
