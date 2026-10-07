import os
import uuid
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import (
    BASE_DIR, UPLOAD_DIR, STATIC_DIR, TEMPLATES_DIR, BENCHMARK_PATH,
    MAX_UPLOAD_SIZE, logger
)
from app import database
from app.core.asr_whisper import whisper_asr
from app.core.preprocessor import preprocessor
from app.core.summarizer_bart import bart_summarizer
from app.core.summarizer_t5 import t5_summarizer
from app.core.report_generator import report_generator
from app.core.evaluator import rouge_evaluator

app = FastAPI(
    title="AI Powered Clinical Report Generation System",
    description="Speech Recognition (Whisper ASR) to Deep Learning Abstractive Summarization (BART/T5) & Clinical Report Extraction",
    version="1.0.0"
)

# Enable CORS for external access / frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dedicated thread pool executor for CPU-intensive ML inferences
_executor = ThreadPoolExecutor(max_workers=2)

# Mount Static assets & Templates
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Pydantic Schemas
class TextSummarizeRequest(BaseModel):
    transcript: str
    model: str = "BART"  # "BART" or "T5"
    patient_id: Optional[str] = "P-1002"
    doctor_info: Optional[str] = "Dr. Alexander Fleming, MD"

class SaveReportRequest(BaseModel):
    consultation_id: str
    patient_id: str
    doctor_info: str
    datetime: Optional[str] = None
    audio_ref: Optional[str] = None
    transcript: str
    clean_transcript: str
    generated_summary: str
    clinical_report: Dict[str, Any]
    selected_model: str

# System Health Endpoint
@app.get("/health")
def health_check():
    return {
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "database": os.path.exists(str(database.DB_PATH))
    }

# Render Front-end Web Interface
@app.get("/", response_class=FileResponse)
def read_root():
    return FileResponse(str(TEMPLATES_DIR / "index.html"))

# 1. Audio Upload & ASR Transcription Endpoint
@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Accept doctor-patient audio file upload, perform Whisper ASR transcription.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected.")

    ext = os.path.splitext(file.filename)[1].lower()
    allowed_exts = [".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"]
    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"Unsupported file format '{ext}'. Allowed: {allowed_exts}")

    # Save uploaded file
    file_id = str(uuid.uuid4())[:8]
    saved_filename = f"{file_id}_{file.filename}"
    saved_path = UPLOAD_DIR / saved_filename

    try:
        content = await file.read()
        if len(content) > MAX_UPLOAD_SIZE:
            max_mb = MAX_UPLOAD_SIZE // (1024 * 1024)
            raise HTTPException(status_code=413, detail=f"File too large. Maximum allowed size is {max_mb} MB.")
        with open(saved_path, "wb") as f:
            f.write(content)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save audio file: {str(e)}")

    # Perform Speech-to-Text Transcription via Whisper ASR (async executor)
    try:
        loop = asyncio.get_event_loop()
        asr_result = await loop.run_in_executor(_executor, whisper_asr.transcribe, str(saved_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Whisper ASR Transcription error: {str(e)}")

    raw_transcript = asr_result["transcript"]

    # Preprocess text
    proc_result = preprocessor.process(raw_transcript)

    return {
        "audio_ref": saved_filename,
        "audio_url": f"/api/audio/{saved_filename}",
        "raw_transcript": raw_transcript,
        "clean_transcript": proc_result["clean_transcript"],
        "formatted_transcript": proc_result["formatted_transcript"],
        "dialogue_turns": proc_result["dialogue_turns"],
        "chunks": proc_result["chunks"],
        "is_long_conversation": proc_result["is_long_conversation"],
        "asr_segments": asr_result.get("segments", []),
        "language": asr_result.get("language", "en")
    }

# Serve uploaded audio files (with Path Traversal protection)
@app.get("/api/audio/{filename}")
def get_audio_file(filename: str):
    # Sanitize filename strictly to prevent path traversal attacks
    safe_filename = os.path.basename(filename)
    if safe_filename != filename or ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid audio filename.")

    file_path = UPLOAD_DIR / safe_filename
    sample_path = STATIC_DIR / "samples" / safe_filename
    
    if os.path.exists(file_path):
        return FileResponse(str(file_path))
    elif os.path.exists(sample_path):
        return FileResponse(str(sample_path))
    else:
        raise HTTPException(status_code=404, detail="Audio file not found")

# 2. Text Preprocessing Endpoint
@app.post("/api/preprocess")
def preprocess_text(data: Dict[str, str]):
    transcript = data.get("transcript", "")
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript text is required")
        
    proc_result = preprocessor.process(transcript)
    return proc_result

# 3. Summarization & Clinical Report Generation Endpoint (BART or T5)
@app.post("/api/summarize")
async def generate_summary_and_report(req: TextSummarizeRequest):
    """
    Summarize transcript using user-selected model (BART or T5)
    and extract structured clinical report. Runs CPU-intensive inference in thread pool.
    """
    transcript = req.transcript.strip()
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript text cannot be empty.")

    selected_model = req.model.upper()
    if selected_model not in ["BART", "T5"]:
        raise HTTPException(status_code=400, detail="Model selection must be 'BART' or 'T5'.")

    loop = asyncio.get_event_loop()

    # Run Preprocessor
    proc_res = await loop.run_in_executor(_executor, preprocessor.process, transcript)
    chunks = proc_res["chunks"]

    # Execute Selected Summarization Model asynchronously
    if selected_model == "BART":
        sum_result = await loop.run_in_executor(_executor, bart_summarizer.summarize, chunks)
    else:
        sum_result = await loop.run_in_executor(_executor, t5_summarizer.summarize, chunks)

    # Fact grounding: add key facts from the transcript that the model's summary left out
    model_summary = sum_result["summary"]
    summary_text = report_generator.ground_summary(
        model_summary, proc_res["dialogue_turns"], proc_res["clean_transcript"]
    )
    consultation_id = f"CONS-{str(uuid.uuid4())[:8].upper()}"

    # Generate Structured Clinical Report
    report = report_generator.generate_report(
        transcript=transcript,
        clean_transcript=proc_res["clean_transcript"],
        dialogue_turns=proc_res["dialogue_turns"],
        summary=summary_text,
        model_used=selected_model,
        patient_id=req.patient_id,
        consultation_id=consultation_id
    )

    return {
        "consultation_id": consultation_id,
        "selected_model": selected_model,
        "summarization_method": sum_result["method"],
        "generated_summary": summary_text,
        "model_summary": model_summary,
        "chunk_summaries": sum_result.get("chunk_summaries", []),
        "clean_transcript": proc_res["clean_transcript"],
        "formatted_transcript": proc_res["formatted_transcript"],
        "clinical_report": report
    }

# 4. Database CRUD Endpoints
@app.post("/api/reports")
def save_clinical_report(req: SaveReportRequest):
    """Save generated clinical report to SQLite Database."""
    try:
        saved_record = database.save_report(
            consultation_id=req.consultation_id,
            patient_id=req.patient_id,
            doctor_info=req.doctor_info,
            datetime_str=req.datetime or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            audio_ref=req.audio_ref,
            transcript=req.transcript,
            clean_transcript=req.clean_transcript,
            generated_summary=req.generated_summary,
            clinical_report=req.clinical_report,
            selected_model=req.selected_model
        )
        return {"status": "success", "message": "Report saved to database successfully", "data": saved_record}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database save failed: {str(e)}")

@app.get("/api/reports")
def list_reports():
    """Retrieve all saved consultation reports from SQLite Database."""
    reports = database.get_all_reports()
    return {"status": "success", "count": len(reports), "reports": reports}

@app.get("/api/reports/search/{query}")
def search_reports(query: str):
    """Search consultation reports by ID, patient ID, doctor info, or transcript content."""
    reports = database.search_reports(query)
    return {"status": "success", "count": len(reports), "reports": reports}

@app.get("/api/reports/{consultation_id}")
def get_report(consultation_id: str):
    """Retrieve a specific consultation report by ID."""
    report = database.get_report_by_id(consultation_id)
    if not report:
        raise HTTPException(status_code=404, detail=f"Report '{consultation_id}' not found.")
    return {"status": "success", "report": report}

@app.get("/api/reports/{consultation_id}/export")
def export_report_text(consultation_id: str):
    """Export a consultation report as a downloadable formatted clinical document."""
    report = database.get_report_by_id(consultation_id)
    if not report:
        raise HTTPException(status_code=404, detail=f"Report '{consultation_id}' not found.")
    
    clinical = report.get("clinical_report", {})
    patient = clinical.get("patient_information", {})
    soap = clinical.get("soap") or report_generator.soap_from_legacy(
        clinical, report.get("generated_summary", "")
    )
    subj, obj, assess, plan = soap["subjective"], soap["objective"], soap["assessment"], soap["plan"]

    def bullets(items):
        return [f"  - {item}" for item in items] or ["  - None"]

    lines = [
        "=" * 60,
        "CLINICAL CONSULTATION REPORT",
        "=" * 60,
        f"Consultation ID: {report.get('consultation_id', 'N/A')}",
        f"Date: {report.get('datetime', 'N/A')}",
        f"Doctor: {report.get('doctor_info', 'N/A')}",
        f"Model Used: {report.get('selected_model', 'N/A')}",
        "",
        "--- PATIENT INFORMATION ---",
        f"Patient ID: {patient.get('patient_id', 'N/A')}",
        f"Name: {patient.get('name', 'Not specified')}",
        f"Age: {patient.get('age', 'Not specified')}",
        f"Gender: {patient.get('gender', 'Not specified')}",
        "",
        "--- S: SUBJECTIVE ---",
        f"Chief Complaint: {subj['chief_complaint']}",
        f"Duration: {subj['history_of_present_illness']['duration']}",
        "Symptoms:",
        *bullets(subj['history_of_present_illness']['symptoms']),
        f"Past Medical History: {', '.join(subj['past_medical_history'])}",
        f"Current Medications: {', '.join(subj['current_medications'])}",
        "",
        "--- O: OBJECTIVE ---",
        *bullets(obj['findings']),
        "",
        "--- A: ASSESSMENT ---",
        "Diagnosis / Impression:",
        *bullets(assess['diagnosis']),
        f"Clinical Summary ({report.get('selected_model', 'N/A')}):",
        f"  {assess['clinical_summary']}",
        "",
        "--- P: PLAN ---",
        f"Prescribed Medications: {', '.join(plan['prescribed_medications'])}",
        "Recommendations & Follow-Up:",
        *bullets(plan['recommendations']),
        "",
        "=" * 60,
        "CONFIDENTIAL MEDICAL RECORD - FOR CLINICAL USE ONLY",
        "=" * 60,
    ]
    
    report_text = "\n".join(lines)
    return Response(
        content=report_text,
        media_type="text/plain; charset=utf-8",
        headers={
            "Content-Disposition": f"attachment; filename={consultation_id}_clinical_report.txt"
        }
    )

@app.delete("/api/reports/{consultation_id}")
def delete_report(consultation_id: str):
    """Delete a consultation report from SQLite Database."""
    success = database.delete_report(consultation_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Report '{consultation_id}' not found.")
    return {"status": "success", "message": f"Report '{consultation_id}' deleted successfully."}

# 5. ROUGE Evaluation Module Endpoints
@app.post("/api/evaluate")
def run_evaluation():
    """
    Run ROUGE-1, ROUGE-2, and ROUGE-L benchmark evaluation comparing BART and T5.
    """
    try:
        results = rouge_evaluator.evaluate_benchmark(str(BENCHMARK_PATH))
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

# Launch Runner entrypoint
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
