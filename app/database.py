import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from app.config import DB_PATH

def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS consultation_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            consultation_id TEXT UNIQUE NOT NULL,
            patient_id TEXT NOT NULL,
            doctor_info TEXT NOT NULL,
            datetime TEXT NOT NULL,
            audio_ref TEXT,
            transcript TEXT NOT NULL,
            clean_transcript TEXT NOT NULL,
            generated_summary TEXT NOT NULL,
            clinical_report_json TEXT NOT NULL,
            selected_model TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_report(
    consultation_id: str,
    patient_id: str,
    doctor_info: str,
    datetime_str: str,
    audio_ref: Optional[str],
    transcript: str,
    clean_transcript: str,
    generated_summary: str,
    clinical_report: Dict[str, Any],
    selected_model: str
) -> Dict[str, Any]:
    conn = get_db_connection()
    cursor = conn.cursor()
    
    report_json_str = json.dumps(clinical_report, indent=2)
    now_str = datetime.now().isoformat()
    
    cursor.execute("""
        INSERT INTO consultation_reports (
            consultation_id, patient_id, doctor_info, datetime, audio_ref,
            transcript, clean_transcript, generated_summary, clinical_report_json, selected_model
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(consultation_id) DO UPDATE SET
            patient_id = excluded.patient_id,
            doctor_info = excluded.doctor_info,
            datetime = excluded.datetime,
            audio_ref = excluded.audio_ref,
            transcript = excluded.transcript,
            clean_transcript = excluded.clean_transcript,
            generated_summary = excluded.generated_summary,
            clinical_report_json = excluded.clinical_report_json,
            selected_model = excluded.selected_model
    """, (
        consultation_id, patient_id, doctor_info, datetime_str or now_str, audio_ref,
        transcript, clean_transcript, generated_summary, report_json_str, selected_model
    ))
    conn.commit()
    
    # Retrieve saved record
    cursor.execute("SELECT * FROM consultation_reports WHERE consultation_id = ?", (consultation_id,))
    row = cursor.fetchone()
    conn.close()
    
    return dict_from_row(row)

def get_report_by_id(consultation_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM consultation_reports WHERE consultation_id = ?", (consultation_id,))
    row = cursor.fetchone()
    conn.close()
    return dict_from_row(row) if row else None

def get_all_reports() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM consultation_reports ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict_from_row(r) for r in rows]

def delete_report(consultation_id: str) -> bool:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM consultation_reports WHERE consultation_id = ?", (consultation_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

def search_reports(query: str) -> List[Dict[str, Any]]:
    """Search reports by consultation_id, patient_id, doctor_info, or transcript content."""
    conn = get_db_connection()
    cursor = conn.cursor()
    search_term = f"%{query}%"
    cursor.execute("""
        SELECT * FROM consultation_reports 
        WHERE consultation_id LIKE ? 
           OR patient_id LIKE ? 
           OR doctor_info LIKE ?
           OR clean_transcript LIKE ?
        ORDER BY id DESC
    """, (search_term, search_term, search_term, search_term))
    rows = cursor.fetchall()
    conn.close()
    return [dict_from_row(r) for r in rows]

def dict_from_row(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    if "clinical_report_json" in d and d["clinical_report_json"]:
        try:
            d["clinical_report"] = json.loads(d["clinical_report_json"])
        except Exception:
            d["clinical_report"] = {}
    return d

# Initialize database on module import
init_db()
