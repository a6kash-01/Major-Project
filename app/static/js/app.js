document.addEventListener("DOMContentLoaded", () => {
  // Global Application State
  const state = {
    selectedAudioFile: null,
    audioRef: null,
    rawTranscript: "",
    cleanTranscript: "",
    formattedTranscript: "",
    dialogueTurns: [],
    chunks: [],
    selectedModel: "BART",
    currentReport: null,
    generatedSummary: "",
    consultationId: null
  };

  // Modern Toast Notification System
  function showToast(message, type = "success") {
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    const icon = type === "success" ? "fa-circle-check" : type === "error" ? "fa-circle-xmark" : "fa-circle-info";
    toast.innerHTML = `<i class="fa-solid ${icon}"></i> <span>${escapeHtml(message)}</span>`;
    document.body.appendChild(toast);
    requestAnimationFrame(() => toast.classList.add("show"));
    setTimeout(() => {
      toast.classList.remove("show");
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  }

  // UI Element References
  const tabs = document.querySelectorAll(".nav-tab");
  const tabPanes = document.querySelectorAll(".tab-pane");

  const dropZone = document.getElementById("drop-zone");
  const audioInput = document.getElementById("audio-input");
  const loadSampleBtn = document.getElementById("load-sample-btn");
  const recordMicBtn = document.getElementById("record-mic-btn");
  const stopRecordBtn = document.getElementById("stop-record-btn");
  const recordingTimer = document.getElementById("recording-timer");
  const recTimeSpan = document.getElementById("rec-time");

  const audioPlayerBox = document.getElementById("audio-player-container");
  const audioPlayer = document.getElementById("audio-player");
  const audioFilename = document.getElementById("audio-filename");

  const transcribeBtn = document.getElementById("transcribe-btn");
  const asrSpinner = document.getElementById("asr-spinner");

  const transcriptDisplay = document.getElementById("transcript-display");
  const rawTranscriptBtn = document.getElementById("raw-transcript-btn");
  const cleanTranscriptBtn = document.getElementById("clean-transcript-btn");

  const proceedBtn = document.getElementById("proceed-to-summarize-btn");
  const generateSummaryBtn = document.getElementById("generate-summary-btn");
  const summarySpinner = document.getElementById("summary-spinner");
  const summarySpinnerText = document.getElementById("summary-spinner-text");

  const modelCards = document.querySelectorAll(".model-card");
  const modelRadios = document.querySelectorAll("input[name='model-choice']");

  const reportViewContainer = document.getElementById("report-view-container");
  const saveDbBtn = document.getElementById("save-db-btn");
  const printReportBtn = document.getElementById("print-report-btn");
  const downloadReportBtn = document.getElementById("download-report-btn");

  const refreshDbBtn = document.getElementById("refresh-db-btn");
  const dbTableBody = document.getElementById("db-table-body");
  const dbSearchInput = document.getElementById("db-search-input");

  const runEvalBtn = document.getElementById("run-eval-btn");
  const evalWinnerBanner = document.getElementById("eval-winner-banner");
  const winnerModelName = document.getElementById("winner-model-name");
  const evalCasesList = document.getElementById("eval-cases-list");

  // Tab Navigation Handling
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      tabPanes.forEach(p => p.classList.remove("active"));
      tab.classList.add("active");
      
      const targetPane = document.getElementById(`tab-${tab.dataset.tab}`);
      if (targetPane) {
        targetPane.classList.add("active");
      }

      if (tab.dataset.tab === "database") {
        fetchDatabaseReports();
      }
    });
  });

  // Audio File Selection & Drag/Drop
  if (dropZone) {
    dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    });
    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
      if (e.dataTransfer.files.length) {
        handleFileSelection(e.dataTransfer.files[0]);
      }
    });
  }

  if (audioInput) {
    audioInput.addEventListener("change", (e) => {
      if (e.target.files.length) {
        handleFileSelection(e.target.files[0]);
      }
    });
  }

  // Load Pre-Recorded Sample Audio
  if (loadSampleBtn) {
    loadSampleBtn.addEventListener("click", async () => {
      const sampleUrl = "/api/audio/sample_doctor_patient.wav";
      audioFilename.textContent = "sample_doctor_patient.wav";
      audioPlayer.src = sampleUrl;
      audioPlayerBox.classList.remove("hidden");
      state.selectedAudioFile = null;
      state.audioRef = "sample_doctor_patient.wav";
      
      // Auto-load raw conversation text for fast sample testing
      const sampleTranscript = 
        "Doctor: Good morning. How can I help you today?\n" +
        "Patient: Good morning Doctor. I have been having severe headaches for 3 days. Um, it gets worse with bright light.\n" +
        "Doctor: I see. Are you experiencing any nausea?\n" +
        "Patient: Yes, Doctor. I feel nauseous whenever the headache peaks.\n" +
        "Doctor: Have you taken any medication?\n" +
        "Patient: I tried taking OTC ibuprofen 400 mg twice a day, but it barely helped.\n" +
        "Doctor: Okay. Your vitals show blood pressure is 120/80 mmHg and temperature is 98.6 F. I am diagnosing acute migraine. I am prescribing Sumatriptan 50 mg and Paracetamol 500 mg. Please rest in a quiet dark room.\n" +
        "Patient: Thank you Doctor.";
        
      state.rawTranscript = sampleTranscript;
      renderTranscriptDisplay("clean");
    });
  }

  function handleFileSelection(file) {
    state.selectedAudioFile = file;
    state.audioRef = file.name;
    audioFilename.textContent = file.name;
    audioPlayer.src = URL.createObjectURL(file);
    audioPlayerBox.classList.remove("hidden");
  }

  // Browser Microphone Live Recording (MediaRecorder API)
  let mediaRecorder = null;
  let audioChunks = [];
  let recordingInterval = null;
  let recordingSeconds = 0;

  if (recordMicBtn) {
    recordMicBtn.addEventListener("click", async () => {
      try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          showToast("Microphone recording is not supported in this browser.", "error");
          return;
        }

        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        recordingSeconds = 0;

        mediaRecorder.ondataavailable = (e) => {
          if (e.data && e.data.size > 0) {
            audioChunks.push(e.data);
          }
        };

        mediaRecorder.onstop = () => {
          clearInterval(recordingInterval);
          stream.getTracks().forEach(track => track.stop());

          const mimeType = mediaRecorder.mimeType || "audio/webm";
          const ext = mimeType.includes("wav") ? "wav" : mimeType.includes("mp4") ? "m4a" : "webm";
          const blob = new Blob(audioChunks, { type: mimeType });
          const recordedFile = new File([blob], `mic_consultation_${Date.now()}.${ext}`, { type: mimeType });

          handleFileSelection(recordedFile);

          recordMicBtn.classList.remove("hidden");
          stopRecordBtn.classList.add("hidden");
          recordingTimer.classList.add("hidden");
          showToast("Audio recording completed successfully!", "success");
        };

        mediaRecorder.start(1000);
        recordMicBtn.classList.add("hidden");
        stopRecordBtn.classList.remove("hidden");
        recordingTimer.classList.remove("hidden");
        recTimeSpan.textContent = "00:00";

        recordingInterval = setInterval(() => {
          recordingSeconds++;
          const mins = String(Math.floor(recordingSeconds / 60)).padStart(2, "0");
          const secs = String(recordingSeconds % 60).padStart(2, "0");
          recTimeSpan.textContent = `${mins}:${secs}`;
        }, 1000);

      } catch (err) {
        showToast("Microphone access denied: " + err.message, "error");
      }
    });
  }

  if (stopRecordBtn) {
    stopRecordBtn.addEventListener("click", () => {
      if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
      }
    });
  }

  // Whisper ASR Transcription Call
  if (transcribeBtn) {
    transcribeBtn.addEventListener("click", async () => {
      if (!state.selectedAudioFile && state.audioRef !== "sample_doctor_patient.wav") {
        showToast("Please select, record, or load an audio file first.", "error");
        return;
      }

      asrSpinner.classList.remove("hidden");
      
      try {
        let response;
        if (state.selectedAudioFile) {
          const formData = new FormData();
          formData.append("file", state.selectedAudioFile);
          response = await fetch("/api/transcribe", { method: "POST", body: formData });
        } else {
          // Pre-processed sample audio call
          const prepRes = await fetch("/api/preprocess", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ transcript: state.rawTranscript })
          });
          const data = await prepRes.json();
          response = {
            ok: true,
            json: async () => ({
              audio_ref: "sample_doctor_patient.wav",
              raw_transcript: state.rawTranscript,
              clean_transcript: data.clean_transcript,
              formatted_transcript: data.formatted_transcript,
              dialogue_turns: data.dialogue_turns,
              chunks: data.chunks
            })
          };
        }

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.detail || "Speech recognition transcription failed.");
        }

        const data = await response.json();
        state.audioRef = data.audio_ref;
        state.rawTranscript = data.raw_transcript;
        state.cleanTranscript = data.clean_transcript;
        state.formattedTranscript = data.formatted_transcript;
        state.dialogueTurns = data.dialogue_turns || [];
        state.chunks = data.chunks || [];

        renderTranscriptDisplay("clean");
        showToast("Whisper ASR transcription complete!", "success");

      } catch (err) {
        showToast("Whisper ASR Error: " + err.message, "error");
      } finally {
        asrSpinner.classList.add("hidden");
      }
    });
  }

  // Transcript Rendering (Raw vs Clean Dialogue Turns)
  function renderTranscriptDisplay(mode = "clean") {
    if (!transcriptDisplay) return;
    
    if (mode === "raw") {
      transcriptDisplay.innerHTML = `<p class="whitespace-pre-line">${escapeHtml(state.rawTranscript || "No transcript available.")}</p>`;
      rawTranscriptBtn.classList.add("active");
      cleanTranscriptBtn.classList.remove("active");
    } else {
      if (state.dialogueTurns && state.dialogueTurns.length > 0) {
        let html = "";
        state.dialogueTurns.forEach(turn => {
          const speakerClass = turn.speaker === "Doctor" ? "Doctor" : "Patient";
          const icon = turn.speaker === "Doctor" ? "fa-user-doctor" : "fa-user-nurse";
          html += `
            <div class="dialogue-turn ${speakerClass}">
              <span class="speaker-name"><i class="fa-solid ${icon}"></i> ${turn.speaker}</span>
              <p>${escapeHtml(turn.text)}</p>
            </div>
          `;
        });
        transcriptDisplay.innerHTML = html;
      } else {
        transcriptDisplay.innerHTML = `<p>${escapeHtml(state.formattedTranscript || state.cleanTranscript || state.rawTranscript)}</p>`;
      }
      cleanTranscriptBtn.classList.add("active");
      rawTranscriptBtn.classList.remove("active");
    }
  }

  if (rawTranscriptBtn) rawTranscriptBtn.addEventListener("click", () => renderTranscriptDisplay("raw"));
  if (cleanTranscriptBtn) cleanTranscriptBtn.addEventListener("click", () => renderTranscriptDisplay("clean"));

  // Proceed to Tab 2
  if (proceedBtn) {
    proceedBtn.addEventListener("click", () => {
      const reportTab = document.querySelector(".nav-tab[data-tab='report']");
      if (reportTab) reportTab.click();
    });
  }

  // Model Selection Toggle (BART vs T5)
  modelRadios.forEach(radio => {
    radio.addEventListener("change", (e) => {
      state.selectedModel = e.target.value;
      modelCards.forEach(card => card.classList.remove("active"));
      const activeLabel = document.getElementById(`label-${state.selectedModel.toLowerCase()}`);
      if (activeLabel) activeLabel.classList.add("active");
    });
  });

  // Summarization & Report Generation Call
  if (generateSummaryBtn) {
    generateSummaryBtn.addEventListener("click", async () => {
      const activeTranscript = state.formattedTranscript || state.cleanTranscript || state.rawTranscript;
      if (!activeTranscript) {
        showToast("Please transcribe or record consultation audio first.", "error");
        return;
      }

      summarySpinnerText.textContent = `Generating Clinical Summary using ${state.selectedModel}...`;
      summarySpinner.classList.remove("hidden");

      const patientId = document.getElementById("patient-id-input").value || "PAT-2026-88";
      const doctorInfo = document.getElementById("doctor-info-input").value || "Dr. Alexander Fleming, MD";

      try {
        const response = await fetch("/api/summarize", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            transcript: activeTranscript,
            model: state.selectedModel,
            patient_id: patientId,
            doctor_info: doctorInfo
          })
        });

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          throw new Error(errData.detail || "Summarization API error.");
        }

        const data = await response.json();
        state.currentReport = data.clinical_report;
        state.generatedSummary = data.generated_summary;
        state.consultationId = data.consultation_id;

        renderClinicalReport(data);

        // Update Metadata box
        document.getElementById("summary-meta-box").classList.remove("hidden");
        document.getElementById("meta-model-name").textContent = data.selected_model;
        document.getElementById("meta-chunk-method").textContent = data.summarization_method;
        document.getElementById("meta-cons-id").textContent = data.consultation_id;

        showToast("Clinical report generated successfully!", "success");

      } catch (err) {
        showToast("Summarization Error: " + err.message, "error");
      } finally {
        summarySpinner.classList.add("hidden");
      }
    });
  }

  // Reports saved before SOAP support have only flat fields; map them onto SOAP sections
  function buildSoapFromLegacy(r, summary) {
    return {
      subjective: {
        chief_complaint: r.chief_complaint || "Not specified",
        history_of_present_illness: { symptoms: r.symptoms || [], duration: r.duration || "Not specified" },
        past_medical_history: r.medical_history || [],
        current_medications: r.medication || []
      },
      objective: { findings: r.doctor_observations || [] },
      assessment: { diagnosis: r.diagnosis || ["Not recorded"], clinical_summary: r.clinical_summary || summary },
      plan: { prescribed_medications: r.medication || [], recommendations: r.followup_recommendations || [] }
    };
  }

  // Render Structured SOAP Clinical Report
  function renderClinicalReport(data) {
    if (!reportViewContainer) return;
    const r = data.clinical_report;
    const soap = r.soap || buildSoapFromLegacy(r, data.generated_summary);
    const list = items => `<ul>${(items || []).map(i => `<li>${escapeHtml(i)}</li>`).join("")}</ul>`;

    const html = `
      <div class="report-paper">
        <div class="report-header-section">
          <div class="report-title-box">
            <h3><i class="fa-solid fa-file-medical"></i> CLINICAL CONSULTATION REPORT</h3>
            <small class="text-muted">Consultation ID: <strong>${data.consultation_id}</strong> | Model: <strong>${data.selected_model}</strong></small>
          </div>
          <div class="report-date">
            <small class="text-muted">Date: ${new Date().toLocaleDateString()}</small>
          </div>
        </div>

        <!-- 1. Patient Information -->
        <div class="report-section">
          <h4><i class="fa-solid fa-user-circle"></i> 1. Patient Information</h4>
          <div class="report-grid-3">
            <div><strong>Patient ID:</strong> ${escapeHtml(r.patient_information?.patient_id || "PAT-2026-88")}</div>
            <div><strong>Name:</strong> ${escapeHtml(r.patient_information?.name || "Not specified")}</div>
            <div><strong>Age:</strong> ${escapeHtml(r.patient_information?.age || "Not specified")}</div>
            <div><strong>Gender:</strong> ${escapeHtml(r.patient_information?.gender || "Not specified")}</div>
            <div><strong>Doctor:</strong> ${escapeHtml(document.getElementById("doctor-info-input")?.value || "Dr. Alexander Fleming")}</div>
          </div>
        </div>

        <!-- S: Subjective -->
        <div class="report-section">
          <h4><i class="fa-solid fa-comment-medical"></i> S — Subjective</h4>
          <p><strong>Chief Complaint:</strong> ${escapeHtml(soap.subjective.chief_complaint)}</p>
          <p class="mt-2"><strong>Onset / Duration:</strong> ${escapeHtml(soap.subjective.history_of_present_illness.duration)}</p>
          <p class="mt-2"><strong>Symptoms:</strong></p>
          ${list(soap.subjective.history_of_present_illness.symptoms)}
          <p class="mt-2"><strong>Past Medical History:</strong> ${escapeHtml(soap.subjective.past_medical_history.join(", "))}</p>
          <p class="mt-2"><strong>Current Medications:</strong> ${escapeHtml(soap.subjective.current_medications.join(", "))}</p>
        </div>

        <!-- O: Objective -->
        <div class="report-section">
          <h4><i class="fa-solid fa-heart-pulse"></i> O — Objective</h4>
          <p><strong>Vitals & Examination Findings:</strong></p>
          ${list(soap.objective.findings)}
        </div>

        <!-- A: Assessment -->
        <div class="report-section">
          <h4><i class="fa-solid fa-brain"></i> A — Assessment</h4>
          <p><strong>Diagnosis / Impression:</strong></p>
          ${list(soap.assessment.diagnosis)}
          <p class="mt-2"><strong>Clinical Summary (${escapeHtml(data.selected_model)}):</strong></p>
          <div class="summary-highlight">
            <p>${escapeHtml(soap.assessment.clinical_summary || data.generated_summary)}</p>
          </div>
        </div>

        <!-- P: Plan -->
        <div class="report-section">
          <h4><i class="fa-solid fa-clipboard-check"></i> P — Plan</h4>
          <p><strong>Prescribed Medications:</strong> ${escapeHtml(soap.plan.prescribed_medications.join(", "))}</p>
          <p class="mt-2"><strong>Recommendations & Follow-Up:</strong></p>
          ${list(soap.plan.recommendations)}
        </div>
      </div>
    `;

    reportViewContainer.innerHTML = html;
  }

  // Save Report to SQLite Database
  if (saveDbBtn) {
    saveDbBtn.addEventListener("click", async () => {
      if (!state.currentReport) {
        showToast("Please generate a clinical report first.", "error");
        return;
      }

      try {
        const payload = {
          consultation_id: state.consultationId,
          patient_id: document.getElementById("patient-id-input").value || "PAT-2026-88",
          doctor_info: document.getElementById("doctor-info-input").value || "Dr. Alexander Fleming, MD",
          audio_ref: state.audioRef,
          transcript: state.rawTranscript,
          clean_transcript: state.cleanTranscript,
          generated_summary: state.generatedSummary,
          clinical_report: state.currentReport,
          selected_model: state.selectedModel
        };

        const res = await fetch("/api/reports", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });

        if (res.ok) {
          showToast(`Report '${state.consultationId}' saved to SQLite database!`, "success");
        } else {
          showToast("Failed to save report to database.", "error");
        }
      } catch (err) {
        showToast("Database save error: " + err.message, "error");
      }
    });
  }

  // Download Report as Formatted Text Document
  if (downloadReportBtn) {
    downloadReportBtn.addEventListener("click", () => {
      if (!state.consultationId) {
        showToast("Please generate a clinical report first.", "error");
        return;
      }
      window.open(`/api/reports/${state.consultationId}/export`, "_blank");
      showToast("Downloading clinical report document...", "info");
    });
  }

  // Print / Export PDF
  if (printReportBtn) {
    printReportBtn.addEventListener("click", () => {
      window.print();
    });
  }

  // Render Database Table Rows
  function renderDbTable(reports) {
    if (!dbTableBody) return;
    if (!reports || reports.length === 0) {
      dbTableBody.innerHTML = `<tr><td colspan="7" class="text-center py-4 text-muted">No matching consultation reports found.</td></tr>`;
      return;
    }

    let html = "";
    reports.forEach(r => {
      const reportObj = r.clinical_report || {};
      html += `
        <tr>
          <td><strong>${escapeHtml(r.consultation_id)}</strong></td>
          <td>${escapeHtml(r.patient_id)}</td>
          <td>${new Date(r.datetime).toLocaleString()}</td>
          <td>${escapeHtml(r.doctor_info)}</td>
          <td><span class="badge ${r.selected_model === 'BART' ? 'badge-accent' : 'badge-purple'}">${escapeHtml(r.selected_model)}</span></td>
          <td>${escapeHtml(reportObj.chief_complaint || 'N/A')}</td>
          <td>
            <button class="btn btn-outline btn-sm view-db-report" title="View Report" data-id="${r.consultation_id}"><i class="fa-solid fa-eye"></i></button>
            <a href="/api/reports/${r.consultation_id}/export" class="btn btn-outline btn-sm" title="Download .txt" target="_blank"><i class="fa-solid fa-download"></i></a>
            <button class="btn btn-outline btn-sm delete-db-report" title="Delete" data-id="${r.consultation_id}" style="color: var(--danger);"><i class="fa-solid fa-trash"></i></button>
          </td>
        </tr>
      `;
    });
    dbTableBody.innerHTML = html;

    // View Report Action
    dbTableBody.querySelectorAll(".view-db-report").forEach(btn => {
      btn.addEventListener("click", async (e) => {
        const id = e.currentTarget.dataset.id;
        try {
          const reportRes = await fetch(`/api/reports/${id}`);
          const reportData = await reportRes.json();
          state.currentReport = reportData.report.clinical_report;
          state.generatedSummary = reportData.report.generated_summary;
          state.consultationId = reportData.report.consultation_id;
          state.selectedModel = reportData.report.selected_model;

          renderClinicalReport({
            consultation_id: state.consultationId,
            selected_model: state.selectedModel,
            generated_summary: state.generatedSummary,
            clinical_report: state.currentReport
          });

          showToast(`Loaded report ${id}`, "info");
          document.querySelector(".nav-tab[data-tab='report']").click();
        } catch (err) {
          showToast("Failed to load report: " + err.message, "error");
        }
      });
    });

    // Delete Report Action
    dbTableBody.querySelectorAll(".delete-db-report").forEach(btn => {
      btn.addEventListener("click", async (e) => {
        const id = e.currentTarget.dataset.id;
        if (confirm(`Are you sure you want to delete report ${id}?`)) {
          try {
            const delRes = await fetch(`/api/reports/${id}`, { method: "DELETE" });
            if (delRes.ok) {
              showToast(`Report ${id} deleted successfully.`, "success");
              fetchDatabaseReports();
            } else {
              showToast("Failed to delete report.", "error");
            }
          } catch (err) {
            showToast("Delete error: " + err.message, "error");
          }
        }
      });
    });
  }

  // Fetch All Database Reports
  async function fetchDatabaseReports() {
    if (!dbTableBody) return;
    try {
      const res = await fetch("/api/reports");
      const data = await res.json();
      renderDbTable(data.reports || []);
    } catch (err) {
      showToast("Database fetch error: " + err.message, "error");
    }
  }

  if (refreshDbBtn) refreshDbBtn.addEventListener("click", fetchDatabaseReports);

  // Real-time Database Search with Debounce
  if (dbSearchInput) {
    let searchTimeout = null;
    dbSearchInput.addEventListener("input", (e) => {
      const query = e.target.value.trim();
      clearTimeout(searchTimeout);
      searchTimeout = setTimeout(async () => {
        if (!query) {
          fetchDatabaseReports();
          return;
        }
        try {
          const res = await fetch(`/api/reports/search/${encodeURIComponent(query)}`);
          if (res.ok) {
            const data = await res.json();
            renderDbTable(data.reports || []);
          } else {
            throw new Error("Search API failed");
          }
        } catch (err) {
          // Fallback to client-side filter
          const rows = dbTableBody.querySelectorAll("tr");
          rows.forEach(row => {
            const text = row.textContent.toLowerCase();
            row.style.display = text.includes(query.toLowerCase()) ? "" : "none";
          });
        }
      }, 300);
    });
  }

  // ROUGE Model Evaluation Call (BART vs T5)
  if (runEvalBtn) {
    runEvalBtn.addEventListener("click", async () => {
      runEvalBtn.disabled = true;
      runEvalBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Evaluating Models on Benchmark Cases...`;

      try {
        const res = await fetch("/api/evaluate", { method: "POST" });
        if (!res.ok) throw new Error("Evaluation endpoint failed.");

        const data = await res.json();
        const results = data.results;

        // Render Overall Winner
        evalWinnerBanner.classList.remove("hidden");
        winnerModelName.textContent = results.overall_winner;

        // Render BART Metrics
        const bart = results.models.BART;
        document.getElementById("bart-r1-p").textContent = bart.rouge1.precision;
        document.getElementById("bart-r1-r").textContent = bart.rouge1.recall;
        document.getElementById("bart-r1-f1").textContent = bart.rouge1.f1;

        document.getElementById("bart-r2-p").textContent = bart.rouge2.precision;
        document.getElementById("bart-r2-r").textContent = bart.rouge2.recall;
        document.getElementById("bart-r2-f1").textContent = bart.rouge2.f1;

        document.getElementById("bart-rL-p").textContent = bart.rougeL.precision;
        document.getElementById("bart-rL-r").textContent = bart.rougeL.recall;
        document.getElementById("bart-rL-f1").textContent = bart.rougeL.f1;

        // Render T5 Metrics
        const t5 = results.models.T5;
        document.getElementById("t5-r1-p").textContent = t5.rouge1.precision;
        document.getElementById("t5-r1-r").textContent = t5.rouge1.recall;
        document.getElementById("t5-r1-f1").textContent = t5.rouge1.f1;

        document.getElementById("t5-r2-p").textContent = t5.rouge2.precision;
        document.getElementById("t5-r2-r").textContent = t5.rouge2.recall;
        document.getElementById("t5-r2-f1").textContent = t5.rouge2.f1;

        document.getElementById("t5-rL-p").textContent = t5.rougeL.precision;
        document.getElementById("t5-rL-r").textContent = t5.rougeL.recall;
        document.getElementById("t5-rL-f1").textContent = t5.rougeL.f1;

        // Render Detailed Case Comparison
        let caseHtml = "";
        results.detailed_cases.forEach((c, idx) => {
          caseHtml += `
            <div class="card mb-3 p-3">
              <h4>Case ${idx+1}: ${escapeHtml(c.title)}</h4>
              <p class="text-sm text-muted">Reference Summary: ${escapeHtml(c.reference_summary)}</p>
              <div class="grid grid-2 mt-2">
                <div class="summary-highlight">
                  <strong>BART Summary (ROUGE-L F1: ${c.bart.metrics.rougeL.f1}):</strong>
                  <p>${escapeHtml(c.bart.summary)}</p>
                </div>
                <div class="summary-highlight" style="border-left-color: var(--purple);">
                  <strong>T5 Summary (ROUGE-L F1: ${c.t5.metrics.rougeL.f1}):</strong>
                  <p>${escapeHtml(c.t5.summary)}</p>
                </div>
              </div>
            </div>
          `;
        });
        evalCasesList.innerHTML = caseHtml;
        showToast(`Evaluation completed over ${results.num_test_cases} benchmark test cases!`, "success");

      } catch (err) {
        showToast("Evaluation Error: " + err.message, "error");
      } finally {
        runEvalBtn.disabled = false;
        runEvalBtn.innerHTML = `<i class="fa-solid fa-play"></i> Run ROUGE Evaluation Benchmark`;
      }
    });
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }
});
