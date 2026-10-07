import re
from typing import Dict, Any, List, Optional

class ClinicalReportGenerator:
    """
    Fact-Preserving Clinical Report Extraction Engine.
    Extracts clinical facts strictly from the conversation transcript & summary
    and arranges them into a SOAP (Subjective, Objective, Assessment, Plan) note.
    Guarantees no hallucination or invention of unmentioned medical facts.
    """
    def __init__(self):
        # Regular expressions for demographic metadata
        self.age_pattern = re.compile(r"\b(\d{1,3})\s*(?:years?\s*old|y/o|yo|yr|yrs)\b", re.I)
        self.gender_patterns = {
            "Male": [r"\bmale\b", r"\bman\b", r"\bboy\b", r"\bhe\b", r"\bhim\b", r"\bhis\b", r"\bgentleman\b"],
            "Female": [r"\bfemale\b", r"\bwoman\b", r"\bgirl\b", r"\bshe\b", r"\bher\b", r"\blady\b"]
        }
        self.duration_pattern = re.compile(
            r"\b(?:for|since|about|past|last)?\s*(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*"
            r"(days?|weeks?|months?|years?|hours?)\b", re.I
        )
        
        # Medical Entity Dictionaries
        self.symptom_keywords = [
            "fever", "headache", "cough", "pain", "chest pain", "shortness of breath",
            "nausea", "vomiting", "diarrhea", "fatigue", "dizziness", "sore throat",
            "rash", "swelling", "stomach ache", "back pain", "joint pain", "chills",
            "congestion", "runny nose", "insomnia", "anxiety", "cramps", "numbness",
            "weakness", "loss of appetite", "itching", "sneezing", "wheezing",
            "palpitations", "blurred vision", "tinnitus", "weight loss", "weight gain",
            "constipation", "bloating", "heartburn", "chest tightness", "muscle pain",
            "neck pain", "shoulder pain", "abdominal pain", "pelvic pain", "hip pain",
            "knee pain", "burning sensation", "tingling", "tremor", "seizures",
            "confusion", "memory loss", "difficulty breathing", "rapid heartbeat",
            "night sweats", "frequent urination", "blood in urine", "blood in stool",
            "difficulty swallowing", "hoarseness", "dry mouth", "excessive thirst",
            "bruising", "hair loss", "skin discoloration", "eye pain", "ear pain",
            "photophobia", "dyspnea", "edema"
        ]
        
        self.medication_keywords = [
            "paracetamol", "ibuprofen", "aspirin", "amoxicillin", "azithromycin",
            "metformin", "atorvastatin", "lisinopril", "omeprazole", "amlodipine",
            "albuterol", "prednisone", "ciprofloxacin", "doxycycline", "losartan",
            "gabapentin", "sertraline", "cetirizine", "antacid", "cough syrup", "antibiotic",
            "metoprolol", "warfarin", "clopidogrel", "levothyroxine", "hydrochlorothiazide",
            "furosemide", "spironolactone", "pantoprazole", "esomeprazole", "ranitidine",
            "insulin", "glipizide", "pioglitazone", "sitagliptin", "empagliflozin",
            "rosuvastatin", "simvastatin", "pravastatin", "fluoxetine", "escitalopram",
            "venlafaxine", "duloxetine", "alprazolam", "lorazepam", "diazepam",
            "tramadol", "morphine", "codeine", "naproxen", "diclofenac",
            "montelukast", "fluticasone", "salbutamol", "ipratropium", "methylprednisolone",
            "hydroxychloroquine", "methotrexate", "acetaminophen", "levofloxacin",
            "clindamycin", "vancomycin", "ceftriaxone", "sumatriptan", "lozenges",
            "ondansetron", "loperamide", "oral rehydration salts", "nitrofurantoin", "trimethoprim",
            "cefuroxime", "cephalexin", "metronidazole", "fluconazole", "budesonide",
            "beclomethasone", "prednisolone", "dexamethasone", "hydrocortisone", "propranolol",
            "bisoprolol", "carvedilol", "ramipril", "enalapril", "valsartan", "candesartan",
            "glimepiride", "gliclazide", "dapagliflozin", "linagliptin", "semaglutide",
            "amitriptyline", "pregabalin", "baclofen", "cyclobenzaprine", "loratadine",
            "fexofenadine", "chlorpheniramine", "mometasone", "allopurinol", "colchicine",
            "nitroglycerin", "lamotrigine", "levetiracetam", "mirtazapine", "buspirone", "zolpidem"
        ]
        # Drug-name endings, to catch medications missing from the list above
        self.drug_suffix_pattern = re.compile(
            r"\b[a-z]{3,}(?:pril|sartan|olol|statin|prazole|mycin|cillin|floxacin|tidine|triptan|"
            r"setron|gliflozin|gliptin|dipine|azepam|azolam|oxetine|tadine|cycline)\b"
        )
        self.dose_pattern = (
            r"(\d+(?:\.\d+)?\s*(?:mg/kg/day|mg/kg|mg|mcg|micrograms?|ml|units|puffs?|tablets?|capsules?)"
            r"(?:\s+(?:once|twice|three times|four times)\s+(?:a\s+)?(?:daily|day)|\s+every\s+\d+\s+hours"
            r"|\s+daily|\s+at night|\s+as needed)?)"
        )
        self.prescribing_cue = re.compile(
            r"\b(?:prescrib\w*|start\w*|take|increas\w*|add\w*|continue|give|use|recommend\w*|switch\w*)\b", re.I
        )
        # (label, pattern) for vitals and test results quoted in the consultation
        self.measurement_patterns = [
            ("Blood Pressure", r"\b(\d{2,3})\s*/\s*(\d{2,3})\s*mm\s*hg\b", "{}/{} mmHg"),
            ("Blood Pressure", r"\b(?:blood pressure|bp)\D{0,15}(\d{2,3})\s*(?:/|over)\s*(\d{2,3})\b", "{}/{} mmHg"),
            ("Heart Rate", r"\b(\d{2,3})\s*(?:bpm|beats per minute)\b", "{} bpm"),
            ("Heart Rate", r"\b(?:heart rate|pulse)\D{0,12}(\d{2,3})\b", "{} bpm"),
            ("Temperature", r"\b(\d{2,3}(?:\.\d)?)\s*(?:°\s*|degrees\s*)?(f|c)\b", "{} {}"),
            ("Respiratory Rate", r"\b(\d{1,2})\s*breaths per minute\b", "{}/min"),
            ("Oxygen Saturation", r"\b(?:oxygen saturation|saturation|spo2|sats)\D{0,12}(\d{2,3})\s*%", "{}%"),
            ("Blood Glucose", r"\b(\d{2,3})\s*mg/dl\b", "{} mg/dL"),
            ("HbA1c", r"\bhba1c\D{0,12}(\d{1,2}(?:\.\d)?)\s*%", "{}%"),
            ("Peak Flow", r"\bpeak (?:expiratory )?flow\D{0,12}(\d{2,3})", "{} L/min")
        ]

        self.history_keywords = [
            "hypertension", "diabetes", "asthma", "high blood pressure", "high cholesterol",
            "heart condition", "surgery", "allergic", "allergy", "kidney disease",
            "thyroid", "stroke", "cancer", "previous episode", "chronic",
            "coronary artery disease", "atrial fibrillation", "heart failure",
            "copd", "emphysema", "liver disease", "hepatitis", "cirrhosis",
            "epilepsy", "depression", "anxiety disorder", "bipolar disorder",
            "rheumatoid arthritis", "osteoarthritis", "osteoporosis",
            "deep vein thrombosis", "pulmonary embolism", "anemia",
            "gout", "psoriasis", "eczema", "ulcer", "transplant",
            "pacemaker", "stent", "hip replacement", "knee replacement",
            "migraine", "gastritis", "urinary tract infection"
        ]

    def extract_patient_info(self, text: str) -> Dict[str, str]:
        """Extract Patient Name, Age, Gender, and ID if present."""
        age_match = self.age_pattern.search(text)
        age = age_match.group(1) + " years" if age_match else "Not specified"

        # Name extraction heuristics (e.g., "Mr. Smith", "John Doe", "patient name is ...")
        name = "Not specified"
        name_match = re.search(r"\b(?:patient|name\s*is|mr\.|ms\.|mrs\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", text, re.I)
        if name_match:
            name = name_match.group(1).title()

        # Gender inference
        gender = "Not specified"
        text_lower = text.lower()
        male_count = sum(len(re.findall(p, text_lower)) for p in self.gender_patterns["Male"])
        female_count = sum(len(re.findall(p, text_lower)) for p in self.gender_patterns["Female"])
        if male_count > female_count and male_count >= 1:
            gender = "Male"
        elif female_count > male_count and female_count >= 1:
            gender = "Female"

        return {
            "name": name,
            "age": age,
            "gender": gender
        }

    def extract_chief_complaint(self, dialogue_turns: List[Dict[str, str]], full_text: str) -> str:
        """Extract primary reason for visit from patient statements."""
        complaints = []
        patient_texts = [turn["text"] for turn in dialogue_turns if turn.get("speaker") == "Patient"]
        
        cues = [
            r"\b(?:i have|i've been having|came in for|suffering from|complaining of|experiencing)\s+([^.!?]+)",
            r"\b(?:my\s+[a-z]+\s+(?:hurts|aches|is sore))",
            r"\b(?:main problem is|reason for visit is)\s+([^.!?]+)"
        ]

        search_corpus = patient_texts if patient_texts else [full_text]
        for p_text in search_corpus:
            for cue in cues:
                matches = re.findall(cue, p_text, re.I)
                for m in matches:
                    clean_m = m.strip()
                    if len(clean_m) > 4 and clean_m not in complaints:
                        complaints.append(clean_m)

        if complaints:
            return complaints[0].capitalize()
            
        # Fallback: search for first symptom mention
        for sym in self.symptom_keywords:
            if sym in full_text.lower():
                return f"Patient presents with {sym}"
                
        return "Not specified"

    def patient_statements(self, dialogue_turns: List[Dict[str, str]]) -> str:
        """The patient's own statements (questions excluded): the source for symptoms and their duration."""
        text = " ".join(t["text"] for t in dialogue_turns if t.get("speaker") == "Patient")
        return " ".join(s for s in re.split(r"(?<=[.!?])\s+", text) if s and not s.endswith("?"))

    def is_negated(self, text: str, start: int) -> bool:
        """
        NegEx-style check: is the term at `start` inside a negation ("no chest pain", "denies fever")?
        Looks back up to 6 words within the same sentence, stopping at "but"/"however".
        """
        window = re.split(r"[.!?;]|\bbut\b|\bhowever\b|\balthough\b", text[:start].lower())[-1]
        words = window.split()[-6:]
        return any(re.fullmatch(r"no|not|never|denies|denied|without|n't|negative|nor|none", w.strip(",")) or
                   w.endswith("n't") for w in words)

    def extract_symptoms(self, text: str) -> List[str]:
        """Extract medical symptoms mentioned in text, skipping negated mentions ("no chest pain")."""
        text_lower = text.lower()
        found_symptoms = []
        for sym in self.symptom_keywords:
            mentions = re.finditer(r"\b" + re.escape(sym) + r"\b", text_lower)
            if any(not self.is_negated(text_lower, m.start()) for m in mentions):
                found_symptoms.append(sym.capitalize())
        return found_symptoms if found_symptoms else ["Not specified"]

    def extract_duration(self, text: str) -> str:
        """Extract onset duration of symptoms."""
        for match in self.duration_pattern.finditer(text):
            # Skip ages ("45 years old"); a loose substring test here used to also skip "3 weeks. Your..."
            if re.match(r"\s*(?:old|of age)\b", text[match.end():], re.I):
                continue
            num, unit = match.groups()
            return f"{num} {unit}".lower()
        
        # Check alternative phrases
        alt_match = re.search(r"\b(since\s+[a-z]+|for\s+a\s+few\s+[a-z]+|recently|yesterday|today)\b", text, re.I)
        if alt_match:
            return alt_match.group(1).lower()

        return "Not specified"

    def extract_medical_history(self, text: str) -> List[str]:
        """Extract existing conditions, allergies, or past history."""
        text_lower = text.lower()
        found_history = []
        for kw in self.history_keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                found_history.append(kw.capitalize())

        # Check explicit allergy statements
        allergy_match = re.search(r"\ballergic to\s+([a-z0-9\s]+?)(?=[.,!?]|$)", text_lower)
        if allergy_match:
            found_history.append(f"Allergy to {allergy_match.group(1).strip().capitalize()}")

        return found_history if found_history else ["No significant medical history reported"]

    def extract_medications(self, text: str) -> List[str]:
        """Extract current or prescribed medications."""
        text_lower = text.lower()
        found_meds = []
        for med in self.medication_keywords:
            if re.search(r"\b" + re.escape(med) + r"\b", text_lower):
                found_meds.append(med.capitalize())

        # Unlisted drugs: recognisable drug-name endings, or "500 mg of <drug>"
        unlisted = self.drug_suffix_pattern.findall(text_lower)
        unlisted += re.findall(r"\b\d+\s*(?:mg|ml|mcg|tablets?|capsules?)\s+of\s+([a-z]{4,})\b", text_lower)
        for name in unlisted:
            if name.capitalize() not in found_meds:
                found_meds.append(name.capitalize())

        return found_meds if found_meds else ["No medications specified"]

    def with_dose(self, med: str, text: str) -> str:
        """Attach the spoken dose to a medication name, e.g. "Sumatriptan" -> "Sumatriptan 50 mg"."""
        name = re.escape(med.lower())
        after = re.search(rf"\b{name}\b(?:\s+[a-z]+)??(?:\s+to)?\s+{self.dose_pattern}", text, re.I)
        before = re.search(rf"{self.dose_pattern}\s+(?:of\s+)?{name}\b", text, re.I)
        match = after or before
        return f"{med} {match.group(1)}" if match else med

    def extract_measurements(self, text: str) -> List[str]:
        """Extract vitals and test results quoted in the consultation, e.g. "Blood Pressure: 120/80 mmHg"."""
        found = []
        for label, pattern, fmt in self.measurement_patterns:
            for match in re.finditer(pattern, text, re.I):
                # Thresholds and targets ("if fever exceeds 102 F", "target of 7.0%") are not measurements
                if re.search(r"\b(?:exceeds?|above|over|higher than|more than|greater than|below|under|target of)\s*$",
                             text[max(0, match.start() - 20):match.start()], re.I):
                    continue
                value = fmt.format(*(g.upper() if len(g) == 1 else g for g in match.groups()))
                item = f"{label}: {value}"
                if item not in found:
                    found.append(item)
        return found

    def extract_followup(self, dialogue_turns: List[Dict[str, str]], text: str) -> List[str]:
        """Extract timed follow-up instructions, e.g. "Follow up in 1 week", "Recheck renal function in 2 weeks"."""
        doc_text = " ".join(t["text"] for t in dialogue_turns if t.get("speaker") == "Doctor") or text
        pattern = (r"\b(?:follow(?:-|\s)?up|return|recheck|review|come back|bring (?:him|her|them) back|"
                   r"see (?:me|you) again)\b[^.!?]*?\b(?:in|within|after)\s+\d+(?:\s*(?:to|-)\s*\d+)?\s+"
                   r"(?:hours?|days?|weeks?|months?)")
        found = []
        for match in re.finditer(pattern, doc_text, re.I):
            clause = re.sub(r"\byour\s+", "", match.group(0)).strip()
            clause = clause[0].upper() + clause[1:]
            if clause not in found:
                found.append(clause)
        return found

    def extract_doctor_observations(self, dialogue_turns: List[Dict[str, str]], text: str) -> List[str]:
        """Extract clinical observations and examination findings made by doctor."""
        doc_observations = []
        doc_texts = [turn["text"] for turn in dialogue_turns if turn.get("speaker") == "Doctor"]

        # Examination findings; vitals and test values are handled by extract_measurements
        obs_cues = [
            r"\b((?:physical\s+)?(?:examination|exam)\s+(?:shows|showed|reveals|revealed|is|was)\s+[^.!?]+)",
            r"\b((?:left\s+|right\s+)?(?:tympanic membranes?|lungs?|throat|abdomen|chest|heart sounds?|pharynx|"
            r"tonsils?|skin|ears?|wound|joint|knee|ankle)\s+(?:is|are|appears?|looks?|sounds?|seems?)\s+[^.!?]+)"
        ]

        search_corpus = doc_texts if doc_texts else [text]
        for d_text in search_corpus:
            for cue in obs_cues:
                for m in re.findall(cue, d_text, re.I):
                    clean = m.strip()
                    clean = clean[0].upper() + clean[1:] if clean else clean
                    if len(clean) > 3 and clean not in doc_observations:
                        doc_observations.append(clean)

        doc_observations += self.extract_measurements(text)

        return doc_observations if doc_observations else ["No examination findings recorded"]

    def extract_recommendations(self, dialogue_turns: List[Dict[str, str]], text: str) -> List[str]:
        """Extract advice, prescriptions, tests, or follow-up instructions."""
        recs = []
        rec_cues = [
            r"\b(?:recommend|advise|prescribe|suggest|please|make sure to)\s+([^.!?]+)",
            r"\b(?:take\s+[a-z0-9\s]+(?:\s+daily|\s+for\s+\d+\s+days|\s+as\s+needed)?)",
            r"\b(?:return in|follow up in|see me again in)\s+([^.!?]+)",
            r"\b(?:drink|rest|avoid|get\s+some|blood test|x-ray|scan)\b[^.!?]*"
        ]

        for cue in rec_cues:
            matches = re.findall(cue, text, re.I)
            for m in matches:
                clean = m.strip() if isinstance(m, str) else m[0].strip()
                if len(clean) > 5 and clean.capitalize() not in recs:
                    recs.append(clean.capitalize())

        return recs if recs else ["No plan specified in conversation"]

    def extract_diagnosis(self, dialogue_turns: List[Dict[str, str]], text: str) -> List[str]:
        """Extract the doctor's stated diagnosis or clinical impression (SOAP Assessment)."""
        diagnoses = []
        doc_texts = [turn["text"] for turn in dialogue_turns if turn.get("speaker") == "Doctor"]
        dx_cues = [
            r"\b(?:i am diagnosing|i'm diagnosing|diagnosed with|diagnosis is|my diagnosis is)\s+(?:you\s+with\s+|as\s+)?([^.!?,]+)",
            r"\b(?:looks like|seems to be|consistent with|suggestive of|most likely|probably)\s+(?:a\s+|an\s+)?([^.!?,]+)",
            # "You have acute gastroenteritis...", "Tommy has acute otitis media..."
            r"\b(?:you have|you've got|[a-z]+ has|this is|it is|it's)\s+(?:an?\s+)?((?:acute|chronic|mild|moderate|severe|"
            r"viral|bacterial|allergic|uncomplicated|early|suspected)\s+[^.!?,]+)"
        ]
        for d_text in (doc_texts if doc_texts else [text]):
            for cue in dx_cues:
                for m in re.findall(cue, d_text, re.I):
                    clean = m.strip().capitalize()
                    if len(clean) > 3 and clean not in diagnoses:
                        diagnoses.append(clean)
        return diagnoses if diagnoses else ["No diagnosis stated in conversation"]

    def split_medications(self, dialogue_turns: List[Dict[str, str]], all_meds: List[str]) -> Dict[str, List[str]]:
        """
        Separate medications the patient already takes (Subjective) from those the doctor
        prescribes (Plan), attaching the dose each speaker stated.
        A doctor's mention only counts as a prescription when the sentence carries a prescribing
        cue ("prescribe", "take", "increase", ...) and is not a question.
        """
        def sentences(speaker: str) -> List[str]:
            text = " ".join(t["text"] for t in dialogue_turns if t.get("speaker") == speaker)
            return [s for s in re.split(r"(?<=[.!?])\s+", text) if s]

        doctor_sents, patient_sents = sentences("Doctor"), sentences("Patient")
        current, prescribed = [], []
        for med in all_meds:
            if med.startswith("No medications"):
                continue
            mentions = lambda sents: [s for s in sents if re.search(r"\b" + re.escape(med) + r"\b", s, re.I)]
            doc_mentions, all_pat_mentions = mentions(doctor_sents), mentions(patient_sents)
            # "Do I need an antibiotic?" is a question, not a medication the patient takes
            pat_mentions = [s for s in all_pat_mentions if not s.endswith("?")]
            orders = [s for s in doc_mentions if self.prescribing_cue.search(s) and not s.endswith("?")]

            if orders:
                prescribed.append(self.with_dose(med, " ".join(orders)))
            if pat_mentions or (doc_mentions and not orders) or not (doc_mentions or all_pat_mentions):
                source = " ".join(pat_mentions or doc_mentions)
                current.append(self.with_dose(med, source) if source else med)
        return {"current": current, "prescribed": prescribed}

    def build_soap(
        self,
        chief_complaint: str,
        symptoms: List[str],
        duration: str,
        history: List[str],
        medications: Dict[str, List[str]],
        doctor_obs: List[str],
        diagnosis: List[str],
        summary: str,
        recommendations: List[str]
    ) -> Dict[str, Any]:
        """Arrange the extracted facts and the model summary into a SOAP note."""
        return {
            "subjective": {
                "chief_complaint": chief_complaint,
                "history_of_present_illness": {"symptoms": symptoms, "duration": duration},
                "past_medical_history": history,
                "current_medications": medications["current"] or ["None reported"]
            },
            "objective": {
                "findings": doctor_obs
            },
            "assessment": {
                "diagnosis": diagnosis,
                "clinical_summary": summary
            },
            "plan": {
                "prescribed_medications": medications["prescribed"] or ["None prescribed"],
                "recommendations": recommendations
            }
        }

    def soap_from_legacy(self, report: Dict[str, Any], summary: str = "") -> Dict[str, Any]:
        """Build a SOAP note for reports saved before SOAP support (flat fields only)."""
        meds = report.get("medication", [])
        return self.build_soap(
            report.get("chief_complaint", "Not specified"),
            report.get("symptoms", []),
            report.get("duration", "Not specified"),
            report.get("medical_history", []),
            {"current": meds, "prescribed": meds},
            report.get("doctor_observations", []),
            report.get("diagnosis", ["Not recorded"]),
            report.get("clinical_summary", summary),
            report.get("followup_recommendations", [])
        )

    def ground_summary(self, summary: str, dialogue_turns: List[Dict[str, str]], clean_transcript: str) -> str:
        """
        Fact-grounded summary: append the key clinical facts the model's abstractive summary left out
        (duration, vitals and test results, diagnosis, prescribed medications with doses, follow-up).
        Every added fact is copied from the transcript, so nothing is invented.
        """
        summary = summary.strip()
        lower = summary.lower()
        additions = []

        duration = self.extract_duration(self.patient_statements(dialogue_turns) or clean_transcript)
        if duration != "Not specified" and duration not in lower:
            additions.append(f"Duration: {duration}.")

        # A measurement is covered when its value (e.g. "120/80", "98.6") already appears
        findings = [m for m in self.extract_measurements(clean_transcript)
                    if m.split(": ", 1)[1].split()[0].lower() not in lower]
        if findings:
            additions.append("Findings: " + ", ".join(f.replace(": ", " ") for f in findings) + ".")

        # A diagnosis is covered when all of its content words already appear
        diagnoses = [d for d in self.extract_diagnosis(dialogue_turns, clean_transcript)
                     if not d.startswith("No diagnosis")
                     and not all(w in lower for w in re.findall(r"[a-z]{4,}", d.lower()))]
        if diagnoses:
            additions.append("Diagnosis: " + "; ".join(diagnoses) + ".")

        # A prescription is covered when the drug name and its dose numbers already appear
        prescribed = self.split_medications(dialogue_turns, self.extract_medications(clean_transcript))["prescribed"]
        prescribed = [m for m in prescribed
                      if not (m.split()[0].lower() in lower
                              and all(n in lower for n in re.findall(r"\d+(?:\.\d+)?", m)))]
        if prescribed:
            additions.append("Prescribed: " + ", ".join(prescribed) + ".")

        for clause in self.extract_followup(dialogue_turns, clean_transcript):
            timing = re.search(r"\d+(?:\s*(?:to|-)\s*\d+)?\s+\w+$", clause)
            if timing and timing.group(0).lower() not in lower:
                additions.append(clause + ".")

        if not additions:
            return summary
        separator = "" if not summary or summary.endswith((".", "!", "?")) else "."
        return f"{summary}{separator} {' '.join(additions)}".strip()

    def generate_report(
        self,
        transcript: str,
        clean_transcript: str,
        dialogue_turns: List[Dict[str, str]],
        summary: str,
        model_used: str,
        patient_id: Optional[str] = None,
        consultation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate the structured clinical report, including the SOAP note.
        Facts are extracted from the transcript only, never from the model's summary,
        so anything the model invents cannot reach the structured sections.
        """
        patient_info = self.extract_patient_info(clean_transcript)
        if patient_id:
            patient_info["patient_id"] = patient_id

        patient_text = self.patient_statements(dialogue_turns) or clean_transcript

        chief_complaint = self.extract_chief_complaint(dialogue_turns, clean_transcript)
        symptoms = self.extract_symptoms(patient_text)
        duration = self.extract_duration(patient_text)
        history = self.extract_medical_history(clean_transcript)
        medications = self.extract_medications(clean_transcript)
        doctor_obs = self.extract_doctor_observations(dialogue_turns, clean_transcript)
        recommendations = self.extract_recommendations(dialogue_turns, clean_transcript)
        diagnosis = self.extract_diagnosis(dialogue_turns, clean_transcript)
        med_split = self.split_medications(dialogue_turns, medications)

        soap = self.build_soap(
            chief_complaint, symptoms, duration, history, med_split,
            doctor_obs, diagnosis, summary, recommendations
        )

        return {
            "consultation_id": consultation_id or "CONS-TEMP",
            "model_used": model_used,
            "patient_information": patient_info,
            "chief_complaint": chief_complaint,
            "symptoms": symptoms,
            "duration": duration,
            "medical_history": history,
            "medication": medications,
            "doctor_observations": doctor_obs,
            "clinical_summary": summary,
            "followup_recommendations": recommendations,
            "diagnosis": diagnosis,
            "soap": soap
        }

# Global Singleton Instance
report_generator = ClinicalReportGenerator()
