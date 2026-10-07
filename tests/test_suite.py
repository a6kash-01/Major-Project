import unittest
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

# Ensure root directory is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.core.preprocessor import preprocessor
from app.core.report_generator import report_generator
from app.core.evaluator import rouge_evaluator
from app import database

class TestClinicalReportSystem(unittest.TestCase):

    def test_01_preprocessor_filler_removal(self):
        raw_text = "Doctor: Good morning. Um, how can I help you today? Uh, what seems to be the problem?"
        cleaned = preprocessor.remove_filler_words(raw_text)
        self.assertNotIn("Um,", cleaned)
        self.assertNotIn("Uh,", cleaned)
        self.assertIn("how can I help you today?", cleaned)

    def test_02_preprocessor_dialogue_identification(self):
        transcript = (
            "Doctor: What brings you in today?\n"
            "Patient: I have been experiencing severe headaches for 3 days.\n"
            "Doctor: Are you having any nausea?"
        )
        turns = preprocessor.identify_speaker_dialogue(transcript)
        self.assertEqual(len(turns), 3)
        self.assertEqual(turns[0]["speaker"], "Doctor")
        self.assertEqual(turns[1]["speaker"], "Patient")

    def test_03_preprocessor_chunking(self):
        long_text = "Patient complains of severe chest pain. " * 50
        chunks = preprocessor.chunk_transcript(long_text, max_chunk_words=100, overlap_words=20)
        self.assertGreater(len(chunks), 1)

    def test_04_report_generator_extraction(self):
        transcript = (
            "Doctor: Good morning Mr. John Doe. Patient is a 45 years old male.\n"
            "Patient: Doctor, I have been having severe throbbing headaches for 4 days.\n"
            "Doctor: Vitals show Blood pressure 130/85 mmHg. I prescribe Paracetamol 500 mg daily."
        )
        proc = preprocessor.process(transcript)
        report = report_generator.generate_report(
            transcript=transcript,
            clean_transcript=proc["clean_transcript"],
            dialogue_turns=proc["dialogue_turns"],
            summary="Patient John Doe presented with 4-day history of throbbing headaches.",
            model_used="BART",
            patient_id="PAT-999",
            consultation_id="CONS-TEST-01"
        )

        self.assertEqual(report["consultation_id"], "CONS-TEST-01")
        self.assertEqual(report["patient_information"]["name"], "John Doe")
        self.assertEqual(report["patient_information"]["age"], "45 years")
        self.assertEqual(report["patient_information"]["gender"], "Male")
        self.assertEqual(report["duration"], "4 days")
        self.assertIn("Paracetamol", report["medication"][0])
        self.assertIn("headache", report["chief_complaint"].lower())

    def test_05_rouge_evaluator_metrics(self):
        reference = "Patient presented with acute migraine headache for 3 days."
        candidate = "Patient has acute migraine headache lasting 3 days."
        scores = rouge_evaluator.calculate_metrics(reference, candidate)

        self.assertIn("rouge1", scores)
        self.assertIn("rouge2", scores)
        self.assertIn("rougeL", scores)
        self.assertGreater(scores["rouge1"]["f1"], 0.5)

    def test_06_database_crud_operations(self):
        cons_id = "CONS-UNITTEST-100"
        saved = database.save_report(
            consultation_id=cons_id,
            patient_id="P-TEST",
            doctor_info="Dr. Smith",
            datetime_str="2026-08-12 10:00:00",
            audio_ref="sample.wav",
            transcript="Test transcript",
            clean_transcript="Clean test transcript",
            generated_summary="Test summary",
            clinical_report={"chief_complaint": "Headache"},
            selected_model="BART"
        )
        self.assertEqual(saved["consultation_id"], cons_id)

        retrieved = database.get_report_by_id(cons_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["patient_id"], "P-TEST")

        # Test search
        search_results = database.search_reports("UNITTEST")
        self.assertGreaterEqual(len(search_results), 1)
        self.assertEqual(search_results[0]["consultation_id"], cons_id)

        # Cleanup
        deleted = database.delete_report(cons_id)
        self.assertTrue(deleted)

    def test_07_database_search_not_found(self):
        results = database.search_reports("NONEXISTENT_QUERY_XYZ_123")
        self.assertEqual(len(results), 0)

    def test_08_expanded_medical_entity_extraction(self):
        text = "Patient experiences severe palpitations and dyspnea. Prescribed Escitalopram 10 mg and Metoprolol 25 mg."
        symptoms = report_generator.extract_symptoms(text)
        meds = report_generator.extract_medications(text)
        
        self.assertIn("Palpitations", symptoms)
        self.assertIn("Dyspnea", symptoms)
        self.assertIn("Escitalopram", meds)
        self.assertIn("Metoprolol", meds)

    def test_09_soap_report_structure(self):
        transcript = (
            "Doctor: Good morning. What brings you in today?\n"
            "Patient: I have been having severe headaches for 3 days. I tried taking ibuprofen.\n"
            "Doctor: Your blood pressure is 120/80 mmHg. I am diagnosing acute migraine. "
            "I am prescribing Sumatriptan 50 mg. Follow up in 1 week."
        )
        proc = preprocessor.process(transcript)
        report = report_generator.generate_report(
            transcript=transcript,
            clean_transcript=proc["clean_transcript"],
            dialogue_turns=proc["dialogue_turns"],
            summary="Patient presented with 3-day headache, diagnosed with acute migraine.",
            model_used="BART"
        )
        soap = report["soap"]

        self.assertEqual(set(soap), {"subjective", "objective", "assessment", "plan"})
        self.assertIn("headache", soap["subjective"]["chief_complaint"].lower())
        self.assertIn("Ibuprofen", soap["subjective"]["current_medications"])
        self.assertIn("Blood Pressure: 120/80 mmHg", soap["objective"]["findings"])
        self.assertIn("Acute migraine", soap["assessment"]["diagnosis"])
        self.assertIn("Sumatriptan 50 mg", soap["plan"]["prescribed_medications"])
        self.assertNotIn("Ibuprofen", soap["plan"]["prescribed_medications"])

    def test_10_no_invented_findings(self):
        """With no exam or diagnosis in the conversation, the report must say so rather than invent one."""
        turns = [{"speaker": "Patient", "text": "I have a cough."}]
        self.assertEqual(report_generator.extract_doctor_observations(turns, "I have a cough."),
                         ["No examination findings recorded"])
        self.assertEqual(report_generator.extract_diagnosis(turns, "I have a cough."),
                         ["No diagnosis stated in conversation"])

    def test_11_full_pipeline_keeps_speaker_turns(self):
        """Cleaning must not merge a tagged conversation into a single turn."""
        transcript = (
            "Doctor: Good morning. Um, how can I help you today?\n"
            "Patient: Uh, I have been having headaches for 3 days.\n"
            "Doctor: Do you know if you are allergic to anything?"
        )
        proc = preprocessor.process(transcript)
        speakers = [turn["speaker"] for turn in proc["dialogue_turns"]]
        self.assertEqual(speakers, ["Doctor", "Patient", "Doctor"])
        self.assertEqual(proc["dialogue_turns"][1]["text"], "I have been having headaches for 3 days.")
        self.assertIn("Do you know if", proc["clean_transcript"])
        self.assertNotIn(".,", proc["clean_transcript"])

    def test_12_normalization_preserves_clinical_numbers(self):
        cleaned = preprocessor.normalize_text("Temperature is 98.6 degrees F. Take 0.5 mg daily, 1,000 units.")
        self.assertIn("98.6", cleaned)
        self.assertIn("0.5 mg", cleaned)
        self.assertIn("1,000", cleaned)

    def test_13_untagged_asr_text_is_split_by_sentence(self):
        """Whisper output has no speaker tags and one segment can hold both speakers."""
        asr_text = ("Um, what brings you in today? Uh, I have been having chest pain for a week. "
                    "Did the pain move anywhere else? No that was it.")
        proc = preprocessor.process(asr_text)
        self.assertEqual(
            proc["formatted_transcript"],
            "Doctor: What brings you in today?\n"
            "Patient: I have been having chest pain for a week.\n"
            "Doctor: Did the pain move anywhere else?\n"
            "Patient: No that was it."
        )

if __name__ == "__main__":
    unittest.main()
