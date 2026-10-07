"""
Measure Whisper speech-recognition accuracy on the spoken benchmark conversations.

Needs data/test_audio/*.wav (create it with scripts/generate_test_audio.py). For each Whisper
model size, with and without the clinical context prompt, it reports:
  - WER: word error rate against the known transcript (lower is better), after Whisper's
    standard English text normalisation
  - Medication names: share of the drugs mentioned that appear correctly in the transcript
  - seconds of processing per recording

Usage:
    python scripts/evaluate_asr.py --models tiny base small

Writes data/asr_results.json.
"""
import os
import re
import sys
import json
import time
import argparse
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from whisper.normalizers import EnglishTextNormalizer
from app.config import BENCHMARK_PATH, DATA_DIR
from app.core.asr_whisper import WhisperASR
from app.core.report_generator import report_generator

AUDIO_DIR = DATA_DIR / "test_audio"
RESULTS_PATH = DATA_DIR / "asr_results.json"
CLINICAL_PROMPT = ("Doctor-patient medical consultation about symptoms, vital signs, diagnosis "
                   "and medications with doses in mg.")
normalize = EnglishTextNormalizer()


def word_error_rate(reference, hypothesis):
    ref, hyp = normalize(reference).split(), normalize(hypothesis).split()
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, 1):
        current = [i] + [0] * len(hyp)
        for j, hyp_word in enumerate(hyp, 1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref_word != hyp_word))
        previous = current
    return previous[-1], len(ref)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["tiny", "base", "small"])
    args = parser.parse_args()

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        cases = [c for c in json.load(f) if (AUDIO_DIR / f"{c['id']}.wav").exists()]
    if not cases:
        sys.exit("No audio found - run scripts/generate_test_audio.py first.")

    rows = []
    for model_name in args.models:
        asr = WhisperASR(model_name=model_name)
        asr._load_model()
        for prompt_label, prompt in (("no prompt", ""), ("clinical prompt", CLINICAL_PROMPT)):
            asr.prompt = prompt
            errors = words = meds_found = meds_total = 0
            seconds = 0.0
            transcripts = {}
            for case in cases:
                reference = " ".join(re.sub(r"^\s*(Doctor|Patient)\s*:\s*", "", line)
                                     for line in case["transcript"].split("\n"))
                start = time.perf_counter()
                hypothesis = asr.transcribe(str(AUDIO_DIR / f"{case['id']}.wav"))["transcript"]
                seconds += time.perf_counter() - start

                e, n = word_error_rate(reference, hypothesis)
                errors, words = errors + e, words + n
                meds = [m.lower() for m in report_generator.extract_medications(reference)
                        if not m.startswith("No medications")]
                meds_total += len(meds)
                meds_found += sum(1 for m in meds if m in hypothesis.lower())
                transcripts[case["id"]] = hypothesis

            row = {"model": model_name, "prompt": prompt_label, "wer": round(errors / words, 4),
                   "medication_recognition": round(meds_found / meds_total, 4) if meds_total else None,
                   "seconds_per_recording": round(seconds / len(cases), 2), "transcripts": transcripts}
            rows.append(row)
            print(f"  whisper-{model_name:<7} {prompt_label:<16} WER {row['wer']:.1%}   "
                  f"medication names {meds_found}/{meds_total}   {row['seconds_per_recording']} s/recording",
                  flush=True)
        del asr

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\nFull results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
