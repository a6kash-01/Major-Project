import re
from typing import List, Dict, Any, Tuple
from app.config import FILLER_WORDS, MAX_CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS

# Comprehensive whitelist regex pattern for key medical terms to protect during normalization
MEDICAL_VOCAB_PATTERNS = [
    r"\b[A-Za-z]+cillin\b", r"\b[A-Za-z]+mycin\b", r"\b[A-Za-z]+olol\b", r"\b[A-Za-z]+tidine\b",
    r"\b[A-Za-z]+prazole\b", r"\b[A-Za-z]+statin\b", r"\b[A-Za-z]+zepam\b", r"\bparacetamol\b",
    r"\bibuprofen\b", r"\baspirin\b", r"\bmetformin\b", r"\bamlodipine\b", r"\blisinopril\b",
    r"\batorvastatin\b", r"\bomeprazole\b", r"\bazithromycin\b", r"\bamoxicillin\b",
    r"\bmg\b", r"\bml\b", r"\bmcg\b", r"\bbid\b", r"\btid\b", r"\bqd\b", r"\bprn\b",
    r"\bhypertension\b", r"\bdiabetes\b", r"\basthma\b", r"\bbronchitis\b", r"\bpneumonia\b",
    r"\barrhythmia\b", r"\bseizure\b", r"\bgastritis\b", r"\bmigraine\b", r"\binsomnia\b"
]

DOCTOR_KEYWORDS = [
    "doctor:", "dr.", "physician:", "dr:", "how can i help", "what brings you",
    "how long have you had", "let me examine", "i am prescribing", "take this",
    "your blood pressure", "your vitals", "recommend", "follow-up", "tests show",
    "does it hurt when", "any family history", "can you tell me", "tell me a bit",
    "tell me more", "can you describe", "on a scale", "any other", "investigations",
    "have you ever", "do you have any", "is there anything", "i see.", "i see,", "i see so",
    "you mentioned", "you've mentioned", "you did mention", "let's", "let me", "diagnos", "prescrib",
    "i recommend", "i suggest", "i'd like you to", "medical student", "confidential", "examination",
    "advise", "we will", "we'll", "we can", "we need to"
]

PATIENT_KEYWORDS = [
    "patient:", "pt:", "i have been having", "i feel", "my head", "my chest",
    "it hurts", "since yesterday", "for 3 days", "for a week", "i noticed",
    "i tried taking", "i am allergic to", "pain is", "doctor, i", "i've never",
    "i was", "i've been", "i'd say", "i don't think so", "my wife", "my husband",
    "my dad", "my mum", "my mother", "my father", "my son", "my daughter"
]

# Pronoun balance: doctors mostly talk about "you/your", patients about "I/my"
SECOND_PERSON = re.compile(r"\b(?:you|your|yours|yourself|you're|you've|you'll|you'd)\b")
FIRST_PERSON = re.compile(r"\b(?:i|me|my|mine|myself|i'm|i've|i'll|i'd)\b")
# Questions about what will happen next ("Will he need an antibiotic?") usually come from the patient
OUTCOME_QUESTION = re.compile(r"\b(?:will|should|would|need)\b.*\b(?:he|she|it|they|this|the)\b|"
                              r"\b(?:he|she|it|they|this|the)\b.*\b(?:will|should|would|need)\b")
# "Doctor" used to address someone ("Thank you Doctor", "Is it serious, doctor?") marks the patient
DOCTOR_ADDRESSED = re.compile(r"\bdoc(?:tor)?\b(?!\s*:)")

class TextPreprocessor:
    """
    Transcript Text Preprocessor for Clinical Conversations.
    Handles filler removal, text normalization, medical term protection,
    speaker dialogue identification, and sentence-aware chunking for long transcripts.
    """
    def __init__(self, filler_patterns: List[str] = FILLER_WORDS):
        self.filler_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in filler_patterns]

    def remove_filler_words(self, text: str) -> str:
        """Filter out common speech fillers while preserving sentence structure and line breaks."""
        cleaned = text
        for pattern in self.filler_patterns:
            cleaned = pattern.sub("", cleaned)
        # Clean extra spaces created by removal (newlines are kept so speaker turns stay separate)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"[ \t]+([,.!?])", r"\1", cleaned)
        # Drop commas left behind by a removed filler, e.g. "morning. , how" or "Patient: , I"
        cleaned = re.sub(r"([.!?:])[ \t]*,", r"\1", cleaned)
        cleaned = re.sub(r"(^|\n)[ \t]*,", r"\1", cleaned)
        cleaned = re.sub(r",[ \t]*,", ",", cleaned)
        return "\n".join(line.strip() for line in cleaned.split("\n")).strip()

    def normalize_text(self, text: str) -> str:
        """
        Normalize punctuation, spacing, and capitalization while protecting medical terms.
        Works line by line so speaker-tagged turns ("Doctor: ...") stay on separate lines.
        """
        lines = [self._normalize_line(line) for line in self.remove_filler_words(text).split("\n")]
        return "\n".join(line for line in lines if line)

    def _normalize_line(self, text: str) -> str:
        # Ensure a space after sentence punctuation, but leave decimals like "98.6" or "0.5 mg" intact
        text = re.sub(r"([.!?])(?=[^\s\d.!?])", r"\1 ", text)
        # Remove repeated punctuation e.g. "!!" -> "!"
        text = re.sub(r"([.!?]){2,}", r"\1", text)
        # Fix spacing around commas, but leave numbers like "1,000" intact
        text = re.sub(r"\s*,(?!\d)\s*", ", ", text)
        # Capitalize the first letter of each sentence (fillers like "Um," often started it)
        text = re.sub(r"(^|[.!?:]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
        return re.sub(r" {2,}", " ", text).strip()

    def identify_speaker_dialogue(self, transcript: str) -> List[Dict[str, str]]:
        """
        Identify Doctor vs Patient dialogue turns using explicit labels or semantic cues.
        Returns a list of structured dialogue turns: [{"speaker": "Doctor"|"Patient", "text": "..."}]
        """
        dialogue_turns = []
        
        # 1. If transcript already has explicit "Doctor:" / "Patient:" tags
        lines = [line.strip() for line in transcript.split("\n") if line.strip()]
        has_tags = any(re.match(r"^(doctor|patient|dr\.|pt\.):", line, re.I) for line in lines)
        
        if has_tags:
            current_speaker = "Doctor"
            for line in lines:
                match = re.match(r"^(doctor|patient|dr\.|pt\.):\s*(.*)", line, re.I)
                if match:
                    tag, content = match.group(1).lower(), match.group(2)
                    current_speaker = "Doctor" if "doc" in tag or "dr" in tag else "Patient"
                    dialogue_turns.append({"speaker": current_speaker, "text": content.strip()})
                else:
                    if dialogue_turns:
                        dialogue_turns[-1]["text"] += " " + line
                    else:
                        dialogue_turns.append({"speaker": current_speaker, "text": line})
            return dialogue_turns

        # 2. Untagged ASR output: attribute each sentence. Whisper segments break at pauses and
        #    often hold both speakers ("Did it move? No that was it."), so sentences are the unit.
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", transcript) if s.strip()]

        current_speaker = "Doctor"  # Consultations usually start with the Doctor greeting
        prev_question_by = None

        for sent_text in sentences:
            sent_lower = sent_text.lower()
            is_question = sent_text.endswith("?")
            second_person = len(SECOND_PERSON.findall(sent_lower.replace("thank you", "")))
            first_person = len(FIRST_PERSON.findall(sent_lower))

            # Phrase cues, plus the pronoun balance as a weaker signal
            doc_score = sum(1 for kw in DOCTOR_KEYWORDS if kw in sent_lower) + 0.5 * second_person
            pat_score = sum(1 for kw in PATIENT_KEYWORDS if kw in sent_lower) + 0.5 * first_person
            if DOCTOR_ADDRESSED.search(sent_lower):
                pat_score += 1

            # Questions about "I/we" or about what will happen next come from the patient; questions
            # about "you" and the remaining ones ("Did the pain move?", "Any fever?") from the doctor
            if is_question:
                if not second_person and (first_person or re.search(r"\bwe\b", sent_lower)
                                          or OUTCOME_QUESTION.search(sent_lower)):
                    pat_score += 1
                else:
                    doc_score += 1

            if doc_score > pat_score:
                speaker = "Doctor"
            elif pat_score > doc_score:
                speaker = "Patient"
            elif prev_question_by and not is_question:
                # A statement right after a question is most likely the other person's answer
                speaker = "Patient" if prev_question_by == "Doctor" else "Doctor"
            else:
                # Retain current speaker for continuation
                speaker = current_speaker

            current_speaker = speaker
            prev_question_by = speaker if is_question else None
            
            # Merge with previous turn if same speaker
            if dialogue_turns and dialogue_turns[-1]["speaker"] == speaker:
                dialogue_turns[-1]["text"] += " " + sent_text
            else:
                dialogue_turns.append({"speaker": speaker, "text": sent_text})
                
        return dialogue_turns

    def format_formatted_transcript(self, dialogue_turns: List[Dict[str, str]]) -> str:
        """Format dialogue turns into a clean readable Doctor-Patient conversation string."""
        return "\n".join([f"{turn['speaker']}: {turn['text']}" for turn in dialogue_turns])

    def chunk_transcript(
        self,
        text: str,
        max_chunk_words: int = 350,
        overlap_words: int = 40
    ) -> List[str]:
        """
        Splits long transcripts into sentence-aware overlapping chunks to handle
        Transformer max length limits without naive truncation.
        """
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sentences:
            return [text]

        chunks = []
        current_chunk_sentences = []
        current_word_count = 0

        for sentence in sentences:
            words = sentence.split()
            sentence_word_count = len(words)

            if current_word_count + sentence_word_count > max_chunk_words and current_chunk_sentences:
                # Package current chunk
                chunk_str = " ".join(current_chunk_sentences)
                chunks.append(chunk_str)

                # Create overlap window for context retention
                overlap_sentences = []
                overlap_count = 0
                for s in reversed(current_chunk_sentences):
                    s_words = len(s.split())
                    if overlap_count + s_words <= overlap_words:
                        overlap_sentences.insert(0, s)
                        overlap_count += s_words
                    else:
                        break

                current_chunk_sentences = overlap_sentences + [sentence]
                current_word_count = overlap_count + sentence_word_count
            else:
                current_chunk_sentences.append(sentence)
                current_word_count += sentence_word_count

        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))

        return chunks

    def process(self, transcript: str) -> Dict[str, Any]:
        """
        Full Preprocessing Pipeline.
        """
        norm_transcript = self.normalize_text(transcript)
        dialogue_turns = self.identify_speaker_dialogue(norm_transcript)
        formatted_transcript = self.format_formatted_transcript(dialogue_turns)
        chunks = self.chunk_transcript(formatted_transcript)

        return {
            "raw_transcript": transcript,
            "clean_transcript": norm_transcript,
            "dialogue_turns": dialogue_turns,
            "formatted_transcript": formatted_transcript,
            "chunks": chunks,
            "is_long_conversation": len(chunks) > 1
        }

# Global Singleton Instance
preprocessor = TextPreprocessor()
