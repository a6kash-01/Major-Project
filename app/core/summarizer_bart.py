import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict, Any
from app.config import BART_MODEL, DEVICE, logger

class BARTSummarizer:
    """
    BART Transformer Abstractive Summarizer for Clinical Conversations.
    Supports single-pass and hierarchical chunk-based summarization for long transcripts.
    Uses direct PyTorch model.generate() for maximum stability across all HuggingFace Transformers versions.
    """
    def __init__(self, model_name: str = BART_MODEL, device: str = DEVICE):
        self.model_name = model_name
        self.device_str = device
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.tokenizer = None
        self.model = None

    def _load_model(self):
        if self.model is None:
            logger.info(f"[BARTSummarizer] Loading BART model '{self.model_name}' on device '{self.device}'...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
            except Exception as e:
                logger.warning(f"[BARTSummarizer] Failed to load '{self.model_name}' ({e}). Falling back to 'facebook/bart-base'...")
                fallback_name = "facebook/bart-base"
                self.model_name = fallback_name
                self.tokenizer = AutoTokenizer.from_pretrained(fallback_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(fallback_name).to(self.device)
            logger.info(f"[BARTSummarizer] BART model '{self.model_name}' loaded successfully.")

    def summarize_chunk(self, text: str, max_length: int = 150, min_length: int = 40) -> str:
        """Summarize a single text chunk using BART PyTorch model."""
        self._load_model()
        words = text.split()
        if len(words) < 25:
            return text.strip()

        num_words = len(words)
        effective_max = min(max_length, max(30, int(num_words * 0.6)))
        effective_min = min(min_length, max(15, int(num_words * 0.25)))

        inputs = self.tokenizer(
            text.strip(),
            return_tensors="pt",
            max_length=1024,
            truncation=True
        ).to(self.device)

        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=effective_max,
            min_length=effective_min,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3
        )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary.strip()

    def summarize(self, chunks: List[str]) -> Dict[str, Any]:
        """
        Hierarchical Chunk-based Summarization for long conversations, or single-pass for short.
        """
        self._load_model()
        if not chunks:
            return {"summary": "", "method": "none", "chunk_summaries": []}

        if len(chunks) == 1:
            summary = self.summarize_chunk(chunks[0], max_length=180, min_length=50)
            return {
                "summary": summary,
                "method": "single_pass",
                "chunk_summaries": [summary]
            }

        # Long conversation: Hierarchical summarization
        logger.info(f"[BARTSummarizer] Performing hierarchical summarization across {len(chunks)} chunks...")
        chunk_summaries = []
        for idx, chunk in enumerate(chunks):
            logger.info(f"[BARTSummarizer] Summarizing chunk {idx + 1}/{len(chunks)}...")
            c_summary = self.summarize_chunk(chunk, max_length=120, min_length=30)
            chunk_summaries.append(c_summary)

        # Meta-summarization pass over combined chunk summaries
        combined_text = " ".join(chunk_summaries)
        logger.info("[BARTSummarizer] Generating final meta-summary...")
        final_summary = self.summarize_chunk(combined_text, max_length=220, min_length=60)

        return {
            "summary": final_summary,
            "method": "hierarchical_chunking",
            "chunk_summaries": chunk_summaries
        }

# Global Singleton Instance
bart_summarizer = BARTSummarizer()
