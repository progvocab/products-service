Below are the **end-to-end training-phase steps** for a **language translation system** (Transformer-based), written in a **system-design / ML-engineering way** and aligned with how models like **Marian, mBART, NLLB, T5** are actually trained.

---

## 1. Problem Definition & Scope

* Define source and target languages
* Define domains (general, legal, retail, medical)
* Define quality metrics (BLEU, COMET, chrF)
* Define latency & model size constraints

---

## 2. Data Collection

* Parallel corpora (source ↔ target pairs)
* Monolingual corpora (for back-translation)
* Domain-specific datasets
* Public datasets (WMT, OPUS, CCMatrix)

---

## 3. Data Cleaning

* Remove duplicates
* Remove misaligned sentence pairs
* Length ratio filtering
* Language ID verification
* Profanity / noise filtering

---

## 4. Data Normalization

* Unicode normalization (NFKC)
* Whitespace cleanup
* Punctuation normalization
* Script normalization (Indic, Arabic)

---

## 5. Sentence Alignment (If Needed)

* Align documents → sentence pairs
* Use length-based + embedding-based alignment

---

## 6. Tokenizer Training (If Custom)

* Train BPE / SentencePiece
* Vocabulary size selection
* Language-agnostic subwords
* Save tokenizer artifacts

---

## 7. Tokenization

* Convert text → subword tokens
* Add language tokens (`<en>`, `<fr>`)
* Create attention masks

---

## 8. Dataset Splitting

* Train / Validation / Test
* Ensure domain and language balance
* Prevent data leakage

---

## 9. Data Augmentation

* Back-translation
* Noising (word dropout, shuffle)
* Paraphrasing
* Synthetic parallel data

---

## 10. Model Architecture Selection

* Encoder-decoder Transformer
* Layers, heads, hidden size
* Positional encoding type
* Shared vs separate embeddings

---

## 11. Parameter Initialization

* Random initialization or pretrained weights
* Load pretrained checkpoints (mBART, NLLB)

---

## 12. Training Setup

* Loss function: Cross-entropy
* Label smoothing
* Optimizer: Adam / AdamW
* Learning rate schedule (warmup + decay)

---

## 13. Forward Pass

* Encoder processes source tokens
* Decoder predicts next target token
* Teacher forcing used

---

## 14. Loss Computation

* Compare predicted tokens vs ground truth
* Ignore padding tokens
* Apply label smoothing

---

## 15. Backpropagation

* Compute gradients
* Gradient clipping
* Mixed precision (fp16/bf16)

---

## 16. Parameter Update

* Optimizer step
* Learning rate scheduler step

---

## 17. Validation Loop

* Run inference on validation set
* Compute BLEU / COMET
* Track overfitting

---

## 18. Checkpointing

* Save best model
* Save periodic checkpoints
* Store optimizer & scheduler state

---

## 19. Early Stopping

* Stop if validation metric plateaus
* Prevent overfitting

---

## 20. Hyperparameter Tuning

* Batch size
* Learning rate
* Beam size (eval)
* Vocabulary size

---

## 21. Evaluation on Test Set

* Final BLEU / chrF / COMET
* Human evaluation (optional)

---

## 22. Model Optimization

* Quantization
* Distillation
* Pruning

---

## 23. Packaging for Inference

* Export model & tokenizer
* Versioning
* Device compatibility checks

---

## 24. Deployment Readiness

* Throughput testing
* Latency benchmarking
* Memory profiling

---

## 25. Continuous Training (Optional)

* Online feedback incorporation
* Domain adaptation
* Periodic retraining

---

## One-Line Summary

> **Training = collect → clean → tokenize → train (encode-decode + loss) → validate → optimize → package.**

---

If you want, I can:

* Map this to **Hugging Face `Trainer`**
* Show **loss & attention flow diagram**
* Compare **training vs fine-tuning**
* Design **multi-language (100+ langs) pipeline**
