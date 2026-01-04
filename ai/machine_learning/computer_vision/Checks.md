Basic CV (Computer Vision) checks at image ingestion are fast, deterministic, non-LLM checks used to filter bad or low-quality images before sending them to an expensive vision-LLM.

Below is a clear, Amazon-interview-ready explanation:


---

1. Image quality checks

These ensure the image is usable at all.

Resolution above minimum threshold

Image not blurry (Laplacian variance / blur score)

Proper lighting (no extreme dark/overexposed images)

No heavy compression artifacts


👉 Purpose: avoid sending unusable images to LLM.


---

2. Format & integrity checks

Supported formats (JPEG, PNG)

Valid image decoding (not corrupted)

Correct orientation (EXIF rotation)


👉 Purpose: fail fast on broken uploads.


---

3. Object presence / framing check

Using lightweight CV models:

Detect if any object exists

Verify object occupies sufficient area (not too small)

Check if image is not empty, blank, or background-only


👉 Purpose: reject random or irrelevant images early.


---

4. Single vs multiple object detection

Detect number of major objects

Ensure expected count (e.g., single product per image)


👉 Purpose: reduce ambiguity before semantic validation.


---

5. Duplicate & near-duplicate detection

Perceptual hashing (pHash, aHash)

Compare against previous uploads


👉 Purpose: cache results and reduce repeated LLM calls.


---

6. Basic content safety check

Detect NSFW or prohibited content

Detect watermarks, screenshots, or text-only images


👉 Purpose: safety + policy compliance.


---

7. Simple product heuristics (domain-specific)

Without reasoning, just detection:

Is there a box-like object?

Is packaging present?

Is background reasonably clean?


👉 Purpose: cheap filtering before deep reasoning.


---

Why these checks matter (interviewer framing)

> “Basic CV checks reduce LLM cost, improve precision, and prevent garbage-in–garbage-out. They act as a deterministic guardrail before semantic reasoning.”




---

What NOT included in basic CV checks

Brand validation

Authenticity checks

Policy interpretation

Complex reasoning


👉 These are LLM responsibilities, not basic CV.


---

One-line summary (great for interview)

“Basic CV checks are fast, deterministic filters for quality, presence, and safety that reduce LLM cost and false positives.”

If you want, I can now:

Ask a deep follow-up question

Show example pseudo-code

Or simulate an Amazon Bar Raiser challenge