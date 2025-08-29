# Orchestrator — the RAG step (deep dive)

Short version: the **orchestrator’s RAG step** is where the conversation engine *turns a user query into a reliable, grounded answer* by (1) reformulating the query, (2) retrieving relevant documents (vector + lexical), (3) re-ranking and selecting the best passages, and (4) assembling a grounded prompt for the reader/generator model — while enforcing ACLs, token budgets, caches, and safety checks. Below I’ll walk through each sub-step, design choices, example code/prompt templates, trade-offs, metrics, and production notes.

---

# 1) High-level pipeline (what the orchestrator does)

1. **Input preprocessing** — normalize, detect language, scrub/flag PII, intent classification.
2. **Query reformulation** — expansion, rewriting, conversation-context injection.
3. **Retrieval**

   * Vector retrieval (semantic via embeddings)
   * Lexical retrieval (BM25)
   * Hybrid score fusion + metadata filtering (ACLs, freshness)
4. **Re-ranking** — cross-encoder or learned ranker to refine top candidates.
5. **Passage selection & assembly** — select, dedupe, trim and format snippets with citations.
6. **Prompt construction** — build the final context-aware prompt for the LLM (respect token budget).
7. **Reader/generation** — model produces an answer grounded in provided passages.
8. **Post-checks** — safety, PII redaction, citation validation, hallucination checks.
9. **Return & persist** — stream answer back, persist minimal conversation, update caches & telemetry.

---

# 2) Input handling & query reformulation (details)

* **Normalize**: strip extra whitespace, unify Unicode, lowercase if needed for lexical retrieval.
* **Language detection**: choose language-specific embeddings and lexical index.
* **PII/sensitive detection**: run regex/classifier; either redact or route to special workflow.
* **Intent classification**: is this Q\&A, a code task, or a tool call? (affects model & retrieval).
* **Context merging**: include conversation memory triggers (recent messages, summarized memory) — but keep token budget in mind.
* **Reformulation**: convert conversational queries (“What about last week?”) into self-contained queries (“sales for product X week of 2025-08-18”). Techniques:

  * Seq2seq rewrite model (fast, small)
  * Use a deterministic template + slot filling from conversation state

Practical tip: precompute expansions for named entities (synonyms, acronyms) to improve recall.

---

# 3) Retrieval (vector + lexical + hybrid)

## Chunking & embeddings

* **Chunk size**: \~512–1024 tokens per chunk with overlap 64–200 tokens (overlap helps keep sentence continuity).
* **Chunk granularity**: sentences, paragraphs, or fixed token windows. Use semantic-aware chunk boundaries where possible.
* **Embedding model**: choose dimension vs cost tradeoff (e.g., 768–1536 or 3072 dims). Use same model for both docs and queries or a lighter bi-encoder for speed.

## Vector DB & index choices

* **HNSW** (fast recall; tune `ef_search` and `M`) — great default for low-latency.
* **IVF-PQ** (better on very large corpora; requires coarse quantization).
* **Faiss / Annoy / Milvus / Pinecone / Weaviate / OpenSearch** — pick based on scale and features (namespace/ACLs, DB persistence).

## Lexical retrieval

* **BM25** (Lucene/OpenSearch) for exact term matching and high precision for named entities and numbers.

## Hybrid retrieval

* Retrieve top-K from both (e.g., k\_vec=100, k\_bm25=100), then **fuse**:

  * Simple linear combination: `score = alpha * vec_score + beta * bm25_score` (normalize scores first)
  * Learned ranker: features = {vec\_score, bm25\_score, freshness, click\_rate, metadata\_match} → ML model to produce final ranking

## Metadata & ACL filtering

* Always pre-filter by `tenant_id`, `document_acl`, `tags`, `freshness_range`. Filtering happens at the vector DB or immediately post-retrieval.

## Parameter examples

* `k_retrieve = 100` (server side), then rerank to `n_select = 5..10`.
* HNSW `ef_search = 64..512` (higher → better recall but higher latency).
* Chunk overlap = 64 tokens; chunk size = 512 tokens.

---

# 4) Re-ranking (why + how)

* **Purpose**: improve ordering and precision before generation (the LLM should see the best few passages).
* **Options**:

  * **Bi-encoder** (fast but coarse) — used to get initial candidates.
  * **Cross-encoder** (BERT-style; concatenates query+passage) — expensive but much higher quality for top candidates; typically GPU-backed, re-ranks top 50→top 5.
  * **Learned fusion**: gradient boosted trees or small MLP mixing lexical & vector scores + metadata.
* **Latency**: cross-encoders are heavier — run only on a small candidate set.
* **Ensembles**: re-ranker score + document importance (e.g., clickthrough, recency).

---

# 5) Passage selection, deduplication, and assembly

* **Deduplicate** overlapping passages (use Jaccard or overlap heuristics) so the LLM isn’t given the same content twice.
* **Trim & prioritize**: select passages that maximize coverage of distinct facts; favor passages with higher source trust/freshness.
* **Citation mapping**: keep mapping chunk → source\_id + char offsets for provenance. Assign IDs `[S1]`, `[S2]`.
* **Format**: put each passage in clearly delimited blocks, e.g.:

```
[Source S1 | doc=policy.pdf | page=3]
This is the passage text...
<<<<end S1>>>>
```

* **Token budget**: ensure the total tokens (system + context + passages + user) < model max — trim the least useful passages if needed.

---

# 6) Prompt construction for the reader / generator

* **System instruction**: rules about grounding, citation usage, refusal policy, answer style.
* **Context block**: include the assembled passages. Use explicit instructions like “Use only the content below. If the answer is not contained, say ‘I don’t know’.”
* **Question**: the final reformulated query.
* **Schema enforcement**: if you need structured output, provide JSON schema and ask the model to reply exactly in that format.

### Example Prompt Template

```
SYSTEM: You are an assistant. Use ONLY the provided source passages. Cite sources inline, e.g., [S1]. If no relevant info exists, respond "I don't know".

CONTEXT:
[S1] (Title: Employee Handbook)
"Employees are entitled to 20 paid vacation days per year."

[S2] (Title: Policy Update 2024)
"Vacation accrual is pro-rated for new employees."

QUESTION:
How many vacation days is a new employee entitled to in year one?

ANSWER:
```

* **Few-shot examples** (optional): include 1–2 examples showing desired grounding & citation style.

---

# 7) Grounding & hallucination mitigation

* **Constrain** the model with explicit “use only provided context” instructions.
* **Verification loop**: after generation, check the answer’s factual claims against sources (string-match, QA models, or a fast extractor). If unsupported, mark low confidence or redact.
* **Abstain threshold**: if top passages’ combined score < threshold, return a “no answer / ask to clarify” response.

---

# 8) Caching & semantic cache

* **Semantic cache**: map `hash(embedding(query))` → previously generated answer & selected passages. Use approximate nearest neighbor in embedding space for cache hits.
* **Response caching**: store final outputs for hot queries; include TTL and invalidation on doc updates.
* **Warm caches**: pre-warm using popular queries or after content ingestion.

---

# 9) Multi-hop & iterative retrieval

* For complex queries, the orchestrator may:

  1. Retrieve initial passages,
  2. Ask the LLM to produce an intermediate query (utils: “What entity to look up next?”),
  3. Re-retrieve using that intermediate information,
  4. Repeat until answerable.
* This enables *compositional reasoning* but increases latency and complexity.

---

# 10) Metrics & evaluation

* **Retrieval metrics**: recall\@k, precision\@k, MRR, NDCG — evaluate if retriever returns relevant docs.
* **End-to-end metrics**: exact match / F1 vs ground-truth answers, human-rated relevance, hallucination rate, citation accuracy.
* **Operational metrics**: latency P50/P95, re-ranker GPU utilization, cache hit rate, cost per query (tokens + GPU time).

---

# 11) Operational concerns, scaling, and cost

* **Latency tradeoffs**: more retrieval candidates & larger re-rankers → better quality but higher latency. Use async streaming to reduce perceived latency: stream partial answer while final checks run.
* **Batching**: batch retrieval & reranker calls to improve throughput.
* **Autoscaling**: scale embedding service separately from re-ranker and LLM.
* **Cost knobs**: reduce `k` for retrieval, use smaller reranker, use quantized indices, tune `ef_search`.
* **Monitoring**: trace token counts, retrieval latencies, reranker latencies, hallucination incidents.

---

# 12) Security, privacy & governance

* **ACL enforcement**: filter results by doc-level ACLs at retrieval time (namespaces in vector store).
* **Redaction**: redact PII from passages before passing to LLM if policy requires.
* **Encryption**: at-rest & in-transit for vector DB and conversation store.
* **Data usage**: enforce “do not use for training” flags for tenant data if requested.

---

# 13) Pseudocode (orchestrator RAG step)

```python
def orchestrator_rag(user_msg, convo_state, tenant_id):
    # 1. Preprocess
    query = normalize(user_msg)
    lang = detect_language(query)
    query = rewrite_with_context(query, convo_state)  # optional

    # 2. Check semantic cache
    cached = semantic_cache.get_similar(query)
    if cached and cached.confidence > CACHE_THRESHOLD:
        return cached.answer  # fast path

    # 3. Retrieve (parallel)
    vec_query = embed(query)
    vec_task = vector_db.search(vec_query, k=100, filter={"tenant": tenant_id})
    bm25_task = bm25.search(query, k=100, filter={"tenant": tenant_id})
    vec_results = vec_task.result()
    bm25_results = bm25_task.result()

    # 4. Fusion & prefilter
    candidates = fuse_and_normalize(vec_results, bm25_results)
    candidates = apply_acl_and_freshness(candidates, tenant_id)

    # 5. Re-rank (cross-encoder on top candidates)
    top_candidates = candidates[:50]
    rerank_scores = cross_encoder.rank(query, top_candidates)  # GPU
    ranked = sort_by_score(top_candidates, rerank_scores)

    # 6. Select & format passages
    selected = select_nonredundant(ranked, n=5)
    context = format_passages_with_metadata(selected)

    # 7. Build prompt & call LLM
    prompt = build_prompt(system_instructions, context, query)
    response = llm.generate(prompt, stream=True)

    # 8. Post-checks
    if not verify_response_against_sources(response, selected):
        response = generate_safe_fallback()

    # 9. Persist & cache
    store_conversation(convo_state.id, user_msg, response, selected)
    semantic_cache.put(query, response, selected)

    return response
```

---

# 14) Example prompt (practical)

```
SYSTEM: Use ONLY the SOURCES below to answer, cite inline like [S1]. If you cannot find the answer in the sources, reply "I don't know".

SOURCES:
[S1] HR Handbook — "Employees get 20 vacation days per year."
[S2] Hiring Policy 2024 — "New hires accrue vacation monthly."

QUESTION: "How many vacation days does a new hire get in their first year, starting July 1?"
ANSWER:
```

---

# 15) Practical tips & debugging checklist

* If hallucinations occur frequently: reduce context, increase re-ranker depth, increase citation requirement, or enforce “always cite” policy.
* If quality is poor for numbers/dates: add strict numeric matchers and prefer lexical retrieval for numerical entities.
* If latency too high: lower `k`, use faster (smaller) re-ranker, cache more aggressively.
* Monitor **cache hit rate**, **reranker latency**, **doc update frequency** (invalidating caches), and **source coverage**.

---

# Trade-offs (short table)

| Design Choice                 | Pros                            | Cons                                      |
| ----------------------------- | ------------------------------- | ----------------------------------------- |
| Large `k` + deep rerank       | Higher recall → better answers  | Higher latency & cost                     |
| Strong cross-encoder reranker | Better top results              | GPU cost & latency                        |
| Small chunk size              | Fine-grained retrieval          | More results to manage; loss of context   |
| High overlap                  | preserves context across chunks | More storage & retrieval overhead         |
| Strict grounding rule         | Fewer hallucinations            | May produce more “I don’t know” responses |

---

If you want, I can:

* produce a concrete **example pipeline** using specific tools (Faiss + BM25 in OpenSearch + cross-encoder reranker + vLLM reader), with sample config values; or
* show **how to instrument metrics and set alarm thresholds** for retrieval quality and hallucination rates.

Which would be most useful next?
