1. What is the difference between orchestration and choreography in multi-agent systems?


2. Why is retrieval augmentation preferred over fine-tuning for dynamic enterprise knowledge?


3. How do you prevent context poisoning in long-running agent workflows?


4. Explain the tradeoff between latency and reasoning depth in agentic architectures.


5. Why can vector similarity search fail for numerically precise queries?


6. How would you implement deterministic execution in stochastic LLM systems?


7. What problem does MCP solve that OpenAI function calling does not?


8. Why is tool selection harder than tool invocation in autonomous agents?


9. Explain semantic caching and its failure scenarios.


10. What causes hallucinations even when using RAG?


11. Why is chunk overlap important in RAG pipelines?


12. How do you detect embedding drift in production systems?


13. Explain the difference between memory and context in agent architectures.


14. Why are graph databases useful in advanced RAG systems?


15. What is the difference between sparse retrieval and dense retrieval?


16. Why is reranking often more important than retrieval itself?


17. How would you reduce token costs in a multi-agent architecture?


18. Why are autonomous recursive agents dangerous in production?


19. What is the difference between planner-executor and ReAct architectures?


20. How do you implement human-in-the-loop approval without blocking scalability?


21. Why does increasing context window not fully solve memory limitations?


22. Explain the tradeoff between centralized and decentralized agent communication.


23. What are the risks of using shared mutable state in agent workflows?


24. How would you handle concurrency conflicts in multi-agent shared memory?


25. Why can embedding models trained on general corpora fail in enterprise search?


26. What is the difference between fine-tuning, instruction tuning, and RLHF?


27. Explain why attention complexity becomes expensive at large context windows.


28. What is speculative decoding and how does it reduce inference latency?


29. How does KV cache improve transformer inference performance?


30. Why do quantized models sometimes degrade tool-calling accuracy?


31. What architectural issue causes agent looping behavior?


32. Explain the difference between synchronous and asynchronous agent execution.


33. Why are event-driven architectures suitable for distributed agents?


34. What is the role of checkpointing in LangGraph-like systems?


35. How would you implement rollback in autonomous workflows?


36. Why can top-k retrieval reduce answer quality?


37. Explain hybrid search and why BM25 is still relevant in GenAI systems.


38. What are the limitations of cosine similarity in vector databases?


39. Why is observability critical in agentic systems?


40. What metrics would you monitor for GenAI production reliability?


41. How do you detect silent failures in LLM pipelines?


42. Why is prompt versioning important in enterprise deployments?


43. What is prompt injection and how would you mitigate it?


44. Explain the difference between guardrails and moderation.


45. Why is schema validation important in structured output generation?


46. What problem does constrained decoding solve?


47. How would you architect multi-tenant RAG securely?


48. Why should embeddings and generation models sometimes be different?


49. Explain the difference between agent memory and vector memory.


50. Why is distributed tracing difficult in multi-agent systems?


1. Orchestration uses a central coordinator, while choreography relies on decentralized event-driven collaboration.


2. RAG is preferred because enterprise knowledge changes frequently and fine-tuning is expensive to update.


3. Context poisoning is prevented using validation, isolation, memory pruning, and trusted retrieval boundaries.


4. Deeper reasoning improves accuracy but increases latency and token consumption.


5. Vector search captures semantics, not exact numerical precision or symbolic relationships.


6. Deterministic execution is achieved through controlled workflows, structured outputs, and fixed seeds/checkpoints.


7. MCP standardizes dynamic tool discovery and interoperability across AI systems.


8. Tool selection requires reasoning and context understanding, while invocation is just execution.


9. Semantic caching fails when semantically similar queries require different factual answers.


10. Hallucinations still occur if retrieved documents are irrelevant, incomplete, or misinterpreted.


11. Chunk overlap preserves contextual continuity across split document segments.


12. Embedding drift is detected by monitoring retrieval relevance degradation over time.


13. Context is temporary prompt data, while memory persists information across interactions.


14. Graph databases model relationships explicitly, improving connected knowledge retrieval.


15. Sparse retrieval uses keyword matching, while dense retrieval uses semantic embeddings.


16. Reranking improves final relevance quality after initial approximate retrieval.


17. Token costs are reduced using summarization, caching, routing, and smaller specialized models.


18. Recursive autonomous agents can create infinite loops, cost explosions, and unsafe actions.


19. Planner-executor separates planning and execution, while ReAct interleaves reasoning and acting iteratively.


20. Human approval is implemented asynchronously using queues, checkpoints, or approval states.


21. Larger context windows still suffer from attention dilution and retrieval inefficiency.


22. Centralized communication improves control, while decentralized communication improves autonomy and scalability.


23. Shared mutable state can create race conditions and inconsistent agent behavior.


24. Concurrency conflicts are handled using locks, transactions, or state versioning.


25. General embeddings may miss domain-specific terminology and enterprise semantics.


26. Fine-tuning adapts models broadly, instruction tuning improves task following, and RLHF aligns human preferences.


27. Attention complexity grows quadratically with token length, increasing compute cost.


28. Speculative decoding uses smaller draft models to accelerate token generation.


29. KV cache avoids recomputing previous attention states during autoregressive inference.


30. Quantization can reduce numerical precision affecting reasoning and tool-selection reliability.


31. Poor stopping conditions and recursive planning often cause agent looping.


32. Synchronous execution blocks sequentially, while asynchronous execution allows parallel non-blocking tasks.


33. Event-driven systems decouple agents and support scalable asynchronous communication.


34. Checkpointing enables workflow recovery, resumability, and fault tolerance.


35. Rollback is implemented by restoring previous checkpoints or state snapshots.


36. Top-k retrieval may introduce noisy irrelevant documents into context.


37. Hybrid search combines semantic and keyword retrieval, while BM25 remains strong for exact matches.


38. Cosine similarity ignores magnitude and may fail for nuanced semantic distinctions.


39. Observability is critical for debugging, monitoring, and tracing autonomous decisions.


40. Key metrics include latency, token usage, hallucination rate, retrieval relevance, and tool success rate.


41. Silent failures are detected through validation checks, monitoring, and output evaluation pipelines.


42. Prompt versioning ensures reproducibility, rollback capability, and controlled experimentation.


43. Prompt injection is mitigated using sanitization, isolation, policy enforcement, and permission boundaries.


44. Guardrails constrain model behavior, while moderation filters unsafe outputs.


45. Schema validation ensures generated outputs remain machine-readable and reliable.


46. Constrained decoding enforces valid structured output generation during inference.


47. Multi-tenant RAG security requires tenant-isolated embeddings, indexes, and access controls.


48. Embedding models optimize retrieval semantics, while generation models optimize reasoning and fluency.


49. Agent memory stores workflow history, while vector memory stores semantic embeddings for retrieval.


50. Distributed tracing is difficult because agents operate asynchronously across multiple services and states.


