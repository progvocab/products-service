

[Prompt Caching in LLM Systems. Table of Contents: - Caching ...](https://medium.com/@nikitaparate9/prompt-caching-in-llm-systems-db1a53dd671f)
[What is prompt caching? How can product managers leverage it ...](https://labs.adaline.ai/p/what-is-prompt-caching-how-can-product)
[What is prompt caching? How can product managers leverage it ...](https://labs.adaline.ai/p/what-is-prompt-caching-how-can-product)
[Prompt caching guide (non-technical) - YouTube](https://www.youtube.com/watch?v=RDjaUJz-uWo)

Prompt caching is a performance optimization technique that stores the processed internal state of a large language model's (LLM) attention layer for specific portions of an input prompt. Instead of recomputing mathematical "key-value" (KV) tensors from scratch for every request, the model reuses these precomputed states for identical prompt prefixes. This approach can significantly reduce Time-to-First-Token (TTFT) latency by up to 80% and lower input token costs by up to 90%. [1, 2, 3, 4, 5, 6] 
## How Prompt Caching Works
LLMs process input in two phases: the prefill stage (processing the input) and the generation stage (producing output tokens). [3, 7] 

* Exact Matching: Caching relies on finding a byte-for-byte identical prefix at the start of a prompt.
* KV Cache Storage: The system stores the intermediate numerical representations (KV tensors) of tokens in the prompt.
* Cache Hit vs. Miss: If a request matches a cached prefix, the model skips the prefill for that section. A minor change (even a single space or timestamp) at the beginning of the prompt can cause a cache miss. [1, 6, 7, 8, 9, 10] 

## Key Benefits and Use Cases
Prompt caching is most effective for workflows with large, static content that is reused frequently. [10, 11] 

* Multi-Turn Conversations: Caching previous messages in a long chat history reduces the cost and speed of each subsequent turn.
* Document Analysis: When asking multiple questions about a single long document (e.g., a 200-page PDF), the document itself is cached once.
* Coding Assistants: Large codebases can be cached to provide near real-time inline suggestions without re-analyzing the entire file for every change.
* Tool/Function Definitions: Complex tool schemas that rarely change are ideal for caching to improve overall agentic performance. [2, 10, 11, 12, 13] 

## Optimizing for Cache Hits
To maximize the benefits of prompt caching, developers should follow specific prompt engineering best practices: [2, 6] 

   1. Structure Strategically: Place static content (system instructions, tool definitions, reference docs) at the beginning of the prompt.
   2. Move Volatile Data: Put dynamic content like user queries, timestamps, and session IDs at the end.
   3. Check Minimum Thresholds: Most providers require a minimum prompt length (often 1024 tokens) before caching is triggered.
   4. Monitor Usage: APIs usually return a cached_tokens or cache_read_input_tokens field in the response to track performance. [1, 2, 3, 10, 12, 14, 15] 

These videos provide technical deep dives and practical tutorials on implementing prompt caching across different platforms:

[Prompt caching guide (non-technical)](https://www.youtube.com/watch?v=RDjaUJz-uWo), YouTube · Dan Cleary · 1970 M01 1
[Build Hour: Prompt Caching](https://www.youtube.com/watch?v=tECAkJAI_Vk), YouTube · OpenAI · 1970 M01 1
[Prompt Caching in Amazon Bedrock: Hands-on Tutorial with ...](https://www.youtube.com/watch?v=TkO9FOXcOhI), YouTube · Fahd Mirza · 1970 M01 1
[What is Prompt Caching? Optimize LLM Latency with AI Transformers](https://www.youtube.com/watch?v=u57EnkQaUTY), YouTube · IBM Technology · 1970 M01 1
[Spring AI Prompt Caching: Stop Wasting Money on Repeated ...](https://www.youtube.com/watch?v=eYb7BKW4QcU), YouTube · Dan Vega · 1970 M01 1

## Comparison with Other Caching Layers
Prompt caching is distinct from other common caching strategies used in AI applications: [11, 16] 

| Feature [1, 2, 3, 5, 10, 11, 12, 17] | Prompt Caching | Semantic Caching | Regular (Exact) Caching |
|---|---|---|---|
| What is Cached | Internal model states (KV tensors) | Embeddings of past queries/responses | Full exact prompt-response pairs |
| How it Matches | Exact prefix match | Meaning similarity (vector search) | Identical string match |
| Savings | Reduces input token costs and latency | Bypasses LLM entirely; saves all tokens | Bypasses LLM entirely; saves all tokens |
| Best For | Multi-turn chats, large documents | Paraphrased user questions (FAQs) | Templated/static queries |

Would you like to see a code example for implementing prompt caching with a specific provider like OpenAI, Anthropic, or Amazon Bedrock?

[1] [https://developers.openai.com](https://developers.openai.com/api/docs/guides/prompt-caching)
[2] [https://console.groq.com](https://console.groq.com/docs/prompt-caching)
[3] [https://redis.io](https://redis.io/blog/what-is-prompt-caching/)
[4] [https://sankalp.bearblog.dev](https://sankalp.bearblog.dev/how-prompt-caching-works/)
[5] [https://www.youtube.com](https://www.youtube.com/watch?v=u57EnkQaUTY&t=1)
[6] [https://developers.openai.com](https://developers.openai.com/cookbook/examples/prompt_caching_201)
[7] [https://developers.cloudflare.com](https://developers.cloudflare.com/workers-ai/features/prompt-caching/)
[8] [https://community.openai.com](https://community.openai.com/t/how-does-prompt-caching-work/992307)
[9] [https://community.openai.com](https://community.openai.com/t/how-prompt-caching-works/970979)
[10] [https://aws.amazon.com](https://aws.amazon.com/blogs/machine-learning/effectively-use-prompt-caching-on-amazon-bedrock/)
[11] [https://redis.io](https://redis.io/blog/prompt-caching-vs-semantic-caching/)
[12] [https://www.vellum.ai](https://www.vellum.ai/llm-parameters/prompt-caching)
[13] [https://www.heroku.com](https://www.heroku.com/blog/faster-agents-automatic-prompt-caching/)
[14] [https://platform.claude.com](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
[15] [https://openai.com](https://openai.com/index/api-prompt-caching/)
[16] [https://www.digitalocean.com](https://www.digitalocean.com/community/tutorials/prompt-caching-explained)
[17] [https://www.ibm.com](https://www.ibm.com/think/topics/prompt-caching)
