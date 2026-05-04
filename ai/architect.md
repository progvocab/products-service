#### LLM
- Tokenization 

### Pretraining 

#### Re-ranking 


#### RAG
- Chunking Strategies 
  - Word
  - Sentence 
  - paragraph 
  - page
- Query Engine
  - take the input and search the database and return most relevant Chunks 
- Semantic Re-ranking 
  - after receiving list of relevant Chunks, using Semantic Re-ranking find Relevance of each Chunk against the query
- Chunks are passed to LLM

- Vector Databases 
  - Faiss 

#### Pre Deployment Evaluation 
- Accuracy 


#### Observability 
- Model Drift
- Data Drift

#### Evaluation 
- Toxicity 
- Grounding 
- Rouge
- Blue
  - Semantic similarity 

#### Guardrails 

- PII data Masking 


#### Security 
- Evaluation before Deployment 
- Prompt Injection 
  - Prompt Template
  - Negative Prompt 

#### Performance 
- GPU architecture 

- Prompt Caching 

- Soft Prompt , part of Fine Tuning 
  - trained by optimising a small set of learnable embedding vector,  prepended to input Token,  using back Propagation,  keeping the main model weight frozen.



#### Agent
- Reasoning Engine, LLM
- Memory,  State
- Short Term and Long Term Context
- Planner , decision making 
- Approval steps


#### Context

LLM has limited Context window
- Context Engineering 
  - Summarization 
  - Forget some part 
  

#### MCP server
- multiple client can connect to it

### Example

User input: “Book me a flight from Bangalore to Delhi tomorrow”

The LLM infers intent → book_flight

In LangChain:

A router chain maps this intent to a flight booking tool/API

The tool is invoked with extracted parameters (source, destination, date)


In LangGraph:

A node handles intent classification

Based on intent, the graph transitions to a “FlightBooking” node

That node calls the tool and returns results



Key idea: Intent → Routing → Tool/Action execution

#### Langgraph 
- multi step Reasoning 
- multi agent system 

#### LangChain 
- Prompt Orchestration 
  - Chaining 
  - Branching 
  - Tool Use 

#### Prompt Refinement using Claude 
