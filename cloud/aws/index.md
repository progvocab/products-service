# AI

## Amazon SageMaker

* Studio
* JumpStart
* Autopilot
* Data Wrangler
* Feature Store
* Training (Managed Training Jobs)
* Hyperparameter Tuning
  - Automated search 
  - Search strategies 
  - Objective metric
  - Parallel search 
  
* Real-time Inference Endpoints
* Batch Transform
* Model Monitor


##  Amazon Bedrock

* Foundation Models (FM access)
* Knowledge Bases
  - Connects your private data to Foundation Models 
  - simpler alternative to RAG,no code
  - built in
    - Document Parsing 
    - citations 
    - IAM Integration 
    - Foundation Models Support 
  - Architecture 
    - User question to Embeddings 
    - Vector search 
    - Retrieve relevant Chunks
    - send to Foundation Models 
    - generate answer 
* Agents for Bedrock
  - allow models to autonomously 
    - plan 
    - reason 
    - execute tasks
  - breaks task into steps , call APIs and returns final response 
  - automatically manage orchestration 
    - tool call
    - prompt engineering 
    - retries 
    - multi step flow

* Guardrails
  - enforce
    - Safety 
    - Compliance
    - Domain restrictions 
    - Data protection 
  - attach to
    - Foundation Model invocation 
    - Agents 
    - Knowledge Base response 
  - controls
    - Content Filter
    - Sensitive Information PII 
    - Topic Restriction
    - Word block , Phrase block
    - Contextual Grounding check
  - Foundation Model can 
    - generate Hallucination 
    - produces unsafe content
    - leak sensitive data
    - respond outside policy boundary 
  - Flow with Guardrails 
    - User input
    - Guardrail input check 
    - Model response 
    - Guardrail output check
    - Final response 
* Model Evaluation
* Fine-tuning
* RAG Integration
 - RAG
   - vector database
   - Retrieval Pipeline 
   - Embeddings 
   - steps to build
     - Chunk Documents
     - Generate Embeddings 
     - Store in vector database 
     - Retrieve top K results 
     - send context to LLM 
     - generate response 
  
* Prompt Management



## Amazon Rekognition

* Image Classification
* Object Detection
* Face Detection
* Face Comparison
* Celebrity Recognition
* Text Detection (OCR)
* Video Analysis
* Content Moderation


##  Amazon Comprehend

* Sentiment Analysis
* Entity Recognition
  - Type focused 
  - performs classification 
  - Pre trained Named Entity Recognition 
  - Pre trained model to detect
    - Person
    - Location 
    - Organization 
    - Date 
    - Quantity 
    - Commercial Item
    - Title
    - Event
  - Custom Entity Recognition 
    - Train your Custom model
* Key Phrase Extraction
  - Works on 1 document 
  - Topic focused 
  - Finds : what is this text about 
  - Extracts : meaningful noun Phrases

* Topic Modeling
  - works on multiple documents
  - discover themes
  - Extracts:
    - Topic clusters 
    - Keywords per Topic

* Syntax Analysis
  - identifies grammatical structure 
  - Detects 
    - Noun 
    - Pronoun
    - Adjective 
    - Verb
    - Adverb
    - Proposition 
    - Punctuation 
* Custom Classification
* Custom Entity Recognition
* PII Detection



##  Amazon Lex

* Chatbot Builder
* Intent Recognition
* Slot Filling
* Multi-language Bots
* Voice Integration
* Lambda Integration
* Conversation Logs


##  Amazon Polly

* Text-to-Speech
* Neural Voices
* Speech Marks
* SSML Support
* Custom Lexicons
* Brand Voice


##  Amazon Transcribe

* Speech-to-Text
* Streaming Transcription
* Speaker Identification
* Call Analytics
* Custom Vocabulary
* Language Identification

---

##  Amazon Translate

* Real-time Translation
* Batch Translation
* Custom Terminology
* Active Custom Translation
* Multi-language Support

---

##  Amazon Textract

* Form Extraction
* Table Extraction
* Key-Value Pair Detection
* Handwriting Recognition
* Expense Analysis
* Identity Document Analysis

---

## Amazon Kendra

* Intelligent Search
* FAQ Matching
* Document Ranking
* Connectors (S3, SharePoint, etc.)
* Custom Document Enrichment
* Semantic Search

---

## Amazon Personalize

* Recommendation Engine
* Real-time Recommendations
* User Personalization
* Similar Items
* Batch Recommendations
* Event Tracking

## Amazon Forecast

* Time Series Forecasting
* Demand Planning
* Dataset Groups
* Predictor Training
* Backtesting
* Explainability


## Amazon Fraud Detector

* Fraud Prediction Models
  - Generates Fraud Score 
  - Online Fraud Insight 
  - Transaction Fraud Insight 
  - Account takeover Insight 
  - Custom Models
* Event Scoring
  - Final score of entire Event 
  - Combines multiple models and rules
* Custom Rules Engine
  - If Else based
* Model Training
* Real-time Detection

##  Amazon Lookout for Vision

* Visual Anomaly Detection
* Defect Detection
* Custom Vision Models
* Model Evaluation



##  Amazon Lookout for Equipment

* Equipment Failure Prediction
* Sensor Data Monitoring
* Anomaly Detection
* Predictive Maintenance

## Amazon Lookout for Metrics

* Anomaly Detection in KPIs
* Root Cause Analysis
* Time Series Monitoring
* Alerting


## Amazon Q

* Business Chat Assistant
* Developer Assistant
* Code Generation
* Documentation Q&A
* AWS Resource Insights

## Amazon CodeWhisperer

* Code Suggestions
* Security Scanning
* IDE Integration
* Reference Tracking
* Unit Test Generation


## AWS DeepRacer

* Reinforcement Learning
* Model Training
* Virtual Racing
* Physical Car Deployment
* Leaderboards



## AWS Panorama

* Edge Computer Vision
* Camera Integration
* On-device Inference
* Industrial Monitoring

##  AWS HealthLake

* Healthcare Data Storage
* FHIR Data Processing
* NLP for Medical Records
* Medical Entity Extraction
* Analytics Integration

### Categories 
* Generative AI
* NLP
* Vision
* Speech
* Forecasting
* Industrial AI
* Developer AI
* Healthcare AI
