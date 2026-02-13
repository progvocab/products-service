# Java
## Object Oriented Design
### Class
### Object
### Abstraction
### Inheritance
### Polymorphism
### Association
### Aggregation
### Composition
### Dependency
## Garbage Collector
### Serial
### Concurrent Mark Sweep
#### Fragmentation
### Parallel
#### Young Generation
#### Old Generation
### Garbage First
## Memory
### Heap
### Metaspace
## Collections
### ArrayList
### LinkedList
### HashMap
#### Bins
#### Bucket
#### Hash Code
#### Hash Function
#### Capacity
#### Load Factor
#### Resize
#### Treeify
#### Red Black Tree
- Black Node
- Red Node
- Balencing
# Spring Framework
# Oracle Database
## ACID
### Atomicity
### 
## Types
### Table
#### Columns
- Number
- Varchar2
- Date
#### Sequence
#### Index
- Unique
- Bitmap
  
#### Constraint
#### Primary Key
#### Foreign Key
### View
### Materialized View

### Function
### Procedure
### Trigger
### Object

## DDL
### CREATE
### TRUNCATE
### DROP
### ALTER
## DML
### SELECT
### INSERT
### UPDATE
### DELETE

## DCL
## Administration
### Replication 
#### Golden Gate

# Design Patterns
## Gang of Four
### Creational
#### Singleton
#### Factory
#### Abstract Factory

### Structural
#### Adaptor
#### Bridge
#### Composite
### Behavioural
#### Proxy
#### Memento

## Distributed Systems

### Service Mesh
### Load Balencer
### 
## Microservices 
### API Gateway
#### Rate Limiter
#### Timeout
#### Retry
### Service Discovery
### Circuit Breaker
### Config Server
### Cache
#### Distributed Lock
#### Cache-Aside
#### Write-Through Cache
#### Write-Behind Cache
### SAGA
#### Orchestration
#### Choreography

# Design Principles
## Solid
### Single Responsibility Principle
### Open Closed Principle
### Liskov Substitution Principle
### Interface Segregation Principle 
### Dependency Inversion Principle
## DRY - Don’t Repeat Yourself
## KISS - Keep It Simple Silly
## YAGNI - You Arent Gonna Need It
## SoC - Separation of Concerns
## Law Of Demeter

# AI
## Machine Learning
### Supervised
#### Classification
- Deterministic 
  - Logistic Regression
  - Naive Bayes
  - K-Nearest Neighbors (KNN)
  - Linear Support Vector Machine (SVM)
  - Kernel based Support Vector Machine (SVM) RBF, polynomial
  - Perceptron
  - Linear Discriminant Analysis (LDA)
  - Quadratic Discriminant Analysis (QDA)
  - Tree-Based 
    - Decision Tree
    - Random Forest
    - Gradient Boost
      - XG Boost
      - Light GBM
      - Cat Boost
- Stochastic 
  - Bayesian Networks (with sampling inference)
  - Probabilistic Graphical Models with Monte Carlo sampling
  - Gaussian Process Classifier (with stochastic inference)
  - Stochastic Gradient Descent Classifier (SGDClassifier, during training randomness)
  - Bagging-based classifiers (with random sampling, if seed not fixed)

#### Regression
- Linear Regression
  - Ordinary Least Square 
  - Weighted Least Square 
  - Ridge Regression
  - Lasso Regression
  - Robust Regression 
  - Gradient Descent Optimization 
  - Elastic Net
- Support Vector Regression (SVR)
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor
### Unsupervised
#### Clustering
- K-Means
- Hierarchical Clustering
- DBSCAN
- Mean Shift
- Gaussian Mixture Model (GMM)
#### Dimention Reduction
- Principal Component Analysis (PCA)
- Linear Discriminant Analysis (LDA)
- t-SNE
- UMAP
- Autoencoders
- ICA (Independent Component Analysis)
#### Association Rule Mining
- Apriori
- FP-Growth
- Eclat
#### Anomaly Detection
- Isolation Forest
- One-Class SVM
- Local Outlier Factor (LOF)
### Semi-Supervised Learning
- Self-Training
- Label Propagation
- Co-Training
### Reinforcement Learning
- Q-Learning
- SARSA
- Deep Q Network (DQN)
- Policy Gradient
- Actor-Critic
- PPO
- A3C
## Deep Learning
### Neural Network Types
- Feedforward Neural Network (ANN)
- Convolutional Neural Network (CNN)
- Recurrent Neural Network (RNN)
- LSTM
- GRU
- Transformer
### Generative Models (Density Estimation)

#### Explicit Density Models (Tractable)
- Autoregressive Models
  - PixelRNN
  - PixelCNN
- Normalizing Flows
  - RealNVP
  - Glow
- NADE
- MADE

#### Latent Variable Models
- Variational Autoencoders (VAE)
- Hierarchical VAE

#### Implicit Density Models
- Generative Adversarial Networks (GAN)
- Conditional GAN
- CycleGAN
- StyleGAN

#### Energy-Based Models
- Boltzmann Machines
- Restricted Boltzmann Machines (RBM)
- Deep Energy-Based Models

#### Diffusion Models
- DDPM
- Score-Based Models
- Stable Diffusion
- Imagen
- DALL·E (diffusion-based)

# ML Lifecycle

##  Data Collection
- Databases (SQL / NoSQL)
- APIs
- Logs
- Sensors / IoT
- Web Scraping
- Data Streams (Kafka, Kinesis)
- Data Lake

##   Data Cleaning (Cleansing)
- Handling Missing Values
- Removing Duplicates
- Outlier Detection
- Noise Removal
- Data Type Conversion
- Consistency Checks
- Data Validation
- Apache Spark
  - Spark SQL
  - DataFrames
  - ETL Jobs


##   Exploratory Data Analysis (EDA)
- Summary Statistics
- Distribution Analysis
- Correlation Analysis
- Visualization (histograms, boxplots, scatter plots)
- Data Drift Detection
- Apache Spark
   - Spark SQL
   - Spark MLlib (stats)


##   Feature Engineering
- Feature Creation
- Feature Selection
- Feature Encoding
  - One-Hot Encoding
  - Label Encoding
  - Target Encoding
- Feature Transformation
  - Log Transform
  - Binning
- Interaction Features
- Text Features (TF-IDF, Embeddings)
- Image Features (CNN embeddings)
- Feature Pipelines
- Feature Store
  - Feast


##   Data Preprocessing

### Normalization / Scaling
- Min-Max Scaling
- Standardization (Z-score)
- Robust Scaling
- Log Scaling

### Sampling
- Train / Test Split
- Cross Validation
- Stratified Sampling
- Oversampling / Undersampling (SMOTE)

##   Model Training

### Algorithms
- Linear Models
- Tree-based Models
- Ensemble Models
- Neural Networks
- Generative Models

## Model Architecture
### Neural Networks
### Layers
* Input Layer
* Linear (Dense) Layer
* Sequential Layer
* Convolutional Layer
  * Conv1D
  * Conv2D
  * Conv3D
* Transposed Convolution (Deconvolution) Layer
* Pooling Layer
  * Max Pooling
  * Average Pooling
  * Global Pooling
* Embedding Layer
* Recurrent Layer (RNN)
* LSTM Layer
* GRU Layer
* Attention Layer
* Multi-Head Attention Layer
* Transformer Encoder Layer
* Transformer Decoder Layer
* Normalization Layer
  * BatchNorm
  * LayerNorm
  * GroupNorm
* Dropout Layer
* Activation Layer
* Residual / Skip Connection Layer
* Flatten Layer
* Reshape Layer
* Concatenation Layer
* Add / Merge Layer
* Masking Layer
* TimeDistributed Layer
* Lambda / Custom Layer
* Output Layer

### Activation Functions
- Binary Step
- Linear
- Sigmoid
- Tanh
- ReLU
- Leaky ReLU
- Parametric ReLU (PReLU)
- ELU
- SELU
- GELU
- Swish
- Mish
- Softplus
- Softsign
- Hard Sigmoid
- Hard Swish
- Maxout
- LogSoftmax
- Softmax
### Training Configuration
#### Loss Function
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Huber Loss
- Log-Cosh Loss
- Binary Cross-Entropy
- Categorical Cross-Entropy
- Sparse Categorical Cross-Entropy
- Negative Log Likelihood (NLL)
- Kullback–Leibler Divergence (KL Loss)
- Hinge Loss
- Squared Hinge Loss
- Focal Loss
- Dice Loss
- Jaccard (IoU) Loss
- Tversky Loss
- Contrastive Loss
- Triplet Loss
- Cosine Similarity Loss
- Poisson Loss
- Quantile Loss
- Earth Mover’s Distance (Wasserstein Loss)
- Perplexity Loss
- CTC Loss
- Label Smoothing Loss
#### Optimizer
* Gradient Descent
* Stochastic Gradient Descent (SGD)
* Mini-Batch Gradient Descent
* Momentum
* Nesterov Accelerated Gradient (NAG)
* Adagrad
* Adadelta
* RMSProp
* Adam
* AdamW
* Adamax
* Nadam
* LBFGS
* Newton’s Method
* Conjugate Gradient
* Coordinate Descent
* Proximal Gradient Descent
* FTRL (Follow-The-Regularized-Leader)

#### Learning Rate
#### Batch Size
#### Epochs
#### Regularization

### Training Loop
- Forward Pass
- Loss Computation
- Backward Pass (Backpropagation)
- Gradient Calculation
- Weight Update by Optimizer 

### Components
- Loss Function
- Optimizer
- Learning Rate Scheduler
- Regularization

### Distributed Training
#### Spark ML Lib
### Hyperparameters
- Batch Size
- Epochs
- Learning Rate
- Optimizer (SGD, Adam, RMSProp)
- Regularization (L1, L2, Dropout)
- Number of Layers
- Hidden Units
- Number of Estimators
- Sample Size (bootstrap size)
- Max Depth
####  Hyperparameter Tuning
- Spark and Ray

###  Ensemble Methods
#### Bagging
#### Random Forest 
- bagging + feature randomness
#### Boosting 
- AdaBoost
- Gradient Boosting
- XGBoost
#### Stacking
#### Voting

##   Model Validation & Evaluation

### Metrics (Regression)
- RMSE
- MAE
- R²

### Metrics (Classification)
- Accuracy
- Precision
- Recall
- F1 Score
- AUC-ROC
- Log Loss

### Validation Techniques
- Holdout Validation
- K-Fold Cross Validation
- Time Series Split
- Bias–Variance Analysis
- Cross Validation
- OOB (Out-of-Bag) Error (Bagging)
  
##   Explainability & Interpretability

- Feature Importance
- Model Coefficients (Weights & Parameters)
- SHAP
- LIME
- Partial Dependence Plot (PDP)
- ICE Plots

### Bias & Variance
- Bias Detection
- Variance Analysis
- Fairness Metrics
- Model Robustness

##  Model Artifacts & Versioning

### Serialization
- Joblib
- Pickle
- ONNX
- TorchScript

### Experiment Tracking
- MLflow
- Weights & Biases
- TensorBoard

### Model Registry
- Model Versioning
- Metadata
- Lineage

##   Deployment

### Serving Patterns
- Batch Inference
- Real-time Inference
- Streaming Inference

### Platforms
- Kubeflow
- Docker
- Kubernetes
- AWS SageMaker
- Azure ML
- GCP Vertex AI

### Jobs
- Spark
### Pipelines
- Airflow (scheduled )

### APIs
- FastAPI
- Flask
- gRPC

##   Inference

### Batch
- Spark
### Streaming Inference 
- Spark Structured Streaming

### Frameworks
- TensorFlow Lite
- TensorRT
- ONNX Runtime
- PyTorch Serve

### Applications
- Web Apps
- Mobile Apps
- Edge Devices
- IoT

### Inference Parameters (Generative Models)
- Top-K
- Top-P (Nucleus Sampling)
- Temperature
- Max Tokens
- Beam Search

##   Monitoring & Maintenance

- Data Drift Monitoring
  - Spark
- Concept Drift Monitoring
- Model Performance Monitoring
- Latency Monitoring
- Error Tracking
- Alerting

##  Retraining & Feedback Loop
- Scheduled Retraining
- Trigger-based Retraining
- Human-in-the-loop Feedback
- Active Learning

##   Governance & Security
- Data Privacy
- Access Control
- Audit Logs
- Compliance (GDPR, HIPAA)
- Model Risk Management

# LLM Lifecycle

##  Data Preparation
- Data Collection
- Data Cleaning & Deduplication
- Filtering (toxicity, quality, language)
- Tokenization
- Data Mixing & Curriculum
- Synthetic Data Generation
## Model Architecture and Design
###  Transformer
### Attention Mechanism
### Positional Encodings
  - Sinusoidal Positional Encoding
  - Learned Positional Embedding
  - Rotary Positional Embedding (RoPE)
  - ALiBi
    
## Pretraining

### Uncontrolled Pretraining
- Web-scale corpora
- Books
- Code
- Wikipedia
- Multimodal data (text, image, audio)

### Controlled Pretraining
- Curated datasets
- Domain-specific corpora (medical, legal, finance)
- Instruction-style data
- Safety-filtered data
- Train with RoPE-enabled attention
- Long-context learning
- Token position generalization

### Objectives
- Next Token Prediction (Causal LM)
- Masked Language Modeling (MLM)
- Multimodal Objectives (CLIP-style)
- Contrastive Learning

##  Mid-Training (Continual / Domain Adaptation)
- Domain adaptation (code, math, biomedical, legal)
- Long-context fine tuning
- Tool-use training
- Multilingual training
- Reasoning datasets (math, logic, chain-of-thought)
- Synthetic data augmentation
- Token Position Generalization
### Long-Context Adaptation
- RoPE scaling
- NTK-aware RoPE
- Long context fine-tuning (32k, 128k, 1M tokens)

##  Benchmarking & Evaluation
### Offline Evaluation 
#### Academic Benchmarks
- MMLU
- HellaSwag
- ARC
- GSM8K
- HumanEval
- BIG-Bench


#### Custom Task Datasets
- Domain QA
- RAG Evaluation Sets
- Tool-Use Datasets
- Reasoning Datasets
- Conversation Datasets
- Safety & Policy Datasets
- Edge Case Datasets
- Adversarial Datasets
- Golden Datasets
- Regression Test Sets

#### Prompt Evaluation
- Prompt Variants
- A/B Testing
- Prompt Templates
- Few-Shot Prompts
- Zero-Shot Prompts
- Chain-of-Thought Prompts
- Structured Output Prompts (JSON / Schema)
- System vs User Prompts
- Prompt Regression Tests

#### LLM-as-a-Judge
- Pairwise Comparison
- Scoring (1–5, 1–10)
- Rubric-Based Evaluation
- Faithfulness Scoring
- Relevance Scoring
- Correctness Scoring
- Helpfulness Scoring
- Safety Scoring
- Hallucination Detection
- Self-Consistency Checks


### Safety Benchmarks
- Toxicity
- Bias
- Hallucination
- Jailbreak resistance

### Performance Metrics - Automated
- Perplexity
- Accuracy
- BLEU / ROUGE (for generation)
- Win-rate vs baseline

### Human Evaluation
- Helpfulness
- Harmlessness
- Honesty
- Annotation UI
- Pairwise Comparison
- Preference Scoring
  
### Capability Evaluation
- Reasoning (GSM8K, ARC)
- Coding (HumanEval)
- Knowledge (MMLU)
  
### Long-Context & Retrieval Evaluation
- Needle-in-a-Haystack (NIAH)
- LongBench
- RULER
- InfiniteBench
### Tooling
- LangSmith
- Weights & Biases (LLM eval)
- OpenAI Evals
- TruLens
- Promptfoo
- DeepEval

##  Fine-Tuning

### Supervised Fine-Tuning (SFT)
- Instruction tuning
- Chat datasets
- Question–Answer datasets
- Task-specific fine-tuning

### Parameter-Efficient Fine-Tuning (PEFT)
- LoRA
- QLoRA
- Adapters
- Prefix Tuning
- Prompt Tuning
- BitFit

#### Quantization 
- INT8
- INT4
- GPTQ
- AWQ


##  Alignment & Safety Training

### RLHF (Reinforcement Learning from Human Feedback)
- Reward Model Training
- Preference Data Collection
- PPO / DPO / RLAIF

### Alternatives to RLHF
- DPO (Direct Preference Optimization)
- RLAIF (AI Feedback)
- Constitutional AI
- Self-Refinement


##   Grounding & Tool Integration

### Retrieval-Augmented Generation (RAG)
- Vector Databases
- Embeddings
- Document Chunking
- Re-ranking

### Tool Use
- Function Calling
- Code Interpreter
- Web Search
- Database Queries
- APIs

### Knowledge Grounding
- Enterprise Knowledge Bases
- Ontologies
- Structured Data (SQL, Graph DB)

 

##  Compression & Optimization

- Distillation
- Pruning
- Quantization
- Sparse Models
- Low-rank Factorization

 

##   Deployment & Serving

- Model Packaging
- ONNX / TensorRT
- vLLM / TGI / Triton
- GPU / TPU / Edge Devices
- Load Balancing
- Caching

 

##  Inference & Decoding

### Decoding Strategies
- Greedy Decoding
- Beam Search
- Top-K
- Top-P (Nucleus Sampling)
- Temperature
- Repetition Penalty

 

##   Monitoring & Feedback Loop

### Usage Analytics



### Feedback Collection
- User Ratings
- Human Review
- Dataset Expansion
### Tracing & Observability
- LangSmith
- OpenTelemetry
- Logs & Metrics
### Production Evaluation
- Drift Detection
- Hallucination Detection
- Safety Evaluation
- Cost & Latency Tracking


##   Continuous Improvement
- Data Flywheel
- Active Learning
- Periodic Re-training
- Model Versioning

  




