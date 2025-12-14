# Learning Resources

Curated resources for ML interview preparation and continuous learning.

## Table of Contents
1. [Interview Preparation](#interview-preparation)
2. [ML Fundamentals](#ml-fundamentals)
3. [Deep Learning](#deep-learning)
4. [NLP & LLMs](#nlp--llms)
5. [ML System Design](#ml-system-design)
6. [MLOps & Production](#mlops--production)
7. [Practice Platforms](#practice-platforms)
8. [Books](#books)
9. [YouTube Channels](#youtube-channels)
10. [Research Papers](#research-papers)

## Interview Preparation

### General ML Interviews
- **Chip Huyen's ML Interviews Book** - Comprehensive guide to ML interviews
- **Introduction to Machine Learning Interviews** - Structured approach to ML case studies
- **Ace the Data Science Interview** (book) - Real interview questions from FAANG

### Coding Interviews
- **LeetCode** - Practice algorithmic problems
  - Focus on: Arrays, Hash Maps, Trees, Dynamic Programming
  - Recommended: Top 150 questions
- **HackerRank** - ML/Data Science track
- **Kaggle** - Real ML competitions and datasets

### System Design
- **Designing Data-Intensive Applications** (book by Martin Kleppmann)
- **System Design Interview** - Volume 1 & 2 by Alex Xu
- **Machine Learning System Design** - Chip Huyen's course

## ML Fundamentals

### Online Courses
- **Machine Learning** - Andrew Ng (Coursera) - Best starting point
- **Fast.ai Practical Deep Learning** - Hands-on approach
- **Stanford CS229** - In-depth ML theory

### Key Topics to Master
- Supervised Learning: Linear/Logistic Regression, Decision Trees, SVMs
- Unsupervised Learning: K-Means, Hierarchical Clustering, PCA
- Ensemble Methods: Random Forest, XGBoost, LightGBM
- Evaluation Metrics: Precision, Recall, F1, AUC-ROC, AUC-PR
- Feature Engineering: Normalization, Encoding, Selection
- Regularization: L1, L2, Dropout

### Practice Datasets
- **UCI ML Repository** - Classic datasets
- **Kaggle Datasets** - Real-world data
- **OpenML** - Curated ML datasets

## Deep Learning

### Courses
- **Deep Learning Specialization** - Andrew Ng (Coursera) - 5-course series
- **CS231n** - Stanford's CNN course - Computer vision focus
- **CS224n** - Stanford's NLP course - Natural language processing
- **Fast.ai** - Practical deep learning for coders

### Frameworks
- **PyTorch** - Preferred for research, increasingly in production
  - Official tutorials
  - PyTorch Lightning for production code
- **TensorFlow/Keras** - Still widely used in industry
  - TensorFlow tutorials
  - Keras documentation

### Key Architectures to Know
- **CNNs**: ResNet, VGG, EfficientNet, Vision Transformer
- **RNNs**: LSTM, GRU, Bidirectional RNNs
- **Transformers**: BERT, GPT, T5, Vision Transformers
- **GANs**: DCGAN, StyleGAN, Pix2Pix

## NLP & LLMs

### Fundamentals
- **Speech and Language Processing** (Jurafsky & Martin) - NLP bible
- **Hugging Face Course** - Transformers and modern NLP
- **CS224n** - Stanford NLP with Deep Learning

### LLM-Specific
- **OpenAI Cookbook** - Practical LLM examples
- **Anthropic's Claude documentation** - Claude API best practices
- **LangChain Documentation** - Building LLM applications
- **Prompt Engineering Guide** - Comprehensive prompting techniques

### Key Concepts
- Word Embeddings: Word2Vec, GloVe, FastText
- Transformers: Attention mechanism, BERT, GPT architecture
- Fine-tuning: LoRA, QLoRA, Prefix tuning
- RAG: Retrieval-Augmented Generation patterns
- Prompt Engineering: Few-shot, chain-of-thought
- LLM Evaluation: BLEU, ROUGE, BERTScore

### Tools & Libraries
- **Hugging Face Transformers** - Pre-trained models
- **LangChain** - LLM application framework
- **LlamaIndex** - Data indexing for LLMs
- **Vector Databases**: Pinecone, Weaviate, ChromaDB
- **Sentence Transformers** - Semantic embeddings

## ML System Design

### Resources
- **Machine Learning Systems Design** - Chip Huyen's book
- **Designing ML Systems** - Comprehensive ML system patterns
- **AWS ML Architecture Best Practices**
- **Google's Rules of ML** - 43 rules for production ML

### Key Topics
- **Feature Stores**: Tecton, Feast documentation
- **Model Serving**: TensorFlow Serving, TorchServe, FastAPI
- **Experiment Tracking**: MLflow, Weights & Biases
- **A/B Testing**: Statistical significance, experiment design
- **Monitoring**: Data drift, model performance, Prometheus

### Company Engineering Blogs
- **Netflix Tech Blog** - Recommendation systems, A/B testing
- **Uber Engineering** - Michelangelo platform, ML infrastructure
- **Airbnb Engineering** - Search ranking, pricing models
- **Meta Engineering** - Feed ranking, recommendation systems
- **Google AI Blog** - Research and production systems
- **LinkedIn Engineering** - Professional network recommendations
- **Spotify Engineering** - Music recommendations, discover weekly

## MLOps & Production

### Courses
- **Made With ML** - MLOps fundamentals
- **Full Stack Deep Learning** - Production ML systems
- **AWS ML Specialty** - Cloud ML infrastructure

### Tools to Learn
- **Docker & Kubernetes** - Containerization and orchestration
- **CI/CD**: GitHub Actions, Jenkins
- **Cloud Platforms**: AWS SageMaker, GCP Vertex AI, Azure ML
- **Feature Stores**: Feast, Tecton
- **Model Registry**: MLflow, W&B
- **Monitoring**: Evidently AI, Whylabs, Prometheus

### Best Practices
- Model versioning and reproducibility
- CI/CD for ML pipelines
- A/B testing methodology
- Monitoring data drift and model performance
- Automated retraining pipelines
- Shadow mode deployment
- Gradual rollout strategies

## Practice Platforms

### Kaggle
- Competitions: Practice on real problems
- Datasets: Find data for projects
- Notebooks: Learn from top solutions
- Recommended competitions:
  - Titanic (beginner)
  - House Prices (regression)
  - MNIST (computer vision)
  - NLP with Disaster Tweets (text classification)

### Interview Practice
- **Pramp** - Mock ML interviews with peers
- **Interviewing.io** - Anonymous technical interviews
- **LeetCode** - ML-specific problems

### Project Ideas for Portfolio
1. **Recommendation System**: Build Netflix/Spotify-like recommender
2. **NLP Project**: Sentiment analysis, text classification
3. **Computer Vision**: Object detection, image classification
4. **Time Series**: Stock prediction, demand forecasting
5. **LLM Application**: RAG system, chatbot, text-to-SQL

## Books

### ML Fundamentals
- **Hands-On Machine Learning** (Géron) - Practical scikit-learn & TensorFlow
- **Pattern Recognition and Machine Learning** (Bishop) - Theoretical depth
- **The Elements of Statistical Learning** (Hastie et al.) - Classic, math-heavy

### Deep Learning
- **Deep Learning** (Goodfellow et al.) - Comprehensive theory
- **Deep Learning with Python** (Chollet) - Practical Keras
- **Dive into Deep Learning** - Interactive, PyTorch/TensorFlow

### NLP
- **Speech and Language Processing** (Jurafsky & Martin) - NLP fundamentals
- **Natural Language Processing with Transformers** - Modern NLP
- **Practical Natural Language Processing** - Applied NLP

### ML System Design
- **Designing Machine Learning Systems** (Chip Huyen) - Must-read for ML engineering
- **Machine Learning Design Patterns** (Lakshmanan et al.) - Practical patterns
- **Building Machine Learning Powered Applications** (Ameisen)

### MLOps
- **Introducing MLOps** (Gift & Deza) - ML in production
- **ML Engineering** (Andriy Burkov) - End-to-end ML systems

## YouTube Channels

### ML Fundamentals
- **StatQuest with Josh Starmer** - Clear explanations of ML concepts
- **3Blue1Brown** - Mathematical intuition (neural networks series)
- **Andrej Karpathy** - Deep learning from scratch

### Practical ML
- **Sentdex** - Python ML tutorials
- **Two Minute Papers** - Research paper summaries
- **Yannic Kilcher** - Deep dives into papers

### Interview Prep
- **CS Dojo** - Interview preparation
- **Back To Back SWE** - Algorithms and data structures
- **Clement Mihailescu** - AlgoExpert founder, interview tips

### Company Tech Talks
- **Google Cloud Tech** - GCP ML tools
- **AWS Events** - AWS ML services
- **Netflix Engineering** - Production ML systems

## Research Papers

### Must-Read Classics
- **Attention Is All You Need** (2017) - Transformer architecture
- **BERT: Pre-training of Deep Bidirectional Transformers** (2018)
- **ImageNet Classification with Deep CNNs** (2012) - AlexNet
- **Deep Residual Learning** (2015) - ResNets
- **Generative Adversarial Networks** (2014) - GANs

### Recommendation Systems
- **Deep Neural Networks for YouTube Recommendations** (2016)
- **Wide & Deep Learning for Recommender Systems** (2016)
- **Neural Collaborative Filtering** (2017)

### NLP & LLMs
- **GPT-3: Language Models are Few-Shot Learners** (2020)
- **RLHF: Learning to summarize from human feedback** (2020)
- **RAG: Retrieval-Augmented Generation** (2020)
- **LoRA: Low-Rank Adaptation of Large Language Models** (2021)

### Computer Vision
- **You Only Look Once (YOLO)** - Object detection
- **Mask R-CNN** - Instance segmentation
- **Vision Transformer (ViT)** - Transformers for vision

### ML Systems
- **Hidden Technical Debt in Machine Learning Systems** (2015) - Must-read
- **TFX: A TensorFlow-Based Production-Scale ML Platform** (2017)
- **Michelangelo: Uber's ML Platform** (2019)

### Where to Find Papers
- **arXiv.org** - Preprints of research papers
- **Papers With Code** - Papers with implementations
- **Google Scholar** - Search academic papers
- **Hugging Face Papers** - NLP/LLM papers

## Stay Updated

### Newsletters
- **The Batch** (DeepLearning.AI) - Weekly AI news
- **Import AI** - Jack Clark's AI newsletter
- **TLDR AI** - Daily AI news
- **The Algorithm** (MIT Tech Review)

### Podcasts
- **Practical AI** - Applied ML discussions
- **Machine Learning Street Talk** - Technical deep dives
- **The TWIML AI Podcast** - Industry trends

### Communities
- **r/MachineLearning** (Reddit) - Research discussions
- **r/LearnMachineLearning** (Reddit) - Learning resources
- **Hugging Face Forums** - NLP/LLM discussions
- **MLOps Community** - Production ML practices
- **Kaggle Forums** - Competition discussions

### Conferences (Follow Online)
- **NeurIPS** - Premier ML conference
- **ICML** - International ML conference
- **ICLR** - Deep learning focused
- **ACL/EMNLP** - NLP conferences
- **CVPR** - Computer vision
- **KDD** - Data science and mining

## Interview-Specific Tips

### Timeline for Preparation
**3 Months Plan:**
- Month 1: ML fundamentals, practice coding
- Month 2: Deep learning, system design
- Month 3: Mock interviews, case studies

**1 Month Plan:**
- Week 1: Review ML concepts, practice top 50 LeetCode
- Week 2: System design patterns, case studies
- Week 3: Mock interviews, review weak areas
- Week 4: Final review, relaxation

### Company-Specific Resources
- **Meta (Facebook)**: Focus on feed ranking, recommendation systems
- **Google**: Search, ads, large-scale ML
- **Amazon**: Personalization, demand forecasting
- **Netflix**: Recommendation systems, A/B testing
- **Uber**: Maps, ETA prediction, pricing
- **Airbnb**: Search ranking, pricing, fraud detection

### What to Prioritize
**For FAANG:**
1. ML System Design (critical)
2. ML Fundamentals (must-have)
3. Coding (LeetCode medium level)
4. ML Theory (probability, statistics)

**For AI Startups:**
1. Hands-on ML skills (critical)
2. LLM/NLP knowledge (often required)
3. Prototyping speed (build things fast)
4. ML Fundamentals (must-have)

## Final Tips

1. **Build Projects**: Portfolio > certifications
2. **Mock Interviews**: Practice explaining your thought process
3. **Stay Current**: LLM/GenAI is hot right now (2024-2025)
4. **Deep > Broad**: Better to know fewer things deeply
5. **Practice Communication**: Explaining > knowing
6. **Real-World Focus**: Production ML > research
7. **Learn from Failures**: Review wrong answers, failed interviews

Good luck with your preparation!
