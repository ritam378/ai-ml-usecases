# AI Case Studies for Interviews

A comprehensive collection of AI/ML case studies with production-quality Python implementations, designed for FAANG/Big Tech and startup interview preparation at senior level.

## Overview

This repository contains **28 case studies** across 8 AI/ML categories. The repository provides a comprehensive framework with complete documentation, structured templates, and **two complete reference implementations** with production-quality code.

**Current Status:**
- ✅ Complete documentation framework (interview guides, glossary, resources)
- ✅ Production-ready infrastructure (testing, linting, pre-commit hooks)
- ✅ Shared utilities library (validation, metrics, base classes)
- ✅ **Text-to-SQL case study: 100% Complete** (code, tests, notebooks, data)
- ✅ **Sentiment Analysis case study: 100% Complete** (code, tests, notebooks, data)
- 📋 26 case studies: Structured and ready for implementation

**Implementation Quality:**
- ~7,000 lines of production Python code
- 260+ unit and integration tests (>80% coverage)
- 6 comprehensive Jupyter notebooks
- Type hints and Google-style docstrings throughout

**Each case study includes:**
- Comprehensive problem statement template
- Solution approach framework
- Structured directories for code, tests, notebooks, and data
- Documentation templates (evaluation, trade-offs, interview Q&A)
- Requirements specifications

## Repository Structure

```
ai-usecases/
├── 01_recommendation-systems/     # 3 case studies
├── 02_classification/             # 3 case studies
├── 03_time-series-forecasting/    # 2 case studies
├── 04_nlp/                        # 3 case studies
├── 05_computer-vision/            # 2 case studies
├── 06_clustering-unsupervised/    # 2 case studies
├── 07_ml-system-design/           # 3 case studies
├── 08_generative-ai-llms/         # 10 case studies
├── docs/                          # Guides and resources
└── templates/                     # Templates for new case studies
```

## ✅ Completed Case Studies

### 1. Text-to-SQL: Natural Language to Database Queries
**Location:** `08_generative-ai-llms/01_text-to-sql/`
**Status:** ✅ 100% Complete

Production-ready LLM-powered SQL generation system:
- Schema management with intelligent table selection
- Prompt engineering with few-shot examples
- Query validation and security checks
- Support for OpenAI and Anthropic models
- Comprehensive test suite (125+ tests, >80% coverage)
- 3 Jupyter notebooks (EDA, prompt engineering, evaluation)

**Key Features:**
- Automatic schema extraction and formatting
- Model routing based on query complexity
- Retry logic with error feedback
- Result caching for performance
- Cost optimization strategies

### 2. Sentiment Analysis at Scale
**Location:** `04_nlp/01_sentiment-analysis/`
**Status:** ✅ 100% Complete

DistilBERT-based sentiment classification system:
- Production-ready text preprocessing pipeline
- Fast inference with result caching
- Batch prediction support
- Synthetic data generation
- Comprehensive test suite (135+ tests, >85% coverage)
- 3 Jupyter notebooks (EDA, training, evaluation)

**Key Features:**
- URL/HTML removal and text normalization
- Confidence-based prediction filtering
- Performance optimization (batching, caching)
- Model save/load functionality
- Production deployment guidelines

---

## 📋 All Case Studies

### 1. Recommendation Systems (3 case studies)
1. **Netflix Movie Recommendations** - Collaborative Filtering + Matrix Factorization
2. **E-commerce Product Recommendations** - Hybrid approach (Content + Collaborative)
3. **Real-time Personalization System** - System design focus

### 2. Classification (3 case studies)
1. **Credit Card Fraud Detection** - Imbalanced data handling
2. **Spam Detection with ML Pipeline** - End-to-end production system
3. **Fake News Detection** - NLP + Graph features

### 3. Time Series & Forecasting (2 case studies)
1. **Demand Forecasting for E-commerce** - Multi-model approach
2. **Anomaly Detection in System Metrics** - Real-time monitoring

### 4. Natural Language Processing (3 case studies)
1. **Sentiment Analysis at Scale** - Twitter/Reviews processing
2. **Named Entity Recognition** - Information extraction
3. **Document Similarity & Search** - Embeddings-based approach

### 5. Computer Vision (2 case studies)
1. **Object Detection for Autonomous Vehicles** - YOLO/R-CNN implementation
2. **Image Classification with Transfer Learning** - Production deployment

### 6. Clustering & Unsupervised Learning (2 case studies)
1. **Customer Segmentation** - E-commerce RFM analysis
2. **Anomaly Detection in User Behavior** - Unsupervised methods

### 7. ML System Design (3 case studies)
1. **End-to-End ML Pipeline Design** - Feature store, serving, monitoring
2. **A/B Testing Framework for ML Models** - Statistical rigor + deployment
3. **Model Serving at Scale** - Latency optimization strategies

### 8. Generative AI & LLMs (10 case studies)
1. **Text-to-SQL: Natural Language to Database Queries** - Schema understanding, query optimization
2. **RAG System for Enterprise Knowledge Base** - Vector DB + retrieval strategies
3. **Chatbot with Intent Classification** - Multi-turn conversations, context management
4. **Fine-tuning LLMs for Domain-Specific Tasks** - LoRA/QLoRA, code generation
5. **Prompt Engineering Framework** - Systematic optimization, few-shot learning
6. **LLM-based Content Moderation System** - Safety + performance trade-offs
7. **Structured Data Extraction from Unstructured Text** - Function calling, schema validation
8. **LLM-powered Semantic Search** - Hybrid search, re-ranking
9. **Multi-Agent LLM System** - Agent orchestration, tool use
10. **LLM Evaluation & Monitoring Pipeline** - Metrics, cost tracking, drift detection

## Learning Paths

### For Interview Preparation
1. **Week 1-2**: Focus on Categories 1-3 (Recommendations, Classification, Time Series)
2. **Week 3-4**: Deep dive into Categories 4-6 (NLP, Computer Vision, Clustering)
3. **Week 5-6**: Master Category 7 (ML System Design)
4. **Week 7-8**: Comprehensive study of Category 8 (Generative AI & LLMs)

### By Experience Level
- **Senior Engineers (5+ years)**: Focus on system design aspects, trade-offs, and production considerations
- **Staff/Principal**: Emphasize Categories 7-8, architectural decisions, and scalability

### By Company Type
- **FAANG/Big Tech**: All categories with emphasis on scale (Categories 7-8)
- **AI Startups**: Deep focus on Categories 4, 8 (NLP and Generative AI)
- **General Tech**: Balanced coverage with Categories 1, 2, 7

## Quick Start

### Prerequisites
```bash
python >= 3.9
pip >= 21.0
```

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd ai-usecases

# Install dependencies for a specific case study
cd 01_recommendation-systems/01_netflix-recommendations
pip install -r requirements.txt

# Run notebooks
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

### Running Tests
```bash
# From any case study directory
pytest tests/ -v --cov=src
```

## Case Study Structure

Each case study follows a consistent structure:

```
case-study-name/
├── README.md                      # Overview and learning objectives
├── problem_statement.md           # Business context and problem definition
├── solution_approach.md           # Architecture and methodology
├── data/                          # Sample datasets
│   ├── sample_data.csv
│   └── data_description.md
├── notebooks/                     # Jupyter notebooks
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── src/                           # Production code
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model.py
│   └── utils.py
├── tests/                         # Unit tests
│   ├── test_data_preprocessing.py
│   ├── test_feature_engineering.py
│   └── test_model.py
├── evaluation.md                  # Results and metrics
├── trade_offs.md                  # Design decisions
├── interview_questions.md         # Common follow-up questions
└── requirements.txt               # Dependencies
```

## Key Features

- **Comprehensive Framework**: Complete structure for 28 AI/ML case studies
- **Interview-Ready Documentation**: ML interview framework, system design guide, glossary
- **Production Focus**: Emphasis on scalability, latency, cost, and deployment
- **Reference Implementation**: Text-to-SQL case study in active development
- **LLM-Heavy Content**: 10 Generative AI case studies including RAG, fine-tuning, agents
- **Structured Templates**: Consistent format across all case studies for easy learning
- **Real Interview Scenarios**: Based on actual FAANG and startup interview questions

## Technologies Used

### Traditional ML
- scikit-learn, XGBoost, LightGBM
- PyTorch, TensorFlow/Keras
- pandas, numpy, polars

### Generative AI & LLMs
- OpenAI API, Anthropic Claude
- Hugging Face Transformers
- LangChain, LlamaIndex
- Vector DBs: Pinecone, ChromaDB, Weaviate, FAISS

### MLOps & Production
- MLflow, Weights & Biases
- FastAPI, Flask
- Docker, Kubernetes
- Prometheus, Grafana

### Data Visualization
- matplotlib, seaborn, plotly
- Jupyter notebooks

## Documentation

- [ML Interview Framework](docs/interview-framework.md) - Step-by-step approach to ML interviews
- [ML System Design Guide](docs/ml-system-design-guide.md) - Comprehensive system design patterns
- [Glossary](docs/glossary.md) - AI/ML terminology reference
- [Resources](docs/resources.md) - Curated learning resources

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This repository is designed for educational purposes and interview preparation. Case studies are inspired by real-world problems from industry-leading companies but contain no proprietary information.

## Contact

For questions or feedback, please open an issue on GitHub.

---

## Repository Status

### ✅ Complete & Ready to Use
- **Documentation**: All core guides complete (30,000+ words)
  - [ML Interview Framework](docs/interview-framework.md)
  - [ML System Design Guide](docs/ml-system-design-guide.md)
  - [AI/ML Glossary](docs/glossary.md) (200+ terms)
  - [Learning Resources](docs/resources.md)
- **Structure**: All 28 case studies organized and templated
- **Templates**: Reusable templates for consistent implementation

### 🚧 In Development
- **Text-to-SQL** (08_generative-ai-llms/01_text-to-sql): ~40% complete
  - ✅ Comprehensive problem statement (10,000+ words)
  - ✅ Detailed solution approach (20,000+ words)
  - ✅ Core query generator implementation
  - ✅ Requirements and dependencies
  - 🚧 Additional source modules (in progress)
  - 🚧 Test suite (planned)
  - 🚧 Jupyter notebooks (planned)
  - 🚧 Sample data (planned)

### 📋 Planned
- **27 Case Studies**: Structured with placeholders, ready for implementation
  - Each has README, folder structure, and documentation templates
  - Contributors can use Text-to-SQL as reference

## How to Use This Repository

### For Interview Preparation
1. **Start**: Read [GETTING_STARTED.md](GETTING_STARTED.md)
2. **Learn**: Study [docs/interview-framework.md](docs/interview-framework.md)
3. **Reference**: Review Text-to-SQL case study structure
4. **Practice**: Use the framework to practice ML case studies

### For Learning
1. Study the comprehensive documentation
2. Follow the Text-to-SQL development as a learning example
3. Use templates to implement your own case studies

### For Contributing
1. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines
2. Use Text-to-SQL as the quality standard
3. Pick a case study from the 27 structured ones
4. Submit pull requests with your implementations

## Getting Started

**New here?**
1. Read [GETTING_STARTED.md](GETTING_STARTED.md) for learning paths
2. Review [docs/interview-framework.md](docs/interview-framework.md)
3. Explore the Text-to-SQL case study
4. Check [docs/resources.md](docs/resources.md) for external resources

**Want to see what's been completed?**
- 📊 See [COMPLETION_SUMMARY.md](COMPLETION_SUMMARY.md) for detailed progress report
- ✅ Text-to-SQL case study: 70% complete with working code and database
- ✅ All core documentation: 100% complete (70,000+ words)
- ✅ All 28 case study structures: Ready for implementation

---

**Last Updated**: December 2025
**Python Version**: 3.9+
**Status**: Active Development - Framework Complete, Content In Progress
