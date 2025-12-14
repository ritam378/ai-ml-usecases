# AI Case Studies for Interviews

A comprehensive collection of AI/ML case studies with production-quality Python implementations, designed for FAANG/Big Tech and startup interview preparation at senior level.

## Overview

This repository contains **28 complete case studies** across 8 AI/ML categories, each with:
- Detailed problem statements and business context
- Complete Python implementations with type hints
- Jupyter notebooks for exploratory analysis
- Unit tests with >80% coverage
- Interview questions and answers
- System design considerations
- Trade-off analyses

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

## Case Studies

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

- **Production-Quality Code**: Type hints, comprehensive docstrings, PEP 8 compliant
- **Complete Documentation**: Theory balanced with practice, senior-level depth
- **Real Interview Scenarios**: Based on actual FAANG and startup interviews
- **Reproducible Results**: Small sample datasets, pinned dependencies
- **System Design Focus**: Scalability, latency, cost considerations
- **LLM Best Practices**: Prompt engineering, cost optimization, evaluation

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

**Last Updated**: December 2025
**Python Version**: 3.9+
**Status**: Active Development
