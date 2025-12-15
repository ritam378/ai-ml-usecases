# AI Use Cases Repository - Quick Reference

**Last Updated:** December 15, 2025
**Status:** 52% Complete - 2 Case Studies Ready for Interviews

---

## ✅ What's Complete and Ready to Use

### 1. Text-to-SQL System
**Location:** `08_generative-ai-llms/01_text-to-sql/`

```bash
# Navigate to the case study
cd 08_generative-ai-llms/01_text-to-sql

# View the implementation
ls src/                     # schema_manager.py, query_generator.py, etc.

# Run tests
pytest tests/ -v

# Explore notebooks
jupyter notebook notebooks/
```

**Interview Talking Points:**
- LLM system design and architecture
- Prompt engineering strategies
- Cost optimization techniques
- Security considerations (SQL injection prevention)
- Performance optimization (caching, model routing)

**Key Files:**
- [Problem Statement](08_generative-ai-llms/01_text-to-sql/problem_statement.md)
- [Solution Approach](08_generative-ai-llms/01_text-to-sql/solution_approach.md)
- [Source Code](08_generative-ai-llms/01_text-to-sql/src/)

---

### 2. Sentiment Analysis System
**Location:** `04_nlp/01_sentiment-analysis/`

```bash
# Navigate to the case study
cd 04_nlp/01_sentiment-analysis

# Generate synthetic data
python3 src/data_generator.py

# Run tests
pytest tests/ -v

# Explore notebooks
jupyter notebook notebooks/
```

**Interview Talking Points:**
- NLP preprocessing pipeline design
- Model selection (DistilBERT vs BERT vs RoBERTa)
- Inference optimization (caching, batching)
- Confidence calibration
- Production deployment strategies

**Key Files:**
- [Problem Statement](04_nlp/01_sentiment-analysis/problem_statement.md)
- [Solution Approach](04_nlp/01_sentiment-analysis/solution_approach.md)
- [Source Code](04_nlp/01_sentiment-analysis/src/)

---

## 🛠️ Shared Utilities

**Location:** `common/`

Available utilities that work across all projects:

```python
# Data validation
from common.data_validation import validate_input, validate_schema

# Model base classes
from common.model_base import BaseMLModel, ModelConfig, PredictionResult

# Metrics
from common.metrics import ClassificationMetrics, RegressionMetrics, MetricTracker
```

**Usage:**
```python
# Example: Using the model base class
from common.model_base import BaseMLModel

class MyModel(BaseMLModel):
    def train(self, X, y):
        # Your training logic
        pass
    
    def predict(self, X):
        # Your prediction logic
        pass
```

---

## 🧪 Running Tests

### All Tests
```bash
# From repository root
make test
```

### Specific Case Study Tests
```bash
# Text-to-SQL tests
cd 08_generative-ai-llms/01_text-to-sql
pytest tests/ -v --cov=src --cov-report=html

# Sentiment Analysis tests
cd 04_nlp/01_sentiment-analysis
pytest tests/ -v --cov=src --cov-report=html
```

### Code Quality Checks
```bash
# Linting
make lint

# Type checking
make type-check

# Format code
make format
```

---

## 📓 Jupyter Notebooks

### Text-to-SQL Notebooks
1. **Exploratory Analysis:** Schema exploration, query patterns
2. **Prompt Engineering:** Template optimization, few-shot examples
3. **Evaluation:** Performance metrics, cost analysis

### Sentiment Analysis Notebooks
1. **Exploratory Analysis:** Data distribution, text characteristics
2. **Model Training:** DistilBERT setup, training workflow
3. **Evaluation:** Performance metrics, optimization strategies

**Launch Notebooks:**
```bash
jupyter notebook <case-study-path>/notebooks/
```

---

## 📊 Key Statistics

| Metric | Value |
|--------|-------|
| Production Code | ~5,500 lines |
| Test Code | ~2,770 lines |
| Total Tests | 260+ |
| Test Coverage | >80% |
| Jupyter Notebooks | 6 |
| Notebook Cells | ~530 |
| Documentation | 70,000+ words |

---

## 🎯 Interview Preparation Checklist

### Before the Interview
- [ ] Read problem statements for both case studies
- [ ] Review solution approaches and trade-offs
- [ ] Run Jupyter notebooks to see results
- [ ] Practice explaining design decisions
- [ ] Review test files for edge cases

### During the Interview
- [ ] Start with problem clarification
- [ ] Discuss trade-offs explicitly
- [ ] Mention production considerations
- [ ] Reference test coverage
- [ ] Discuss monitoring and observability

### Topics to Cover
- **System Design:** Architecture, scalability, latency
- **ML/AI:** Model selection, evaluation metrics, optimization
- **Software Engineering:** Testing, error handling, code quality
- **Production:** Deployment, monitoring, cost optimization

---

## 🚀 Quick Start for Each Case Study

### Text-to-SQL
```python
from src.schema_manager import SchemaManager
from src.query_generator import QueryGenerator

# Initialize
schema_mgr = SchemaManager('data/sample_database.db')
query_gen = QueryGenerator(schema_manager=schema_mgr)

# Generate SQL
result = query_gen.generate(
    "What are the top 5 customers by total order value?"
)
print(result.sql_query)
```

### Sentiment Analysis
```python
from src.sentiment_predictor import SentimentPredictor

# Initialize
predictor = SentimentPredictor()

# Predict
result = predictor.predict("This product is amazing!")
print(f"Sentiment: {result.sentiment}")
print(f"Confidence: {result.confidence:.2f}")
```

---

## 📚 Documentation Structure

Each case study has:
```
case-study/
├── README.md                   # Overview
├── problem_statement.md        # Business context, requirements
├── solution_approach.md        # Architecture, design decisions
├── evaluation.md               # Metrics, results
├── trade_offs.md              # Design trade-offs
├── interview_questions.md     # Common Q&A
├── requirements.txt           # Dependencies
├── src/                       # Source code
├── tests/                     # Test suite
├── notebooks/                 # Jupyter notebooks
└── data/                      # Sample data
```

---

## 🔗 Important Links

- **Main README:** [README.md](README.md)
- **Progress Tracking:** [PROGRESS_SUMMARY.md](PROGRESS_SUMMARY.md)
- **Implementation Status:** [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)
- **Final Summary:** [FINAL_SUMMARY.md](FINAL_SUMMARY.md)
- **Interview Guide:** [docs/interview-framework.md](docs/interview-framework.md)
- **System Design Guide:** [docs/ml-system-design-guide.md](docs/ml-system-design-guide.md)

---

## ⚡ Common Commands

```bash
# Installation
make install          # Install dependencies
make install-dev      # Install dev dependencies + pre-commit hooks

# Testing
make test            # Run all tests
make test-cov        # Run tests with coverage report

# Code Quality
make lint            # Run linters (flake8, pylint)
make format          # Format code (black, isort)
make type-check      # Run type checker (mypy)

# Cleanup
make clean           # Remove build artifacts
make clean-test      # Remove test artifacts
```

---

## 💡 Tips for Interview Success

1. **Understand the Why**
   - Don't just explain what the code does
   - Explain why you made specific design choices
   - Discuss alternative approaches considered

2. **Show Production Awareness**
   - Mention monitoring and observability
   - Discuss error handling strategies
   - Talk about scalability considerations

3. **Demonstrate Testing Knowledge**
   - Explain your testing strategy
   - Discuss edge cases covered
   - Mention test coverage goals

4. **Discuss Trade-offs**
   - Every decision has trade-offs
   - Be explicit about what you optimized for
   - Acknowledge limitations

5. **Use Concrete Examples**
   - Reference specific code from these projects
   - Share metrics and results
   - Discuss real challenges faced

---

**Ready to ace your ML interviews!** 🎯

For more details, see [FINAL_SUMMARY.md](FINAL_SUMMARY.md)
