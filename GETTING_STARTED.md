# Getting Started with AI Case Studies

Welcome! This guide will help you navigate the repository and make the most of these case studies for interview preparation.

## Quick Navigation

### By Your Goal

**🎯 Preparing for FAANG Interviews?**
- Start with: [ML System Design Guide](docs/ml-system-design-guide.md)
- Focus on: Categories 7 (ML System Design) and 8 (Generative AI)
- Practice: Text-to-SQL, RAG System, End-to-End Pipeline

**🎯 Preparing for AI Startup Interviews?**
- Start with: [Interview Framework](docs/interview-framework.md)
- Focus on: Categories 4 (NLP) and 8 (Generative AI)
- Practice: All LLM case studies, NLP projects

**🎯 Learning ML from Scratch?**
- Start with: Category 6 (Clustering) - Customer Segmentation
- Then: Category 2 (Classification) - Spam Detection
- Build up to: More complex case studies

**🎯 Building Your Portfolio?**
- Pick 3-4 case studies from different categories
- Complete them fully (code + tests + documentation)
- Deploy at least one (add link to README)

### By Time Available

**📅 1 Week Crash Course**
1. Day 1-2: Read [ML Interview Framework](docs/interview-framework.md)
2. Day 3-4: Complete 1 case study from your target area
3. Day 5-6: Review [ML System Design Guide](docs/ml-system-design-guide.md)
4. Day 7: Mock interview practice

**📅 1 Month Preparation**
- Week 1: Fundamentals (Customer Segmentation, Spam Detection)
- Week 2: Advanced ML (Fraud Detection, Sentiment Analysis)
- Week 3: System Design (End-to-End Pipeline, A/B Testing)
- Week 4: LLMs (Text-to-SQL, RAG System)

**📅 3 Month Deep Dive**
- Month 1: Complete 6-8 case studies across categories 1-4
- Month 2: Complete 4-6 case studies from categories 5-7
- Month 3: Deep dive into Generative AI (all 10 LLM case studies)

## Repository Structure

```
ai-usecases/
├── docs/                              # 📚 Learning guides
│   ├── interview-framework.md         # ML interview approach
│   ├── ml-system-design-guide.md      # System design patterns
│   ├── glossary.md                    # ML terminology
│   └── resources.md                   # Learning resources
│
├── templates/                         # 📋 Templates for new case studies
│
├── 01_recommendation-systems/         # 3 case studies
├── 02_classification/                 # 3 case studies
├── 03_time-series-forecasting/        # 2 case studies
├── 04_nlp/                           # 3 case studies
├── 05_computer-vision/                # 2 case studies
├── 06_clustering-unsupervised/        # 2 case studies
├── 07_ml-system-design/               # 3 case studies
└── 08_generative-ai-llms/             # 10 case studies ⭐
```

## How to Use Each Case Study

### 1. Start with README.md
- Understand the business problem
- Check difficulty level and time required
- Review learning objectives

### 2. Read problem_statement.md
- Deep dive into business context
- Understand requirements and constraints
- Study success metrics

### 3. Study solution_approach.md
- Learn the architecture
- Understand design decisions
- Review trade-offs

### 4. Explore Notebooks
- Run exploratory analysis notebook
- Understand feature engineering
- See model training process

### 5. Review Source Code
- Read production-quality code
- Understand best practices
- Note code organization

### 6. Run Tests
```bash
cd [case-study-directory]
pip install -r requirements.txt
pytest tests/ -v
```

### 7. Study interview_questions.md
- Review common follow-up questions
- Practice explaining your approach
- Prepare for deep dives

## Recommended Learning Paths

### Path 1: ML Generalist (FAANG)
Focus on breadth and system design.

**Week 1-2: Foundations**
- Customer Segmentation (Beginner)
- Spam Detection (Intermediate)

**Week 3-4: Core ML**
- Fraud Detection (Imbalanced data)
- Sentiment Analysis (NLP)
- Demand Forecasting (Time series)

**Week 5-6: System Design**
- End-to-End ML Pipeline
- A/B Testing Framework
- Model Serving at Scale

**Week 7-8: Modern ML**
- Text-to-SQL (LLMs)
- RAG System (LLMs + Vector DBs)
- LLM Evaluation Pipeline

**Total Time**: 8 weeks, 8 case studies

### Path 2: LLM/NLP Specialist (AI Startups)
Focus on depth in NLP and Generative AI.

**Week 1: NLP Fundamentals**
- Sentiment Analysis
- Document Similarity

**Week 2-3: Advanced NLP**
- Named Entity Recognition
- Fake News Detection (Multi-modal)

**Week 4-8: LLM Deep Dive**
- Text-to-SQL
- RAG System
- Chatbot with Intent Classification
- Prompt Engineering Framework
- Structured Data Extraction
- Semantic Search
- Multi-Agent System
- LLM Evaluation

**Total Time**: 8 weeks, 12 case studies

### Path 3: Computer Vision Specialist
Focus on CV with ML engineering.

**Week 1-2: Fundamentals**
- Image Classification with Transfer Learning
- Customer Segmentation (for comparison)

**Week 3-4: Advanced CV**
- Object Detection (YOLO/R-CNN)

**Week 5-6: System Design**
- End-to-End ML Pipeline (adapt for CV)
- Model Serving at Scale (CV-specific)

**Week 7-8: Production**
- Real-time Personalization
- A/B Testing Framework

**Total Time**: 8 weeks, 7 case studies

### Path 4: Quick Interview Prep (1 Week)
Maximize impact with limited time.

**Day 1: Theory**
- Read ML Interview Framework
- Review ML System Design Guide
- Study Glossary

**Day 2-3: One Complete Case Study**
- Choose from your focus area
- Complete all components
- Practice explaining it

**Day 4-5: System Design**
- Study End-to-End ML Pipeline
- Review Text-to-SQL (for LLM knowledge)
- Practice drawing architectures

**Day 6-7: Mock Interviews**
- Practice with interview_questions.md
- Do mock interviews (Pramp, Interviewing.io)
- Review and iterate

## Tips for Success

### Do's ✅

1. **Understand, Don't Memorize**
   - Focus on why, not just what
   - Understand trade-offs
   - Be able to explain alternatives

2. **Code Along**
   - Don't just read, implement
   - Modify and experiment
   - Break things and fix them

3. **Practice Explaining**
   - Explain to a rubber duck
   - Record yourself
   - Do mock interviews

4. **Connect to Real World**
   - Think about production deployment
   - Consider scale and cost
   - Discuss monitoring and maintenance

5. **Review Multiple Times**
   - First pass: Understand
   - Second pass: Implement
   - Third pass: Optimize

### Don'ts ❌

1. **Don't Skip Fundamentals**
   - Don't jump to LLMs without ML basics
   - Interviewers will test foundations

2. **Don't Just Read**
   - Must write code
   - Must run experiments
   - Must understand results

3. **Don't Ignore System Design**
   - Production matters more than accuracy
   - Interviewers care about deployment
   - Scale is always important

4. **Don't Memorize Solutions**
   - Understand the approach
   - Be ready for variations
   - Know multiple solutions

## Case Study Status

### ✅ Complete Examples
- **Text-to-SQL** (08_generative-ai-llms/01_text-to-sql)
  - Full documentation
  - Complete source code
  - Comprehensive explanations
  - Use this as a reference!

### 🚧 In Progress
- All other case studies have structure and placeholders
- README files provide overview and context
- You can contribute to completing them!

### How to Contribute

Want to help complete case studies?

1. Pick a case study from the structure
2. Follow the Text-to-SQL example
3. Implement documentation, code, and tests
4. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Common Interview Questions by Category

### Recommendation Systems
- "Design YouTube's recommendation system"
- "How would you handle cold start problem?"
- "Explain collaborative filtering vs content-based filtering"

### Classification
- "How do you handle imbalanced datasets?"
- "Design a fraud detection system"
- "Explain precision-recall trade-off"

### LLMs/Generative AI
- "Design a Text-to-SQL system"
- "How would you reduce LLM hallucinations?"
- "Explain RAG architecture"
- "How do you evaluate LLM outputs?"

### ML System Design
- "Design an end-to-end ML pipeline"
- "How would you serve models at scale?"
- "Explain A/B testing for ML models"

## Interview Preparation Checklist

### Technical Knowledge
- [ ] Understand core ML algorithms
- [ ] Know common evaluation metrics
- [ ] Understand feature engineering
- [ ] Know how to handle imbalanced data
- [ ] Understand regularization techniques
- [ ] Know ensemble methods
- [ ] Understand neural networks
- [ ] Know transformer architecture (for NLP/LLM roles)

### System Design
- [ ] Can draw end-to-end ML architecture
- [ ] Understand feature stores
- [ ] Know model serving patterns
- [ ] Understand monitoring and retraining
- [ ] Can discuss scale and performance
- [ ] Know A/B testing methodology

### Coding
- [ ] Can implement ML algorithms from scratch
- [ ] Comfortable with pandas/numpy
- [ ] Know scikit-learn API
- [ ] Can use PyTorch or TensorFlow
- [ ] Write clean, documented code
- [ ] Can write unit tests

### Communication
- [ ] Can explain technical concepts simply
- [ ] Can discuss trade-offs clearly
- [ ] Can ask clarifying questions
- [ ] Can think out loud
- [ ] Can handle follow-up questions

## Resources

### Within This Repo
- [Interview Framework](docs/interview-framework.md) - Interview approach
- [ML System Design Guide](docs/ml-system-design-guide.md) - System patterns
- [Glossary](docs/glossary.md) - ML terminology
- [Resources](docs/resources.md) - External resources

### External Resources
- **Courses**: Fast.ai, DeepLearning.AI, Stanford CS229
- **Books**: "Designing ML Systems" (Chip Huyen), "Hands-On ML" (Géron)
- **Practice**: LeetCode, Kaggle, Mock interviews
- **Blogs**: Company engineering blogs (Netflix, Uber, Airbnb)

## Getting Help

### Questions?
- Open an issue on GitHub
- Check existing discussions
- Review the documentation

### Found a Bug?
- Report it in GitHub Issues
- Include case study name
- Describe the problem clearly

### Want to Contribute?
- See [CONTRIBUTING.md](CONTRIBUTING.md)
- Start with a small case study
- Follow the Text-to-SQL example

## Next Steps

1. **Choose Your Path**: Pick a learning path above
2. **Set a Schedule**: Commit specific hours per week
3. **Start Small**: Begin with one case study
4. **Practice Explaining**: Do mock interviews
5. **Iterate**: Review, improve, repeat

Good luck with your preparation! 🚀

Remember: The goal isn't to memorize solutions, but to understand approaches and trade-offs. Interviewers want to see how you think, not what you've memorized.

---

**Last Updated**: December 2025
**Contributors**: Welcome!
**License**: MIT
