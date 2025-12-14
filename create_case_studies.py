#!/usr/bin/env python3
"""
Script to create folder structure for all AI case studies.

This script generates the complete directory structure and placeholder files
for all 28 case studies in the repository.
"""

import os
from pathlib import Path
from typing import List, Dict

# Define all case studies
CASE_STUDIES = {
    "01_recommendation-systems": [
        {
            "name": "01_netflix-recommendations",
            "title": "Netflix Movie Recommendations",
            "description": "Collaborative Filtering + Matrix Factorization for movie recommendations",
            "difficulty": "Intermediate",
            "hours": "5-6"
        },
        {
            "name": "02_ecommerce-recommendations",
            "title": "E-commerce Product Recommendations",
            "description": "Hybrid recommendation system combining content-based and collaborative filtering",
            "difficulty": "Intermediate",
            "hours": "5-6"
        },
        {
            "name": "03_realtime-personalization",
            "title": "Real-time Personalization System",
            "description": "System design for real-time personalized content delivery",
            "difficulty": "Advanced",
            "hours": "6-8"
        }
    ],
    "02_classification": [
        {
            "name": "01_fraud-detection",
            "title": "Credit Card Fraud Detection",
            "description": "Binary classification with severe class imbalance",
            "difficulty": "Intermediate",
            "hours": "4-5"
        },
        {
            "name": "02_spam-detection",
            "title": "Spam Detection ML Pipeline",
            "description": "End-to-end spam classification system with deployment",
            "difficulty": "Intermediate",
            "hours": "4-5"
        },
        {
            "name": "03_fake-news-detection",
            "title": "Fake News Detection",
            "description": "Multi-modal classification using NLP and graph features",
            "difficulty": "Advanced",
            "hours": "6-7"
        }
    ],
    "03_time-series-forecasting": [
        {
            "name": "01_demand-forecasting",
            "title": "E-commerce Demand Forecasting",
            "description": "Multi-step time series forecasting for inventory management",
            "difficulty": "Intermediate",
            "hours": "5-6"
        },
        {
            "name": "02_anomaly-detection",
            "title": "System Metrics Anomaly Detection",
            "description": "Real-time anomaly detection in time series data",
            "difficulty": "Advanced",
            "hours": "5-6"
        }
    ],
    "04_nlp": [
        {
            "name": "01_sentiment-analysis",
            "title": "Sentiment Analysis at Scale",
            "description": "Large-scale sentiment classification for social media/reviews",
            "difficulty": "Intermediate",
            "hours": "4-5"
        },
        {
            "name": "02_named-entity-recognition",
            "title": "Named Entity Recognition",
            "description": "Information extraction from unstructured text",
            "difficulty": "Advanced",
            "hours": "5-6"
        },
        {
            "name": "03_document-similarity",
            "title": "Document Similarity & Search",
            "description": "Embedding-based semantic search and document matching",
            "difficulty": "Intermediate",
            "hours": "4-5"
        }
    ],
    "05_computer-vision": [
        {
            "name": "01_object-detection",
            "title": "Object Detection for Autonomous Vehicles",
            "description": "Real-time object detection using YOLO/R-CNN",
            "difficulty": "Advanced",
            "hours": "6-8"
        },
        {
            "name": "02_image-classification",
            "title": "Image Classification with Transfer Learning",
            "description": "Production-ready image classification using pre-trained models",
            "difficulty": "Intermediate",
            "hours": "4-5"
        }
    ],
    "06_clustering-unsupervised": [
        {
            "name": "01_customer-segmentation",
            "title": "Customer Segmentation",
            "description": "RFM analysis and clustering for targeted marketing",
            "difficulty": "Beginner",
            "hours": "3-4"
        },
        {
            "name": "02_behavioral-anomaly-detection",
            "title": "User Behavior Anomaly Detection",
            "description": "Unsupervised anomaly detection in user activity patterns",
            "difficulty": "Intermediate",
            "hours": "4-5"
        }
    ],
    "07_ml-system-design": [
        {
            "name": "01_end-to-end-pipeline",
            "title": "End-to-End ML Pipeline",
            "description": "Complete ML pipeline with feature store, serving, and monitoring",
            "difficulty": "Advanced",
            "hours": "7-9"
        },
        {
            "name": "02_ab-testing-framework",
            "title": "A/B Testing Framework for ML",
            "description": "Statistical rigorous experimentation for ML models",
            "difficulty": "Advanced",
            "hours": "6-7"
        },
        {
            "name": "03_model-serving-at-scale",
            "title": "Model Serving at Scale",
            "description": "Low-latency, high-throughput model serving infrastructure",
            "difficulty": "Advanced",
            "hours": "6-8"
        }
    ],
    "08_generative-ai-llms": [
        {
            "name": "01_text-to-sql",
            "title": "Text-to-SQL System",
            "description": "Natural language to database queries with LLMs",
            "difficulty": "Advanced",
            "hours": "6-8"
        },
        {
            "name": "02_rag-system",
            "title": "RAG System for Enterprise Knowledge Base",
            "description": "Retrieval-Augmented Generation with vector databases",
            "difficulty": "Advanced",
            "hours": "7-9"
        },
        {
            "name": "03_chatbot-intent-classification",
            "title": "Chatbot with Intent Classification",
            "description": "Multi-turn conversational AI with context management",
            "difficulty": "Advanced",
            "hours": "6-8"
        },
        {
            "name": "04_llm-fine-tuning",
            "title": "Fine-tuning LLMs for Domain Tasks",
            "description": "Efficient fine-tuning with LoRA/QLoRA for code generation",
            "difficulty": "Advanced",
            "hours": "7-9"
        },
        {
            "name": "05_prompt-engineering-framework",
            "title": "Prompt Engineering Framework",
            "description": "Systematic prompt optimization and few-shot learning",
            "difficulty": "Intermediate",
            "hours": "5-6"
        },
        {
            "name": "06_content-moderation",
            "title": "LLM-based Content Moderation",
            "description": "Multi-lingual content safety with performance optimization",
            "difficulty": "Advanced",
            "hours": "6-7"
        },
        {
            "name": "07_structured-data-extraction",
            "title": "Structured Data Extraction",
            "description": "Extract structured data from unstructured text using LLMs",
            "difficulty": "Intermediate",
            "hours": "5-6"
        },
        {
            "name": "08_semantic-search",
            "title": "LLM-powered Semantic Search",
            "description": "Hybrid search with embeddings and re-ranking",
            "difficulty": "Advanced",
            "hours": "6-7"
        },
        {
            "name": "09_multi-agent-system",
            "title": "Multi-Agent LLM System",
            "description": "Agent orchestration with planning, execution, and reflection",
            "difficulty": "Advanced",
            "hours": "8-10"
        },
        {
            "name": "10_llm-evaluation-pipeline",
            "title": "LLM Evaluation & Monitoring",
            "description": "Automated evaluation, cost tracking, and drift detection",
            "difficulty": "Advanced",
            "hours": "6-8"
        }
    ]
}


def create_case_study_structure(base_path: Path, category: str, case_study: Dict):
    """Create folder structure and placeholder files for a case study."""

    # Create main directory
    case_path = base_path / category / case_study["name"]
    case_path.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (case_path / "data").mkdir(exist_ok=True)
    (case_path / "notebooks").mkdir(exist_ok=True)
    (case_path / "src").mkdir(exist_ok=True)
    (case_path / "tests").mkdir(exist_ok=True)

    # Create __init__.py files
    (case_path / "src" / "__init__.py").touch()
    (case_path / "tests" / "__init__.py").touch()

    # Create README.md
    readme_content = f"""# {case_study['title']}

## Overview

**Business Problem**: [To be documented]

**ML Problem**: {case_study['description']}

**Difficulty**: {case_study['difficulty']}

**Time to Complete**: {case_study['hours']} hours

**Key Skills**: [To be documented]

## Quick Start

```bash
# Install dependencies
cd {category}/{case_study['name']}
pip install -r requirements.txt

# Run notebooks
jupyter notebook notebooks/
```

## Structure

```
{case_study['name']}/
├── README.md
├── problem_statement.md
├── solution_approach.md
├── data/                 # Sample data and descriptions
├── notebooks/            # Jupyter notebooks for exploration
├── src/                  # Production Python code
├── tests/                # Unit tests
├── evaluation.md         # Results and metrics
├── trade_offs.md         # Design decisions
├── interview_questions.md
└── requirements.txt
```

## Status

🚧 **Under Development** - This case study is being developed.

For a complete example, see: [Text-to-SQL](../../08_generative-ai-llms/01_text-to-sql/)

## Contributing

Want to help complete this case study? See [CONTRIBUTING.md](../../CONTRIBUTING.md)
"""

    with open(case_path / "README.md", "w") as f:
        f.write(readme_content)

    # Create placeholder files
    placeholder_files = [
        "problem_statement.md",
        "solution_approach.md",
        "evaluation.md",
        "trade_offs.md",
        "interview_questions.md",
        "requirements.txt",
        "data/data_description.md"
    ]

    for file in placeholder_files:
        filepath = case_path / file
        if not filepath.exists():
            filepath.touch()
            with open(filepath, "w") as f:
                f.write(f"# {file.replace('_', ' ').replace('.md', '').title()}\n\n")
                f.write("🚧 **To be documented**\n\n")
                f.write(f"This file is part of the **{case_study['title']}** case study.\n")

    print(f"✓ Created: {category}/{case_study['name']}")


def create_all_case_studies():
    """Create folder structure for all case studies."""

    base_path = Path(__file__).parent

    total = sum(len(studies) for studies in CASE_STUDIES.values())
    current = 0

    print(f"\n🚀 Creating structure for {total} case studies...\n")

    for category, studies in CASE_STUDIES.items():
        print(f"\n📁 {category}")
        print("─" * 50)

        for study in studies:
            create_case_study_structure(base_path, category, study)
            current += 1

        print()

    print(f"\n✅ Successfully created {total} case study structures!\n")
    print("Next steps:")
    print("  1. Review the generated structure")
    print("  2. Complete the placeholder files")
    print("  3. Add sample data and notebooks")
    print("  4. Implement source code")
    print("  5. Write tests")
    print("\n📖 See the Text-to-SQL case study for a complete example.\n")


if __name__ == "__main__":
    create_all_case_studies()
