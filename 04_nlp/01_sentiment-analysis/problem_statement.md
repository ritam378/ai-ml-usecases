# Problem Statement: Sentiment Analysis at Scale

## Overview

Build a production-ready sentiment analysis system capable of processing millions of user reviews, social media posts, or customer feedback to extract sentiment signals at scale. This is a common interview question at companies like Meta, Twitter/X, Google, and e-commerce platforms like Amazon.

## Business Context

You are a Machine Learning Engineer at a large e-commerce platform processing 10 million+ product reviews monthly. The business needs:

1. **Real-time sentiment classification** on new reviews (< 200ms latency)
2. **Multi-class sentiment detection** (Positive, Negative, Neutral)
3. **Confidence scores** for each prediction
4. **Batch processing** for historical data analysis
5. **Explainability** for content moderation decisions

## Problem Requirements

### Functional Requirements

1. **Sentiment Classification**
   - Classify text into Positive, Negative, or Neutral sentiment
   - Support variable text lengths (10-5000 characters)
   - Handle multiple languages (start with English)
   - Return confidence scores for each class

2. **Performance Requirements**
   - Real-time inference: < 200ms P95 latency
   - Batch processing: 10,000+ reviews/minute
   - High accuracy: > 85% on test set
   - Handle traffic spikes (10x normal load)

3. **Data Requirements**
   - Training data: 100K+ labeled reviews
   - Handle imbalanced classes
   - Deal with noisy, informal text (typos, slang, emojis)
   - Support domain-specific language (product reviews)

### Non-Functional Requirements

1. **Scalability**
   - Scale to 1M+ predictions/day
   - Horizontal scaling capability
   - Efficient resource utilization

2. **Reliability**
   - 99.9% uptime SLA
   - Graceful degradation under load
   - Fallback mechanisms

3. **Maintainability**
   - Easy model updates/retraining
   - A/B testing support
   - Monitoring and alerting
   - Model versioning

## Technical Constraints

1. **Latency Budget**: 200ms P95 (includes preprocessing, inference, postprocessing)
2. **Model Size**: < 500MB for deployment
3. **Infrastructure**: Deploy on CPU instances (cost optimization)
4. **Dependencies**: Minimize external dependencies
5. **Memory**: < 4GB RAM per instance

## Input/Output Specification

### Input
```python
{
    "text": "This product exceeded my expectations! Great quality.",
    "metadata": {
        "review_id": "12345",
        "product_id": "ABC789",
        "timestamp": "2024-01-15T10:30:00Z"
    }
}
```

### Output
```python
{
    "sentiment": "positive",
    "confidence": 0.94,
    "probabilities": {
        "positive": 0.94,
        "neutral": 0.04,
        "negative": 0.02
    },
    "processing_time_ms": 145,
    "model_version": "v2.1.0",
    "metadata": {
        "review_id": "12345"
    }
}
```

## Data Challenges

1. **Class Imbalance**
   - Positive reviews: 60%
   - Neutral reviews: 25%
   - Negative reviews: 15%

2. **Text Characteristics**
   - Short texts (< 50 words): 40%
   - Medium texts (50-200 words): 45%
   - Long texts (> 200 words): 15%
   - Contains emojis: 30%
   - Contains sarcasm: 5%

3. **Domain Specificity**
   - Product-specific jargon
   - Comparative statements
   - Multi-aspect reviews (price vs. quality)

## Success Metrics

### Model Metrics
- **Accuracy**: > 85% on test set
- **F1 Score**: > 0.83 (weighted average)
- **Per-class Recall**: > 0.80 for each sentiment class
- **ROC-AUC**: > 0.90 (one-vs-rest)

### System Metrics
- **Latency**: P50 < 100ms, P95 < 200ms, P99 < 500ms
- **Throughput**: > 10,000 predictions/minute
- **Availability**: 99.9% uptime
- **Cost**: < $0.0001 per prediction

### Business Metrics
- **Reduction in manual review time**: 70%
- **Accuracy vs. human labelers**: > 90% agreement
- **Time to insights**: < 1 hour for new products

## Evaluation Scenarios

Your solution will be evaluated on:

1. **Model Design**
   - Choice of architecture (traditional ML vs. deep learning)
   - Feature engineering strategy
   - Handling of edge cases

2. **System Design**
   - Scalability approach
   - Latency optimization
   - Monitoring strategy

3. **Trade-offs**
   - Accuracy vs. latency
   - Model complexity vs. interpretability
   - Cost vs. performance

## Example Test Cases

### Test Case 1: Clear Positive
```
Input: "Amazing product! Best purchase I've made this year. Highly recommend!"
Expected: sentiment="positive", confidence > 0.90
```

### Test Case 2: Clear Negative
```
Input: "Terrible quality. Broke after one use. Complete waste of money."
Expected: sentiment="negative", confidence > 0.90
```

### Test Case 3: Neutral/Mixed
```
Input: "The product works as described. Nothing special, but it does the job."
Expected: sentiment="neutral", confidence > 0.70
```

### Test Case 4: Sarcasm (Challenging)
```
Input: "Oh great, another product that doesn't work. Just what I needed."
Expected: sentiment="negative" (difficult to detect sarcasm)
```

### Test Case 5: Short Text
```
Input: "Good"
Expected: sentiment="positive", but with lower confidence
```

### Test Case 6: Emoji-Heavy
```
Input: "😍😍😍 Love it! 🔥🔥🔥"
Expected: sentiment="positive", confidence > 0.85
```

## Interview Discussion Points

During the interview, be prepared to discuss:

1. **Model Selection**
   - Why choose BERT over simpler models?
   - When to use transfer learning vs. training from scratch?
   - Trade-offs between model families

2. **Feature Engineering**
   - Text preprocessing strategies
   - Handling special characters and emojis
   - N-gram features vs. embeddings

3. **Training Strategy**
   - Handling class imbalance
   - Data augmentation techniques
   - Hyperparameter tuning approach

4. **Deployment**
   - Model serving architecture
   - Caching strategies
   - A/B testing framework

5. **Monitoring**
   - Detecting data drift
   - Model degradation
   - Performance metrics to track

6. **Scaling**
   - Horizontal vs. vertical scaling
   - Batch vs. real-time processing
   - Cost optimization

## Extensions and Follow-ups

Interviewers may ask about:

1. **Multi-label Classification**: Detect multiple aspects (quality, price, service)
2. **Aspect-Based Sentiment**: Extract sentiment for specific product features
3. **Multilingual Support**: Extend to non-English languages
4. **Real-time Learning**: Update model with new data
5. **Explainability**: Highlight words contributing to sentiment
6. **Sarcasm Detection**: Handle irony and sarcasm
7. **Context-Aware Sentiment**: Consider conversation history

## Resources

- **Dataset**: Amazon Product Reviews, Yelp Reviews, IMDB Reviews
- **Pre-trained Models**: BERT, RoBERTa, DistilBERT, ELECTRA
- **Frameworks**: Transformers (Hugging Face), TensorFlow, PyTorch
- **Deployment**: FastAPI, TensorFlow Serving, TorchServe

## Time Expectations

- **45-minute interview**: Focus on model design and key trade-offs
- **System design round**: Architecture, scaling, monitoring
- **Take-home assignment**: Full implementation with 3-5 days
