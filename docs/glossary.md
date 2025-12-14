# AI/ML Glossary

A comprehensive glossary of AI/ML terms commonly used in interviews and this repository.

## A

**A/B Testing**: Comparing two versions (A and B) of a model or system to determine which performs better based on statistical significance.

**Accuracy**: The proportion of correct predictions out of all predictions. Formula: (TP + TN) / (TP + TN + FP + FN)

**Activation Function**: Function applied to neuron outputs in neural networks (ReLU, sigmoid, tanh).

**Adam**: Adaptive learning rate optimization algorithm that combines momentum and RMSProp.

**ANN (Approximate Nearest Neighbors)**: Algorithms for finding nearest neighbors efficiently in high-dimensional spaces (FAISS, Annoy).

**AUC-ROC (Area Under ROC Curve)**: Metric measuring classifier performance across all classification thresholds.

**AUC-PR (Area Under Precision-Recall Curve)**: Better than AUC-ROC for imbalanced datasets.

## B

**Backpropagation**: Algorithm for computing gradients in neural networks using chain rule.

**Batch Normalization**: Technique to normalize layer inputs, improving training speed and stability.

**Batch Size**: Number of samples processed before model weights are updated.

**Bagging**: Bootstrap aggregating - ensemble method training models on random data subsets.

**BERT (Bidirectional Encoder Representations from Transformers)**: Pre-trained language model using transformers.

**Bias (ML)**: Systematic error in model predictions. High bias = underfitting.

**Bias (Fairness)**: Systematic unfairness in model predictions across demographic groups.

**Boosting**: Ensemble method training models sequentially, each correcting previous errors (XGBoost, AdaBoost).

## C

**Catastrophic Forgetting**: Neural networks forgetting previously learned information when learning new tasks.

**Categorical Cross-Entropy**: Loss function for multi-class classification.

**Class Imbalance**: Unequal distribution of classes in dataset (e.g., 99% negative, 1% positive).

**Clustering**: Unsupervised learning task of grouping similar data points (K-means, DBSCAN).

**Cold Start Problem**: Challenge of making predictions for new users/items with no historical data.

**Collaborative Filtering**: Recommendation technique based on user-item interactions.

**Confusion Matrix**: Table showing TP, TN, FP, FN for classification evaluation.

**Content-Based Filtering**: Recommendation technique based on item/user features.

**Convergence**: When model training reaches a point where loss stops decreasing significantly.

**Cosine Similarity**: Similarity measure based on angle between vectors, often used for embeddings.

**Cross-Validation**: Model evaluation technique splitting data into K folds for robust performance estimation.

## D

**Data Augmentation**: Creating additional training data through transformations (rotation, flipping, etc.).

**Data Drift**: Change in input data distribution over time.

**Data Leakage**: Information from test set leaking into training, causing overly optimistic performance.

**DBSCAN**: Density-based clustering algorithm that can find arbitrary-shaped clusters.

**Decision Tree**: Tree-based model making predictions through series of if-else decisions.

**Deep Learning**: Machine learning using neural networks with multiple layers.

**Dimensionality Reduction**: Reducing number of features while preserving information (PCA, t-SNE).

**Dropout**: Regularization technique randomly dropping neurons during training.

## E

**Early Stopping**: Stopping training when validation performance stops improving.

**Embedding**: Dense vector representation of discrete objects (words, users, products).

**Ensemble Learning**: Combining multiple models for better performance (bagging, boosting, stacking).

**Epoch**: One complete pass through entire training dataset.

**Evaluation Metrics**: Measures of model performance (accuracy, precision, recall, F1, etc.).

**Explainability**: Ability to understand and interpret model predictions (SHAP, LIME).

## F

**F1 Score**: Harmonic mean of precision and recall. Formula: 2 * (Precision * Recall) / (Precision + Recall)

**False Negative (FN)**: Model predicts negative, but actual is positive (Type II error).

**False Positive (FP)**: Model predicts positive, but actual is negative (Type I error).

**Feature Engineering**: Creating new features from raw data to improve model performance.

**Feature Importance**: Measure of how much each feature contributes to predictions.

**Feature Store**: Centralized repository for storing and serving ML features.

**Few-Shot Learning**: Learning from very few examples (3-5 samples per class).

**Fine-Tuning**: Adapting pre-trained model to specific task by training on task-specific data.

## G

**Generalization**: Model's ability to perform well on unseen data.

**Gradient**: Vector of partial derivatives indicating direction of steepest increase in loss.

**Gradient Boosting**: Ensemble method building models sequentially to correct previous errors (XGBoost, LightGBM).

**Gradient Descent**: Optimization algorithm updating weights in direction of negative gradient.

**Grid Search**: Hyperparameter tuning by exhaustively trying all combinations.

**GRU (Gated Recurrent Unit)**: Simpler alternative to LSTM for sequence modeling.

## H

**Hallucination**: When LLMs generate plausible but factually incorrect information.

**Hyperparameter**: Parameter set before training (learning rate, batch size, etc.).

**Hyperparameter Tuning**: Process of finding optimal hyperparameters (grid search, Bayesian optimization).

## I

**Imbalanced Dataset**: Dataset where class distribution is skewed (e.g., 99:1 ratio).

**Imputation**: Filling missing values in data.

**Inference**: Using trained model to make predictions on new data.

**IoU (Intersection over Union)**: Metric for object detection measuring overlap between predicted and ground truth boxes.

## K

**K-Fold Cross-Validation**: Splitting data into K subsets, training K times with different validation fold.

**K-Means**: Clustering algorithm partitioning data into K clusters.

**K-Nearest Neighbors (KNN)**: Classification/regression based on K closest training examples.

## L

**L1 Regularization (Lasso)**: Penalty proportional to absolute value of weights, promotes sparsity.

**L2 Regularization (Ridge)**: Penalty proportional to square of weights, shrinks weights.

**Label**: Ground truth output value in supervised learning.

**Label Smoothing**: Technique replacing hard targets (0/1) with soft targets (0.1/0.9).

**LangChain**: Framework for developing applications with LLMs.

**Latency**: Time between request and response.

**Learning Rate**: Step size in gradient descent. Too high = unstable, too low = slow.

**LIME (Local Interpretable Model-Agnostic Explanations)**: Technique for explaining individual predictions.

**Logistic Regression**: Linear model for binary classification using sigmoid function.

**Loss Function**: Measure of model error to minimize during training (cross-entropy, MSE).

**LSTM (Long Short-Term Memory)**: RNN architecture handling long-term dependencies.

**LLM (Large Language Model)**: Transformer-based language model trained on massive text (GPT, Claude).

**LoRA (Low-Rank Adaptation)**: Efficient fine-tuning method for large models.

## M

**MAE (Mean Absolute Error)**: Average absolute difference between predictions and actuals.

**MAP (Mean Average Precision)**: Ranking metric averaging precision across different recall levels.

**Matrix Factorization**: Decomposing user-item matrix into latent factors for recommendations.

**MLOps**: Practices for deploying and maintaining ML systems in production.

**MSE (Mean Squared Error)**: Average squared difference between predictions and actuals.

**Multi-Task Learning**: Training single model on multiple related tasks simultaneously.

## N

**NDCG (Normalized Discounted Cumulative Gain)**: Ranking metric considering position and relevance.

**Neural Network**: Model composed of layers of interconnected neurons.

**NLP (Natural Language Processing)**: AI field focused on understanding and generating text.

**Normalization**: Scaling features to standard range (e.g., 0-1 or mean=0, std=1).

## O

**Offline Evaluation**: Evaluating model on held-out test data before deployment.

**Online Learning**: Updating model continuously as new data arrives.

**One-Hot Encoding**: Representing categorical variables as binary vectors.

**Outlier**: Data point significantly different from others.

**Overfitting**: Model performs well on training data but poorly on unseen data (high variance).

## P

**Precision**: Proportion of positive predictions that are correct. Formula: TP / (TP + FP)

**Pre-training**: Training model on large dataset before fine-tuning on specific task.

**Prompt Engineering**: Crafting effective prompts for LLMs to get desired outputs.

**Pruning**: Removing unnecessary weights/neurons from neural network.

## Q

**Quantization**: Reducing precision of model weights (FP32 → INT8) for faster inference.

**Query**: In information retrieval, the user's search or question.

## R

**R² (R-Squared)**: Regression metric indicating proportion of variance explained by model.

**RAG (Retrieval-Augmented Generation)**: LLM technique retrieving relevant documents before generating response.

**Random Forest**: Ensemble of decision trees trained on random data/feature subsets.

**Recall**: Proportion of actual positives correctly identified. Formula: TP / (TP + FN)

**Recommendation System**: System suggesting items to users based on preferences.

**Recurrent Neural Network (RNN)**: Neural network for sequence data with recurrent connections.

**Regularization**: Techniques preventing overfitting (L1, L2, dropout).

**Reinforcement Learning**: Learning through interaction with environment to maximize reward.

**Re-ranking**: Refining initial ranking using more sophisticated model.

**RMSE (Root Mean Squared Error)**: Square root of MSE, in same units as target.

**ROC Curve**: Plot of True Positive Rate vs False Positive Rate at different thresholds.

## S

**Semantic Search**: Search based on meaning rather than keywords, often using embeddings.

**SGD (Stochastic Gradient Descent)**: Gradient descent updating weights on mini-batches.

**SHAP (SHapley Additive exPlanations)**: Game theory-based method for explaining predictions.

**Sigmoid**: Activation function mapping values to (0, 1), used in logistic regression.

**Softmax**: Function converting logits to probability distribution.

**SMOTE (Synthetic Minority Over-sampling Technique)**: Creating synthetic samples for imbalanced data.

**Supervised Learning**: Learning from labeled data (inputs and outputs).

## T

**T-SNE**: Dimensionality reduction for visualization, preserving local structure.

**Tensor**: Multi-dimensional array, generalization of matrices.

**TF-IDF (Term Frequency-Inverse Document Frequency)**: Text feature weighting scheme.

**Transfer Learning**: Using knowledge from one task to improve learning on related task.

**Transformer**: Neural architecture using self-attention, basis for LLMs (BERT, GPT).

**True Negative (TN)**: Model correctly predicts negative class.

**True Positive (TP)**: Model correctly predicts positive class.

## U

**Underfitting**: Model too simple to capture data patterns (high bias).

**Unsupervised Learning**: Learning from unlabeled data (clustering, dimensionality reduction).

## V

**Validation Set**: Data subset for tuning hyperparameters and early stopping.

**Vanishing Gradient**: Gradients becoming very small in deep networks, hampering learning.

**Variance**: Model's sensitivity to fluctuations in training data. High variance = overfitting.

**Vector Database**: Database optimized for storing and querying high-dimensional vectors (Pinecone, Weaviate).

## W

**Weight Decay**: L2 regularization term in loss function.

**Word2Vec**: Technique for learning word embeddings from text.

## X

**XGBoost**: Gradient boosting library optimized for speed and performance.

## Z

**Zero-Shot Learning**: Making predictions on classes not seen during training, often using LLMs.

---

## Interview-Specific Terms

**Business Metric**: Metric measuring business impact (revenue, engagement, satisfaction).

**Cold Start**: Challenge of making predictions for new users/items with no history.

**Concept Drift**: Change in relationship between features and target over time.

**Data Flywheel**: Virtuous cycle where more users → more data → better models → better product → more users.

**Feature Leakage**: Features that won't be available at prediction time (data leakage).

**Model Drift**: Model performance degrading over time due to data/concept drift.

**Multi-Armed Bandit**: Framework for balancing exploration and exploitation.

**Point-in-Time Correctness**: Ensuring training features use only data available at that time.

**Production Model**: Model deployed and serving real users.

**Shadow Mode**: Running new model alongside production model without affecting users, for validation.

**Technical Debt**: Cost of maintaining ML systems (retraining, monitoring, debugging).

---

**Pro Tip**: In interviews, use precise terminology to demonstrate expertise. For example, say "AUC-PR is better than AUC-ROC for imbalanced datasets" rather than just "use AUC."
