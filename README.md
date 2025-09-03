# Classification-of-Text-Documents-using-Machine-Learning-Algorithms

## Project Description
This project implements a comprehensive text classification system that categorizes news articles into four categories: Technology, Sports, Politics, and Entertainment. The system compares the performance of six different machine learning algorithms and includes extensive analysis of model performance, overfitting detection, and hyperparameter tuning.

## Key Features
- Multi-model Comparison: Evaluates six different ML algorithms on the same dataset
- Comprehensive Preprocessing: Implements text cleaning, tokenization, stopword removal, and stemming
- Overfitting Analysis: Includes training vs test accuracy comparison and learning curves
- Hyperparameter Tuning: Optimizes model parameters using GridSearchCV
- Visualization: Generates detailed visualizations of model performance and confusion matrices

## Techniques Used
### 1. Data Preprocessing
- Text cleaning and normalization
- Tokenization using NLTK
- Stopword removal
- Porter stemming
- Special character and digit removal

### 2. Feature Engineering
- Count Vectorization for text-to-numeric conversion
- N-gram support (configurable)
- TF-IDF transformation (optional)

### 3. Machine Learning Models
- Naive Bayes (MultinomialNB)
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest
- Support Vector Machine (SVM)
- Decision Tree

### 4. Model Evaluation
- Accuracy scoring
- Confusion matrices
- Classification reports
- Cross-validation

### 5. Overfitting Detection
- Training vs test accuracy comparison
- Learning curves for all models
- Overfitting gap calculation
- Regularization techniques

### 6. Hyperparameter Optimization
- GridSearchCV for parameter tuning
- Regularization parameters (C value, penalty)
- Tree depth and estimator count optimization
- Kernel and gamma selection for SVM

### 7. Visualization
- Accuracy comparison bar charts
- Confusion matrix heatmaps
- Learning curves
- Overfitting gap visualization

## Dataset
The project uses a synthetic dataset of news articles labeled into four categories:
- Technology
- Sports
- Politics
- Entertainment
The dataset includes 367 examples with balanced distribution across categories.

## Project Structure
```
text-classification-ml/
│
├── main.py                 # Main implementation script
├── synthetic_text_data.csv # Dataset
└── README.md              # Project documentation
```

## Results
The project generates multiple visualizations:
1. Accuracy comparison across all models
2. Training vs test accuracy comparison
3. Overfitting gap analysis
4. Learning curves for all models
5. Confusion matrices for each model
6. Hyperparameter tuning results

## Applications
This text classification system can be adapted for:
- News categorization
- Content moderation
- Sentiment analysis
- Spam detection
- Topic modeling

## Future Enhancements
Potential improvements include:
- Deep learning approaches (RNNs, Transformers)
- Advanced word embeddings (Word2Vec, GloVe, BERT)
- Additional text preprocessing techniques
- Real-time classification API
- Deployment as a web service

## Dependencies
- Python 3.6+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- nltk

This project demonstrates a complete machine learning workflow from data preprocessing to model evaluation, making it an excellent educational resource for understanding text classification with multiple algorithms.
