# Importing Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download NLTK resources (run once if not already downloaded)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Loading the Dataset
try:
    data = pd.read_csv('synthetic_text_data.csv')
except pd.errors.ParserError:
    # If there's a parsing error, try reading with error handling
    data = pd.read_csv('synthetic_text_data.csv', on_bad_lines='skip')

# Check if we have the expected columns
if 'text' not in data.columns or 'label' not in data.columns:
    # If the CSV structure is different, try to handle it
    data = pd.read_csv('synthetic_text_data.csv', header=None, names=['text', 'label'])

# Check class distribution
print("Class distribution:")
print(data['label'].value_counts())

# Remove classes with insufficient samples (less than 2)
class_counts = data['label'].value_counts()
valid_classes = class_counts[class_counts >= 2].index
data = data[data['label'].isin(valid_classes)]

X = data['text']
y = data['label']

# Check class distribution again
print("Final class distribution:")
print(y.value_counts())

# Text Preprocessing Function
def preprocess_text(text):
    # Check if text is a string
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    try:
        tokens = word_tokenize(text)
    except:
        tokens = text.split()
    
    # Remove stopwords
    try:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    except:
        pass
    
    # Stemming
    try:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
    except:
        pass
    
    return ' '.join(tokens)

# Apply preprocessing
print("Preprocessing text data...")
X_processed = X.apply(preprocess_text)

# Check if we have enough samples for stratified split
if len(y.unique()) < 2 or min(y.value_counts()) < 2:
    # Use regular split without stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.4, random_state=42
    )
else:
    # Use stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.4, random_state=42, stratify=y
    )

# Using CountVectorizer (simpler than TF-IDF for this example)
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Initialize Models
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'KNN': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'Decision Tree': DecisionTreeClassifier()
}

# Train and evaluate models
accuracies = {}
train_accuracies = {}
conf_matrices = {}

print("Training and evaluating models...")
for name, model in models.items():
    try:
        # Train model
        model.fit(X_train_vectorized, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_vectorized)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        accuracies[name] = accuracy
        
        # Calculate training accuracy
        y_train_pred = model.predict(X_train_vectorized)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_accuracies[name] = train_accuracy
        
        # Store confusion matrix
        conf_matrices[name] = confusion_matrix(y_test, y_pred)
        
        print(f'{name} Accuracy: {accuracy * 100:.2f}%')
        print(f'{name} Training Accuracy: {train_accuracy * 100:.2f}%')
        print(f'{name} Overfitting Gap: {(train_accuracy - accuracy) * 100:.2f}%')
        print('-' * 40)
    except Exception as e:
        print(f"Error with {name}: {str(e)}")
        accuracies[name] = 0
        train_accuracies[name] = 0

# 1. Compare Training and Test Performance
plt.figure(figsize=(12, 6))
models_names = list(accuracies.keys())
train_values = list(train_accuracies.values())
test_values = list(accuracies.values())

x = np.arange(len(models_names))
width = 0.35

plt.bar(x - width/2, train_values, width, label='Training Accuracy', color='blue', alpha=0.7)
plt.bar(x + width/2, test_values, width, label='Test Accuracy', color='red', alpha=0.7)

plt.ylabel('Accuracy')
plt.title('Training vs Test Accuracy by Model')
plt.xticks(x, models_names, rotation=45)
plt.legend()
plt.ylim(0, 1)

# Add accuracy values on top of bars
for i, (train_acc, test_acc) in enumerate(zip(train_values, test_values)):
    plt.text(i - width/2, train_acc + 0.01, f'{train_acc:.3f}', ha='center', va='bottom', fontsize=8)
    plt.text(i + width/2, test_acc + 0.01, f'{test_acc:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()

# Plot overfitting gaps
overfitting_gaps = {name: train_accuracies[name] - accuracies[name] for name in accuracies}
plt.figure(figsize=(10, 6))
models_names = list(overfitting_gaps.keys())
gap_values = list(overfitting_gaps.values())

bars = plt.bar(models_names, gap_values, color=['red' if gap > 0.1 else 'orange' if gap > 0.05 else 'green' for gap in gap_values])
plt.ylabel('Overfitting Gap (Training Accuracy - Test Accuracy)')
plt.title('Overfitting Gap by Model')
plt.xticks(rotation=45)

# Add values on top of bars
for bar, gap in zip(bars, gap_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{gap:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# 2. Learning Curves for All Models on the Same Graph
print("Generating learning curves for all models...")
plt.figure(figsize=(12, 8))

# Define colors for each model
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
model_colors = dict(zip(models.keys(), colors))

# Generate learning curves for each model
for i, (name, model) in enumerate(models.items()):
    try:
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_train_vectorized, y_train, cv=3,
            train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1, random_state=42
        )
        
        # Calculate mean and standard deviation
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # Plot learning curve for this model
        color = model_colors[name]
        plt.plot(train_sizes, train_mean, 'o-', color=color, label=f'{name} Training')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color=color)
        plt.plot(train_sizes, test_mean, 's--', color=color, label=f'{name} CV')
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color=color)
        
    except Exception as e:
        print(f"Error generating learning curve for {name}: {str(e)}")

plt.xlabel('Training examples')
plt.ylabel('Accuracy score')
plt.title('Learning Curves for All Models')
plt.legend(loc='best', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Regularization and Hyperparameter Tuning
print("Performing hyperparameter tuning...")

# Define parameter grids for tuning
param_grids = {
    'Logistic Regression': {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    'SVM': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
}

# Perform hyperparameter tuning for selected models
best_models = {}
for name in ['Logistic Regression', 'Random Forest', 'SVM']:
    if name in models:
        print(f"Tuning hyperparameters for {name}...")
        grid_search = GridSearchCV(
            models[name], param_grids[name], cv=3, 
            scoring='accuracy', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train_vectorized, y_train)
        best_models[name] = grid_search.best_estimator_
        print(f"Best parameters for {name}: {grid_search.best_params_}")
        print(f"Best CV score for {name}: {grid_search.best_score_:.3f}")
        
        # Evaluate on test set
        y_pred = grid_search.predict(X_test_vectorized)
        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"Test accuracy for {name}: {test_accuracy:.3f}")
        print("-" * 50)

# Plot accuracy comparison
if accuracies:
    plt.figure(figsize=(10, 6))
    models_names = list(accuracies.keys())
    accuracy_values = list(accuracies.values())

    bars = plt.bar(models_names, accuracy_values, color=['blue', 'orange', 'green', 'red', 'purple', 'brown'])
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)

    # Add accuracy values on top of bars
    for bar, accuracy in zip(bars, accuracy_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{accuracy:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

# Plot confusion matrices for models that worked
if conf_matrices:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    class_labels = np.unique(y_test)

    for idx, (name, cm) in enumerate(conf_matrices.items()):
        if idx < len(axes):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_labels, 
                        yticklabels=class_labels, 
                        ax=axes[idx],
                        annot_kws={"size": 8})
            axes[idx].set_title(f'{name}', fontsize=10)
            axes[idx].set_xlabel('Predicted', fontsize=9)
            axes[idx].set_ylabel('True', fontsize=9)

    # Hide any unused subplots
    for idx in range(len(conf_matrices), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.show()
else:
    print("No confusion matrices to display.")