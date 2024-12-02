import os
import re
from string import punctuation
from collections import OrderedDict
import unidecode
import numpy as np
import logging
import matplotlib.pyplot as plt
import gensim
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from gensim.models import KeyedVectors
import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
import pandas as pd
from numpy.linalg import svd
import sklearn
from sklearn.metrics.pairwise import cosine_similarity as cos 
from numpy import dot
from numpy.linalg import norm
from gensim.models.word2vec import PathLineSentences
from nltk.tokenize import word_tokenize
import nltk
from nltk.tokenize import punkt 
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
import multiprocessing
import gensim.downloader as api
from scipy.linalg import orthogonal_procrustes
from gensim.models import Word2Vec
from gensim.similarities import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from scipy.spatial.distance import jensenshannon
from gensim.similarities import WmdSimilarity
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from collections import Counter
from nltk.corpus import stopwords
from gensim.models import Word2Vec, KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, 
                              ExtraTreesClassifier, BaggingClassifier, VotingClassifier, StackingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler



# Function to apply SMOTE
def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X_train, y_train)

# Function to apply undersampling
def apply_undersampling(X_train, y_train):
    undersampler = RandomUnderSampler(random_state=42)
    return undersampler.fit_resample(X_train, y_train)

# Function to train and evaluate models
def train_and_evaluate(X_train, X_val, y_train, y_val, strategy_name):
    val_results = []

    for name, clf in classifiers.items():
        # Adjust for class weights if strategy is 'Class Weights'
        if strategy_name == 'Class Weights' and hasattr(clf, 'class_weight'):
            clf.set_params(class_weight='balanced')

        # Train the classifier
        clf.fit(X_train, y_train)

        # Evaluate on validation set
        y_val_pred = clf.predict(X_val)
        y_val_proba = clf.predict_proba(X_val)[:, 1] if hasattr(clf, 'predict_proba') else None
        
        # Metrics
        accuracy = accuracy_score(y_val, y_val_pred)
        precision = precision_score(y_val, y_val_pred, pos_label=1, zero_division=0)
        recall = recall_score(y_val, y_val_pred, pos_label=1)
        f1 = f1_score(y_val, y_val_pred, pos_label=1)
        roc_auc = roc_auc_score(y_val, y_val_proba) if y_val_proba is not None else "N/A"

        val_results.append({
            "Strategy": strategy_name,
            "Classifier": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "ROC-AUC": roc_auc
        })

    return pd.DataFrame(val_results)

# Load the feature dataset for training and validation
df = pd.read_csv('final_features4.csv')

# Separate features and target
X = df.drop(columns=['word', 'annotation'])  # Drop 'word' and keep 'annotation' as the target
y = df['annotation']

# Replace `inf` and `-inf` values with NaN
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Handle missing values in the features (Option 1: Imputation)
imputer = SimpleImputer(strategy='mean')  # You can change 'mean' to 'median' or another strategy
X = imputer.fit_transform(X)

# Split the data into training and validation sets (70% train, 30% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, random_state=42),
    "Histogram Gradient Boosting": HistGradientBoostingClassifier(max_iter=200, random_state=42),
    "SVM": SVC(probability=True, max_iter=500, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
    "SGD Classifier": SGDClassifier(max_iter=1000, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=150, random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', random_state=42),
    "MLP": MLPClassifier(max_iter=500, random_state=42),
    
    "Bagging (DecisionTree)": BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42),
    
    "Voting Classifier (Hard Voting)": VotingClassifier(estimators=[
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('svc', SVC(probability=True, random_state=42))
    ], voting='hard'),
    
    "Voting Classifier (Soft Voting)": VotingClassifier(estimators=[
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('svc', SVC(probability=True, random_state=42))
    ], voting='soft'),

    "Stacking Classifier": StackingClassifier(estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=150, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', random_state=42))
    ], final_estimator=LogisticRegression(), cv=5)
}

# Apply strategies and collect validation results
all_results = []

# Strategy 1: SMOTE
X_train_smote, y_train_smote = apply_smote(X_train, y_train)
all_results.append(train_and_evaluate(X_train_smote, X_val, y_train_smote, y_val, "SMOTE"))

# Strategy 2: Undersampling
X_train_under, y_train_under = apply_undersampling(X_train, y_train)
all_results.append(train_and_evaluate(X_train_under, X_val, y_train_under, y_val, "Undersampling"))

# Strategy 3: Class Weights
all_results.append(train_and_evaluate(X_train, X_val, y_train, y_val, "Class Weights"))

# Combine all results into a single DataFrame
val_results_df = pd.concat(all_results, ignore_index=True)

# Save and display validation results
val_results_df.to_csv('validation_results_with_strategies_own_incre_OP.csv', index=False)
print(val_results_df)

# Select the best model based on F1-Score
best_model_row = val_results_df.loc[val_results_df['F1-Score'].idxmax()]
best_model_name = best_model_row['Classifier']
best_model_strategy = best_model_row['Strategy']
best_model = classifiers[best_model_name]

print(f"\nBest model based on F1-Score: {best_model_name} using {best_model_strategy} strategy")

# Evaluate the best model on the test set
test_set_df = pd.read_csv('final_features_test_set6.csv')
X_test = test_set_df.drop(columns=['word', 'annotation'])
y_test = test_set_df['annotation']

# Handle missing values in the test set
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test = imputer.transform(X_test)

# Use the best model strategy for the test set
if best_model_strategy == "SMOTE":
    X_train_final, y_train_final = apply_smote(X_train, y_train)
elif best_model_strategy == "Undersampling":
    X_train_final, y_train_final = apply_undersampling(X_train, y_train)
else:  # Class Weights
    X_train_final, y_train_final = X_train, y_train

# Retrain the best model on the final training set
best_model.fit(X_train_final, y_train_final)

# Evaluate on the test set
y_test_pred = best_model.predict(X_test)
y_test_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None

# Test set metrics
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, pos_label=1, zero_division=0)
test_recall = recall_score(y_test, y_test_pred, pos_label=1)
test_f1 = f1_score(y_test, y_test_pred, pos_label=1)
test_roc_auc = roc_auc_score(y_test, y_test_proba) if y_test_proba is not None else "N/A"

# Save test set results
test_results = pd.DataFrame([{
    "Classifier": best_model_name,
    "Strategy": best_model_strategy,
    "Accuracy": test_accuracy,
    "Precision": test_precision,
    "Recall": test_recall,
    "F1-Score": test_f1,
    "ROC-AUC": test_roc_auc
}])

test_results.to_csv('test_results_best_model_own_incre.csv', index=False)
print("\nTest set results for the best model:")
print(test_results)


#####################################################################################################################################################################################



########################################################################
# Fine tune best model
########################################################################

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import numpy as np
import pandas as pd
import joblib

# Function to evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    print(f"Model: {model.__class__.__name__}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred, pos_label=1))
    print("ROC AUC:", roc_auc_score(y_test, y_proba) if y_proba is not None else "N/A")
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Apply the appropriate data balancing strategy to training data
if best_model_strategy == "SMOTE":
    X_train_final, y_train_final = apply_smote(X_train, y_train)
elif best_model_strategy == "Undersampling":
    X_train_final, y_train_final = apply_undersampling(X_train, y_train)
else:  # Class Weights
    X_train_final, y_train_final = X_train, y_train

# Define parameter grids for models
if best_model_name == "Random Forest":
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
elif best_model_name == "SVM":
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['linear', 'rbf']
    }
elif best_model_name == "Logistic Regression":
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs']
    }
elif best_model_name == "Decision Tree":
    param_grid = {
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 6]
    }
elif best_model_name == "Gradient Boosting":
    param_grid = {
        'n_estimators': [100, 150, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
elif best_model_name == "XGBoost":
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
elif best_model_name == "AdaBoost":
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1]
    }
elif best_model_name == "Extra Trees":
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
elif best_model_name == "MLP":
    param_grid = {
        'hidden_layer_sizes': [(50, 50), (100,), (100, 100)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive']
    }
elif best_model_name == "K-Nearest Neighbors":
    param_grid = {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
else:
    param_grid = {}  # Empty grid if no parameters are defined

# Only perform GridSearchCV if there are parameters to tune
if param_grid:
    grid_search = GridSearchCV(estimator=best_model, param_grid=param_grid, cv=5, scoring='f1', verbose=2, n_jobs=-1)
    grid_search.fit(X_train_final, y_train_final)
    
    # Print best parameters from grid search
    print(f"Best parameters found by GridSearchCV for {best_model_name}:")
    print(grid_search.best_params_)
    
    # Update best_model with the fine-tuned version
    best_model = grid_search.best_estimator_

# Evaluate the fine-tuned model on the validation set
print(f"\nEvaluating the fine-tuned model: {best_model_name}")
evaluate_model(best_model, X_train_final, X_val, y_train_final, y_val)

# Perform cross-validation to ensure robustness
cv_scores = cross_val_score(best_model, X, y, cv=10, scoring='accuracy')
print(f"10-fold cross-validation accuracy for {best_model_name}: {np.mean(cv_scores):.3f}")

# Save the fine-tuned model to a file
joblib.dump(best_model, f'{best_model_name}_fine_tuned_model.pkl')
print(f"Fine-tuned model saved as {best_model_name}_fine_tuned_model.pkl")

# ================= Final Check on Test Set =================

# Load the unseen test set
test_set_df = pd.read_csv('final_features_test_set6.csv')
X_test = test_set_df.drop(columns=['word', 'annotation'])
y_test = test_set_df['annotation']

# Handle any missing values in the test set
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns)  # Ensure feature names are retained

# Evaluate the fine-tuned model on the unseen test set
print(f"\nFinal evaluation on the unseen test set with the fine-tuned {best_model_name}:")
y_test_pred = best_model.predict(X_test)
y_test_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None

# Calculate and print metrics for the test set
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, pos_label=1, zero_division=0)
test_recall = recall_score(y_test, y_test_pred, pos_label=1)
test_f1 = f1_score(y_test, y_test_pred, pos_label=1)
test_roc_auc = roc_auc_score(y_test, y_test_proba) if y_test_proba is not None else "N/A"

print("Test Accuracy:", test_accuracy)
print("Test Precision:", test_precision)
print("Test Recall:", test_recall)
print("Test F1-Score:", test_f1)
print("Test ROC AUC:", test_roc_auc)

# Save test set results
test_results = pd.DataFrame([{
    "Classifier": best_model_name,
    "Strategy": best_model_strategy,
    "Accuracy": test_accuracy,
    "Precision": test_precision,
    "Recall": test_recall,
    "F1-Score": test_f1,
    "ROC-AUC": test_roc_auc
}])

test_results.to_csv('final_test_results_own_incre.csv', index=False)
print("\nTest set results saved as 'final_test_results.csv'")












































