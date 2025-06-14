from preprocessor import SmartPreprocessor
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print('Loading dataset...')
dataset = pd.read_csv('Training.csv')
X = dataset.drop('prognosis', axis=1)
y = dataset['prognosis']

print('Preprocessing with SmartPreprocessor...')
preprocessor = SmartPreprocessor(feature_selection_method='mutual_info', n_features=80)
X_processed, y_encoded = preprocessor.fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print('Training models...')

# Train RandomForest
rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_accuracy = accuracy_score(y_test, rf.predict(X_test))

# Train SVM
svm = SVC(C=100, kernel='rbf', probability=True, random_state=42)
svm.fit(X_train, y_train)
svm_accuracy = accuracy_score(y_test, svm.predict(X_test))

print(f'RandomForest Accuracy: {rf_accuracy:.4f}')
print(f'SVM Accuracy: {svm_accuracy:.4f}')

# Select best model
best_model = rf if rf_accuracy >= svm_accuracy else svm
best_name = 'RandomForest' if rf_accuracy >= svm_accuracy else 'SVM'

print(f'Best model: {best_name}')

# Save models with correct module context
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
    
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)
    
# Simple SVC for compatibility
simple_svc = SVC(kernel='linear', random_state=42)
simple_svc.fit(X_train, y_train)
with open('svc.pkl', 'wb') as f:
    pickle.dump(simple_svc, f)

print('Models saved successfully with correct module context!')