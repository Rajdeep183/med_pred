import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif

class SmartPreprocessor:
    def __init__(self, feature_selection_method='mutual_info', n_features=100):
        self.label_encoder = LabelEncoder()
        self.scaler = RobustScaler()  # More robust than StandardScaler
        self.feature_selector = None
        self.feature_selection_method = feature_selection_method
        self.n_features = n_features
        self.selected_features = None
        
    def fit_transform(self, X, y):
        # Encode target variable
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Feature selection
        if self.feature_selection_method == 'chi2':
            self.feature_selector = SelectKBest(chi2, k=self.n_features)
        else:
            self.feature_selector = SelectKBest(mutual_info_classif, k=self.n_features)
            
        X_selected = self.feature_selector.fit_transform(X, y_encoded)
        self.selected_features = X.columns[self.feature_selector.get_support()]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_selected)
        
        return X_scaled, y_encoded
    
    def transform(self, X):
        X_selected = self.feature_selector.transform(X)
        X_scaled = self.scaler.transform(X_selected)
        return X_scaled