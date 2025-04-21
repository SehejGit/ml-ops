from sklearn.datasets import load_wine
import numpy as np
import pandas as pd

def get_wine_data():
    """Load and prepare the wine dataset"""
    wine = load_wine()
    X = wine.data
    y = wine.target
    feature_names = list(wine.feature_names)
    target_names = list(wine.target_names)
    
    return X, y, feature_names, target_names