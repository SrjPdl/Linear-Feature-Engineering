import numpy as np
from utils import create_polynomial_features, plot_fold_errors, fit_polynomial_regression

def estimate_errors_kfold(X, y, degree, k_folds=5, plot_state=""):
    np.random.seed(696969)  
    shuffled_indices = np.random.permutation(len(X)) 
    X = X[shuffled_indices]
    y = y[shuffled_indices]
    n_samples = len(X)
    fold_size = n_samples // k_folds
    train_errors = []
    val_errors = []

    for i in range(k_folds):
        X_val = X[i*fold_size:(i+1)*fold_size]
        y_val = y[i*fold_size:(i+1)*fold_size]
        X_train = np.vstack([X[:i*fold_size], X[(i+1)*fold_size:]])
        y_train = np.concatenate([y[:i*fold_size], y[(i+1)*fold_size:]])

        beta, X_train_poly = fit_polynomial_regression(X_train, y_train, degree)
        X_val_poly = create_polynomial_features(X_val, degree)
        y_val_pred = X_val_poly @ beta
        y_train_pred = X_train_poly @ beta
        error_train = np.mean((y_train - y_train_pred)**2)
        error_val = np.mean((y_val - y_val_pred) ** 2)
        train_errors.append(error_train)
        val_errors.append(error_val)
    
    plot_fold_errors(train_errors, val_errors, plot_state)

    return np.mean(train_errors), np.mean(val_errors)
