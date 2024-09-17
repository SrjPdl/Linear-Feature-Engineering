import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from least_squares import least_squares

def get_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, sep=r"\s{3,}", index_col=False, header=None, engine='python')


def process_data(data_frame: pd.DataFrame):
    X = data_frame.iloc[:, :8]
    y = data_frame.iloc[:, -1]
    return X.values, y.values


def plot_p_vs_R(errors, plot_state):

    degrees = [x[0] for x in errors]
    train_error = [x[1] for x in errors]
    val_error = [x[2] for x in errors]
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, train_error, marker='o', label='Training Error')
    plt.plot(degrees, val_error, marker='x', label='Validation Error')
    plt.xticks(range(min(degrees), max(degrees) + 1))

    plt.xlabel('Polynomial Degree')
    plt.ylabel('Residual Sum of Squares (R)')
    plt.title('Errors vs Polynomial Degree')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./artifacts/p_vs_R_{plot_state}.png")
    plt.close()


def plot_k_fold_errors(train_errors, test_errors):
    ks = range(1, len(train_errors) + 1)
    plt.figure();
    plt.plot(ks, train_errors, label='Train Error')
    plt.plot(ks, test_errors, label='Test Error')
    plt.xlabel('K (Folds)')
    plt.ylabel('Error')

    plt.xticks(ks)  
    plt.grid(True)  
    plt.title('K vs Train/Test Error')
    plt.legend()
    plt.savefig("./artifacts/k_fold_errors.png")
    plt.close()

def plot_fold_errors(train_errors, val_errors, plot_state):
    folds = np.arange(1, len(train_errors) + 1)
    mean_train = np.mean(train_errors)
    var_train = np.var(train_errors)
    mean_val = np.mean(val_errors)
    var_val = np.var(val_errors)

    # Plot train errors
    plt.figure(figsize=(10, 6))
    plt.bar(folds, train_errors, color='blue', label='Train Error')
    plt.axhline(mean_train, color='red', linestyle='--', label=f'Mean Train Error: {mean_train:.4f}')
    plt.text(len(folds) + 0.5, max(train_errors), f'σ² = {var_train:.4f}', fontsize=12, color='red', ha='right')
    plt.xlabel('Fold')
    plt.ylabel('Train Error')
    plt.title('Train Errors across Folds')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./artifacts/train_error_k_fold_{plot_state}.png")
    plt.close()
    
    # Plot validation errors
    plt.figure(figsize=(10, 6))
    plt.bar(folds, val_errors, color='green', label='Validation Error')
    plt.axhline(mean_val, color='orange', linestyle='--', label=f'Mean Validation Error: {mean_val:.4f}')
    plt.text(len(folds) + 0.5, max(val_errors), f'σ² = {var_val:.4f}', fontsize=12, color='orange', ha='right')
    plt.xlabel('Fold')
    plt.ylabel('Validation Error')
    plt.title('Validation Errors across Folds')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./artifacts/val_error_k_fold_{plot_state}.png")
    plt.close()

def create_polynomial_features(X, degree):
    X_poly = np.ones((X.shape[0], 1))  # Bias term
    for d in range(1, degree + 1):
        X_poly = np.hstack([X_poly, X ** d])
    return X_poly

def fit_polynomial_regression(X, y, degree):
    X_poly = np.ones((X.shape[0], 1)) 
    for d in range(1, degree + 1):
        X_poly = np.hstack([X_poly, X ** d])
    
    beta = least_squares(X_poly, y)
    return beta, X_poly

def write_predictions_to_file(predictions, filename):
    with open(filename, 'w') as file:
        for prediction in predictions:
            file.write(f"{prediction}\n")


def plot_features_vs_errors(errors, best_feature_indices):

    feature_indices = [x[0] for x in errors]
    error_values = [x[1] for x in errors]

    plt.figure(figsize=(10, 6))
    plt.bar(feature_indices, error_values, color='b')
    plt.xlabel('Feature Index')
    plt.ylabel('Mean Squared Error')
    plt.title('Feature Index vs. Mean Squared Error')
    plt.grid(True)
    plt.savefig("./artifacts/best_features.png")
    plt.close()